import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import trans

def mean_channels_c(F):
    assert(F.dim() == 4)
    # 对 C 通道维度进行求和，保留 H 和 W 维度
    channel_sum = F.sum(1, keepdim=True)
    # 对 C 通道维度进行平均
    return channel_sum / F.size(1)

def sub_mean_c(F):
    assert (F.dim() == 4)
    F_mean = mean_channels_c(F)  # [B, 1, H, W]

    # 对原始特征图的每个通道减去均值特征图，并保持原始尺寸 [B, C, H, W]
    F_mean_expanded = F_mean.expand_as(F)  # 将 [B, 1, H, W] 扩展为 [B, C, H, W]
    F_diff = F - F_mean_expanded  # 进行减法操作，尺寸保持为 [B, C, H, W]

    return F_diff

def stdv_channels_c(F):
    assert (F.dim() == 4)
    F_mean = mean_channels_c(F)  # [B, 1, H, W]

    # 对原始特征图的每个通道减去均值特征图，并保持原始尺寸 [B, C, H, W]
    F_mean_expanded = F_mean.expand_as(F)  # 将 [B, 1, H, W] 扩展为 [B, C, H, W]
    F_diff = F - F_mean_expanded  # 进行减法操作，尺寸保持为 [B, C, H, W]

    # 计算平方差并对 C 维度求和，得到 [B, 1, H, W]
    F_variance = (F_diff.pow(2)).sum(1, keepdim=True) / F.size(1)  # [B, 1, H, W]

    return F_variance

class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class SCA(nn.Module):#SCA简单通道注意力
    def __init__(self, in_channel):
        super(SCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        out = torch.mul(x, y)
        return out

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class EPA(nn.Module):
    '''EPA is enhancement pixel attention'''
    def __init__(self, nf):

        super(EPA, self).__init__()
        self.lconv = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, dilation=2, padding=2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.lconv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class asy_atten(nn.Module):
    def __init__(self, in_channels):
        super(asy_atten, self).__init__()

        self.depth_conv_5x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=in_channels
        )

        self.depth_conv_1x5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=in_channels
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_3x3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        attn_branch = self.depth_conv_5x1(x)
        attn_branch = self.depth_conv_1x5(attn_branch)

        attn_weights = self.sigmoid(attn_branch)

        feature_branch = self.conv_3x3(x)
        output = feature_branch * attn_weights

        return output

class PIA(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA, self).__init__()
        self.upper_conv1 = nn.Conv2d(nf // 3, nf // 3, 1, bias=True)
        self.mid_bsconv = nn.Sequential(
            nn.Conv2d(nf // 3, nf // 3, 1, bias=True),
            nn.Conv2d(nf // 3, nf // 3, 3, 1, 1, groups=nf // 3)
        )
        self.bott_dconv_3 = nn.Conv2d(in_channels=nf // 3, out_channels=nf // 3, kernel_size=3, padding=2, dilation=2)
        self.conv_1_1 = nn.Conv2d(nf // 3, nf // 3, 1, bias=True)
        self.conv_1_2 = nn.Conv2d(nf // 3, nf // 3, 1, bias=True)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x
        u_out, m_out, b_out = torch.chunk(x, 3, dim=1)
        u_out = self.upper_conv1(u_out)
        m_out = self.mid_bsconv(m_out)
        b_out = self.bott_dconv_3(b_out)
        mid = self.conv_1_1(u_out + m_out)
        bottom = self.conv_1_2(mid + b_out)
        out = torch.cat([u_out, mid, bottom], dim=1)
        out = self.conv_1_last(out)

        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class SA_2(nn.Module):
    '''SA is sand attention'''
    def __init__(self, nf, factor):
        super(SA_2, self).__init__()
        mid_c = int(nf * factor)

        self.upper_conv1 = nn.Conv2d(nf, mid_c, 1, bias=True)
        self.conv_1_last = nn.Conv2d(mid_c, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x
        mean_residual = sub_mean_c(resiudal)
        fusion_residual = resiudal + mean_residual

        u_out = self.upper_conv1(x)
        out = self.conv_1_last(u_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, fusion_residual)
        return out

class SA_1(nn.Module):
    '''SA is sand attention'''
    def __init__(self, nf, factor):
        super(SA_1, self).__init__()
        mid_c = int(nf * factor)

        self.upper_conv1 = nn.Conv2d(nf, mid_c, 1, bias=True)
        self.conv_1_last = nn.Conv2d(mid_c, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x
        mean_residual = sub_mean_c(resiudal)

        u_out = self.upper_conv1(x)
        out = self.conv_1_last(u_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, mean_residual)
        return out

class SA(nn.Module):
    '''SA is sand attention'''
    def __init__(self, nf, factor):
        super(SA, self).__init__()
        mid_c = int(nf * factor)

        self.upper_conv1 = nn.Conv2d(nf, mid_c, 1, bias=True)
        self.conv_1_last = nn.Conv2d(mid_c, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        u_out = self.upper_conv1(x)
        out = self.conv_1_last(u_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_1(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):
        super(PIA_1, self).__init__()

        self.upper_conv1 = nn.Conv2d(nf, nf, 1, bias=True)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        u_out = self.upper_conv1(x)
        out = self.conv_1_last(u_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_1_lrelu(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_1_lrelu, self).__init__()
        self.upper_conv1 = nn.Conv2d(nf, nf, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        u_out = self.upper_conv1(x)
        u_out = self.lrelu(u_out)
        out = self.conv_1_last(u_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_1_1(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf, down):

        super(PIA_1_1, self).__init__()
        self.upper_conv1 = nn.Conv2d(nf, down, 1, bias=True)
        self.conv_1_last = nn.Conv2d(down, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        u_out = self.upper_conv1(x)
        out = self.conv_1_last(u_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_2(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_2, self).__init__()
        self.mid_bsconv = nn.Sequential(
            nn.Conv2d(nf, nf, 1, bias=True),
            nn.Conv2d(nf, nf, 3, 1, 1, groups=nf)
        )
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        m_out = self.mid_bsconv(x)
        out = self.conv_1_last(m_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_3(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_3, self).__init__()

        self.bott_dconv_3 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=2, dilation=2)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        b_out = self.bott_dconv_3(x)
        out = self.conv_1_last(b_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_3_1(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_3_1, self).__init__()

        self.bott_dconv_3 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, bias=True),
            nn.Conv2d(nf, nf, 3, 1, 1, groups=nf),
            nn.Conv2d(nf, nf, 1, bias=True),
            nn.Conv2d(nf, nf, 3, 1, 1, groups=nf)
        )
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        b_out = self.bott_dconv_3(x)
        out = self.conv_1_last(b_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_3_2(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_3_2, self).__init__()

        self.bott_dconv_3 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, bias=True),
            nn.Conv2d(
                in_channels=nf,
                out_channels=nf,
                kernel_size=5,
                stride=1,
                padding=2,  # 确保输入输出尺寸一致
                groups=nf  # 分组数为输入通道数，实现深度卷积
            )
        )
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x

        b_out = self.bott_dconv_3(x)
        out = self.conv_1_last(b_out)
        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_4(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_4, self).__init__()
        self.upper_conv1 = nn.Conv2d(nf // 2, nf // 2, 1, bias=True)
        self.mid_bsconv = nn.Sequential(
            nn.Conv2d(nf // 2, nf // 2, 1, bias=True),
            nn.Conv2d(nf // 2, nf // 2, 3, 1, 1, groups=nf // 2)
        )
        self.conv_1_1 = nn.Conv2d(nf // 2, nf // 2, 1, bias=True)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x
        u_out, m_out = torch.chunk(x, 2, dim=1)
        u_out = self.upper_conv1(u_out)
        m_out = self.mid_bsconv(m_out)
        mid = self.conv_1_1(u_out + m_out)
        out = torch.cat([u_out, mid], dim=1)
        out = self.conv_1_last(out)

        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_5(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_5, self).__init__()
        self.upper_conv1 = nn.Conv2d(nf // 2, nf // 2, 1, bias=True)
        self.bott_dconv_3 = nn.Conv2d(in_channels=nf // 2, out_channels=nf // 2, kernel_size=3, padding=2, dilation=2)
        self.conv_1_1 = nn.Conv2d(nf // 2, nf // 2, 1, bias=True)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x
        u_out, b_out = torch.chunk(x, 2, dim=1)
        u_out = self.upper_conv1(u_out)
        b_out = self.bott_dconv_3(b_out)
        bottom = self.conv_1_1(u_out + b_out)
        out = torch.cat([u_out, bottom], dim=1)
        out = self.conv_1_last(out)

        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

class PIA_6(nn.Module):
    '''PIA is progressive interactive attention'''
    def __init__(self, nf):

        super(PIA_6, self).__init__()
        self.mid_bsconv = nn.Sequential(
            nn.Conv2d(nf // 2, nf // 2, 1, bias=True),
            nn.Conv2d(nf // 2, nf // 2, 3, 1, 1, groups=nf // 2)
        )
        self.bott_dconv_3 = nn.Conv2d(in_channels=nf // 2, out_channels=nf // 2, kernel_size=3, padding=2, dilation=2)
        self.conv_1_2 = nn.Conv2d(nf // 2, nf // 2, 1, bias=True)
        self.conv_1_last = nn.Conv2d(nf, nf, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resiudal = x
        m_out, b_out = torch.chunk(x, 2, dim=1)
        m_out = self.mid_bsconv(m_out)
        b_out = self.bott_dconv_3(b_out)
        mid = self.conv_1_2(m_out + b_out)
        out = torch.cat([m_out, mid], dim=1)
        out = self.conv_1_last(out)

        atten = self.sigmoid(out)
        out = torch.mul(atten, resiudal)
        return out

#CCA_fun
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

#CCA_fun
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SELayer_me(nn.Module):#SE通道注意力
    def __init__(self, in_channel, reduction=16):#reduction为MLP通道降维和升维的倍数
        super(SELayer_me, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        out = torch.mul(x, y)
        return out

class SELayer(nn.Module):#SE通道注意力
    def __init__(self, in_channel, reduction=16):#reduction为MLP通道降维和升维的倍数
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),#降维
            nn.ReLU(),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()#获取tensor（x）的个数（bstchsize）、深度（c）、高和宽
        y = self.avg_pool(x).view(b, c)#全局平均池化输出四维tensor（b，c，h，w）view（h，w）将矩阵reshape为h与w大小矩阵
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)#expand_as将tensor（y）形状变为与tensor（x）一样，每个通道用同一个值填充；与expand（）类似功能

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class GCT(nn.Module):#使用高斯函数的通道注意力
    def __init__(self, learnable=False):#learnable表示高斯函数的标准差是否可学习，不可学习默认为c=2
        super(GCT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.learnable = learnable
        if self.learnable:
            self.c = nn.Parameter(torch.ones(1) * 0)#随机生成一个1维张量，值为1，*0表示初始化为0
        else:
            self.c = 2

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape#b=1
        attn = self.avg_pool(x).view(b, c)#全局平均池化是四维输出，用view改变形状
        # norm
        attn = self.norm(attn)
        # gaussian function
        if self.learnable:
            attn = self.gaussian(attn, 3 * torch.sigmoid(self.c) + 1)
        else:
            attn = self.gaussian(attn, self.c)
        attn = attn.unsqueeze(-1).unsqueeze(-1)
        out = residual * attn
        return out

    @staticmethod
    def norm(x):
        mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = x.std(dim=-1, keepdim=True).expand_as(x)
        rst = (x - mean) / std
        return rst

    @staticmethod
    def gaussian(x, c):
        return torch.exp(-(x ** 2) / (2 * c))

#相似注意力（myself）
class SimiLayer(nn.Module):
    def __init__(self, in_channel):
        super(SimiLayer, self).__init__()
        channel = in_channel
        self.num = channel
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.conv3_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv3_g = nn.Conv2d(channel * 3, channel, kernel_size=3, stride=1, padding=1, groups=channel)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        out1 = self.conv1(out)
        out1 = self.relu(out1) + x


        out2 = self.relu(self.conv3_1(out)) + x
        out3 = self.relu(self.conv3_2(out2)) + out2
        out_all = torch.cat([out1, out2, out3], dim=1)
        out = trans.shift_channel(out_all, self.num * 3, Concat_num=3)
        out = self.sigmoid(self.conv3_g(out) + x)
        return out