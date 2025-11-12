import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# ESA 类定义
from src.model import atten
from src.option import args
from torch.nn import init

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

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

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
        c1_ = self.conv1(x)
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

class high(nn.Module):
    def __init__(self, in_channel, scale=2):
        super(high, self).__init__()
        self.scale = scale

        self.up_down = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.AvgPool2d(scale, stride=scale, padding=0)
        )

        self.sca = atten.SCA(in_channel)
        self.pa = atten.PA(in_channel)
        self.conv_1 = nn.Sequential(
            #nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # out = x
        up = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        high_info = x - self.up_down(up)

        atten_add = self.pa(high_info) + self.sca(high_info)
        atten = self.conv_1(atten_add)
        # high_out = torch.mul(atten, out)

        return atten

class high2(nn.Module):
    def __init__(self, in_channel, scale=2):
        super(high2, self).__init__()
        self.scale = scale

        # self.up_down = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, 1, bias=True),
        #     nn.AvgPool2d(scale, stride=scale, padding=0),
        # )

        self.sca = atten.SCA(in_channel)
        self.pa = atten.PA(in_channel)
        self.conv_1 = nn.Sequential(
            #nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x
        #up = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # high_info = x - self.up_down(up)
        high_info = x

        atten_add = self.pa(high_info) + self.sca(high_info)
        atten = self.conv_1(atten_add)
        #high_out = torch.mul(atten, out)

        return atten

class high3(nn.Module):
    def __init__(self, in_channel, scale=2):
        super(high3, self).__init__()
        self.scale = scale

        self.up_down = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.AvgPool2d(scale, stride=scale, padding=0),
        )

        # self.sca = atten.SCA(in_channel)
        # self.pa = atten.PA(in_channel)
        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, 1, bias=True),
        #     nn.Sigmoid()
        # )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        up = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        high_info = x - self.up_down(up)

        # atten_add = self.pa(high_info) + self.sca(high_info)
        # atten = self.conv_1(atten_add)
        atten = self.sigmoid(high_info)
        high_out = torch.mul(atten, out)

        return high_out

# 定义输入
n_feats = 32
scale = 4
input = torch.randn(1, n_feats, 1280//scale, 720//scale)  # 输入为 1x32x640x360

# # 创建 ESA 模型实例
# esa_model = ESA(n_feats, nn.Conv2d)
# # 计算 FLOPs 和参数个数
# macs, params = profile(esa_model, inputs=(input,))

# 创建 high 模型实例
high_model = high2(n_feats, scale)
# 计算 FLOPs 和参数个数
macs, params = profile(high_model, inputs=(input,))

# # 创建 cbam 模型实例
# cbam_model = CBAMBlock(channel=n_feats)
# # 计算 FLOPs 和参数个数
# macs, params = profile(cbam_model, inputs=(input,))

# # 创建 se 模型实例
# se_model = SELayer(in_channel=n_feats)
# # 计算 FLOPs 和参数个数
# macs, params = profile(se_model, inputs=(input,))

print('params =', params)
print('MACs =', macs / 1e9, 'G')
