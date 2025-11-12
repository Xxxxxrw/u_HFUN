import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../..//")
from src.model import common
from src.model import atten
from src.option import args


""""
    Args:输入输出特征图数量、放大尺度因子、输入输出通道不变3*3Conv
"""

def make_model(args, parent=False):
    return MYMODEL(args)

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


class SobelOperator(nn.Module):
    def __init__(self, in_channels):
        super(SobelOperator, self).__init__()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        # sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.sobel_x = nn.Parameter(
            sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))  # [C, 1, 3, 3] for C channels
        # self.sobel_y = nn.Parameter(
        #     sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))  # [C, 1, 3, 3] for C channels

    def forward(self, x):
        sobel_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])  # 对每个通道应用sobel_x
        # sobel_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])  # 对每个通道应用sobel_y
        return sobel_x #, sobel_y


# class LaplacianOperator(nn.Module):
#     def __init__(self, in_channels):
#         super(LaplacianOperator, self).__init__()
#         self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         # 将Laplacian算子扩展为[C, 1, 3, 3]，为每个通道创建独立的卷积核
#         self.laplacian = nn.Parameter(self.laplacian.repeat(in_channels, 1, 1, 1))
#
#     def forward(self, x):
#         laplacian = F.conv2d(x, self.laplacian, padding=1, groups=x.shape[1])  # 对每个通道应用laplacian
#         return laplacian


class Upsampling_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling_Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 定义1x1卷积层
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Sobel和Laplacian算子
        self.sobel_operator = SobelOperator(in_channels)
        # self.laplacian_operator = LaplacianOperator(in_channels // 2)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=out_channels)
        )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1,
        #               groups=out_channels // 2)
        # )
        # self.conv_3 = nn.Sequential(
        #     nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1,
        #               groups=out_channels // 2)
        # )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        # 通过1x1卷积处理
        F_f = self.conv1x1(x)  # F_f[B, C, H, W]

        # # 沿着通道维度拆分为F_X和F_Y
        # F_X, F_Y = torch.chunk(F_f, 2, dim=1)  # F_X, F_Y[B, C//2, H, W]

        # 处理F_X，使用Sobel算子提取高频特征
        sobel_x = self.sobel_operator(F_f)

        # 使用卷积进行进一步处理
        sobel_x = self.sigmoid(self.conv_1(sobel_x)) * F_f  # [B, C//2, H, W]
        # sobel_y = self.sigmoid(self.conv_2(sobel_y)) * F_X  # [B, C//2, H, W]

        # # 将sobel_x和sobel_y相加融合
        # fused_sobel = sobel_x + sobel_y  # [B, C//2, H, W]
        #
        # # 处理F_Y，使用Laplacian算子提取高频特征
        # laplacian = self.laplacian_operator(F_Y)
        #
        # # 使用卷积进行进一步处理
        # laplacian = self.sigmoid(self.conv_3(laplacian)) * F_Y  # [B, C//2, H, W]
        #
        # # 将两条分支的特征拼接在一起
        # fused_features = torch.cat([fused_sobel, laplacian], dim=1)  # [B, C, H, W]
        #
        # # 最后通过1x1卷积得到最终输出
        # output = self.conv1x1(fused_features) + residual  # [B, C, H, W]
        output = self.conv1x1(sobel_x) + residual  # [B, C, H, W]

        return output

class Low_Module(nn.Module):
    def __init__(self, in_channels, scale, d_f):
        super(Low_Module, self).__init__()
        self.df = d_f
        self.scale = scale

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.depth_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

        self.att = atten.SA(in_channels, d_f)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.depth_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

        self.esa = atten.ESA(in_channels, nn.Conv2d)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.depth_conv1(out)

        out = self.att(out)

        out = self.conv2(out)
        out = self.depth_conv2(out)
        out = self.leaky_relu(out)

        out = self.esa(out)
        output = identity + out

        output = self.conv3(output)

        return output

class Block(nn.Module):
    def __init__(self, in_channels, scale, down_factor):
        super(Block, self).__init__()

        self.upsampling_module = Upsampling_Module(in_channels, in_channels)

        self.low_module = Low_Module(in_channels, scale, down_factor)

    def forward(self, x):
        F_u = self.upsampling_module(x)

        final_output = self.low_module(F_u)

        return final_output

#####Net
class MYMODEL(nn.Module):
    def __init__(self, args):
        super(MYMODEL, self).__init__()

        self.scale = args.scale[0]
        self.num_blocks = 8
        self.feat = args.n_feats
        unf = 24
        down_factor = 0.25

        self.initial_conv = nn.Conv2d(3, self.feat, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.ModuleList(
            [Block(in_channels=self.feat, scale=self.scale, down_factor=down_factor) for _ in range(self.num_blocks)])

        self.reduce_conv_low = nn.Conv2d(self.num_blocks * self.feat, self.feat, kernel_size=1, stride=1, padding=0)

        self.reduce_conv_high = nn.Conv2d((self.num_blocks + 1) * self.feat, 3, kernel_size=1, stride=1, padding=0)
        self.conv_3_f = nn.Conv2d(self.feat, self.feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

        ### upsampling  sub_pixel
        self.sub_pixel1 = nn.Sequential(
            nn.Conv2d(self.feat, self.feat, 1, bias=True),
            nn.Conv2d(self.feat, self.feat * ((self.scale) ** 2), 3, 1, 1, groups=self.feat),
            nn.PixelShuffle((self.scale)),
            nn.PReLU()
        )
        self.sub_pixel2 = nn.Sequential(
            nn.Conv2d(self.feat, self.feat, 1, bias=True),
            nn.Conv2d(self.feat, self.feat * ((self.scale) ** 2), 3, 1, 1, groups=self.feat),
            nn.PixelShuffle((self.scale)),
            nn.PReLU()
        )
        self.sub_pixel3 = nn.Sequential(
            nn.Conv2d(self.feat, self.feat, 1, bias=True),
            nn.Conv2d(self.feat, self.feat * ((self.scale) ** 2), 3, 1, 1, groups=self.feat),
            nn.PixelShuffle((self.scale)),
            nn.PReLU()
        )
        self.upconv = nn.Sequential(
            nn.Conv2d(self.feat * 2, unf, 1, bias=True),
            nn.Conv2d(unf, unf, 3, 1, 1, groups=unf),
        )
        self.att = atten.SA(unf, down_factor)
        self.HRconv = nn.Sequential(
            nn.Conv2d(unf, self.feat, 1, bias=True),
            nn.Conv2d(self.feat, self.feat, 3, 1, 1, groups=self.feat),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)

        coarse_feature = self.initial_conv(x)
        coarse_feature = self.lrelu(coarse_feature)

        final_outputs = []

        feature = coarse_feature
        for block in self.blocks:
            final_output = block(feature)
            final_outputs.append(final_output)
            feature = final_output

        concat_final_output = torch.cat(final_outputs, dim=1)

        # 经过 1x1 卷积将通道数从 6C 降低到 C
        refined_feature = self.reduce_conv_low(concat_final_output)
        fm = self.conv_3_f(refined_feature)

        ff = fm + coarse_feature

        fm = self.sub_pixel1(fm)
        f0 = self.sub_pixel2(coarse_feature)
        ff = self.sub_pixel3(ff)

        out = fm + f0
        out = torch.cat([out, ff], dim=1)
        out = self.upconv(out)
        out = self.lrelu(self.att(out))
        out = self.lrelu(self.HRconv(out)) #32


        final_feature = self.conv_last(out)

        bilinear_upsampled_x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        output = final_feature + bilinear_upsampled_x

        out = self.add_mean(output)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == '__main__':
    from thop import profile

    model = MYMODEL(args)
    scale = 4
    input = torch.randn(1, 3, 1280//scale, 720//scale)
    macs, params = profile(model, inputs=(input,))

    print('params=', params)
    print("MACs=", str(macs / 1e9) + '{}'.format("G"))
    print("MACs=", str(macs / 1e6) + '{}'.format("M"))