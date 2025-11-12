import math

import torch
import torch.nn as nn
from src.option import args

class Add_Dim(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Add_Dim, self).__init__()
        self.conv = nn.Conv2d(in_feat, out_feat, 1)

    def forward(self, x):
        return self.conv(x)

"""
    Args:
        通道丢弃率(default = 0.1)
"""
#drop+copy to all
class tensor_process_copy(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(tensor_process_copy, self).__init__()
        self.dr = drop_rate

    def forward(self, x):
        out = x
        b, c, h, w = x.size()
        ###求方差(返回值var是二维张量)
        var = torch.var(x, dim=(2, 3))
        ###
        #丢弃通道数量（向上取整）
        drop_num = math.ceil(c * self.dr)
        # 获取每个通道的var降序排序索引
        sorted_indices = torch.argsort(var, descending=True, dim=1)
        # 根据排序索引对通道进行排序
        out = torch.gather(out, dim=1, index=sorted_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w))
        # 将第一个通道的数据复制到最后三个通道
        var_max_channel = out[:, 0:1, :, :]  # 提取第一个通道
        out[:, c-drop_num:, :, :] = var_max_channel.expand(-1, drop_num, -1, -1)

        return out

# drop+copy to all
class tensor_process_copy_right(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(tensor_process_copy_right, self).__init__()
        self.dr = drop_rate

    def forward(self, x):
        # 获取每个批次中每个通道的方差
        batch_channel_variances = torch.var(x, dim=(2, 3), unbiased=False)

        # 计算每个批次中方差最大的通道索引
        max_variance_indices = torch.argmax(batch_channel_variances, dim=1)

        # 计算每个批次中方差最小的三个通道索引
        min_variance_indices = torch.argsort(batch_channel_variances, dim=1)[:, :3]

        # 创建一个用于替换的目标张量
        target_tensor = x.clone()

        # 替换每个批次中最小方差的三个通道为方差最大的通道
        for batch_index in range(x.size(0)):
            target_tensor[batch_index, min_variance_indices[batch_index], :, :] = x[batch_index, max_variance_indices[batch_index], :, :]

        return target_tensor

# drop+copy to all
class tensor_process_copy_mean(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(tensor_process_copy_mean, self).__init__()
        self.dr = drop_rate

    def forward(self, x):
        # 获取每一批次的通道内方差
        batch_channel_variances = torch.var(x, dim=(2, 3), unbiased=False)

        # 获取每一批次中方差最小的三个通道索引
        min_variance_indices = torch.argsort(batch_channel_variances, dim=1)[:, :3]

        # 计算每一批次的通道均值
        batch_channel_means = torch.mean(x, dim=1, keepdim=True)

        # 创建一个用于替换的目标张量
        target_tensor = x.clone()

        # 替换每一批次中最小方差的三个通道为该批次的通道均值
        for batch_index in range(x.size(0)):
            target_tensor[batch_index, min_variance_indices[batch_index], :, :] = batch_channel_means[batch_index]

        return target_tensor

"""
    Args:
        通道丢弃率(default = 0.1)
"""
#drop+conv
class tensor_process_conv(nn.Module):
    def __init__(self, drop_rate=0.1, in_feat=args.n_feats):#20
        super(tensor_process_conv, self).__init__()
        self.dr = drop_rate
        self.conv_1 = Add_Dim(in_feat - math.ceil(in_feat * drop_rate), in_feat)

    def forward(self, x):
        out = x
        b, c, h, w = x.size()
        ###求方差(返回值var是二维张量)
        var = torch.var(x, dim=(2, 3))
        ###
        #丢弃通道数量（向上取整）
        drop_num = math.ceil(c * self.dr)
        # 获取每个通道的var降序排序索引
        sorted_indices = torch.argsort(var, descending=True, dim=1)
        # 根据排序索引对通道进行排序
        out = torch.gather(out, dim=1, index=sorted_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w))
        # 将drop的tensor通过Conv升维
        droped_channel = out[:, :c-drop_num, :, :]  # 提取drop后的tensor
        out = self.conv_1(droped_channel)

        return out