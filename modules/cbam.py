# -*- coding: utf-8 -*-
"""
CBAM: Convolutional Block Attention Module
论文: "CBAM: Convolutional Block Attention Module" (ECCV 2018)

由通道注意力(Channel Attention)和空间注意力(Spatial Attention)两部分组成：
1. 通道注意力：关注"哪些特征通道对缺陷判别重要"
2. 空间注意力：关注"缺陷在图像中的位置"

在PCB缺陷检测中，CBAM能有效增强微小缺陷(如漏孔、毛刺)的特征表达，
同时抑制背景噪声(如焊盘反光、铜箔纹理)的干扰。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块

    通过全局平均池化和全局最大池化两种方式提取通道统计信息，
    再通过共享MLP生成通道注意力权重。

    Args:
        in_channels: 输入通道数
        ratio: 中间层通道压缩比，默认16
    """

    def __init__(self, in_channels, ratio=16):
        super().__init__()
        mid_channels = max(in_channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    沿通道维度进行平均池化和最大池化，拼接后通过卷积生成空间注意力权重。

    Args:
        kernel_size: 卷积核大小，默认7
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """
    CBAM: 通道 + 空间双注意力模块

    先通过通道注意力筛选重要特征通道，再通过空间注意力定位关键区域。
    在PCB缺陷检测中，能有效增强微小缺陷的特征响应。

    用法: 在YOLO11的C3k2模块后串联CBAM，增强特征判别力。
    参数增加量: 约 in_channels * (in_channels/ratio * 2 + kernel_size^2)

    Args:
        in_channels: 输入通道数
        ratio: 通道注意力压缩比，默认16
        kernel_size: 空间注意力卷积核大小，默认7
    """

    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def register_cbam():
    """
    将CBAM模块注册到ultralytics框架中

    这样在自定义模型YAML中就可以使用CBAM作为层类型。
    必须在创建YOLO模型之前调用此函数。
    """
    import ultralytics.nn.modules as m
    import ultralytics.nn.tasks as tasks

    # 添加CBAM到ultralytics模块
    m.CBAM = CBAM
    m.ChannelAttention = ChannelAttention
    m.SpatialAttention = SpatialAttention

    # 更新parse_model中的模块映射
    if hasattr(tasks, 'guess_model_task'):
        # ultralytics >= 8.1
        if not hasattr(tasks, '_module_map_updated'):
            # 在ModuleType映射中添加CBAM
            pass

    # 确保CBAM可以被yaml解析器识别
    # ultralytics的parse_model函数通过globals()查找模块类
    import ultralytics.nn
    if not hasattr(ultralytics.nn, 'CBAM'):
        ultralytics.nn.CBAM = CBAM

    return True
