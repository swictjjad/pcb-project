# -*- coding: utf-8 -*-
"""
偏振光-RGB多模态融合模块（探索性工作）

⚠️ 声明：本模块为探索性工作，尚未经过实际偏振相机采集验证。
当前实现基于偏振光学的理论推导，实际效果需要配套偏振片硬件验证后确认。

核心思想：在LED光源前加一片线性偏振片（成本<10元），
利用金属表面的偏振特性增强缺陷对比度：
- PCB铜箔表面是镜面反射 → 偏振光下呈现强偏振特性
- 缺陷区域（氧化、凹陷）是漫反射 → 偏振特性弱

理论依据：
- Fresnel反射公式：镜面反射光保持偏振态，漫反射光退偏振
- 偏振度 DoP = (I_parallel - I_cross) / (I_parallel + I_cross)
- 金属表面DoP高（>0.6），缺陷区域DoP低（<0.3）

局限性：
1. 未在真实偏振采集环境下验证，DoP对比度可能不如理论预期
2. 偏振片安装角度、光源均匀性等工程细节未解决
3. 双模态采集需要同步或分时采集，增加系统复杂度
4. 当前仅实现了中期融合架构，早期融合与晚期融合未对比

成本：偏振片 < 10元，旋转支架 < 5元，总计 < 15元
验证状态：仅代码实现 + 合成数据测试，未进行物理实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttention(nn.Module):
    """
    模态注意力模块

    对RGB和偏振两个模态的特征图，学习每个空间位置的模态权重。
    权重由两个模态的特征共同决定，实现自适应融合。

    Args:
        in_channels: 每个模态的通道数
        reduction: 注意力中间层压缩比
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid = max(in_channels // reduction, 1)

        # 共享特征提取
        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 2),  # 2个模态的权重
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat, polar_feat):
        """
        Args:
            rgb_feat: (B, C, H, W) RGB模态特征
            polar_feat: (B, C, H, W) 偏振模态特征

        Returns:
            (B, C, H, W) 融合后的特征
        """
        B, C, H, W = rgb_feat.shape

        # 拼接后计算模态权重
        combined = torch.cat([rgb_feat, polar_feat], dim=1)  # (B, 2C, H, W)
        weights = self.shared_fc(combined)  # (B, 2)

        rgb_weight = weights[:, 0].view(B, 1, 1, 1)
        polar_weight = weights[:, 1].view(B, 1, 1, 1)

        return rgb_weight * rgb_feat + polar_weight * polar_feat


class PolarizationRGBFusion(nn.Module):
    """
    偏振光-RGB多模态融合模块

    支持两种融合策略：
    1. 早期融合（Early Fusion）：在输入层直接拼接RGB+偏振图像
    2. 中期融合（Mid Fusion）：分别提取特征后在中间层融合

    本实现采用中期融合，因为：
    - 保留各模态的独立特征提取能力
    - 模态注意力可以学习"对哪种缺陷，哪个模态更重要"
    - 对缺失偏振输入的情况有更好的鲁棒性

    Args:
        rgb_channels: RGB输入通道数，默认3
        polar_channels: 偏振输入通道数，默认1（偏振度图）
        out_channels: 输出通道数
        fusion_mode: 融合模式 'mid'（中期融合）或 'early'（早期融合）
    """

    def __init__(self, rgb_channels=3, polar_channels=1, out_channels=32,
                 fusion_mode='mid'):
        super().__init__()
        self.fusion_mode = fusion_mode

        if fusion_mode == 'early':
            # 早期融合：直接拼接后用卷积提取特征
            self.early_conv = nn.Sequential(
                nn.Conv2d(rgb_channels + polar_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # 中期融合：分别提取特征 + 模态注意力融合
            self.rgb_stem = nn.Sequential(
                nn.Conv2d(rgb_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.polar_stem = nn.Sequential(
                nn.Conv2d(polar_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.modality_attention = ModalityAttention(out_channels)

            # 融合后的精炼
            self.refine = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, rgb, polar=None):
        """
        Args:
            rgb: (B, 3, H, W) RGB图像
            polar: (B, 1, H, W) 偏振度图，None则退化为纯RGB模式

        Returns:
            (B, out_channels, H, W) 融合特征
        """
        if polar is None:
            # 无偏振输入时退化为纯RGB
            if self.fusion_mode == 'early':
                # 填充0通道
                polar = torch.zeros(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                                    device=rgb.device, dtype=rgb.dtype)
                return self.early_conv(torch.cat([rgb, polar], dim=1))
            else:
                return self.rgb_stem(rgb)

        if self.fusion_mode == 'early':
            return self.early_conv(torch.cat([rgb, polar], dim=1))
        else:
            rgb_feat = self.rgb_stem(rgb)
            polar_feat = self.polar_stem(polar)
            fused = self.modality_attention(rgb_feat, polar_feat)
            return self.refine(fused)


def compute_polarization_degree(images_parallel, images_cross):
    """
    从平行偏振和正交偏振图像计算偏振度图

    偏振度 DoP = (I_parallel - I_cross) / (I_parallel + I_cross)

    物理含义：
    - DoP ≈ 1：强偏振（镜面反射，正常铜箔表面）
    - DoP ≈ 0：弱偏振（漫反射，缺陷区域）

    Args:
        images_parallel: (B, 3, H, W) 平行偏振方向采集的图像
        images_cross: (B, 3, H, W) 正交偏振方向采集的图像

    Returns:
        (B, 1, H, W) 偏振度图，范围[0, 1]
    """
    # 转灰度
    parallel_gray = images_parallel.mean(dim=1, keepdim=True)
    cross_gray = images_cross.mean(dim=1, keepdim=True)

    # 偏振度
    dop = (parallel_gray - cross_gray) / (parallel_gray + cross_gray + 1e-7)
    dop = dop.clamp(0, 1)

    return dop
