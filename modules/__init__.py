# -*- coding: utf-8 -*-
"""PCB-YOLO 自定义模块

模块性质分类：
- 核心创新：DAGE（缺陷几何先验特征增强）
- 采用已有方法：WIoU（Tong et al., 2024）、可变形卷积（Dai et al., 2017）
- 探索性工作：偏振光融合（未验证）
- 辅助工具：置信度波动检测、Grad-CAM++可视化、根因推测
"""

from .dage import DAGE, GaborFilterBank, DoGRingDetector, DefectDeformConv, AdaptivePathFusion, register_dage
from .wiou import WIoULoss, patch_wiou_loss
from .uncertainty import MCDropoutFluctuationDetector, FluctuationDrivenDecision
from .polarization_fusion import PolarizationRGBFusion, ModalityAttention, compute_polarization_degree
from .explainability import GradCAMPlusPlus, DefectRootCauseAnalyzer, visualize_cam

__all__ = [
    # 核心创新
    'DAGE', 'GaborFilterBank', 'DoGRingDetector', 'DefectDeformConv',
    'AdaptivePathFusion', 'register_dage',
    # 采用已有方法（PCB适配）
    'WIoULoss', 'patch_wiou_loss',
    # 辅助工具
    'MCDropoutFluctuationDetector', 'FluctuationDrivenDecision',
    # 探索性工作（未验证）
    'PolarizationRGBFusion', 'ModalityAttention', 'compute_polarization_degree',
    # 可视化与推测
    'GradCAMPlusPlus', 'DefectRootCauseAnalyzer', 'visualize_cam',
]
