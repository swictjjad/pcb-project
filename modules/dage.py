# -*- coding: utf-8 -*-
"""
DAGE: Defect-Aware Geometric Enhancement Module
缺陷几何先验嵌入特征增强模块

=== 模块性质声明 ===
本模块是本作品的核心算法贡献。与CBAM的本质区别不在于"拆分注意力路径"，
而在于将PCB缺陷的物理几何先验显式编码为可学习的检测核：

1. Gabor滤波器组 → 检测方向性边缘（毛刺、开路）
   物理依据：毛刺沿导线法线方向突出，开路导致横向边缘不连续
   创新点：可学习Gabor核（初始化为物理核，训练中微调），对照实验验证有效性

2. DoG圆环检测核 → 检测圆形缺失（漏孔）
   物理依据：漏孔=焊盘中心的圆形暗区，DoG对圆环边缘响应精确
   注意：使用完整的DoG（Difference of Gaussians），非简化高斯blob

3. 可变形卷积 → 自适应几何形变（鼠咬、杂铜的不规则形状）
   性质说明：可变形卷积(Dai et al., 2017)是已有方法，本作品将其作为
   DAGE的一条路径使用，并非创新点。创新在于"为什么把可变形卷积放在这里"——
   因为鼠咬和杂铜的几何形态不规则，标准卷积的规则采样网格无法贴合。

=== 与CBAM的本质区别 ===
- CBAM问"哪里重要" → 数据驱动的通用注意力，无物理先验
- DAGE问"哪里像漏孔/毛刺/开路" → 物理先验驱动的几何检测，每种核对应一类缺陷

=== 自适应路径权重 ===
三条路径的融合权重不是固定的1:1:1，而是通过通道注意力学习得到。
这使得模型可以根据输入图像自适应地调整各路径的贡献：
- 毛刺多的图像 → Gabor路径权重增大
- 漏孔多的图像 → Ring路径权重增大
- 不规则缺陷多的图像 → Deform路径权重增大

=== 统计显著性 ===
所有"提升X%"的结论必须通过bootstrap检验(p<0.05)才有意义。
参见 utils/statistical_testing.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaborFilterBank(nn.Module):
    """
    可学习Gabor滤波器组

    初始化为标准Gabor核（方向0°/45°/90°/135°），训练中微调参数。
    Gabor核天然适合检测方向性边缘，对毛刺（细长突起）和开路（线宽突变）特别有效。

    物理依据：
    - 毛刺沿导线法线方向突出，Gabor核可捕捉该方向梯度
    - 开路导致导线截断，Gabor核可检测横向边缘不连续

    可学习性验证：
    - learnable=True: 训练中微调Gabor核（默认）
    - learnable=False: 冻结Gabor核（对照实验基线）
    - 对照实验需配合bootstrap统计检验，确认提升非随机波动

    Args:
        in_channels: 输入通道数
        num_orientations: Gabor核方向数，默认4（0°/45°/90°/135°）
        kernel_size: 核大小，默认5
        learnable: 是否可学习（False用于对照实验）
    """

    def __init__(self, in_channels, num_orientations=4, kernel_size=5, learnable=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        self.learnable = learnable

        # 可学习参数：每个方向一个卷积核
        self.weight = nn.Parameter(
            torch.zeros(in_channels * num_orientations, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(in_channels * num_orientations))

        # 用Gabor核初始化
        self._init_gabor_weights()

        # 若不可学习，冻结权重（用于对照实验：fixed Gabor vs learnable Gabor）
        if not learnable:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

    def _init_gabor_weights(self):
        """用标准Gabor核初始化权重"""
        with torch.no_grad():
            for i in range(self.in_channels):
                for j in range(self.num_orientations):
                    theta = j * math.pi / self.num_orientations
                    kernel = self._make_gabor_kernel(theta)
                    idx = i * self.num_orientations + j
                    self.weight[idx, i, :, :] = torch.from_numpy(kernel).float()

    @staticmethod
    def _make_gabor_kernel(theta, sigma=2.0, lambd=4.0, gamma=0.5, psi=0.0, ksize=5):
        """生成单个Gabor核"""
        half = ksize // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gaussian = np.exp(-0.5 * (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / sigma ** 2)
        sinusoid = np.cos(2 * np.pi * x_theta / lambd + psi)
        kernel = gaussian * sinusoid
        kernel = kernel / (np.abs(kernel).sum() + 1e-6)
        return kernel

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — Gabor滤波后的特征（多方向响应取L2范数聚合）
        """
        B, C, H, W = x.shape
        out = F.conv2d(x, self.weight, self.bias, padding=self.kernel_size // 2,
                       groups=1)
        # 输出形状 (B, C*N, H, W) → reshape 为 (B, N, C, H, W) → permute 为 (B, C, N, H, W)
        out = out.view(B, self.num_orientations, C, H, W).permute(0, 2, 1, 3, 4)
        out = torch.norm(out, dim=2)  # 方向不变性聚合 → (B, C, H, W)
        return out


class DoGRingDetector(nn.Module):
    """
    DoG圆环检测核（Difference of Gaussians）

    专门检测圆形/椭圆形缺失（漏孔缺陷）。
    使用完整的DoG（外高斯 - 内高斯），对圆环边缘响应精确。

    物理依据：
    - 漏孔 = 焊盘中心的圆形暗区，DoG核的"中心抑制-边缘兴奋"特性天然匹配
    - 导线 = 线性亮区，DoG核对线性结构响应弱，减少误检

    与简化高斯blob的区别：
    - 高斯blob只检测"暗斑"，对任何暗区都有响应（包括导线间隙）
    - DoG圆环核检测"环形边缘"，只对圆形暗斑的边缘有强响应
    - 对漏孔的特异性更高，误检率更低

    Args:
        in_channels: 输入通道数
        kernel_size: 核大小，默认7
        num_scales: 多尺度DoG核数量，默认3（覆盖不同大小的漏孔）
    """

    def __init__(self, in_channels, kernel_size=7, num_scales=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_scales = num_scales

        # 多尺度DoG参数：不同sigma比例覆盖不同大小的漏孔
        # sigma_inner控制内高斯（中心抑制区），sigma_outer控制外高斯（边缘兴奋区）
        # 使用 register_buffer 而非 Parameter：物理尺寸不应在训练中更新
        sigma_inner_vals = torch.tensor([0.8, 1.2, 1.8])[:num_scales]
        sigma_outer_vals = torch.tensor([2.0, 3.0, 4.5])[:num_scales]
        self.register_buffer('sigma_inner', sigma_inner_vals)
        self.register_buffer('sigma_outer', sigma_outer_vals)

        # 可学习权重：多尺度DoG核
        self.weight = nn.Parameter(
            torch.zeros(in_channels * num_scales, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(in_channels * num_scales))

        self._init_dog_weights()

    def _init_dog_weights(self):
        """用DoG圆环核初始化"""
        with torch.no_grad():
            for i in range(self.in_channels):
                for s in range(self.num_scales):
                    kernel = self._make_dog_kernel(
                        self.sigma_inner[s].item(),
                        self.sigma_outer[s].item()
                    )
                    idx = i * self.num_scales + s
                    self.weight[idx, i, :, :] = torch.from_numpy(kernel).float()

    @staticmethod
    def _make_dog_kernel(sigma_inner, sigma_outer, ksize=7):
        """
        生成DoG（Difference of Gaussians）圆环核

        DoG = 外高斯 - 内高斯 → 中心抑制、边缘兴奋的圆环
        对圆形暗斑的边缘有精确响应，对线性结构响应弱。
        """
        half = ksize // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        r_sq = x ** 2 + y ** 2

        gauss_inner = np.exp(-r_sq / (2 * sigma_inner ** 2))
        gauss_outer = np.exp(-r_sq / (2 * sigma_outer ** 2))

        # DoG = 外高斯 - 内高斯 → 圆环形状
        kernel = gauss_outer - gauss_inner

        # 归一化：使正负部分分别归一化，保持圆环特性
        pos_sum = kernel[kernel > 0].sum()
        neg_sum = np.abs(kernel[kernel < 0].sum())
        if pos_sum > 0 and neg_sum > 0:
            # 平衡正负响应，使圆环核的兴奋和抑制等强
            scale = min(pos_sum, neg_sum)
            kernel = kernel / (scale + 1e-6)
        else:
            kernel = kernel / (np.abs(kernel).sum() + 1e-6)

        return kernel

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — 多尺度DoG圆环检测特征（取最大响应聚合）
        """
        B, C, H, W = x.shape

        # 分组卷积
        out = F.conv2d(x, self.weight, self.bias, padding=self.kernel_size // 2,
                       groups=self.in_channels)

        # 重塑为 (B, C, num_scales, H, W)
        out = out.view(B, C, self.num_scales, H, W)

        # 多尺度取最大响应（尺度不变性）
        out = out.max(dim=2)[0]

        return out


class DefectDeformConv(nn.Module):
    """
    缺陷感知可变形卷积

    性质说明：可变形卷积(Dai et al., 2017)是已有方法，本作品将其作为DAGE的
    一条路径使用。创新不在于可变形卷积本身，而在于"为什么放在这里"——
    鼠咬和杂铜的几何形态不规则，标准卷积的规则采样网格无法贴合缺陷轮廓。

    对于DeepPCB数据集中缺陷形状相对规则的情况，可变形卷积的增益有限，
    但在更复杂的真实产线数据上，其自适应采样能力将更有价值。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，默认3
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # 偏移量预测网络
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 1)
        )

        # 初始化偏移量为0（等价于标准卷积）
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

        # 主卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        offset = self.offset_conv(x)

        try:
            from torchvision.ops import deform_conv2d
            B, C, H, W = x.shape
            mask = torch.ones(B, self.kernel_size * self.kernel_size, H, W,
                              device=x.device, dtype=x.dtype)
            out = deform_conv2d(x, offset, self.conv.weight, self.conv.bias,
                                stride=1, padding=self.kernel_size // 2, mask=mask)
        except ImportError:
            # torchvision 不支持 deform_conv2d，退化为标准卷积
            out = self.conv(x)

        out = self.bn(out)
        return out


class AdaptivePathFusion(nn.Module):
    """
    自适应路径融合模块

    学习三条路径（Gabor/Ring/Deform）的通道级权重，
    使模型可以根据输入图像自适应调整各路径的贡献。

    不同于简单的1x1卷积拼接融合，自适应融合：
    1. 对每条路径独立进行通道注意力（SE-like）
    2. 在通道维度上学习路径间的竞争关系
    3. 输出加权后的融合特征

    这使得：
    - 毛刺多的图像 → Gabor路径权重自动增大
    - 漏孔多的图像 → Ring路径权重自动增大
    - 不规则缺陷多的图像 → Deform路径权重自动增大
    """

    def __init__(self, in_channels, out_channels, num_paths=3, reduction=4):
        super().__init__()
        self.num_paths = num_paths
        mid_channels = max(in_channels // reduction, 1)

        # 每条路径的通道注意力
        self.path_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, in_channels),
                nn.Sigmoid(),
            ) for _ in range(num_paths)
        ])

        # 融合后的精炼
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, *path_features):
        """
        Args:
            path_features: 多条路径的特征图，每个 (B, C, H, W)
        Returns:
            (B, out_channels, H, W) 融合特征
        """
        assert len(path_features) == self.num_paths

        B, C, H, W = path_features[0].shape

        # 对每条路径施加通道注意力
        gated_features = []
        for i, feat in enumerate(path_features):
            gate = self.path_gates[i](feat)  # (B, C)
            gate = gate.view(B, C, 1, 1)
            gated_features.append(feat * gate)

        # 逐元素求和融合（而非拼接，更参数高效）
        fused = sum(gated_features)

        # 精炼
        out = self.refine(fused)
        return out


class DAGE(nn.Module):
    """
    DAGE: Defect-Aware Geometric Enhancement Module
    缺陷几何先验嵌入特征增强模块

    将Gabor滤波器组（方向边缘）、DoG圆环检测核（圆形缺失）、可变形卷积（不规则形状）
    三条路径通过自适应融合机制组合，针对PCB缺陷的几何先验进行特征增强。

    === 创新点 ===
    1. 物理先验驱动的检测核设计：每种核对应一类缺陷的几何特征
    2. 可学习Gabor核：初始化为物理核，训练中微调（对照实验验证有效性）
    3. 多尺度DoG圆环检测：覆盖不同大小的漏孔
    4. 自适应路径融合：模型自动学习各路径的权重

    === 非创新点（诚实标注）===
    - 可变形卷积：Dai et al., 2017的已有方法，本作品作为DAGE的一条路径使用
    - P2检测头：目标检测领域应对小目标的标准做法

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（默认与输入相同）
        num_gabor_orientations: Gabor核方向数
        num_dog_scales: DoG圆环核尺度数
        reduction: 通道压缩比
        learnable_gabor: Gabor核是否可学习（False用于对照实验）
    """

    def __init__(self, in_channels, out_channels=None, num_gabor_orientations=4,
                 num_dog_scales=3, reduction=4, learnable_gabor=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 路径1: Gabor滤波器组 → 方向性边缘（毛刺、开路）
        self.gabor_path = GaborFilterBank(in_channels, num_gabor_orientations,
                                           kernel_size=5, learnable=learnable_gabor)

        # 路径2: DoG圆环检测核 → 圆形缺失（漏孔）
        self.ring_path = DoGRingDetector(in_channels, kernel_size=7, num_scales=num_dog_scales)

        # 路径3: 可变形卷积 → 不规则形状（鼠咬、杂铜）
        # 注：可变形卷积是已有方法(Dai et al., 2017)，本作品作为DAGE的一条路径使用
        self.deform_path = DefectDeformConv(in_channels, in_channels, kernel_size=3)

        # 自适应路径融合（替代简单的拼接+1x1卷积）
        self.fusion = AdaptivePathFusion(
            in_channels, out_channels, num_paths=3, reduction=reduction
        )

        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels \
            else nn.Identity()

    def forward(self, x):
        # 三条并行路径
        gabor_out = self.gabor_path(x)      # 方向性边缘特征
        ring_out = self.ring_path(x)         # 圆形缺失特征
        deform_out = self.deform_path(x)     # 不规则形状特征

        # 自适应融合（学习路径权重）
        out = self.fusion(gabor_out, ring_out, deform_out)

        # 残差连接
        return out + self.residual(x)

    def get_path_weights(self, x):
        """
        获取三条路径的注意力权重（用于可视化和分析）

        Returns:
            dict: 各路径的通道注意力权重统计
        """
        with torch.no_grad():
            gabor_out = self.gabor_path(x)
            ring_out = self.ring_path(x)
            deform_out = self.deform_path(x)

            weights = {}
            for i, (name, feat) in enumerate([
                ('gabor', gabor_out), ('ring', ring_out), ('deform', deform_out)
            ]):
                gate = self.fusion.path_gates[i](feat)
                weights[name] = {
                    'mean': float(gate.mean()),
                    'std': float(gate.std()),
                    'max': float(gate.max()),
                    'min': float(gate.min()),
                }
            return weights


def register_dage():
    """
    将DAGE模块注册到ultralytics框架中
    必须在创建YOLO模型之前调用此函数。
    """
    import ultralytics.nn.modules as m

    m.DAGE = DAGE
    m.GaborFilterBank = GaborFilterBank
    m.DoGRingDetector = DoGRingDetector
    m.DefectDeformConv = DefectDeformConv
    m.AdaptivePathFusion = AdaptivePathFusion

    import ultralytics.nn
    if not hasattr(ultralytics.nn, 'DAGE'):
        ultralytics.nn.DAGE = DAGE

    return True
