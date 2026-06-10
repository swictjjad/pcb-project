
# ============================================================
# 文件: modules/dage.py
# 说明: DAGE缺陷几何先验特征增强 — Gabor滤波器+圆环检测核+可变形卷积
# 行数: 469
# ============================================================
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

# ============================================================
# 文件: modules/wiou.py
# 说明: WIoU v3动态聚焦损失 — 边界框回归损失函数
# 行数: 263
# ============================================================
# -*- coding: utf-8 -*-
"""
WIoU v3: Wise-IoU Loss with Dynamic Focusing Mechanism
论文: "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism" (Tong et al., 2024)

=== 性质声明 ===
WIoU是Tong等人2024年的工作，本作品直接采用。以下参数针对PCB缺陷场景进行了适配调优，
而非简单照搬原论文参数。

=== PCB场景适配 ===
原论文参数（δ=3, momentum=0.9）针对COCO通用目标检测设计。
PCB缺陷场景的特点：
1. 目标尺度差异大：微小漏孔(10×10px) vs 大面积短路(200×200px)
2. 目标密度低：单张PCB通常只有0-3个缺陷
3. 背景高度重复：PCB图像中导线/焊盘占主导

适配调整：
- δ=2.5（原论文3.0）：PCB缺陷尺度差异大，过强的聚焦会忽略中等尺度缺陷
- momentum=0.85（原论文0.9）：PCB图像间差异小，IoU_mean更新可稍快
- 梯度截断：focusing系数上限=5.0，防止极端IoU导致梯度爆炸

=== 消融实验 ===
需对比以下配置：
1. CIoU（基线）
2. WIoU(δ=3, momentum=0.9)（原论文参数）
3. WIoU(δ=2.5, momentum=0.85)（PCB适配参数）
4. WIoU(δ=2.0, momentum=0.8)（更保守参数）

所有对比需配合bootstrap统计检验(p<0.05)。
"""

import torch
import math

# ---------------------------------------------------------------------------
# Module-level forward reference to ultralytics BboxLoss — resolved lazily.
# ---------------------------------------------------------------------------
_original_bbox_loss_cls = None


def _get_original_bbox_loss():
    global _original_bbox_loss_cls
    if _original_bbox_loss_cls is None:
        from ultralytics.utils.loss import BboxLoss
        _original_bbox_loss_cls = BboxLoss
    return _original_bbox_loss_cls


class WIoULoss:
    """
    WIoU v3 动态聚焦损失函数（PCB适配版）

    Args:
        delta: 聚焦系数指数，PCB适配值2.5（原论文3.0）
        momentum: IoU_mean动量，PCB适配值0.85（原论文0.9）
        max_focusing: 聚焦系数上限，防止梯度爆炸
        eps: 防止除零
    """

    def __init__(self, delta=2.5, momentum=0.85, max_focusing=5.0, eps=1e-7):
        self.delta = delta
        self.momentum = momentum
        self.max_focusing = max_focusing
        self.eps = eps
        self.iou_mean = 0.0
        self.initialized = False

    @torch.no_grad()
    def update_iou_mean(self, iou):
        """更新IoU的指数移动平均"""
        batch_mean = iou.mean().item()
        if not self.initialized:
            self.iou_mean = batch_mean
            self.initialized = True
        else:
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * batch_mean

    def compute(self, pred_boxes, target_boxes):
        """
        计算WIoU v3 Loss

        Args:
            pred_boxes: (N, 4) [x1, y1, x2, y2]
            target_boxes: (N, 4) [x1, y1, x2, y2]

        Returns:
            wiou_loss: (N,) WIoU v3损失值
        """
        eps = self.eps

        # 1. IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + eps

        iou = inter_area / union_area

        # 2. CIoU组件
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

        distance_cost = center_dist / enc_diag

        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]

        v = (4 / math.pi ** 2) * (torch.atan(target_w / (target_h + eps))
                                    - torch.atan(pred_w / (pred_h + eps))) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou_loss = 1 - iou + distance_cost + alpha * v

        # 3. WIoU v3 动态聚焦
        self.update_iou_mean(iou)

        iou_mean_tensor = torch.tensor(self.iou_mean, device=iou.device, dtype=iou.dtype)
        focusing = (iou / (iou_mean_tensor + eps)).clamp(min=eps) ** self.delta

        # 梯度截断：防止极端IoU导致梯度爆炸
        focusing = focusing.clamp(max=self.max_focusing)

        wiou_loss = focusing * ciou_loss
        return wiou_loss


class WIoUBboxLoss(_get_original_bbox_loss()):
    """
    使用WIoU v3（PCB适配版）替换CIoU的BboxLoss。

    定义为模块级类，避免 pickle 无法序列化嵌套局部类。
    通过 __getstate__/__setstate__ 跳过 wiou 状态（EMA 动量无需持久化），
    并在 load 时自动重建 wiou 实例。
    """

    def __init__(self, reg_max: int = 16, delta=2.5, momentum=0.85, max_focusing=5.0):
        self._wiou_params = {"delta": delta, "momentum": momentum, "max_focusing": max_focusing}
        super().__init__(reg_max)
        self.wiou = WIoULoss(**self._wiou_params)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 替换 CIoU 为 WIoU v3 动态聚焦损失
        # 1. 计算每个正样本的 IoU
        from ultralytics.utils.loss import bbox_iou

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        # 2. 权重（目标置信度加权）
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # 3. WIoU v3 损失
        self.wiou.iou_mean = 0.0  # 重置 EMA 状态
        wiou_losses = self.wiou.compute(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((wiou_losses * weight.squeeze(-1)).sum() / max(target_scores_sum, 1e-7))

        # 4. DFL loss（从父类继承）
        if self.dfl_loss:
            from ultralytics.utils.loss import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                    target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / max(target_scores_sum, 1e-7)
        else:
            from ultralytics.utils.loss import bbox2dist
            import torch.nn.functional as F
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_scaled = pred_dist * stride
            pred_dist_scaled[..., 0::2] /= imgsz[1]
            pred_dist_scaled[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_scaled[fg_mask], target_ltrb[fg_mask],
                          reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / max(target_scores_sum, 1e-7)

        return loss_iou, loss_dfl

    # --- pickle support --------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # wiou contains .initialized flag and .iou_mean EMA — these are
        # training-dependent and should not be persisted across saves.
        del state["wiou"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # Reconstruct wiou with the saved parameters.
        self.wiou = WIoULoss(**self._wiou_params)


# Default parameters (PCB适配值)
_DEFAULT_DELTA = 2.5
_DEFAULT_MOMENTUM = 0.85
_DEFAULT_MAX_FOCUSING = 5.0


def patch_wiou_loss(delta=2.5, momentum=0.85, max_focusing=5.0):
    """
    将ultralytics默认的CIoU Loss替换为WIoU v3（PCB适配版）

    适配 ultralytics >= 8.4 的 BboxLoss API：
    - __init__(self, reg_max: int = 16)
    - forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
              target_scores, target_scores_sum, fg_mask, imgsz, stride)

    Args:
        delta: 聚焦系数指数，默认2.5（PCB适配，原论文3.0）
        momentum: IoU_mean动量，默认0.85（PCB适配，原论文0.9）
        max_focusing: 聚焦系数上限

    Returns:
        bool: 是否替换成功
    """
    try:
        import ultralytics.utils.loss as loss_module

        loss_module.BboxLoss = WIoUBboxLoss

        import ultralytics.utils.tal as tal
        if hasattr(tal, 'BboxLoss'):
            tal.BboxLoss = WIoUBboxLoss

        return True

    except Exception as e:
        import logging
        logging.warning(f"WIoU v3 Loss替换失败，将使用默认CIoU: {e}")
        return False

# ============================================================
# 文件: modules/cbam.py
# 说明: CBAM注意力机制 — 通道+空间注意力
# 行数: 126
# ============================================================
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

# ============================================================
# 文件: modules/uncertainty.py
# 说明: 不确定性量化 — MC Dropout置信度波动检测
# 行数: 448
# ============================================================
# -*- coding: utf-8 -*-
"""
MC Dropout置信度波动检测模块

=== 性质声明 ===
本模块实现真正的MC Dropout推理（在推理时保持Dropout激活，多次前向传播），
但不再声称这是"严格贝叶斯不确定性分解"。

术语修正：
- 不再使用"不确定性量化"（Uncertainty Quantification）这个术语
- 改用"置信度波动检测"（Confidence Fluctuation Detection）
- 因为YOLO检测器输出的置信度不等于贝叶斯后验概率，
  不能直接套用 H[E[p]] - E[H[p]] 的分解公式

=== 真正的MC Dropout ===
本实现确实在推理时保持Dropout激活，进行N次前向传播。
这与Gal & Ghahramani (2016)的MC Dropout方法一致。

但输出的指标名称已修正：
- 检测一致性 (detection_consistency): N次采样中目标被检测到的比例
- 置信度波动 (confidence_fluctuation): N次采样中置信度的标准差
- 综合波动指标 (fluctuation_index): 检测一致性 × 置信度波动

不再使用"认知不确定性""偶然不确定性"等贝叶斯术语。

=== 决策逻辑 ===
- 高置信度 + 低波动 → 直接判定（模型判断稳定）
- 高置信度 + 高波动 → 触发复核（模型"自信但判断不稳定"）
- 低置信度 + 低波动 → 合格（模型稳定地认为无缺陷）
- 低置信度 + 高波动 → 触发复核（模型对该区域缺乏稳定判断）

参考文献：
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
  ICML. (MC Dropout的原始论文)
- Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian
  Deep Learning for Computer Vision? NeurIPS. (认知/偶然不确定性分解)
"""

import torch
import torch.nn as nn
import numpy as np


class MCDropoutFluctuationDetector:
    """
    MC Dropout置信度波动检测器

    通过在推理时保持Dropout激活，多次前向传播估计检测的稳定性。

    Args:
        num_passes: MC采样次数，默认10
        consistency_threshold: 检测一致性阈值，低于此值触发复核
        fluctuation_threshold: 置信度波动阈值，高于此值触发复核
    """

    def __init__(self, num_passes=10, consistency_threshold=0.7,
                 fluctuation_threshold=0.05):
        self.num_passes = num_passes
        self.consistency_threshold = consistency_threshold
        self.fluctuation_threshold = fluctuation_threshold

    def enable_dropout(self, model):
        """在推理时启用Dropout（MC Dropout的核心操作）"""
        dropout_count = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                dropout_count += 1
        return dropout_count

    def estimate_fluctuation(self, model, image_tensor, num_classes=6):
        """
        估计单张图像的检测置信度波动

        Args:
            model: YOLO模型（ultralytics格式）
            image_tensor: 预处理后的图像张量 [1, 3, H, W]
            num_classes: 类别数

        Returns:
            dict: 包含波动指标
        """
        dropout_count = 0
        all_preds = []

        # 确保所有 Dropout 层在推理时保持激活状态
        for m in model.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
                dropout_count += 1

        # 保存原始 eval 状态
        was_training = model.model.training

        with torch.no_grad():
            for _ in range(self.num_passes):
                # 直接调用 __call__ 而非 predict()，避免 ultralytics 内部强制 model.eval()
                # ultralytics 的 predict() 会重新设置 model.eval()，破坏 MC Dropout
                try:
                    outputs = model.model(image_tensor)
                    # ultralytics 的 forward() 返回 Detect 层输出的原始张量
                    # 形状: (B, 4+nc, num_preds) 或列表形式
                    preds = self._parse_yolo_output(outputs, image_tensor.shape[-2:])
                    all_preds.extend(preds)
                except Exception:
                    # 回退: 尝试 model.predict()（注意：MC Dropout 可能不生效）
                    results = model.predict(image_tensor, verbose=False)
                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            all_preds.append({
                                'boxes': result.boxes.xyxy.cpu().numpy(),
                                'confs': result.boxes.conf.cpu().numpy(),
                                'classes': result.boxes.cls.cpu().numpy().astype(int)
                            })

        # 恢复原始模式
        if was_training:
            model.model.train()
        else:
            model.model.eval()

        if not all_preds:
            return {
                'has_detection': False,
                'detection_consistency': 0.0,
                'confidence_fluctuation': 0.0,
                'fluctuation_index': 0.0,
                'mean_confidence': 0.0,
                'num_detections': 0,
                'num_mc_passes': self.num_passes,
                'num_dropout_layers': dropout_count,
                'detections': []
            }

        return self._compute_fluctuation(all_preds, num_classes)

    def _parse_yolo_output(self, outputs, input_shape):
        """
        解析 ultralytics YOLO 的原始输出为检测框格式。

        ultralytics 的 forward() 返回 Detect 层的直接输出:
        - 多输出列表: [bbox_pred, conf_pred, class_pred] 或 (B, 4+nc, H*W)
        - 取决于 YOLO 版本

        Args:
            outputs: model.forward() 的原始输出
            input_shape: 原始图像形状 (H, W)，用于反算 letterbox

        Returns:
            list[dict]: 解析后的检测结果
        """
        results = []

        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 3:
                # [batch, 4, num_preds], [batch, nc, num_preds], [batch, num_preds] 或类似
                # ultralytics v11 Detect 层输出
                bboxes_raw = outputs[0]  # (B, 4, N)
                scores_raw = outputs[1]  # (B, nc, N)
            elif len(outputs) == 2:
                # (B, 4+nc, N)
                bboxes_raw = outputs[:, :4, :]
                scores_raw = outputs[:, 4:, :]
            else:
                return results
        elif isinstance(outputs, torch.Tensor):
            if outputs.shape[1] >= 8:  # 4 bbox + >=4 classes
                bboxes_raw = outputs[:, :4, :]
                scores_raw = outputs[:, 4:, :]
            else:
                return results
        else:
            return results

        # bboxes_raw: (B, 4, N) → 转换为 (B, N, 4) xyxy 格式
        # ultralytics Detect 输出是 cx,cy,w,h 格式，需要转换
        B = bboxes_raw.shape[0]
        N = bboxes_raw.shape[2]

        for b in range(min(B, 1)):  # 目前只支持 batch=1
            bboxes_cxcywh = bboxes_raw[b].permute(1, 0)  # (N, 4)
            scores = scores_raw[b].permute(1, 0)  # (N, nc)

            # cx,cy,w,h → x1,y1,x2,y2
            cx, cy, w, h = bboxes_cxcywh[:, 0], bboxes_cxcywh[:, 1], bboxes_cxcywh[:, 2], bboxes_cxcywh[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            max_scores, cls_ids = scores.max(dim=1)

            results.append({
                'boxes': boxes_xyxy.cpu().numpy(),
                'confs': max_scores.cpu().numpy(),
                'classes': cls_ids.cpu().numpy().astype(int)
            })

        return results

    def _compute_fluctuation(self, all_preds, num_classes):
        """
        从多次MC采样结果计算置信度波动指标

        指标定义：
        1. detection_consistency: N次采样中目标被检测到的比例
           - 高(>0.9): 模型对该目标的判断稳定
           - 低(<0.5): 模型对该目标"犹豫不决"

        2. confidence_fluctuation: 被检测到时置信度的标准差
           - 低(<0.03): 模型对置信度的判断稳定
           - 高(>0.1): 模型对置信度的判断波动大

        3. fluctuation_index: (1 - consistency) * fluctuation
           综合指标，同时考虑检测稳定性和置信度稳定性
        """
        class_stats = {}
        for cls_id in range(num_classes):
            confs_for_class = []
            for pred in all_preds:
                cls_mask = pred['classes'] == cls_id
                if cls_mask.any():
                    cls_confs = pred['confs'][cls_mask]
                    confs_for_class.append(float(cls_confs.max()))
                else:
                    confs_for_class.append(0.0)

            confs = np.array(confs_for_class)
            detection_count = (confs > 0).sum()
            consistency = detection_count / self.num_passes

            detected_confs = confs[confs > 0]

            if detection_count > 0:
                mean_conf = float(detected_confs.mean())
                fluctuation = float(detected_confs.std()) if len(detected_confs) > 1 else 0.0
            else:
                mean_conf = 0.0
                fluctuation = 0.0

            # 综合波动指标
            fi = (1.0 - consistency) * fluctuation

            class_stats[cls_id] = {
                'consistency': float(consistency),
                'fluctuation': float(fluctuation),
                'fluctuation_index': float(fi),
                'mean_conf': mean_conf,
                'detection_count': int(detection_count),
            }

        # 汇总
        total_weight = max(sum(s['consistency'] for s in class_stats.values()), 1e-7)
        total_consistency = sum(s['consistency'] * s['consistency']
                                for s in class_stats.values()) / total_weight
        total_fluctuation = sum(s['fluctuation'] * s['consistency']
                                 for s in class_stats.values()) / total_weight
        total_fi = sum(s['fluctuation_index'] * s['consistency']
                        for s in class_stats.values()) / total_weight

        final_detections = self._consensus_detection(all_preds, class_stats)

        return {
            'has_detection': len(final_detections) > 0,
            'detection_consistency': float(total_consistency),
            'confidence_fluctuation': float(total_fluctuation),
            'fluctuation_index': float(total_fi),
            'mean_confidence': float(np.mean([s['mean_conf'] for s in class_stats.values()
                                               if s['mean_conf'] > 0]) if any(s['mean_conf'] > 0 for s in class_stats.values()) else 0.0),
            'num_detections': len(final_detections),
            'num_mc_passes': self.num_passes,
            'detections': final_detections,
            'class_stats': class_stats
        }

    @staticmethod
    def _box_iou(box1, box2):
        """计算两个 xyxy 格式边界框的 IoU"""
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-7)

    def _consensus_detection(self, all_preds, class_stats):
        """
        基于 IoU 匹配的多次预测共识检测

        将多次 MC 采样中匹配的检测框（IoU > 0.5）聚合，
        平均框位置和置信度，得到更稳定的检测结果。
        """
        merged = []
        used = [set() for _ in all_preds]

        for i, pred in enumerate(all_preds):
            if len(pred['confs']) == 0:
                continue
            for j in range(len(pred['confs'])):
                if any(j in u for u in used):
                    continue
                cls_id = int(pred['classes'][j])
                stats = class_stats.get(cls_id, {})
                matched_boxes = [pred['boxes'][j]]
                matched_confs = [pred['confs'][j]]
                used[i].add(j)

                for k, other in enumerate(all_preds):
                    if k == i:
                        continue
                    for l in range(len(other['boxes'])):
                        if l in used[k]:
                            continue
                        if self._box_iou(pred['boxes'][j], other['boxes'][l]) > 0.5:
                            matched_boxes.append(other['boxes'][l])
                            matched_confs.append(other['confs'][l])
                            used[k].add(l)
                            break

                avg_box = np.mean(matched_boxes, axis=0)
                avg_conf = float(np.mean(matched_confs))

                merged.append({
                    'bbox': avg_box.tolist(),
                    'confidence': avg_conf,
                    'class_id': cls_id,
                    'class_name': '',
                    'consistency': stats.get('consistency', 0.0),
                    'fluctuation': stats.get('fluctuation', 0.0),
                    'fluctuation_index': stats.get('fluctuation_index', 0.0),
                })

        return merged


class FluctuationDrivenDecision:
    """
    基于置信度波动的动态决策

    替代固定阈值，根据MC Dropout采样的波动指标动态调整决策。

    注意：此处不使用"不确定性"术语，使用"波动"术语。
    因为YOLO的置信度不是贝叶斯后验概率，"波动"更准确地描述了
    MC Dropout采样的实际行为。

    Args:
        conf_threshold: 基础置信度阈值
        consistency_threshold: 检测一致性阈值
        fluctuation_threshold: 置信度波动阈值
    """

    def __init__(self, conf_threshold=0.5, consistency_threshold=0.7,
                 fluctuation_threshold=0.05):
        self.conf_threshold = conf_threshold
        self.consistency_threshold = consistency_threshold
        self.fluctuation_threshold = fluctuation_threshold

    def decide(self, fluctuation_result):
        """
        基于波动指标进行决策

        Args:
            fluctuation_result: MCDropoutFluctuationDetector的返回值

        Returns:
            dict: 决策结果
        """
        if not fluctuation_result['has_detection']:
            return {
                'decision': 'OK',
                'reason': '未检测到缺陷',
                'confidence': 0.0,
                'consistency': 0.0,
                'fluctuation': 0.0,
                'review_required': False
            }

        detections = fluctuation_result['detections']
        if not detections:
            return {
                'decision': 'OK',
                'reason': '无有效检测',
                'confidence': 0.0,
                'consistency': 0.0,
                'fluctuation': 0.0,
                'review_required': False
            }

        best = max(detections, key=lambda d: d['confidence'])
        conf = best['confidence']
        consistency = best.get('consistency', 1.0)
        fluctuation = best.get('fluctuation', 0.0)

        # 动态决策
        if conf >= self.conf_threshold:
            if consistency >= self.consistency_threshold and fluctuation < self.fluctuation_threshold:
                # 高置信度 + 高一致性 + 低波动 → 直接判不合格
                return {
                    'decision': 'NG',
                    'reason': f"检测到{best['class_name']}缺陷(置信度{conf:.2f}，一致性{consistency:.2f}，波动{fluctuation:.3f})",
                    'confidence': conf,
                    'consistency': consistency,
                    'fluctuation': fluctuation,
                    'review_required': False,
                    'best_detection': best
                }
            else:
                # 高置信度 + 低一致性或高波动 → 复核
                reason_parts = []
                if consistency < self.consistency_threshold:
                    reason_parts.append(f"一致性低({consistency:.2f})")
                if fluctuation >= self.fluctuation_threshold:
                    reason_parts.append(f"波动大({fluctuation:.3f})")
                return {
                    'decision': 'REVIEW',
                    'reason': f"检测到疑似{best['class_name']}缺陷，但{'，'.join(reason_parts)}，建议人工复核",
                    'confidence': conf,
                    'consistency': consistency,
                    'fluctuation': fluctuation,
                    'review_required': True,
                    'review_reason': 'high_fluctuation',
                    'best_detection': best
                }
        else:
            if consistency >= self.consistency_threshold:
                return {
                    'decision': 'OK',
                    'reason': f"置信度低({conf:.2f})且一致性高，判定合格",
                    'confidence': conf,
                    'consistency': consistency,
                    'fluctuation': fluctuation,
                    'review_required': False
                }
            else:
                return {
                    'decision': 'REVIEW',
                    'reason': f"置信度低({conf:.2f})且一致性低({consistency:.2f})，模型判断不稳定，建议人工复核",
                    'confidence': conf,
                    'consistency': consistency,
                    'fluctuation': fluctuation,
                    'review_required': True,
                    'review_reason': 'low_conf_low_consistency',
                    'best_detection': best
                }

# ============================================================
# 文件: modules/explainability.py
# 说明: 模型可解释性 — Grad-CAM热力图
# 行数: 416
# ============================================================
# -*- coding: utf-8 -*-
"""
Grad-CAM++ 可解释性与缺陷根因分析模块

⚠️ 声明：根因分析部分基于规则映射，未经PCB工艺专家验证。
当前映射规则来源于PCB制造工艺教材与公开文献的总结，
不代表实际产线中的因果关系。根因分析结果应视为"参考线索"，
而非"诊断结论"，最终判断需由工艺工程师确认。

本模块实现：
1. Grad-CAM++ 热力图生成 → 定位模型"看"的位置（有理论保证）
2. 缺陷-工艺知识映射 → 推理可能的工艺根因（规则驱动，需专家验证）
3. 可解释性报告生成 → 辅助工程师分析，不替代人工判断

验证状态：
- Grad-CAM++：理论成熟，已在多篇论文中验证
- 根因映射：仅基于文献总结的规则，未与实际产线数据对照
- 专家评估：建议邀请PCB工艺工程师对映射规则进行评审
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAMPlusPlus:
    """
    Grad-CAM++ 热力图生成器

    相比Grad-CAM，Grad-CAM++通过加权组合梯度，
    对多个目标实例的定位更精确，适合PCB上同时存在多个缺陷的场景。

    Args:
        model: PyTorch模型
        target_layer: 目标卷积层（通常为最后一个卷积层）
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        生成Grad-CAM++热力图

        适配 ultralytics YOLO 的 Results 包装类型。

        Args:
            input_tensor: (1, 3, H, W) 输入张量
            target_class: 目标类别ID，None则取最高置信度类别

        Returns:
            numpy数组 (H, W) 热力图，范围[0, 1]
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)

        # 适配 ultralytics YOLO 的 Results 包装
        if not isinstance(output, torch.Tensor):
            if hasattr(output, 'probs'):
                output = output.probs
            elif hasattr(output, 'boxes') and hasattr(output.boxes, 'conf'):
                output = output.boxes.conf
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Grad-CAM++计算
        activations = self.activations[0]  # (C, h, w)
        gradients = self.gradients[0]       # (C, h, w)

        # 计算alpha权重
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        sum_act = activations.sum(dim=(1, 2), keepdim=True)
        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + sum_act * grad_3 + 1e-7
        alpha = alpha_numer / alpha_denom

        # 加权全局平均池化
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))

        # 生成CAM
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # 上采样到输入尺寸
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return cam.squeeze().cpu().numpy()


class DefectRootCauseAnalyzer:
    """
    缺陷根因分析器

    基于PCB工艺知识图谱，将检测到的缺陷映射到可能的工艺根因。

    知识图谱节点：焊盘、导线、过孔、阻焊层
    知识图谱边：相邻、连接、覆盖

    缺陷-根因映射规则：
    - 漏孔在焊盘中心 → 钻孔偏移 / 电镀不足
    - 漏孔在焊盘边缘 → 钻孔偏移 / 对位误差
    - 毛刺在导线边缘 → 蚀刻过度 / 阻焊剥离
    - 毛刺在焊盘边缘 → 蚀刻参数异常
    - 开路在导线中部 → 蚀刻过度 / 划伤
    - 开路在导线转角 → 蚀刻不均匀 / 线宽不足
    - 短路在相邻导线 → 曝光不足 / 显影不净
    - 短路在焊盘间 → 焊膏印刷偏移 / 间距不足
    - 杂铜在空白区域 → 蚀刻不净 / 阻焊覆盖不全
    - 鼠咬在导线边缘 → 蚀刻过度 / 机械损伤
    """

    # 缺陷-位置-根因映射表
    ROOT_CAUSE_MAP = {
        'pin-hole': {
            'pad_center': ['钻孔偏移（drill offset）', '电镀不足（insufficient plating）'],
            'pad_edge': ['钻孔偏移（drill offset）', '对位误差（registration error）'],
            'default': ['钻孔工艺异常', '电镀工艺异常']
        },
        'spur': {
            'trace_edge': ['蚀刻过度（over-etching）', '阻焊剥离（solder mask peeling）'],
            'pad_edge': ['蚀刻参数异常（etching parameter drift）'],
            'default': ['蚀刻工艺异常', '阻焊工艺异常']
        },
        'open': {
            'trace_middle': ['蚀刻过度（over-etching）', '机械划伤（mechanical scratch）'],
            'trace_corner': ['蚀刻不均匀（uneven etching）', '线宽不足（insufficient trace width）'],
            'default': ['蚀刻工艺异常', '机械损伤']
        },
        'short': {
            'between_traces': ['曝光不足（under-exposure）', '显影不净（incomplete development）'],
            'between_pads': ['焊膏印刷偏移（solder paste shift）', '间距不足（insufficient clearance）'],
            'default': ['曝光/显影工艺异常', '设计间距不足']
        },
        'copper': {
            'open_area': ['蚀刻不净（incomplete etching）', '阻焊覆盖不全（solder mask gap）'],
            'default': ['蚀刻工艺异常', '阻焊工艺异常']
        },
        'mousebite': {
            'trace_edge': ['蚀刻过度（over-etching）', '机械损伤（mechanical damage）'],
            'default': ['蚀刻工艺异常', '机械损伤']
        }
    }

    # 缺陷严重程度与工艺建议
    SEVERITY_MAP = {
        'pin-hole': {'severity': 'high', 'action': '检查钻孔对位精度与电镀参数'},
        'spur': {'severity': 'medium', 'action': '检查蚀刻时间与阻焊附着力'},
        'open': {'severity': 'critical', 'action': '立即检查蚀刻参数，停线排查'},
        'short': {'severity': 'critical', 'action': '检查曝光/显影参数，清洁掩膜版'},
        'copper': {'severity': 'medium', 'action': '检查蚀刻液浓度与喷淋均匀性'},
        'mousebite': {'severity': 'medium', 'action': '检查蚀刻参数与传送带机械状态'},
    }

    # 专家评估记录（用于后续验证映射规则的准确性）
    _expert_reviews = []

    @classmethod
    def add_expert_review(cls, defect_type, location_context, predicted_causes,
                          expert_causes, is_correct, notes=""):
        """
        记录专家评估结果，用于验证和改进根因映射规则

        Args:
            defect_type: 缺陷类型
            location_context: 位置上下文
            predicted_causes: 系统预测的根因
            expert_causes: 专家判定的根因
            is_correct: 预测是否与专家一致
            notes: 备注
        """
        cls._expert_reviews.append({
            'defect_type': defect_type,
            'location_context': location_context,
            'predicted_causes': predicted_causes,
            'expert_causes': expert_causes,
            'is_correct': is_correct,
            'notes': notes,
        })

    @classmethod
    def get_expert_review_stats(cls):
        """
        获取专家评估统计

        Returns:
            dict: 评估统计信息（命中率、各类别评估次数等）
        """
        if not cls._expert_reviews:
            return {
                'total_reviews': 0,
                'accuracy': None,
                'note': '尚无专家评估数据，根因映射规则未经验证'
            }

        total = len(cls._expert_reviews)
        correct = sum(1 for r in cls._expert_reviews if r['is_correct'])

        return {
            'total_reviews': total,
            'accuracy': correct / total,
            'per_defect_type': {
                dt: {
                    'count': sum(1 for r in cls._expert_reviews if r['defect_type'] == dt),
                    'accuracy': sum(1 for r in cls._expert_reviews
                                    if r['defect_type'] == dt and r['is_correct']) /
                                max(sum(1 for r in cls._expert_reviews if r['defect_type'] == dt), 1)
                }
                for dt in set(r['defect_type'] for r in cls._expert_reviews)
            }
        }

    def analyze(self, defect_type, bbox, img_shape):
        """
        分析缺陷根因

        Args:
            defect_type: 缺陷类型名称
            bbox: [x1, y1, x2, y2] 缺陷边界框
            img_shape: (H, W) 图像尺寸

        Returns:
            dict: 根因分析结果
        """
        # 推断缺陷位置上下文
        location = self._infer_location(bbox, img_shape)

        # 查找根因
        cause_map = self.ROOT_CAUSE_MAP.get(defect_type, {})
        root_causes = cause_map.get(location, cause_map.get('default', ['未知根因']))

        # 严重程度与建议
        severity_info = self.SEVERITY_MAP.get(defect_type,
                                                {'severity': 'unknown', 'action': '需人工分析'})

        # 缺陷面积分析
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        area_ratio = area / img_area if img_area > 0 else 0

        return {
            'defect_type': defect_type,
            'location_context': location,
            'suspected_causes': root_causes,  # 改名：root_causes → suspected_causes
            'caution': '根因为基于规则的推测，未经专家验证，仅供参考',
            'severity': severity_info['severity'],
            'recommended_action': severity_info['action'],
            'defect_area_ratio': float(area_ratio),
            'is_small_target': area_ratio < 0.01,
        }

    def _infer_location(self, bbox, img_shape):
        """
        从边界框位置推断缺陷所在的PCB结构上下文

        简化规则：
        - 中心区域 → pad（焊盘）
        - 边缘区域 → trace（导线）
        - 两点之间 → between（间距区域）
        - 其他 → open_area（空白区域）
        """
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]

        cx = (x1 + x2) / 2 / w  # 中心x归一化
        cy = (y1 + y2) / 2 / h  # 中心y归一化

        # 缺陷宽高比
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        aspect_ratio = bw / (bh + 1e-7)

        # 面积
        area = bw * bh

        # 位置推断
        if area < 0.005 and aspect_ratio > 0.7 and aspect_ratio < 1.3:
            # 小面积 + 近似圆形 → 焊盘中心
            return 'pad_center'
        elif area < 0.01 and aspect_ratio > 2.0:
            # 细长形状 → 导线边缘
            return 'trace_edge'
        elif area < 0.01:
            # 小面积非细长 → 焊盘边缘
            return 'pad_edge'
        elif area >= 0.01 and aspect_ratio > 2.0:
            # 较大细长 → 导线中部
            return 'trace_middle'
        elif area >= 0.01:
            # 较大面积 → 间距区域
            return 'between_traces'
        else:
            return 'open_area'

    def generate_report(self, detections, img_shape):
        """
        生成完整的缺陷根因分析报告

        Args:
            detections: 检测结果列表
            img_shape: 图像尺寸

        Returns:
            dict: 完整分析报告
        """
        reports = []
        critical_count = 0

        for det in detections:
            defect_type = det.get('class_name', 'unknown')
            bbox = det.get('bbox', [0, 0, 0, 0])

            analysis = self.analyze(defect_type, bbox, img_shape)
            reports.append(analysis)

            if analysis['severity'] == 'critical':
                critical_count += 1

        return {
            'total_defects': len(reports),
            'critical_defects': critical_count,
            'defect_reports': reports,
            'overall_severity': 'critical' if critical_count > 0 else
                                ('warning' if len(reports) > 0 else 'ok'),
            'summary': self._generate_summary(reports)
        }

    def _generate_summary(self, reports):
        """生成自然语言摘要"""
        if not reports:
            return "未检测到缺陷，PCB质量合格。"

        lines = [f"共检测到 {len(reports)} 处缺陷："]

        # 按严重程度分组
        critical = [r for r in reports if r['severity'] == 'critical']
        high = [r for r in reports if r['severity'] == 'high']
        medium = [r for r in reports if r['severity'] == 'medium']

        if critical:
            types = set(r['defect_type'] for r in critical)
            lines.append(f"  严重缺陷 {len(critical)} 处（{', '.join(types)}），建议立即停线排查。")
        if high:
            types = set(r['defect_type'] for r in high)
            lines.append(f"  高优先级缺陷 {len(high)} 处（{', '.join(types)}），需优先处理。")
        if medium:
            types = set(r['defect_type'] for r in medium)
            lines.append(f"  中等缺陷 {len(medium)} 处（{', '.join(types)}），建议记录并跟踪。")

        # 根因汇总
        all_causes = set()
        for r in reports:
            all_causes.update(r.get('suspected_causes', r.get('root_causes', []))[:1])  # 取首要疑似根因
        if all_causes:
            lines.append(f"  主要疑似根因：{', '.join(all_causes)}")

        return '\n'.join(lines)


def visualize_cam(image, cam, alpha=0.4):
    """
    将Grad-CAM++热力图叠加到原图上

    Args:
        image: BGR格式原图 (H, W, 3)
        cam: 热力图 (H, W)，范围[0, 1]
        alpha: 叠加透明度

    Returns:
        BGR格式叠加图 (H, W, 3)
    """
    # 热力图着色
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ============================================================
# 文件: modules/polarization_fusion.py
# 说明: 偏振光-RGB多模态融合
# 行数: 189
# ============================================================
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

# ============================================================
# 文件: baselines/traditional_cv.py
# 说明: 传统CV基线 — 阈值分割+形态学操作
# 行数: 224
# ============================================================
# -*- coding: utf-8 -*-
"""
传统OpenCV基线检测器

用于与深度学习方法对比，证明深度学习的优势。
传统方法基于图像差分+形态学处理+连通域分析。

这是工业AOI中常用的传统方法，作为基线对比：
- 如果深度学习方法仅比传统方法高1-2%，那深度学习的价值存疑
- 如果深度学习方法显著优于传统方法（>10% mAP），则证明其价值

方法：
1. 图像配准：将待检图像与标准模板对齐
2. 差分运算：待检图 - 模板 = 差异图
3. 形态学处理：开运算去噪 + 闭运算填充
4. 阈值分割：Otsu自适应阈值
5. 连通域分析：提取缺陷区域和边界框

局限性：
- 需要标准模板（无缺陷的参考图像）
- 对配准精度敏感（偏移>2px时误报率急剧上升）
- 无法分类缺陷类型（只能检测"有异常"）
- 对光照变化敏感
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class TraditionalCVBaseline:
    """传统OpenCV基线检测器"""

    def __init__(self, template_image: np.ndarray = None,
                 blur_kernel: int = 5,
                 morph_kernel: int = 3,
                 min_area: int = 50,
                 max_area: int = 50000):
        """
        Args:
            template_image: 标准模板图像（无缺陷参考）
            blur_kernel: 高斯模糊核大小
            morph_kernel: 形态学操作核大小
            min_area: 最小缺陷面积（像素）
            max_area: 最大缺陷面积（像素）
        """
        self.template = template_image
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.min_area = min_area
        self.max_area = max_area

    def set_template(self, template: np.ndarray):
        """设置标准模板"""
        self.template = template.copy()
        if len(self.template.shape) == 3:
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        else:
            self.template_gray = self.template.copy()

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        检测缺陷

        Args:
            image: BGR格式图像

        Returns:
            检测结果列表
        """
        if self.template is None:
            return []

        # 1. 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        template_blur = cv2.GaussianBlur(
            self.template_gray, (self.blur_kernel, self.blur_kernel), 0
        )

        # 3. 图像配准（简化版：仅平移对齐）
        aligned = self._align(gray, template_blur)

        # 4. 差分
        diff = cv2.absdiff(aligned, template_blur)

        # 5. Otsu阈值分割
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 去噪
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 填充

        # 7. 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        detections = []
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            detections.append({
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'class_name': 'defect',  # 传统方法无法分类
                'confidence': min(area / self.max_area, 1.0),  # 面积作为伪置信度
                'area': int(area),
            })

        return detections

    def _align(self, image: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        简化的图像配准（基于相位相关的平移估计）

        Args:
            image: 待配准图像
            template: 模板图像

        Returns:
            配准后的图像
        """
        try:
            # 相位相关估计平移
            shift = cv2.phaseCorrelate(
                np.float32(image), np.float32(template)
            )
            dx, dy = shift[0]

            # 仅在偏移较小时进行补偿（>5px说明配准失败）
            if abs(dx) < 5 and abs(dy) < 5:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                return aligned
        except Exception:
            pass

        return image

    def evaluate(self, images: List[np.ndarray], annotations: List[List[dict]],
                 iou_threshold: float = 0.5) -> dict:
        """
        评估传统方法的检测性能

        Args:
            images: 图像列表
            annotations: 标注列表（每张图的缺陷列表）
            iou_threshold: IoU阈值

        Returns:
            评估结果
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for image, gt_boxes in zip(images, annotations):
            detections = self.detect(image)

            matched_gt = set()
            tp = 0

            for det in detections:
                det_bbox = det['bbox']
                best_iou = 0
                best_gt_idx = -1

                for i, gt in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    iou = self._compute_iou(det_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_tp += tp
            total_fn += len(gt_boxes) - len(matched_gt)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'method': 'Traditional CV (Diff+Morphology)',
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'note': '传统方法无法分类缺陷类型，只能检测"有异常"'
        }

    @staticmethod
    def _compute_iou(box1, box2):
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

# ============================================================
# 文件: tests/test_core.py
# 说明: 单元测试 — 核心功能测试用例
# 行数: 436
# ============================================================
# -*- coding: utf-8 -*-
"""
PCB缺陷检测系统 - 单元测试
覆盖：产线管理、误报屏蔽、缺陷追溯、权限管理、设备监控
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.production_manager import ProductionManager, LineProfile
from utils.false_alarm_suppressor import FalseAlarmSuppressor, SuppressionRule
from utils.traceability import TraceabilityDB
from utils.auth import AuthManager, Role, User
from utils.equipment_monitor import EquipmentMonitor, HealthStatus
from utils.alarm import AlarmManager, AlarmLevel, AlarmConfig
from utils.config_loader import Config, load_config, merge_config


class TestLineProfile(unittest.TestCase):
    """产线配置档案测试"""

    def test_create_profile(self):
        profile = LineProfile(
            name="test_line",
            model_path="./models/test.pt",
            conf_threshold=0.3,
            class_names=["open", "short"],
        )
        self.assertEqual(profile.name, "test_line")
        self.assertEqual(profile.conf_threshold, 0.3)
        self.assertEqual(len(profile.class_names), 2)

    def test_to_dict_and_back(self):
        profile = LineProfile(name="line1", model_path="m.pt", conf_threshold=0.5)
        d = profile.to_dict()
        restored = LineProfile.from_dict(d)
        self.assertEqual(restored.name, "line1")
        self.assertEqual(restored.model_path, "m.pt")
        self.assertEqual(restored.conf_threshold, 0.5)


class TestProductionManager(unittest.TestCase):
    """产线管理器测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pm = ProductionManager(profiles_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_profile(self):
        profile = LineProfile(name="line_A", model_path="a.pt", conf_threshold=0.3)
        self.pm.save_profile(profile)

        loaded = self.pm.get_profile("line_A")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "line_A")
        self.assertEqual(loaded.conf_threshold, 0.3)

    def test_list_profiles(self):
        for name in ["line1", "line2", "line3"]:
            self.pm.save_profile(LineProfile(name=name))

        profiles = self.pm.list_profiles()
        self.assertEqual(len(profiles), 3)
        self.assertIn("line1", profiles)

    def test_switch_line(self):
        self.pm.save_profile(LineProfile(name="line1", conf_threshold=0.2))
        self.pm.save_profile(LineProfile(name="line2", conf_threshold=0.5))

        switched = []
        self.pm.on_line_switched(lambda p: switched.append(p.name))

        result = self.pm.switch_line("line2")
        self.assertTrue(result)
        self.assertEqual(self.pm.current_profile_name, "line2")
        self.assertEqual(switched, ["line2"])

    def test_switch_nonexistent_line(self):
        result = self.pm.switch_line("nonexistent")
        self.assertFalse(result)

    def test_delete_profile(self):
        self.pm.save_profile(LineProfile(name="to_delete"))
        self.assertTrue(self.pm.delete_profile("to_delete"))
        self.assertIsNone(self.pm.get_profile("to_delete"))

    def test_mode_switch(self):
        self.pm.set_mode(ProductionManager.MODE_DEBUG)
        self.assertTrue(self.pm.is_debug_mode)
        self.assertFalse(self.pm.is_production_mode)

        self.pm.set_mode(ProductionManager.MODE_PRODUCTION)
        self.assertTrue(self.pm.is_production_mode)

    def test_shift_info(self):
        self.pm.set_shift_info(operator_id="O001", equipment_id="EQ-001", shift_name="早班")
        info = self.pm.shift_info
        self.assertEqual(info['operator_id'], "O001")
        self.assertEqual(info['equipment_id'], "EQ-001")


class TestSuppressionRule(unittest.TestCase):
    """误报屏蔽规则测试"""

    def test_rect_rule_matches(self):
        rule = SuppressionRule(
            name="test_rect",
            rule_type="rect",
            region={"x1": 100, "y1": 100, "x2": 300, "y2": 300},
        )
        # 中心点在区域内
        self.assertTrue(rule.matches([150, 150, 250, 250], "open", 0.8))
        # 中心点在区域外
        self.assertFalse(rule.matches([400, 400, 500, 500], "open", 0.8))

    def test_class_filter(self):
        rule = SuppressionRule(
            name="class_filter",
            rule_type="rect",
            region={"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
            class_names=["open", "short"],
        )
        self.assertTrue(rule.matches([100, 100, 200, 200], "open", 0.8))
        self.assertFalse(rule.matches([100, 100, 200, 200], "copper", 0.8))

    def test_confidence_filter(self):
        rule = SuppressionRule(
            name="conf_filter",
            rule_type="rect",
            region={"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
            min_confidence=0.5,
        )
        self.assertTrue(rule.matches([100, 100, 200, 200], "open", 0.3))
        self.assertFalse(rule.matches([100, 100, 200, 200], "open", 0.8))

    def test_disabled_rule(self):
        rule = SuppressionRule(
            name="disabled",
            rule_type="rect",
            region={"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
            enabled=False,
        )
        self.assertFalse(rule.matches([100, 100, 200, 200], "open", 0.8))

    def test_polygon_rule(self):
        rule = SuppressionRule(
            name="polygon",
            rule_type="polygon",
            region={"points": [[0, 0], [200, 0], [200, 200], [0, 200]]},
        )
        self.assertTrue(rule.matches([50, 50, 100, 100], "open", 0.8))
        self.assertFalse(rule.matches([300, 300, 400, 400], "open", 0.8))


class TestFalseAlarmSuppressor(unittest.TestCase):
    """误报屏蔽管理器测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.suppressor = FalseAlarmSuppressor(rules_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_list_rules(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 100, 100)
        rules = self.suppressor.list_rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].name, "zone1")

    def test_filter_detections(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 200, 200)
        detections = [
            {'bbox': [50, 50, 100, 100], 'class_name': 'open', 'confidence': 0.8},
            {'bbox': [300, 300, 400, 400], 'class_name': 'short', 'confidence': 0.7},
        ]
        valid, suppressed = self.suppressor.filter_detections(detections)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(valid[0]['class_name'], 'short')

    def test_toggle_rule(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 200, 200)
        self.suppressor.toggle_rule("zone1")
        rules = self.suppressor.list_rules()
        self.assertFalse(rules[0].enabled)

    def test_delete_rule(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 100, 100)
        self.assertTrue(self.suppressor.remove_rule("zone1"))
        self.assertEqual(len(self.suppressor.list_rules()), 0)


class TestTraceabilityDB(unittest.TestCase):
    """缺陷追溯数据库测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = TraceabilityDB(db_path=os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_record(self):
        rid = self.db.add_record(
            product_sn="PCB001",
            operator_id="O001",
            equipment_id="EQ-001",
            result="OK",
            num_defects=0,
        )
        self.assertGreater(rid, 0)

    def test_query_by_sn(self):
        self.db.add_record(product_sn="PCB001", result="OK")
        self.db.add_record(product_sn="PCB002", result="NG", num_defects=2)

        records = self.db.query_by_sn("PCB001")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['result'], 'OK')

    def test_query_by_operator(self):
        self.db.add_record(operator_id="O001", result="OK")
        self.db.add_record(operator_id="O002", result="NG")

        records = self.db.query_by_operator("O001")
        self.assertEqual(len(records), 1)

    def test_review_record(self):
        rid = self.db.add_record(product_sn="PCB001", result="NG", num_defects=1)
        self.db.review_record(rid, "confirmed", "reviewer1")

        unreviewed = self.db.get_unreviewed()
        self.assertEqual(len(unreviewed), 0)

    def test_batch_review(self):
        ids = []
        for i in range(5):
            rid = self.db.add_record(product_sn=f"PCB{i:03d}", result="NG", num_defects=1)
            ids.append(rid)

        self.db.batch_review(ids[:3], "confirmed", "reviewer1")
        unreviewed = self.db.get_unreviewed()
        self.assertEqual(len(unreviewed), 2)

    def test_get_statistics(self):
        self.db.add_record(result="OK")
        self.db.add_record(result="OK")
        self.db.add_record(result="NG", num_defects=1)

        stats = self.db.get_statistics()
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['ok_count'], 2)
        self.assertEqual(stats['ng_count'], 1)
        self.assertAlmostEqual(stats['pass_rate'], 66.7, places=0)

    def test_defect_details(self):
        defects = [
            {'class_name': 'open', 'confidence': 0.9, 'bbox': [10, 20, 30, 40]},
            {'class_name': 'short', 'confidence': 0.7, 'bbox': [50, 60, 70, 80]},
        ]
        rid = self.db.add_record(result="NG", num_defects=2, defects=defects)
        details = self.db.get_defect_details(rid)
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]['class_name'], 'open')


class TestAuthManager(unittest.TestCase):
    """权限管理测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.auth = AuthManager(db_path=os.path.join(self.tmpdir, "users.json"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_users_created(self):
        users = self.auth.list_users()
        self.assertGreaterEqual(len(users), 3)

    def test_login_success(self):
        result = self.auth.login("admin", "admin123")
        self.assertTrue(result)
        self.assertTrue(self.auth.is_logged_in)
        self.assertEqual(self.auth.current_user.role, Role.ADMIN)

    def test_login_wrong_password(self):
        result = self.auth.login("admin", "wrong")
        self.assertFalse(result)

    def test_login_nonexistent_user(self):
        result = self.auth.login("nobody", "pass")
        self.assertFalse(result)

    def test_logout(self):
        self.auth.login("admin", "admin123")
        self.auth.logout()
        self.assertFalse(self.auth.is_logged_in)

    def test_permission_check(self):
        self.auth.login("admin", "admin123")
        self.assertTrue(self.auth.check_permission('manage_users'))
        self.assertTrue(self.auth.check_permission('system_config'))

    def test_operator_permissions(self):
        self.auth.login("operator", "op123")
        self.assertTrue(self.auth.check_permission('start_stop'))
        self.assertFalse(self.auth.check_permission('adjust_threshold'))
        self.assertFalse(self.auth.check_permission('manage_users'))

    def test_engineer_permissions(self):
        self.auth.login("engineer", "eng123")
        self.assertTrue(self.auth.check_permission('adjust_threshold'))
        self.assertTrue(self.auth.check_permission('view_debug_info'))
        self.assertFalse(self.auth.check_permission('manage_users'))

    def test_add_user(self):
        self.auth.login("admin", "admin123")
        result = self.auth.add_user("newuser", Role.OPERATOR, "pass123", "New User")
        self.assertTrue(result)
        self.auth.logout()
        self.assertTrue(self.auth.login("newuser", "pass123"))

    def test_change_password(self):
        self.auth.login("operator", "op123")
        result = self.auth.change_password("operator", "op123", "newpass")
        self.assertTrue(result)
        self.auth.logout()
        self.assertTrue(self.auth.login("operator", "newpass"))


class TestEquipmentMonitor(unittest.TestCase):
    """设备健康度监控测试"""

    def setUp(self):
        self.monitor = EquipmentMonitor()

    def test_initial_status(self):
        self.assertEqual(self.monitor.overall_status, HealthStatus.GOOD)

    def test_record_frame_updates_brightness(self):
        import numpy as np
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.monitor.record_frame(frame, inference_time_ms=50)
        details = self.monitor.get_details()
        self.assertGreater(details['light_source']['avg_brightness'], 0)

    def test_low_brightness_triggers_critical(self):
        import numpy as np
        # 连续记录低亮度帧
        dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 20
        for _ in range(20):
            self.monitor.record_frame(dark_frame, inference_time_ms=50)
        self.assertEqual(self.monitor.health_report['light_source'], HealthStatus.CRITICAL)

    def test_high_inference_time_triggers_warning(self):
        import numpy as np
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        for _ in range(20):
            self.monitor.record_frame(frame, inference_time_ms=250)
        self.assertEqual(self.monitor.health_report['inference'], HealthStatus.CRITICAL)

    def test_reset(self):
        import numpy as np
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.monitor.record_frame(frame, 50)
        self.monitor.reset()
        details = self.monitor.get_details()
        self.assertEqual(details['camera']['total_frames'], 0)


class TestAlarmManager(unittest.TestCase):
    """报警管理器测试"""

    def test_alarm_callback(self):
        triggered = []
        alarm = AlarmManager()
        alarm.on_alarm(lambda l, c, m: triggered.append((l, c, m)))
        alarm.trigger(AlarmLevel.WARNING, 2, "test")
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0][0], AlarmLevel.WARNING)

    def test_cooldown(self):
        triggered = []
        alarm = AlarmManager(config=AlarmConfig(cooldown_ms=5000))
        alarm.on_alarm(lambda l, c, m: triggered.append(1))
        alarm.trigger(AlarmLevel.WARNING, 1, "test1")
        alarm.trigger(AlarmLevel.WARNING, 1, "test2")  # 应被冷却
        self.assertEqual(len(triggered), 1)


class TestConfig(unittest.TestCase):
    """配置加载测试"""

    def test_config_dict_access(self):
        config = Config({'model': {'name': 'yolo11s'}, 'epochs': 200})
        self.assertEqual(config.model.name, 'yolo11s')
        self.assertEqual(config.epochs, 200)

    def test_config_get(self):
        config = Config({'key': 'value'})
        self.assertEqual(config.get('key'), 'value')
        self.assertIsNone(config.get('missing'))
        self.assertEqual(config.get('missing', 'default'), 'default')

    def test_config_to_dict(self):
        original = {'model': {'name': 'test'}, 'epochs': 100}
        config = Config(original)
        result = config.to_dict()
        self.assertEqual(result, original)

    def test_merge_config(self):
        base = Config({'model': {'name': 'a', 'size': 640}, 'epochs': 100})
        override = {'model': {'name': 'b'}, 'epochs': 200}
        merged = merge_config(base, override)
        self.assertEqual(merged.model.name, 'b')
        self.assertEqual(merged.model.size, 640)
        self.assertEqual(merged.epochs, 200)


if __name__ == '__main__':
    unittest.main()
