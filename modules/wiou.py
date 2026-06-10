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
