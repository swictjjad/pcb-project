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
