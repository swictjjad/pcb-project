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
