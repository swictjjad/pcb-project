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
