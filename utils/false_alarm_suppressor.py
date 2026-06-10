# -*- coding: utf-8 -*-
"""
误报屏蔽模块
功能：
1. 区域忽略规则 - 某个区域总是误报时，可一键屏蔽
2. 规则持久化存储
3. 检测结果过滤
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("false_alarm_suppressor")

# 规则存储路径
SUPPRESSION_RULES_DIR = Path("./configs/suppression_rules")


class SuppressionRule:
    """误报屏蔽规则"""

    RULE_TYPE_RECT = "rect"       # 矩形区域
    RULE_TYPE_POLYGON = "polygon" # 多边形区域

    def __init__(self, name: str, rule_type: str = "rect",
                 region: dict = None, class_names: List[str] = None,
                 min_confidence: float = 0.0, enabled: bool = True,
                 created_by: str = "", note: str = ""):
        """
        Args:
            name: 规则名称
            rule_type: 规则类型 "rect" 或 "polygon"
            region: 区域定义
                rect: {"x1": int, "y1": int, "x2": int, "y2": int}
                polygon: {"points": [[x1,y1], [x2,y2], ...]}
            class_names: 要屏蔽的缺陷类别，空列表=屏蔽所有类别
            min_confidence: 低于此置信度的检测才屏蔽，0=屏蔽所有
            enabled: 是否启用
            created_by: 创建者
            note: 备注
        """
        self.name = name
        self.rule_type = rule_type
        self.region = region or {}
        self.class_names = class_names or []
        self.min_confidence = min_confidence
        self.enabled = enabled
        self.created_by = created_by
        self.note = note
        self.created_at = datetime.now().isoformat()
        self.hit_count = 0  # 命中次数

    def matches(self, bbox: List[float], class_name: str, confidence: float) -> bool:
        """
        检查检测结果是否匹配此屏蔽规则

        Args:
            bbox: [x1, y1, x2, y2]
            class_name: 缺陷类别名
            confidence: 置信度

        Returns:
            True=应屏蔽
        """
        if not self.enabled:
            return False

        # 类别过滤
        if self.class_names and class_name not in self.class_names:
            return False

        # 置信度过滤：min_confidence>0时，只屏蔽置信度低于此值的检测（低置信度=更可能是误报）
        if self.min_confidence > 0 and confidence >= self.min_confidence:
            return False

        # 区域匹配
        if self.rule_type == self.RULE_TYPE_RECT:
            return self._match_rect(bbox)
        elif self.rule_type == self.RULE_TYPE_POLYGON:
            return self._match_polygon(bbox)
        return False

    def _match_rect(self, bbox: List[float]) -> bool:
        """矩形区域匹配 - 检测框中心点在屏蔽区域内"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        r = self.region
        return r.get('x1', 0) <= cx <= r.get('x2', 99999) and \
               r.get('y1', 0) <= cy <= r.get('y2', 99999)

    def _match_polygon(self, bbox: List[float]) -> bool:
        """多边形区域匹配 - 检测框中心点在多边形内"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        points = self.region.get('points', [])
        if len(points) < 3:
            return False
        return self._point_in_polygon(cx, cy, points)

    @staticmethod
    def _point_in_polygon(px: float, py: float, polygon: List[List[float]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'rule_type': self.rule_type,
            'region': self.region,
            'class_names': self.class_names,
            'min_confidence': self.min_confidence,
            'enabled': self.enabled,
            'created_by': self.created_by,
            'note': self.note,
            'created_at': self.created_at,
            'hit_count': self.hit_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SuppressionRule':
        rule = cls(
            name=data['name'],
            rule_type=data.get('rule_type', 'rect'),
            region=data.get('region', {}),
            class_names=data.get('class_names', []),
            min_confidence=data.get('min_confidence', 0.0),
            enabled=data.get('enabled', True),
            created_by=data.get('created_by', ''),
            note=data.get('note', ''),
        )
        rule.created_at = data.get('created_at', rule.created_at)
        rule.hit_count = data.get('hit_count', 0)
        return rule


class FalseAlarmSuppressor:
    """误报屏蔽管理器"""

    def __init__(self, rules_dir: str = None, profile_name: str = "default"):
        self.rules_dir = Path(rules_dir) if rules_dir else SUPPRESSION_RULES_DIR
        self.rules_dir.mkdir(parents=True, exist_ok=True)

        self._profile_name = profile_name
        self._rules: Dict[str, SuppressionRule] = {}
        self._load_rules()

    @property
    def rules_file(self) -> Path:
        return self.rules_dir / f"{self._profile_name}.json"

    def _load_rules(self):
        """加载屏蔽规则"""
        if not self.rules_file.exists():
            return
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for rule_data in data.get('rules', []):
                rule = SuppressionRule.from_dict(rule_data)
                self._rules[rule.name] = rule
            logger.info(f"已加载 {len(self._rules)} 条屏蔽规则")
        except Exception as e:
            logger.error(f"加载屏蔽规则失败: {e}")

    def _save_rules(self):
        """保存屏蔽规则"""
        try:
            data = {
                'profile': self._profile_name,
                'updated_at': datetime.now().isoformat(),
                'rules': [r.to_dict() for r in self._rules.values()],
            }
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存屏蔽规则失败: {e}")

    def set_profile(self, profile_name: str):
        """切换产线时更新规则文件"""
        self._profile_name = profile_name
        self._rules.clear()
        self._load_rules()

    def add_rule(self, rule: SuppressionRule) -> bool:
        """添加屏蔽规则"""
        self._rules[rule.name] = rule
        self._save_rules()
        logger.info(f"添加屏蔽规则: {rule.name} (类型={rule.rule_type})")
        return True

    def add_rect_rule(self, name: str, x1: int, y1: int, x2: int, y2: int,
                      class_names: List[str] = None, note: str = "") -> SuppressionRule:
        """快捷方法：添加矩形屏蔽区域"""
        rule = SuppressionRule(
            name=name,
            rule_type=SuppressionRule.RULE_TYPE_RECT,
            region={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            class_names=class_names or [],
            note=note,
        )
        self.add_rule(rule)
        return rule

    def remove_rule(self, name: str) -> bool:
        """删除屏蔽规则"""
        if name in self._rules:
            del self._rules[name]
            self._save_rules()
            logger.info(f"删除屏蔽规则: {name}")
            return True
        return False

    def toggle_rule(self, name: str) -> bool:
        """启用/禁用屏蔽规则"""
        if name in self._rules:
            self._rules[name].enabled = not self._rules[name].enabled
            self._save_rules()
            status = "启用" if self._rules[name].enabled else "禁用"
            logger.info(f"屏蔽规则 {name}: {status}")
            return True
        return False

    def list_rules(self) -> List[SuppressionRule]:
        """列出所有屏蔽规则"""
        return list(self._rules.values())

    def filter_detections(self, detections: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        过滤检测结果，屏蔽误报

        Args:
            detections: 检测结果列表，每项包含 bbox/class_name/confidence

        Returns:
            (有效检测, 被屏蔽的检测)
        """
        if not self._rules:
            return detections, []

        valid = []
        suppressed = []

        for det in detections:
            bbox = det.get('bbox', [])
            class_name = det.get('class_name', '')
            confidence = det.get('confidence', 0)

            is_suppressed = False
            for rule in self._rules.values():
                if rule.matches(bbox, class_name, confidence):
                    is_suppressed = True
                    rule.hit_count += 1
                    break

            if is_suppressed:
                det_copy = dict(det)
                det_copy['suppressed_by'] = rule.name
                suppressed.append(det_copy)
            else:
                valid.append(det)

        if suppressed:
            self._save_rules()  # 保存命中计数

        return valid, suppressed

    def draw_suppression_zones(self, image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """在图像上绘制屏蔽区域（半透明红色）"""
        overlay = image.copy()

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.rule_type == SuppressionRule.RULE_TYPE_RECT:
                r = rule.region
                x1, y1 = int(r.get('x1', 0)), int(r.get('y1', 0))
                x2, y2 = int(r.get('x2', 0)), int(r.get('y2', 0))
                cv2_rect = overlay[y1:y2, x1:x2]
                if cv2_rect.size > 0:
                    cv2_rect[:] = (cv2_rect * (1 - alpha) + np.array([0, 0, 200]) * alpha).astype(np.uint8)
                    overlay[y1:y2, x1:x2] = cv2_rect

            elif rule.rule_type == SuppressionRule.RULE_TYPE_POLYGON:
                points = rule.region.get('points', [])
                if len(points) >= 3:
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 0, 200))

        # 添加标签
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.rule_type == SuppressionRule.RULE_TYPE_RECT:
                r = rule.region
                x1, y1 = int(r.get('x1', 0)), int(r.get('y1', 0))
                cv2.putText(overlay, f"[X] {rule.name}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return overlay
