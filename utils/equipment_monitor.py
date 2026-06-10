# -*- coding: utf-8 -*-
"""
设备健康度监控模块
功能：
1. 光源亮度衰减检测
2. 摄像头帧率/丢帧监控
3. 推理耗时趋势
4. 设备异常告警
"""

import time
import threading
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Callable

import numpy as np

from utils.logger import setup_logger

logger = setup_logger("equipment_monitor")


class HealthStatus:
    """健康状态"""
    GOOD = "good"           # 正常
    WARNING = "warning"     # 需要关注
    CRITICAL = "critical"   # 需要维护


class EquipmentMonitor:
    """设备健康度监控"""

    def __init__(self, config=None):
        self.config = config

        # 监控指标
        self._brightness_history = deque(maxlen=100)     # 图像亮度历史
        self._fps_history = deque(maxlen=100)            # 帧率历史
        self._inference_time_history = deque(maxlen=100) # 推理耗时历史
        self._drop_frame_count = 0                       # 丢帧计数
        self._total_frames = 0                           # 总帧数

        # 阈值配置
        self._brightness_min = 40        # 亮度下限
        self._brightness_max = 220       # 亮度上限
        self._brightness_trend_window = 50  # 亮度趋势窗口
        self._fps_min = 10               # 帧率下限
        self._inference_max_ms = 200     # 推理耗时上限

        # 状态
        self._last_frame_time = time.time()
        self._alert_callbacks = []

        # 健康度报告
        self._health_report = {
            'light_source': HealthStatus.GOOD,
            'camera': HealthStatus.GOOD,
            'inference': HealthStatus.GOOD,
            'overall': HealthStatus.GOOD,
        }

    def record_frame(self, frame: np.ndarray, inference_time_ms: float = 0):
        """
        记录一帧的监控数据

        Args:
            frame: 图像帧
            inference_time_ms: 推理耗时
        """
        self._total_frames += 1
        now = time.time()

        # 1. 亮度检测
        if frame is not None and frame.size > 0:
            try:
                gray = np.mean(frame)  # 简单平均亮度
                self._brightness_history.append(gray)
            except Exception:
                pass

        # 2. 帧率计算
        dt = now - self._last_frame_time
        if dt > 0:
            fps = 1.0 / dt
            self._fps_history.append(fps)

            # 丢帧检测（帧间隔超过2倍预期）
            if dt > 0.2:  # 超过200ms算丢帧
                self._drop_frame_count += 1

        self._last_frame_time = now

        # 3. 推理耗时
        if inference_time_ms > 0:
            self._inference_time_history.append(inference_time_ms)

        # 4. 更新健康度
        self._update_health()

    def _update_health(self):
        """更新设备健康度"""
        # 光源健康度
        if len(self._brightness_history) >= 10:
            recent = list(self._brightness_history)[-10:]
            avg_brightness = np.mean(recent)

            if avg_brightness < self._brightness_min:
                self._health_report['light_source'] = HealthStatus.CRITICAL
            elif self._is_brightness_declining():
                self._health_report['light_source'] = HealthStatus.WARNING
            else:
                self._health_report['light_source'] = HealthStatus.GOOD

        # 摄像头健康度
        if len(self._fps_history) >= 10:
            recent_fps = list(self._fps_history)[-10:]
            avg_fps = np.mean(recent_fps)

            if avg_fps < self._fps_min:
                self._health_report['camera'] = HealthStatus.CRITICAL
            elif self._drop_frame_count > 10:
                self._health_report['camera'] = HealthStatus.WARNING
            else:
                self._health_report['camera'] = HealthStatus.GOOD

        # 推理健康度
        if len(self._inference_time_history) >= 10:
            recent_times = list(self._inference_time_history)[-10:]
            avg_time = np.mean(recent_times)

            if avg_time > self._inference_max_ms:
                self._health_report['inference'] = HealthStatus.CRITICAL
            elif avg_time > self._inference_max_ms * 0.7:
                self._health_report['inference'] = HealthStatus.WARNING
            else:
                self._health_report['inference'] = HealthStatus.GOOD

        # 综合健康度
        statuses = [
            self._health_report['light_source'],
            self._health_report['camera'],
            self._health_report['inference'],
        ]
        if HealthStatus.CRITICAL in statuses:
            self._health_report['overall'] = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            self._health_report['overall'] = HealthStatus.WARNING
        else:
            self._health_report['overall'] = HealthStatus.GOOD

    def _is_brightness_declining(self) -> bool:
        """检测亮度是否在持续衰减（光源老化）"""
        if len(self._brightness_history) < self._brightness_trend_window:
            return False

        recent = list(self._brightness_history)[-self._brightness_trend_window:]
        # 简单线性回归斜率
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]

        # 斜率显著为负 → 光源在衰减
        return slope < -0.3

    @property
    def health_report(self) -> dict:
        return dict(self._health_report)

    @property
    def overall_status(self) -> str:
        return self._health_report['overall']

    def get_details(self) -> dict:
        """获取详细监控数据"""
        details = {
            'light_source': {
                'status': self._health_report['light_source'],
                'avg_brightness': float(np.mean(self._brightness_history)) if self._brightness_history else 0,
                'is_declining': self._is_brightness_declining() if len(self._brightness_history) >= self._brightness_trend_window else False,
            },
            'camera': {
                'status': self._health_report['camera'],
                'avg_fps': float(np.mean(self._fps_history)) if self._fps_history else 0,
                'drop_frames': self._drop_frame_count,
                'total_frames': self._total_frames,
            },
            'inference': {
                'status': self._health_report['inference'],
                'avg_time_ms': float(np.mean(self._inference_time_history)) if self._inference_time_history else 0,
                'max_time_ms': float(max(self._inference_time_history)) if self._inference_time_history else 0,
            },
        }
        return details

    def on_alert(self, callback: Callable):
        """注册健康告警回调"""
        self._alert_callbacks.append(callback)

    def reset(self):
        """重置监控数据"""
        self._brightness_history.clear()
        self._fps_history.clear()
        self._inference_time_history.clear()
        self._drop_frame_count = 0
        self._total_frames = 0
