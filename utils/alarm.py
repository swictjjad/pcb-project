# -*- coding: utf-8 -*-
"""
声光报警模块
功能：
1. 缺陷检出时声光报警
2. 报警级别配置
3. 支持蜂鸣器（串口控制）和系统声音
"""

import time
import threading
from typing import Optional, Callable

from utils.logger import setup_logger

logger = setup_logger("alarm")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AlarmLevel:
    """报警级别"""
    INFO = "info"           # 提示：合格品通过
    WARNING = "warning"     # 警告：低置信度缺陷
    CRITICAL = "critical"   # 严重：高置信度缺陷


class AlarmConfig:
    """报警配置"""
    def __init__(self, level: str = AlarmLevel.WARNING,
                 sound_enabled: bool = True, light_enabled: bool = True,
                 duration_ms: int = 500, cooldown_ms: int = 2000,
                 beep_freq: int = 1000, beep_duration: int = 300):
        self.level = level
        self.sound_enabled = sound_enabled
        self.light_enabled = light_enabled
        self.duration_ms = duration_ms
        self.cooldown_ms = cooldown_ms
        self.beep_freq = beep_freq
        self.beep_duration = beep_duration


class AlarmManager:
    """声光报警管理器"""

    def __init__(self, controller=None, config: AlarmConfig = None):
        """
        Args:
            controller: Arduino控制器（用于控制外部声光设备）
            config: 报警配置
        """
        self.controller = controller
        self.config = config or AlarmConfig()
        self._last_alarm_time = 0
        self._lock = threading.Lock()
        self._alarm_callbacks = []

    def trigger(self, level: str = AlarmLevel.WARNING, defect_count: int = 0,
                message: str = ""):
        """
        触发报警

        Args:
            level: 报警级别
            defect_count: 缺陷数量
            message: 报警消息
        """
        now = time.time() * 1000

        # 冷却期检查
        if now - self._last_alarm_time < self.config.cooldown_ms:
            return

        with self._lock:
            self._last_alarm_time = now

            # 声音报警
            if self.config.sound_enabled:
                self._play_sound(level, defect_count)

            # 外部灯光报警（通过Arduino控制）
            if self.config.light_enabled and self.controller:
                self._trigger_light(level)

            # 通知回调
            for callback in self._alarm_callbacks:
                try:
                    callback(level, defect_count, message)
                except Exception as e:
                    logger.error(f"报警回调执行失败: {e}")

            level_names = {
                AlarmLevel.INFO: "提示",
                AlarmLevel.WARNING: "警告",
                AlarmLevel.CRITICAL: "严重",
            }
            logger.info(f"报警触发: [{level_names.get(level, level)}] {message}")

    def _play_sound(self, level: str, defect_count: int):
        """播放报警声音"""
        if not WINSOUND_AVAILABLE:
            return

        try:
            if level == AlarmLevel.CRITICAL:
                # 严重：急促连续蜂鸣
                for _ in range(min(defect_count, 5)):
                    winsound.Beep(self.config.beep_freq, self.config.beep_duration)
                    time.sleep(0.05)
            elif level == AlarmLevel.WARNING:
                # 警告：两声短蜂鸣
                winsound.Beep(self.config.beep_freq, self.config.beep_duration)
                time.sleep(0.1)
                winsound.Beep(self.config.beep_freq, self.config.beep_duration)
            else:
                # 提示：一声短蜂鸣
                winsound.Beep(800, 150)
        except Exception as e:
            logger.debug(f"播放声音失败: {e}")

    def _trigger_light(self, level: str):
        """触发外部灯光报警"""
        if not self.controller:
            return
        try:
            # 通过串口发送灯光控制命令
            if hasattr(self.controller, '_send_command'):
                if level == AlarmLevel.CRITICAL:
                    self.controller._send_command(b'ALM_C\n')  # 红灯闪烁
                elif level == AlarmLevel.WARNING:
                    self.controller._send_command(b'ALM_W\n')  # 黄灯
                else:
                    self.controller._send_command(b'ALM_I\n')  # 绿灯
        except Exception as e:
            logger.debug(f"触发灯光报警失败: {e}")

    def on_alarm(self, callback: Callable):
        """注册报警回调"""
        self._alarm_callbacks.append(callback)

    def test_alarm(self):
        """测试报警"""
        self.trigger(AlarmLevel.WARNING, 1, "报警测试")
