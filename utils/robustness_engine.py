# -*- coding: utf-8 -*-
"""
车间鲁棒性引擎
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger

logger = setup_logger("robustness")


class ImageQualityChecker:
    """图像质量检查器"""

    def __init__(self, brightness_min=40, brightness_max=220, sharpness_threshold=30):
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.sharpness_threshold = sharpness_threshold

    def check(self, image):
        issues = []
        scores = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.brightness_min:
            issues.append(f"亮度过低: {mean_brightness:.1f} < {self.brightness_min}")
            scores.append(0.3)
        elif mean_brightness > self.brightness_max:
            issues.append(f"亮度过高: {mean_brightness:.1f} > {self.brightness_max}")
            scores.append(0.3)
        else:
            scores.append(1.0)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.sharpness_threshold:
            issues.append(f"图像模糊: Laplacian方差={laplacian_var:.1f} < {self.sharpness_threshold}")
            scores.append(0.3)
        else:
            scores.append(1.0)
        contrast = float(np.std(gray))
        if contrast < 15:
            issues.append(f"对比度过低: {contrast:.1f}")
            scores.append(0.4)
        else:
            scores.append(1.0)
        quality_score = float(np.mean(scores))
        passed = quality_score >= 0.6 and len(issues) <= 1
        return {
            "quality_score": round(quality_score, 4),
            "brightness": round(mean_brightness, 2),
            "sharpness": round(laplacian_var, 2),
            "contrast": round(contrast, 2),
            "issues": issues,
            "passed": passed,
        }


class LightAdaptationEngine:
    """光照自适应引擎"""

    def __init__(self, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))

    def normalize(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = float(np.mean(gray))
        adaptation_info = {"original_brightness": round(mean_brightness, 2), "method": "none"}
        if mean_brightness < 50 or mean_brightness > 200:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.multiply(v, 255.0 / max(mean_brightness, 1))
                v = np.clip(v, 0, 255).astype(np.uint8)
                image = cv2.merge([h, s, v])
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                adaptation_info["method"] = "hsv_adjust"
            else:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                adaptation_info["method"] = "minmax_normalize"
        elif mean_brightness < 80 or mean_brightness > 180:
            if len(image.shape) == 3:
                channels = cv2.split(image)
                equalized = [self.clahe.apply(ch) for ch in channels]
                image = cv2.merge(equalized)
            else:
                image = self.clahe.apply(image)
            adaptation_info["method"] = "clahe"
        else:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            adaptation_info["method"] = "minmax_light"
        adj_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        adaptation_info["adjusted_brightness"] = round(float(np.mean(adj_gray)), 2)
        return image, adaptation_info


class CameraReconnectionManager:
    """相机重连管理器"""

    def __init__(self, max_retries=5, retry_interval=2, camera_index=0):
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.camera_index = camera_index
        self.cap = None
        self.connected = False
        self.reconnect_count = 0
        self.last_error = None

    def connect(self):
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                ret, _ = self.cap.read()
                if ret:
                    self.connected = True
                    self.reconnect_count += 1
                    logger.info(f"相机连接成功 (第{attempt}次尝试, 累计重连{self.reconnect_count}次)")
                    return True
                else:
                    logger.warning(f"相机打开成功但无法读取帧 (尝试 {attempt}/{self.max_retries})")
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"相机连接失败 (尝试 {attempt}/{self.max_retries}): {e}")
            if attempt < self.max_retries:
                time.sleep(self.retry_interval)
        self.connected = False
        logger.error(f"相机连接失败，已达最大重试次数 {self.max_retries}")
        return False

    def read_frame(self):
        if not self.connected or self.cap is None or not self.cap.isOpened():
            if self.connect():
                pass
            else:
                return None, "camera_disconnected"
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("读取帧失败，尝试重连...")
            self.connected = False
            if self.connect():
                ret, frame = self.cap.read()
                if not ret:
                    return None, "reconnect_failed"
            else:
                return None, "reconnect_failed"
        return frame, None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.connected = False

    def get_status(self):
        return {
            "connected": self.connected,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "camera_index": self.camera_index,
        }


class RobustnessEngine:
    """车间鲁棒性综合引擎"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.project_root = get_project_root()
        monitor_cfg = getattr(self.config, 'monitor', None)
        brightness_min = getattr(monitor_cfg, 'brightness_min', 40) if monitor_cfg else 40
        brightness_max = getattr(monitor_cfg, 'brightness_max', 220) if monitor_cfg else 220
        sharpness_thresh = getattr(monitor_cfg, 'sharpness_threshold', 30) if monitor_cfg else 30
        self.quality_checker = ImageQualityChecker(brightness_min, brightness_max, sharpness_thresh)
        self.light_adaptor = LightAdaptationEngine()
        hw_cfg = getattr(self.config, 'hardware', None)
        cam_cfg = getattr(hw_cfg, 'camera', None) if hw_cfg else None
        cam_index = getattr(cam_cfg, 'index', 0) if cam_cfg else 0
        self.camera_mgr = CameraReconnectionManager(camera_index=cam_index)
        self.stats = {
            "total_frames": 0, "quality_passed": 0, "quality_failed": 0,
            "light_adaptations": 0, "camera_reconnections": 0,
        }

    def process_frame(self, image):
        self.stats["total_frames"] += 1
        quality = self.quality_checker.check(image)
        if not quality["passed"]:
            self.stats["quality_failed"] += 1
            logger.warning(f"图像质量不达标: {quality['issues']}")
        else:
            self.stats["quality_passed"] += 1
        processed, adaptation = self.light_adaptor.normalize(image)
        if adaptation["method"] != "none":
            self.stats["light_adaptations"] += 1
        return processed, quality, adaptation, not quality["passed"]

    def get_report(self):
        total = max(self.stats["total_frames"], 1)
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats.copy(),
            "quality_pass_rate": round(self.stats["quality_passed"] / total, 4),
            "adaptation_rate": round(self.stats["light_adaptations"] / total, 4),
            "camera_status": self.camera_mgr.get_status() if hasattr(self, 'camera_mgr') else None,
        }
