# -*- coding: utf-8 -*-
"""
一键换线增强模块
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger

logger = setup_logger("line_swap")


class LineSwapEnhancer:
    """一键换线增强器"""

    DEFAULT_TEMPLATES = {
        "FR4_standard": {
            "description": "FR4标准板 (1-2层)",
            "img_size": 640, "conf_threshold": 0.30, "iou_threshold": 0.40,
            "max_det": 300, "clahe_enabled": True, "clahe_clip_limit": 2.0,
            "servo_angle_ok": 45, "servo_angle_ng": 135,
            "conveyor_speed": 100, "sensor_debounce_ms": 50,
        },
        "HDI_high_density": {
            "description": "HDI高密度板 (4-8层)",
            "img_size": 640, "conf_threshold": 0.35, "iou_threshold": 0.35,
            "max_det": 500, "clahe_enabled": True, "clahe_clip_limit": 3.0,
            "servo_angle_ok": 45, "servo_angle_ng": 135,
            "conveyor_speed": 80, "sensor_debounce_ms": 30,
        },
        "flex_pcb": {
            "description": "柔性电路板 (FPC)",
            "img_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.45,
            "max_det": 400, "clahe_enabled": True, "clahe_clip_limit": 1.5,
            "servo_angle_ok": 30, "servo_angle_ng": 150,
            "conveyor_speed": 120, "sensor_debounce_ms": 80,
        },
        "heavy_current": {
            "description": "厚铜板 (电源板)",
            "img_size": 640, "conf_threshold": 0.30, "iou_threshold": 0.40,
            "max_det": 200, "clahe_enabled": False, "clahe_clip_limit": 2.0,
            "servo_angle_ok": 60, "servo_angle_ng": 120,
            "conveyor_speed": 60, "sensor_debounce_ms": 100,
        },
    }

    def __init__(self, config=None):
        self.config = config or get_config()
        self.project_root = get_project_root()
        self.profiles_dir = Path(self.config.production.profiles_dir) if hasattr(self.config, 'production') and self.config.production.profiles_dir else self.project_root / "configs" / "line_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.current_profile = None
        self.swap_history = []

    def get_template_names(self):
        return {name: info["description"] for name, info in self.DEFAULT_TEMPLATES.items()}

    def recommend_template(self, pcb_type, layer_count=None):
        pcb_lower = pcb_type.lower()
        matched = None
        for key, template in self.DEFAULT_TEMPLATES.items():
            if key.lower() in pcb_lower or pcb_lower in key.lower():
                matched = template
                break
        if matched is None:
            matched = self.DEFAULT_TEMPLATES["FR4_standard"]
        if layer_count is not None and layer_count > 4:
            adjusted = matched.copy()
            adjusted["conf_threshold"] = max(matched["conf_threshold"] - 0.05, 0.20)
            adjusted["iou_threshold"] = max(matched["iou_threshold"] - 0.05, 0.30)
            return adjusted
        return matched

    def save_profile(self, profile_name, config_dict):
        profile_path = self.profiles_dir / f"{profile_name}.json"
        profile_data = {
            "name": profile_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": config_dict,
        }
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        logger.info(f"产线配置已保存: {profile_path}")
        return str(profile_path)

    def load_profile(self, profile_name):
        profile_path = self.profiles_dir / f"{profile_name}.json"
        if not profile_path.exists():
            if profile_name in self.DEFAULT_TEMPLATES:
                template = self.DEFAULT_TEMPLATES[profile_name]
                self.current_profile = {
                    "name": profile_name, "template": True,
                    "config": template, "loaded_at": datetime.now().isoformat(),
                }
                return self.current_profile
            logger.error(f"配置档案不存在: {profile_name}")
            return None
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.current_profile = data
        logger.info(f"产线配置已加载: {profile_name}")
        return data

    def validate_config(self, config_dict):
        errors = []
        warnings = []
        conf = config_dict.get("conf_threshold", 0.30)
        if not (0.01 <= conf <= 0.99):
            errors.append(f"conf_threshold 必须在 0.01-0.99 之间，当前: {conf}")
        iou = config_dict.get("iou_threshold", 0.40)
        if not (0.01 <= iou <= 0.99):
            errors.append(f"IoU阈值必须在 0.01-0.99 之间，当前: {iou}")
        max_det = config_dict.get("max_det", 300)
        if not (1 <= max_det <= 1000):
            errors.append(f"max_det 必须在 1-1000 之间，当前: {max_det}")
        angle_ok = config_dict.get("servo_angle_ok", 45)
        angle_ng = config_dict.get("servo_angle_ng", 135)
        if not (0 <= angle_ok <= 180):
            errors.append(f"合格舵机角度必须在 0-180 之间，当前: {angle_ok}")
        if not (0 <= angle_ng <= 180):
            errors.append(f"不合格舵机角度必须在 0-180 之间，当前: {angle_ng}")
        if abs(angle_ok - angle_ng) < 20:
            warnings.append("合格/不合格舵机角度差小于20度")
        speed = config_dict.get("conveyor_speed", 100)
        if not (10 <= speed <= 500):
            errors.append(f"传送带速度必须在 10-500 之间，当前: {speed}")
        clip = config_dict.get("clahe_clip_limit", 2.0)
        if not (0.5 <= clip <= 8.0):
            warnings.append(f"CLAHE clip_limit 超出推荐范围 [0.5, 8.0]")
        return len(errors) == 0, errors, warnings

    def activate_profile(self, profile_name):
        start_time = time.time()
        profile = self.load_profile(profile_name)
        if profile is None:
            return {"success": False, "error": f"无法加载配置: {profile_name}"}
        config = profile.get("config", profile)
        is_valid, errors, warnings = self.validate_config(config)
        if not is_valid:
            return {"success": False, "error": "配置校验失败", "errors": errors, "warnings": warnings}
        if self.current_profile:
            self.save_profile(f"{profile_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                              self.current_profile.get("config", self.current_profile))
        last_profile_path = self.profiles_dir / "_last_profile.json"
        with open(last_profile_path, "w", encoding="utf-8") as f:
            json.dump({"last_profile": profile_name, "activated_at": datetime.now().isoformat()}, f, indent=2)
        elapsed = time.time() - start_time
        self.swap_history.append({
            "from": self.current_profile.get("name", "unknown") if self.current_profile else None,
            "to": profile_name, "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(), "valid": is_valid, "warnings": warnings,
        })
        logger.info(f"换线完成: {profile_name} (耗时 {elapsed:.2f}s)")
        return {"success": True, "profile": profile_name, "swap_time_seconds": round(elapsed, 2),
                "warnings": warnings, "config_applied": config}

    def export_all_profiles(self, output_dir=None):
        if output_dir is None:
            output_dir = self.project_root / "results" / "exports" / "line_profiles"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        for name in self.DEFAULT_TEMPLATES:
            config = self.DEFAULT_TEMPLATES[name]
            profile_data = {"name": name, "description": config.get("description", ""),
                            "config": {k: v for k, v in config.items() if k != "description"}}
            path = output_dir / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            exported.append(str(path))
        logger.info(f"已导出 {len(exported)} 个配置档案到 {output_dir}")
        return exported

    def get_swap_history(self):
        return {"total_swaps": len(self.swap_history), "history": self.swap_history[-20:]}
