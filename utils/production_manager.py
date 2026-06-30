# -*- coding: utf-8 -*-
"""
产线管理模块
功能：
1. 一键换线 - 产线配置热切换，无需改YAML文件
2. 产线/调试模式切换
3. 产线配置持久化存储
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from utils.logger import setup_logger
from utils.config_loader import get_project_root

logger = setup_logger("production_manager")

# 产线配置存储路径（自动基于项目根目录）
PROFILES_DIR = get_project_root() / "configs" / "line_profiles"


class LineProfile:
    """产线配置档案"""

    def __init__(self, name: str, model_path: str = "", conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, class_names: List[str] = None,
                 camera_index: int = 0, servo_angle_ok: int = 45,
                 servo_angle_ng: int = 135, conveyor_speed: int = 100,
                 custom_settings: Dict[str, Any] = None):
        self.name = name
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        self.camera_index = camera_index
        self.servo_angle_ok = servo_angle_ok
        self.servo_angle_ng = servo_angle_ng
        self.conveyor_speed = conveyor_speed
        self.custom_settings = custom_settings or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'class_names': self.class_names,
            'camera_index': self.camera_index,
            'servo_angle_ok': self.servo_angle_ok,
            'servo_angle_ng': self.servo_angle_ng,
            'conveyor_speed': self.conveyor_speed,
            'custom_settings': self.custom_settings,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LineProfile':
        profile = cls(
            name=data['name'],
            model_path=data.get('model_path', ''),
            conf_threshold=data.get('conf_threshold', 0.25),
            iou_threshold=data.get('iou_threshold', 0.45),
            class_names=data.get('class_names', []),
            camera_index=data.get('camera_index', 0),
            servo_angle_ok=data.get('servo_angle_ok', 45),
            servo_angle_ng=data.get('servo_angle_ng', 135),
            conveyor_speed=data.get('conveyor_speed', 100),
            custom_settings=data.get('custom_settings', {}),
        )
        profile.created_at = data.get('created_at', profile.created_at)
        profile.updated_at = data.get('updated_at', profile.updated_at)
        return profile


class ProductionManager:
    """产线管理器 - 一键换线 + 模式切换"""

    MODE_PRODUCTION = "production"   # 产线模式：简化界面，只显示关键信息
    MODE_DEBUG = "debug"             # 调试模式：显示Grad-CAM++、不确定性等详细信息

    def __init__(self, profiles_dir: str = None):
        self.profiles_dir = Path(profiles_dir) if profiles_dir else PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._current_profile: Optional[LineProfile] = None
        self._current_mode: str = self.MODE_PRODUCTION
        self._lock = threading.Lock()
        self._switch_callbacks: List[callable] = []
        self._mode_callbacks: List[callable] = []

        # 产线运行状态
        self._shift_info = {
            'operator_id': '',       # 操作员工号
            'equipment_id': '',      # 设备编号
            'shift_name': '',        # 班次名称
            'shift_start': None,     # 班次开始时间
        }

        # 加载上次使用的产线配置
        self._load_last_profile()

    def _load_last_profile(self):
        """加载上次使用的产线配置"""
        last_file = self.profiles_dir / "_last_profile.json"
        if last_file.exists():
            try:
                with open(last_file, 'r', encoding='utf-8') as f:
                    name = json.load(f).get('name', '')
                if name:
                    self.load_profile(name)
                    logger.info(f"已恢复上次产线配置: {name}")
            except Exception as e:
                logger.warning(f"恢复上次产线配置失败: {e}")

    def _save_last_profile(self):
        """记录当前使用的产线配置名"""
        last_file = self.profiles_dir / "_last_profile.json"
        try:
            with open(last_file, 'w', encoding='utf-8') as f:
                json.dump({'name': self._current_profile.name}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存产线配置记录失败: {e}")

    # ==================== 产线配置管理 ====================

    def list_profiles(self) -> List[str]:
        """列出所有可用的产线配置"""
        profiles = []
        for f in self.profiles_dir.glob("*.json"):
            if f.name != "_last_profile.json":
                profiles.append(f.stem)
        return sorted(profiles)

    def get_profile(self, name: str) -> Optional[LineProfile]:
        """获取指定产线配置"""
        filepath = self.profiles_dir / f"{name}.json"
        if not filepath.exists():
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return LineProfile.from_dict(json.load(f))
        except Exception as e:
            logger.error(f"加载产线配置失败 [{name}]: {e}")
            return None

    def save_profile(self, profile: LineProfile) -> bool:
        """保存产线配置"""
        filepath = self.profiles_dir / f"{profile.name}.json"
        profile.updated_at = datetime.now().isoformat()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"产线配置已保存: {profile.name}")
            return True
        except Exception as e:
            logger.error(f"保存产线配置失败: {e}")
            return False

    def delete_profile(self, name: str) -> bool:
        """删除产线配置"""
        filepath = self.profiles_dir / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"产线配置已删除: {name}")
            if self._current_profile and self._current_profile.name == name:
                self._current_profile = None
            return True
        return False

    def load_profile(self, name: str) -> Optional[LineProfile]:
        """加载产线配置（不切换，仅加载）"""
        profile = self.get_profile(name)
        if profile:
            self._current_profile = profile
        return profile

    def switch_line(self, name: str) -> bool:
        """
        一键换线 - 切换到指定产线配置

        切换流程：
        1. 加载目标产线配置
        2. 通知所有回调函数（用于更新检测器参数、硬件参数等）
        3. 记录切换日志

        Args:
            name: 产线配置名称

        Returns:
            是否切换成功
        """
        with self._lock:
            old_name = self._current_profile.name if self._current_profile else "(无)"
            profile = self.load_profile(name)
            if not profile:
                logger.error(f"一键换线失败: 产线配置 '{name}' 不存在")
                return False

            self._save_last_profile()

            # 通知回调
            for callback in self._switch_callbacks:
                try:
                    callback(profile)
                except Exception as e:
                    logger.error(f"换线回调执行失败: {e}")

            logger.info(f"一键换线: {old_name} → {name}")
            return True

    def create_profile_from_current(self, name: str, config=None) -> LineProfile:
        """从当前配置创建新的产线配置"""
        profile = LineProfile(name=name)
        if config:
            profile.model_path = getattr(getattr(config, 'inference', None), 'model_path', '')
            profile.conf_threshold = getattr(getattr(config, 'inference', None), 'conf_threshold', 0.25)
            profile.iou_threshold = getattr(getattr(config, 'inference', None), 'iou_threshold', 0.45)
            class_names = getattr(config, 'data', None)
            profile.class_names = list(class_names.class_names) if class_names and hasattr(class_names, 'class_names') else []
            hw = getattr(config, 'hardware', None)
            cam = getattr(hw, 'camera', None) if hw else None
            profile.camera_index = cam.index if cam else 0
            servo = getattr(hw, 'servo', None) if hw else None
            profile.servo_angle_ok = servo.angle_ok if servo else 45
            profile.servo_angle_ng = servo.angle_ng if servo else 135
            conveyor = getattr(hw, 'conveyor', None) if hw else None
            profile.conveyor_speed = conveyor.speed if conveyor else 100
        self.save_profile(profile)
        return profile

    @property
    def current_profile(self) -> Optional[LineProfile]:
        return self._current_profile

    @property
    def current_profile_name(self) -> str:
        return self._current_profile.name if self._current_profile else "(未选择)"

    def on_line_switched(self, callback: callable):
        """注册换线回调函数"""
        self._switch_callbacks.append(callback)

    # ==================== 模式切换 ====================

    def set_mode(self, mode: str):
        """
        切换产线/调试模式

        Args:
            mode: "production" 或 "debug"
        """
        if mode not in (self.MODE_PRODUCTION, self.MODE_DEBUG):
            logger.error(f"无效模式: {mode}")
            return

        old_mode = self._current_mode
        self._current_mode = mode

        for callback in self._mode_callbacks:
            try:
                callback(mode)
            except Exception as e:
                logger.error(f"模式切换回调执行失败: {e}")

        mode_names = {self.MODE_PRODUCTION: "产线模式", self.MODE_DEBUG: "调试模式"}
        logger.info(f"模式切换: {mode_names.get(old_mode, old_mode)} → {mode_names[mode]}")

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def is_production_mode(self) -> bool:
        return self._current_mode == self.MODE_PRODUCTION

    @property
    def is_debug_mode(self) -> bool:
        return self._current_mode == self.MODE_DEBUG

    def on_mode_changed(self, callback: callable):
        """注册模式切换回调函数"""
        self._mode_callbacks.append(callback)

    # ==================== 班次信息 ====================

    def set_shift_info(self, operator_id: str = "", equipment_id: str = "",
                       shift_name: str = ""):
        """设置班次信息"""
        self._shift_info['operator_id'] = operator_id
        self._shift_info['equipment_id'] = equipment_id
        self._shift_info['shift_name'] = shift_name
        self._shift_info['shift_start'] = datetime.now().isoformat()
        logger.info(f"班次信息已设置: 操作员={operator_id}, 设备={equipment_id}, 班次={shift_name}")

    @property
    def shift_info(self) -> dict:
        return dict(self._shift_info)
