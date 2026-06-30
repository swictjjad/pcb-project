# -*- coding: utf-8 -*-
"""
配置加载模块
用于加载和解析YAML配置文件
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any


# 项目根目录自动检测（支持从任意工作目录运行）
_PROJECT_ROOT = None


def get_project_root() -> Path:
    """获取项目根目录（configs/config.yaml 所在的目录）"""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    # 方案1: 从当前脚本所在目录向上查找（最可靠）
    script_dir = Path(__file__).resolve().parent
    for ancestor in [script_dir, *script_dir.parents]:
        config_yaml = ancestor / "configs" / "config.yaml"
        if config_yaml.exists():
            # 确保这是真正的根目录（不是子目录里的 configs）
            # 真正的根目录应该有 main.py、inference.py 等文件
            if (ancestor / "main.py").exists() or (ancestor / "inference.py").exists():
                _PROJECT_ROOT = ancestor
                return _PROJECT_ROOT

    # 方案2: 检查当前工作目录
    for candidate in [Path.cwd(), Path.cwd() / "pcb_defect_system"]:
        config_yaml = candidate / "configs" / "config.yaml"
        if config_yaml.exists() and (candidate / "main.py").exists():
            _PROJECT_ROOT = candidate
            return _PROJECT_ROOT

    # 方案3: 从 sys.path 中的项目目录查找
    for p in sys.path:
        candidate = Path(p)
        config_yaml = candidate / "configs" / "config.yaml"
        if config_yaml.exists() and (candidate / "main.py").exists():
            _PROJECT_ROOT = candidate
            return _PROJECT_ROOT

    # 最后回退到当前目录
    _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """将相对路径解析为基于项目根目录的绝对路径"""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return get_project_root() / p


class Config:
    """配置类，支持字典和属性访问，所有相对路径自动解析为绝对路径"""

    # 配置中可能包含相对路径的键名
    PATH_KEYS = {
        "model_path", "dataset_path", "db_path", "log_file",
        "save_dir", "rules_dir", "profiles_dir", "audit_log",
    }

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, str) and key in self.PATH_KEYS:
                # 相对路径自动解析为绝对路径
                setattr(self, key, str(resolve_path(value)))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self):
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """
    加载YAML配置文件（支持相对路径自动解析为绝对路径）

    Args:
        config_path: 配置文件路径（相对或绝对）

    Returns:
        Config对象
    """
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")

    with open(resolved, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_config(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    合并配置，override_dict中的值会覆盖base_config中的值
    
    Args:
        base_config: 基础配置
        override_dict: 覆盖配置字典
        
    Returns:
        合并后的Config对象
    """
    base_dict = base_config.to_dict()
    
    def recursive_merge(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                recursive_merge(base[key], value)
            else:
                base[key] = value
    
    recursive_merge(base_dict, override_dict)
    return Config(base_dict)


# 全局配置实例
_config_instance = None


def get_config(config_path: str = "configs/config.yaml", reload: bool = False) -> Config:
    """
    获取全局配置实例（单例模式）

    Args:
        config_path: 配置文件路径（相对或绝对）
        reload: 是否强制重新加载

    Returns:
        Config对象
    """
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = load_config(config_path)
    return _config_instance


if __name__ == "__main__":
    # 测试配置加载
    config = load_config("../configs/config.yaml")
    print("模型名称:", config.model.name)
    print("训练轮数:", config.train.epochs)
    print("类别数:", config.data.num_classes)
    print("类别名称:", config.data.class_names)
