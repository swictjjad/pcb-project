# -*- coding: utf-8 -*-
"""
配置加载模块
用于加载和解析YAML配置文件
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """配置类，支持字典和属性访问"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
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
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
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
        config_path: 配置文件路径
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
