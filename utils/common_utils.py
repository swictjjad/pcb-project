# -*- coding: utf-8 -*-
"""
通用工具函数
"""
import cv2
import numpy as np
from pathlib import Path


def load_image_broken_path(image_path):
    """
    解决 OpenCV 无法读取含中文/特殊字符路径的问题
    
    Args:
        image_path: 图片文件路径（可以是 str 或 Path）
    
    Returns:
        numpy array (BGR) or None if failed
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    # Fallback: use np.fromfile + cv2.imdecode
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_image_unchanged(image_path):
    """Same but IMREAD_ANYCOLOR|IMREAD_ANYDEPTH"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    except Exception:
        return None
