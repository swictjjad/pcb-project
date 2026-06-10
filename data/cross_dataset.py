# -*- coding: utf-8 -*-
"""
跨数据集验证支持

DeepPCB数据集只有1500对图像，在深度学习时代规模偏小。
为验证模型不是过拟合DeepPCB，需在额外数据集上测试。

支持的数据集：
1. DeepPCB: 1500对图像，6类缺陷（主数据集）
2. PKU-Market-PCB: ~1300张图像，6类缺陷（跨数据集验证）

使用方法：
    from data.cross_dataset import CrossDatasetValidator

    validator = CrossDatasetValidator(model_path='./models/best.pt')
    results = validator.evaluate_on_pku_market(data_dir='./data/PKU-Market-PCB')
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import setup_logger
from utils.statistical_testing import bootstrap_test, paired_t_test

logger = setup_logger("cross_dataset")


class CrossDatasetValidator:
    """跨数据集验证器"""

    def __init__(self, model_path: str = None, device: str = "0"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def _load_model(self):
        """加载YOLO模型"""
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"模型已加载: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def evaluate_on_dataset(self, data_dir: str, dataset_name: str = "unknown",
                            img_size: int = 640, conf: float = 0.25,
                            iou: float = 0.45) -> dict:
        """
        在指定数据集上评估模型

        Args:
            data_dir: 数据集目录
            dataset_name: 数据集名称
            img_size: 输入图像尺寸
            conf: 置信度阈值
            iou: NMS IoU阈值

        Returns:
            评估结果
        """
        self._load_model()

        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"数据集目录不存在: {data_dir}")
            return {'error': f'数据集目录不存在: {data_dir}'}

        # 尝试使用ultralytics的验证功能
        try:
            results = self.model.val(
                data=str(data_path / "dataset.yaml") if (data_path / "dataset.yaml").exists() else str(data_dir),
                imgsz=img_size,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
            )

            return {
                'dataset': dataset_name,
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'per_class_map50': {k: float(v) for k, v in zip(
                    results.names.values(), results.box.maps
                )} if hasattr(results.box, 'maps') else {},
                'num_images': len(results.box.maps) if hasattr(results.box, 'maps') else 0,
            }
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {'error': str(e)}

    def evaluate_on_pku_market(self, data_dir: str = "./data/PKU-Market-PCB") -> dict:
        """
        在PKU-Market-PCB数据集上评估

        PKU-Market-PCB数据集：
        - 来源：北京大学+Marketech
        - 规模：约1300张图像
        - 缺陷类别：missing hole, mouse bite, open circuit, short, spur, spurious copper
        - 与DeepPCB的6类缺陷有对应关系

        Args:
            data_dir: PKU-Market-PCB数据集目录
        """
        return self.evaluate_on_dataset(data_dir, "PKU-Market-PCB")

    def cross_dataset_comparison(self, datasets: Dict[str, str]) -> dict:
        """
        多数据集对比

        Args:
            datasets: {数据集名: 数据集目录}

        Returns:
            对比结果
        """
        results = {}
        for name, data_dir in datasets.items():
            logger.info(f"正在评估 {name}...")
            result = self.evaluate_on_dataset(data_dir, name)
            results[name] = result

        # 对比分析
        comparison = {
            'datasets': results,
            'analysis': self._analyze_cross_dataset(results),
        }

        return comparison

    def _analyze_cross_dataset(self, results: Dict[str, dict]) -> dict:
        """分析跨数据集结果"""
        analysis = {
            'performance_drop': {},
            'generalization_assessment': '',
        }

        # 计算性能下降
        maps = {k: v.get('map50', 0) for k, v in results.items() if 'error' not in v}
        if len(maps) >= 2:
            best_dataset = max(maps, key=maps.get)
            for name, map50 in maps.items():
                if name != best_dataset:
                    drop = maps[best_dataset] - map50
                    analysis['performance_drop'][f"{best_dataset}→{name}"] = float(drop)

            # 泛化能力评估
            min_map = min(maps.values())
            max_map = max(maps.values())
            if max_map - min_map < 0.05:
                analysis['generalization_assessment'] = (
                    f"跨数据集性能差异<5%({max_map:.3f}→{min_map:.3f})，"
                    f"模型泛化能力良好，未过拟合特定数据集"
                )
            elif max_map - min_map < 0.10:
                analysis['generalization_assessment'] = (
                    f"跨数据集性能差异5-10%({max_map:.3f}→{min_map:.3f})，"
                    f"模型有一定泛化能力，但存在数据集偏移"
                )
            else:
                analysis['generalization_assessment'] = (
                    f"跨数据集性能差异>10%({max_map:.3f}→{min_map:.3f})，"
                    f"模型可能过拟合训练数据集，泛化能力不足"
                )

        return analysis


def prepare_pku_market_yaml(data_dir: str, output_path: str = None) -> str:
    """
    为PKU-Market-PCB数据集生成ultralytics格式的dataset.yaml

    Args:
        data_dir: PKU-Market-PCB数据集目录
        output_path: 输出yaml路径

    Returns:
        yaml文件路径
    """
    data_path = Path(data_dir)

    yaml_content = f"""# PKU-Market-PCB Dataset
path: {data_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: 6
names:
  - missing_hole
  - mouse_bite
  - open_circuit
  - short
  - spur
  - spurious_copper
"""

    if output_path is None:
        output_path = str(data_path / "dataset.yaml")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    logger.info(f"PKU-Market-PCB dataset.yaml 已生成: {output_path}")
    return output_path
