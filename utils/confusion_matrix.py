# -*- coding: utf-8 -*-
"""
混淆矩阵与误报分析模块
功能：
1. 在验证集上生成逐类混淆矩阵
2. 计算每类假阳性率(FP Rate)
3. 误报成本计算器：每周浪费的检测工时
4. 集成到现有 FalseAlarmSuppressor
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger
from utils.common_utils import load_image_broken_path

logger = setup_logger("confusion_matrix")


class ConfusionMatrixAnalyzer:
    """PCB缺陷检测混淆矩阵分析器"""

    def __init__(self, config=None, detector=None):
        self.config = config or get_config()
        self.detector = detector
        self.project_root = get_project_root()
        self.class_names = self.config.data.class_names
        self.num_classes = len(self.class_names)

    def _load_val_images(self):
        """加载验证集图片"""
        val_img_dir = self.project_root / self.config.data.dataset_path / "val" / "images"
        val_lbl_dir = self.project_root / self.config.data.dataset_path / "val" / "labels"
        if not val_img_dir.exists():
            logger.error(f"验证集目录不存在: {val_img_dir}")
            return [], []
        img_files = sorted([f for f in val_img_dir.iterdir()
                           if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        lbl_files = []
        for img_f in img_files:
            lbl_f = val_lbl_dir / f"{img_f.stem}.txt"
            lbl_files.append(lbl_f if lbl_f.exists() else None)
        logger.info(f"加载验证集: {len(img_files)} 张图片")
        return img_files, lbl_files

    def _parse_yolo_labels(self, label_path):
        """解析YOLO标注，返回类别列表"""
        if label_path is None or not label_path.exists():
            return []
        classes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes.append(int(parts[0]))
        return classes

    def compute_confusion_matrix(self, detector=None):
        """
        在验证集上计算混淆矩阵

        Returns:
            dict: 包含混淆矩阵、逐类指标、FP率分析
        """
        logger.info("=" * 60)
        logger.info("开始计算混淆矩阵...")
        logger.info("=" * 60)

        if detector is None:
            if self.detector is None:
                from inference import PCBDefectDetector
                detector = PCBDefectDetector(self.config)
            else:
                detector = self.detector

        img_files, lbl_files = self._load_val_images()
        if len(img_files) == 0:
            logger.warning("验证集为空")
            return {}

        # TP/FP/FN per class
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for idx, (img_path, lbl_path) in enumerate(zip(img_files, lbl_files)):
            try:
                img = load_image_broken_path(str(img_path))
                if img is None:
                    continue
                result = detector.detect(img)
                detections = (result[0] if isinstance(result, tuple) else result.get("detections", []))

                gt_classes = set(self._parse_yolo_labels(lbl_path))
                pred_classes = set(d["class_id"] for d in detections)

                # 填充混淆矩阵：每张图片只统计是否存在该类别
                for cls_id in gt_classes:
                    if cls_id in pred_classes:
                        cm[cls_id, cls_id] += 1  # TP
                    else:
                        cm[cls_id, cls_id] += 0  # FN counted below
                    # FN: GT存在但预测不存在
                    if cls_id not in pred_classes:
                        cm[cls_id, -1 if cls_id == self.num_classes - 1 else cls_id]  # placeholder
                for cls_id in pred_classes:
                    if cls_id not in gt_classes:
                        pass  # FP
            except Exception as e:
                logger.warning(f"图片 {img_path.name} 处理失败: {e}")

        # 更精确的逐类统计
        tp = [0] * self.num_classes
        fp = [0] * self.num_classes
        fn = [0] * self.num_classes

        for idx, (img_path, lbl_path) in enumerate(zip(img_files, lbl_files)):
            try:
                img = load_image_broken_path(str(img_path))
                if img is None:
                    continue
                result = detector.detect(img)
                detections = (result[0] if isinstance(result, tuple) else result.get("detections", []))
                gt_classes = set(self._parse_yolo_labels(lbl_path))
                pred_classes = set(d["class_id"] for d in detections)

                for cls_id in range(self.num_classes):
                    in_gt = cls_id in gt_classes
                    in_pred = cls_id in pred_classes
                    if in_gt and in_pred:
                        tp[cls_id] += 1
                    elif not in_gt and in_pred:
                        fp[cls_id] += 1
                    elif in_gt and not in_pred:
                        fn[cls_id] += 1
            except Exception as e:
                logger.warning(f"图片 {img_path.name} 处理失败: {e}")

        # 计算指标
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        fp_rate_per_class = []

        per_class = {}
        for i, name in enumerate(self.class_names):
            p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            fp_rate = fp[i] / (fp[i] + tp[i]) if (fp[i] + tp[i]) > 0 else 0.0

            precision_per_class.append(p)
            recall_per_class.append(r)
            f1_per_class.append(f1)
            fp_rate_per_class.append(fp_rate)

            per_class[name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "fp_rate": round(fp_rate, 4),
                "tp": tp[i],
                "fp": fp[i],
                "fn": fn[i],
            }

        macro_p = round(float(np.mean(precision_per_class)), 4)
        macro_r = round(float(np.mean(recall_per_class)), 4)
        macro_f1 = round(float(np.mean(f1_per_class)), 4)

        result = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(img_files),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "class_names": self.class_names,
        }

        logger.info(f"混淆矩阵计算完成: 宏平均F1={macro_f1:.4f}")
        return result

    def generate_confusion_matrix_image(self, result, output_path=None):
        """生成混淆矩阵可视化图片"""
        if result is None or "confusion_matrix" not in result:
            logger.warning("没有混淆矩阵数据")
            return None

        cm = np.array(result["confusion_matrix"])
        class_names = result.get("class_names", [f"c{i}" for i in range(cm.shape[0])])
        n = cm.shape[0]

        cell_size = 80
        margin = 100
        font_scale = 0.6
        img_size = margin + n * cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # 标题
        cv2.putText(img, "Confusion Matrix", (margin // 2, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        max_val = cm.max() if cm.max() > 0 else 1

        for i in range(n):
            # 行标签
            cv2.putText(img, class_names[i], (5, margin + i * cell_size + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
            # 列标签
            cv2.putText(img, class_names[i], (i * cell_size + cell_size // 2 - 20, margin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

            for j in range(n):
                x = j * cell_size + margin
                y = i * cell_size + margin
                val = cm[i][j]
                ratio = val / max_val

                # 热力图颜色
                color = (int(255 * (1 - ratio)), int(255 * ratio), int(255 * (1 - ratio * 0.5)))
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), color, -1)
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 1)
                cv2.putText(img, str(val), (x + cell_size // 2 - 10, y + cell_size // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / "confusion_matrix.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), img)
        logger.info(f"混淆矩阵图片已保存: {output_path}")
        return str(output_path)

    def export_json(self, result, output_path=None):
        """导出JSON报告"""
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"confusion_matrix_{ts}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"混淆矩阵JSON已保存: {output_path}")
        return str(output_path)


class FalseAlarmCostCalculator:
    """误报成本计算器"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.project_root = get_project_root()
        # 默认参数（可通过配置文件覆盖）
        self.inspector_hourly_rate = self.config.get("false_alarm", {}).get("inspector_rate", 50)
        self.inspectors_per_shift = self.config.get("false_alarm", {}).get("inspectors", 2)
        self.shifts_per_day = self.config.get("false_alarm", {}).get("shifts", 2)
        self.working_days_per_week = self.config.get("false_alarm", {}).get("days_per_week", 6)

    def calculate(self, fp_per_image=0, images_per_hour=10, weeks=4):
        """
        计算误报成本

        Args:
            fp_per_image: 平均每张图片的假阳性数
            images_per_hour: 每小时处理的图片数
            weeks: 计算周期（周）

        Returns:
            dict: 成本分析结果
        """
        images_per_week = images_per_hour * 8 * self.shifts_per_day * self.working_days_per_week * weeks
        total_fp = int(fp_per_image * images_per_week)

        # 每次误报需要人工复检的时间（分钟）
        recheck_time_min = 5
        total_recheck_hours = total_fp * recheck_time_min / 60
        total_cost = total_recheck_hours * self.inspector_hourly_rate * self.inspectors_per_shift

        weekly_fp = int(fp_per_image * images_per_hour * 8 * self.shifts_per_day * self.working_days_per_week)
        weekly_cost = total_cost / weeks

        result = {
            "calculation_params": {
                "fp_per_image": fp_per_image,
                "images_per_hour": images_per_hour,
                "shifts_per_day": self.shifts_per_day,
                "working_days_per_week": self.working_days_per_week,
                "weeks": weeks,
                "inspector_hourly_rate": self.inspector_hourly_rate,
                "recheck_time_min": recheck_time_min,
            },
            "weekly_fp_count": weekly_fp,
            "weekly_recheck_hours": round(total_recheck_hours / weeks, 2),
            "weekly_false_alarm_cost": round(weekly_cost, 2),
            "total_period_fp_count": total_fp,
            "total_period_cost": round(total_cost, 2),
            "currency": "CNY",
        }

        logger.info(f"误报成本: 每周FP={weekly_fp}, 每周成本={weekly_cost:.2f} CNY")
        return result

    def calculate_from_analysis(self, confusion_result, images_per_hour=10, weeks=4):
        """从混淆矩阵分析结果计算误报成本"""
        if not confusion_result or "per_class" not in confusion_result:
            return {"error": "no confusion matrix data"}

        total_fp = sum(v.get("fp", 0) for v in confusion_result["per_class"].values())
        total_images = confusion_result.get("total_images", 1)
        fp_per_image = total_fp / total_images

        return self.calculate(fp_per_image=fp_per_image, images_per_hour=images_per_hour, weeks=weeks)

    def export_report(self, cost_result, output_path=None):
        """导出误报成本报告"""
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"false_alarm_cost_{ts}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cost_result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"误报成本报告已保存: {output_path}")
        return str(output_path)
