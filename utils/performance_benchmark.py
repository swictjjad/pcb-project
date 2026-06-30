# -*- coding: utf-8 -*-
"""
性能基准测试模块
"""

import json
import os
import sys
import time
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger
from utils.common_utils import load_image_broken_path

logger = setup_logger("benchmark")


class PerformanceBenchmark:
    """PCB缺陷检测系统性能基准测试器"""

    def __init__(self, config=None, detector=None):
        self.config = config or get_config()
        self.detector = detector
        self.project_root = get_project_root()
        self.results = {}
        self.class_names = self.config.data.class_names

    def _load_val_images(self):
        """加载验证集图片路径和标注"""
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
        """解析YOLO格式标注文件"""
        if label_path is None or not label_path.exists():
            return []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    labels.append({
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    })
        return labels

    def _compute_per_class_metrics(self, predictions, ground_truths, img_shapes):
        """计算逐类 Precision/Recall/F1"""
        num_classes = len(self.class_names)
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        for preds, gts, shape in zip(predictions, ground_truths, img_shapes):
            pred_classes = set(p["class_id"] for p in preds)
            gt_classes = set(g["class_id"] for g in gts)
            for cls_id in gt_classes:
                if cls_id in pred_classes:
                    tp[cls_id] += 1
                else:
                    fn[cls_id] += 1
            for cls_id in pred_classes:
                if cls_id not in gt_classes:
                    fp[cls_id] += 1
        precisions = []
        recalls = []
        f1_scores = []
        per_class = {}
        for i, name in enumerate(self.class_names):
            p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(round(p, 4))
            recalls.append(round(r, 4))
            f1_scores.append(round(f1, 4))
            per_class[name] = {"precision": p, "recall": r, "f1": f1,
                               "tp": tp[i], "fp": fp[i], "fn": fn[i]}
        macro_p = round(float(np.mean(precisions)), 4) if precisions else 0
        macro_r = round(float(np.mean(recalls)), 4) if recalls else 0
        macro_f1 = round(float(np.mean(f1_scores)), 4) if f1_scores else 0
        return {
            "per_class": per_class,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "class_names": self.class_names
        }

    def benchmark_latency(self, runs=100):
        """推理延迟基准测试（P50/P95/P99）"""
        logger.info(f"开始延迟测试 ({runs} 次推理)...")
        if self.detector is None:
            from inference import ONNXPCBDefectDetector, PCBDefectDetector
            from pathlib import Path
            onnx_path = Path(self.config.inference.model_path).with_suffix(".onnx")
            if onnx_path.exists():
                logger.info("Detected ONNX model, using ONNX backend for benchmark")
                self.detector = ONNXPCBDefectDetector(
                    str(onnx_path),
                    img_size=self.config.inference.img_size,
                    conf_threshold=self.config.inference.conf_threshold,
                    iou_threshold=self.config.inference.iou_threshold,
                    class_names=self.config.data.class_names,
                )
                self.detector.warmup(runs=3)
            else:
                logger.info("No ONNX, using PyTorch")
                self.detector = PCBDefectDetector(self.config)
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.detector.detect(dummy)
        times_ms = []
        for i in range(runs):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            self.detector.detect(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed_ms)
        times_ms = np.array(times_ms)
        result = {
            "runs": runs,
            "mean_ms": round(float(np.mean(times_ms)), 2),
            "std_ms": round(float(np.std(times_ms)), 2),
            "min_ms": round(float(np.min(times_ms)), 2),
            "max_ms": round(float(np.max(times_ms)), 2),
            "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
            "p99_ms": round(float(np.percentile(times_ms, 99)), 2),
            "mean_fps": round(float(1000.0 / np.mean(times_ms)), 2),
        }
        logger.info(f"延迟结果: 均值={result['mean_ms']:.1f}ms, P95={result['p95_ms']:.1f}ms, FPS={result['mean_fps']:.1f}")
        return result

    def benchmark_throughput_stability(self, duration_minutes=5, detector=None):
        """长时间吞吐稳定性测试"""
        logger.info(f"开始稳定性测试 ({duration_minutes} 分钟)...")
        if detector is None:
            detector = self.detector
            if detector is None:
                from inference import ONNXPCBDefectDetector, PCBDefectDetector
                from pathlib import Path
                onnx_path = Path(self.config.inference.model_path).with_suffix('.onnx')
                if onnx_path.exists():
                    logger.info('Using ONNX backend for stability test')
                    detector = ONNXPCBDefectDetector(
                        str(onnx_path),
                        img_size=self.config.inference.img_size,
                        conf_threshold=self.config.inference.conf_threshold,
                        iou_threshold=self.config.inference.iou_threshold,
                        class_names=self.config.data.class_names,
                    )
                    detector.warmup(runs=3)
                else:
                    detector = PCBDefectDetector(self.config)
        end_time = time.time() + duration_minutes * 60
        fps_list = []
        latencies = []
        error_count = 0
        total_frames = 0
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        while time.time() < end_time:
            try:
                t0 = time.perf_counter()
                detector.detect(dummy)
                elapsed = time.perf_counter() - t0
                fps_list.append(1.0 / elapsed if elapsed > 0 else 0)
                latencies.append(elapsed * 1000)
                total_frames += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"稳定性测试帧 {total_frames} 出错: {e}")
        latencies = np.array(latencies) if latencies else np.array([0])
        fps_arr = np.array(fps_list) if fps_list else np.array([0])
        result = {
            "duration_minutes": duration_minutes,
            "total_frames": total_frames,
            "error_count": error_count,
            "avg_fps": round(float(np.mean(fps_arr)), 2),
            "min_fps": round(float(np.min(fps_arr)), 2),
            "max_fps": round(float(np.max(fps_arr)), 2),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "fps_std": round(float(np.std(fps_arr)), 2),
            "fps_stability_ratio": round(float(np.min(fps_arr) / np.max(fps_arr)), 4) if np.max(fps_arr) > 0 else 0,
        }
        logger.info(f"稳定性结果: 共 {total_frames} 帧, 错误 {error_count}, 平均 FPS={result['avg_fps']:.1f}")
        return result

    def benchmark_memory(self, runs=10):
        """内存占用测试"""
        logger.info("开始内存测试...")
        tracemalloc.start()
        if self.detector is None:
            from inference import PCBDefectDetector
            self.detector = PCBDefectDetector(self.config)
        peak_before = tracemalloc.get_traced_memory()[1]
        for i in range(runs):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            self.detector.detect(img)
            if (i + 1) % 3 == 0:
                import gc
                gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_delta = peak - peak_before
        result = {
            "iterations": runs,
            "current_memory_mb": round(current / 1024 / 1024, 2),
            "peak_memory_mb": round(peak / 1024 / 1024, 2),
            "peak_increment_mb": round(peak_delta / 1024 / 1024, 2),
        }
        logger.info(f"内存结果: 当前={result['current_memory_mb']:.1f}MB, 峰值={result['peak_memory_mb']:.1f}MB")
        return result

    def benchmark_on_validation_set(self):
        """在验证集上运行完整基准测试（精度+延迟）"""
        logger.info("=" * 60)
        logger.info("开始在验证集上运行完整基准测试...")
        logger.info("=" * 60)
        img_files, lbl_files = self._load_val_images()
        if len(img_files) == 0:
            logger.warning("验证集为空，跳过精度测试")
            return {"accuracy": {}, "note": "validation set empty"}
        if self.detector is None:
            from inference import ONNXPCBDefectDetector, PCBDefectDetector
            from pathlib import Path
            onnx_path = Path(self.config.inference.model_path).with_suffix(".onnx")
            if onnx_path.exists():
                logger.info("Using ONNX backend for validation benchmark")
                self.detector = ONNXPCBDefectDetector(
                    str(onnx_path),
                    img_size=self.config.inference.img_size,
                    conf_threshold=self.config.inference.conf_threshold,
                    iou_threshold=self.config.inference.iou_threshold,
                    class_names=self.config.data.class_names,
                )
                self.detector.warmup(runs=3)
            else:
                self.detector = PCBDefectDetector(self.config)
        predictions = []
        ground_truths = []
        latencies = []
        detected_count = 0
        for idx, (img_path, lbl_path) in enumerate(zip(img_files, lbl_files)):
            try:
                img = load_image_broken_path(str(img_path))
                if img is None:
                    continue
                t0 = time.perf_counter()
                result = self.detector.detect(img)
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)
                preds = [{"class_id": d["class_id"], "confidence": d["confidence"]}
                          for d in (result[0] if isinstance(result, tuple) else result.get("detections", []))]
                gts = self._parse_yolo_labels(lbl_path)
                predictions.append(preds)
                ground_truths.append(gts)
                detected_count += len(preds)
            except Exception as e:
                logger.warning(f"图片 {img_path.name} 推理失败: {e}")
        latencies = np.array(latencies)
        latency_stats = {
            "samples": len(latencies),
            "mean_ms": round(float(np.mean(latencies)), 2),
            "std_ms": round(float(np.std(latencies)), 2),
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
        }
        img_shapes = [(640, 640)] * len(img_files)
        accuracy = self._compute_per_class_metrics(predictions, ground_truths, img_shapes)
        full_result = {
            "test_type": "validation_set_benchmark",
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.detector.model_path) if self.detector else "unknown",
            "backend": "onnx" if "ONNX" in str(self.detector.__class__) else "pytorch",
            "latency": latency_stats,
            "accuracy": accuracy,
            "total_images": len(img_files),
            "total_detections": detected_count,
            "avg_detections_per_image": round(detected_count / max(len(img_files), 1), 2),
        }
        logger.info(f"验证集基准完成: {len(img_files)} 张图片, 检测到 {detected_count} 个缺陷, 宏平均F1={accuracy['macro_f1']:.4f}")
        return full_result

    def generate_json_report(self, result, output_path=None):
        """生成JSON格式报告"""
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"benchmark_report_{ts}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON报告已保存: {output_path}")
        return str(output_path)

    def generate_markdown_report(self, result, output_path=None):
        """生成Markdown格式报告"""
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"benchmark_report_{ts}.md"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# PCB\u7f3a\u9677\u68c0\u6d4b\u7cfb\u7edf \u6027\u80fd\u57fa\u51c6\u6d4b\u8bd5\u62a5\u544a\n")
        lines.append(f"**\u751f\u6210\u65f6\u95f4**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")
        lines.append("## \u6a21\u578b\u4fe1\u606f\n")
        lines.append(f"- **\u6a21\u578b\u8def\u5f84**: {result.get('model_path', 'N/A')}\n")
        lines.append(f"- **\u63a8\u7406\u540e\u7aef**: {result.get('backend', 'N/A')}\n")
        lines.append(f"- **\u6d4b\u8bd5\u7c7b\u578b**: {result.get('test_type', 'N/A')}\n")
        lines.append(f"- **\u9a8c\u8bc1\u96c6\u56fe\u7247\u6570**: {result.get('total_images', 'N/A')}\n")
        lines.append(f"- **\u603b\u68c0\u6d4b\u6570**: {result.get('total_detections', 'N/A')}\n")
        lines.append("")
        if "latency" in result:
            lat = result["latency"]
            lines.append("## \u63a8\u7406\u5ef6\u8fdf\u7edf\u8ba1\n")
            lines.append("| \u6307\u6807 | \u6570\u503c |")
            lines.append("|------|------|")
            lines.append(f"| \u6837\u672c\u6570 | {lat.get('samples', 'N/A')} |")
            lines.append(f"| \u5e73\u5747\u5ef6\u8fdf | {lat.get('mean_ms', 'N/A')} ms |")
            lines.append(f"| \u6807\u51c6\u5dee | {lat.get('std_ms', 'N/A')} ms |")
            lines.append(f"| P50 (\u4e2d\u4f4d\u6570) | {lat.get('p50_ms', 'N/A')} ms |")
            lines.append(f"| P95 | {lat.get('p95_ms', 'N/A')} ms |")
            lines.append(f"| P99 | {lat.get('p99_ms', 'N/A')} ms |")
            lines.append(f"| \u6700\u5c0f\u503c | {lat.get('min_ms', 'N/A')} ms |")
            lines.append(f"| \u6700\u5927\u503c | {lat.get('max_ms', 'N/A')} ms |")
            mean_val = lat.get('mean_ms', 1)
            if mean_val == 0: mean_val = 1
            lines.append(f"| \u5e73\u5747FPS | {1000.0 / mean_val:.1f} |")
            lines.append("")
        if "accuracy" in result and result["accuracy"]:
            acc = result["accuracy"]
            lines.append("## \u68c0\u6d4b\u7cbe\u5ea6\n")
            lines.append(f"- **\u5b8f\u5e73\u5747 Precision**: {acc.get('macro_precision', 0):.4f}")
            lines.append(f"- **\u5b8f\u5e73\u5747 Recall**: {acc.get('macro_recall', 0):.4f}")
            lines.append(f"- **\u5b8f\u5e73\u5747 F1**: {acc.get('macro_f1', 0):.4f}\n")
            pc = acc.get("per_class", {})
            if pc:
                lines.append("### \u9010\u7c7b\u6307\u6807\n")
                lines.append("| \u7c7b\u522b | Precision | Recall | F1 | TP | FP | FN |")
                lines.append("|------|-----------|--------|-----|----|----|----|")
                for name in acc.get("class_names", []):
                    if name in pc:
                        c = pc[name]
                        lines.append(f"| {name} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['tp']} | {c['fp']} | {c['fn']} |")
                lines.append("")
        if "stability" in result:
            st = result["stability"]
            lines.append("## \u541e\u5410\u7a33\u5b9a\u6027\n")
            lines.append("| \u6307\u6807 | \u6570\u503c |")
            lines.append("|------|------|")
            lines.append(f"| \u6d4b\u8bd5\u65f6\u957f | {st.get('duration_minutes', 'N/A')} \u5206\u949f |")
            lines.append(f"| \u603b\u5e27\u6570 | {st.get('total_frames', 'N/A')} |")
            lines.append(f"| \u9519\u8bef\u6570 | {st.get('error_count', 'N/A')} |")
            lines.append(f"| \u5e73\u5747FPS | {st.get('avg_fps', 'N/A')} |")
            lines.append(f"| FPS\u6807\u51c6\u5dee | {st.get('fps_std', 'N/A')} |")
            lines.append(f"| \u7a33\u5b9a\u6027\u6bd4 | {st.get('fps_stability_ratio', 'N/A')} |")
            lines.append("")
        lines.append("---\n")
        lines.append("*\u672c\u62a5\u544a\u7531 PCB-Defect-Benchmark \u81ea\u52a8\u751f\u6210*\n")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        logger.info(f"Markdown\u62a5\u544a\u5df2\u4fdd\u5b58: {output_path}")
        return str(output_path)

    def run_full_benchmark(self, stability_minutes=3, output_dir=None):
        """运行完整基准测试套件"""
        logger.info("=" * 70)
        logger.info("开始完整性能基准测试套件")
        logger.info("=" * 70)
        full_result = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
        }
        logger.info("\n[1/4] 延迟基准测试...")
        latency = self.benchmark_latency(runs=100)
        full_result["latency"] = latency
        logger.info("\n[2/4] 验证集精度测试...")
        accuracy = self.benchmark_on_validation_set()
        full_result.update(accuracy)
        logger.info("\n[3/4] 吞吐稳定性测试...")
        if self.detector is None:
            from inference import ONNXPCBDefectDetector, PCBDefectDetector
            from pathlib import Path
            onnx_path = Path(self.config.inference.model_path).with_suffix(".onnx")
            if onnx_path.exists():
                logger.info("Detected ONNX model, using ONNX backend for benchmark")
                self.detector = ONNXPCBDefectDetector(
                    str(onnx_path),
                    img_size=self.config.inference.img_size,
                    conf_threshold=self.config.inference.conf_threshold,
                    iou_threshold=self.config.inference.iou_threshold,
                    class_names=self.config.data.class_names,
                )
                self.detector.warmup(runs=3)
            else:
                logger.info("No ONNX, using PyTorch")
                self.detector = PCBDefectDetector(self.config)
        stability = self.benchmark_throughput_stability(duration_minutes=stability_minutes, detector=self.detector)
        full_result["stability"] = stability
        logger.info("\n[4/4] 内存占用测试...")
        memory = self.benchmark_memory(runs=20)
        full_result["memory"] = memory
        results_dir = self.project_root / "results" / "benchmarks"
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.generate_json_report(full_result, results_dir / "latest_benchmark.json")
        md_path = self.generate_markdown_report(full_result, results_dir / "latest_benchmark.md")
        full_result["reports"] = {"json": json_path, "markdown": md_path}
        logger.info("\n" + "=" * 70)
        logger.info("完整基准测试完成！")
        logger.info(f"  JSON报告: {json_path}")
        logger.info(f"  Markdown报告: {md_path}")
        logger.info("=" * 70)
        return full_result
