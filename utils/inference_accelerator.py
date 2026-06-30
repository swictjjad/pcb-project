# -*- coding: utf-8 -*-
"""
推理加速模块
"""

import os
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger

logger = setup_logger("accelerator")


class QuantizationEngine:
    """INT8 后训练量化引擎"""

    def __init__(self, detector=None):
        self.detector = detector
        self.quantized_model_path = None

    def export_onnx(self, detector, output_path=None):
        if detector is None:
            from inference import PCBDefectDetector
            detector = PCBDefectDetector(get_config())
        if output_path is None:
            results_dir = get_project_root() / "models"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / "model_quantizable.onnx"
        try:
            detector.model.export(format="onnx", half=True, dynamic=False)
            logger.info(f"ONNX 导出成功: {output_path}")
            self.quantized_model_path = str(output_path)
            return str(output_path)
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            return None

    def apply_dynamic_quantization(self, model=None):
        if model is None:
            if self.detector is None:
                from inference import PCBDefectDetector
                self.detector = PCBDefectDetector(get_config())
            model = self.detector.model
        try:
            import torch
            quantized_model = torch.quantization.quantize_dynamic(
                model.model if hasattr(model, 'model') else model,
                {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("动态 INT8 量化完成")
            return quantized_model
        except ImportError:
            logger.warning("torch.quantization 不可用")
            return None
        except Exception as e:
            logger.error(f"量化失败: {e}")
            return None


class BatchProcessor:
    """批处理推理优化器"""

    def __init__(self, batch_size=4, detector=None):
        self.batch_size = batch_size
        self.detector = detector
        self.frame_buffer = deque(maxlen=batch_size)
        self.lock = threading.Lock()

    def submit_frame(self, image):
        with self.lock:
            self.frame_buffer.append(image)
            if len(self.frame_buffer) >= self.batch_size:
                return self._process_batch()
        return None

    def _process_batch(self):
        with self.lock:
            batch = list(self.frame_buffer)
            self.frame_buffer.clear()
        if self.detector is None:
            from inference import PCBDefectDetector
            self.detector = PCBDefectDetector(get_config())
        results = []
        for img in batch:
            try:
                results.append(self.detector.detect(img))
            except Exception as e:
                logger.warning(f"批处理帧推理失败: {e}")
                results.append({"detections": [], "num_detections": 0, "inference_time": 0, "fps": 0})
        return results

    def flush(self):
        with self.lock:
            batch = list(self.frame_buffer)
            self.frame_buffer.clear()
        if not batch or self.detector is None:
            return []
        from inference import PCBDefectDetector
        if self.detector is None:
            self.detector = PCBDefectDetector(get_config())
        results = []
        for img in batch:
            try:
                results.append(self.detector.detect(img))
            except Exception as e:
                logger.warning(f"flush 推理失败: {e}")
                results.append({"detections": [], "num_detections": 0, "inference_time": 0, "fps": 0})
        return results


class InferencePipeline:
    """多线程推理流水线"""

    def __init__(self, detector=None, buffer_size=2):
        self.detector = detector
        self.buffer_size = buffer_size
        self.preprocess_queue = queue.Queue(maxsize=buffer_size)
        self.infer_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.threads = []
        self.stats = {"frames_captured": 0, "frames_processed": 0, "frames_dropped": 0, "start_time": None}

    def _preprocess_worker(self):
        while self.running:
            try:
                frame_id, image = self.preprocess_queue.get(timeout=0.1)
                if image is not None:
                    model_input_size = 640
                    h, w = image.shape[:2]
                    ratio = min(model_input_size / w, model_input_size / h)
                    new_w, new_h = int(w * ratio), int(h * ratio)
                    resized = cv2.resize(image, (new_w, new_h))
                    pad_w = (model_input_size - new_w) // 2
                    pad_h = (model_input_size - new_h) // 2
                    padded = np.full((model_input_size, model_input_size, 3), 114, dtype=np.uint8)
                    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
                    self.infer_queue.put((frame_id, padded))
                self.preprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"预处理线程异常: {e}")

    def _infer_worker(self):
        if self.detector is None:
            from inference import PCBDefectDetector
            self.detector = PCBDefectDetector(get_config())
        while self.running:
            try:
                frame_id, image = self.infer_queue.get(timeout=0.1)
                if image is not None:
                    t0 = time.perf_counter()
                    result = self.detector.detect(image)
                    result["frame_id"] = frame_id
                    result["infer_time_ms"] = (time.perf_counter() - t0) * 1000
                    self.result_queue.put(result)
                self.infer_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"推理线程异常: {e}")

    def start(self):
        self.running = True
        self.stats["start_time"] = datetime.now().isoformat()
        t1 = threading.Thread(target=self._preprocess_worker, daemon=True)
        t2 = threading.Thread(target=self._infer_worker, daemon=True)
        self.threads = [t1, t2]
        for t in self.threads:
            t.start()
        logger.info("推理流水线已启动")

    def submit(self, frame_id, image):
        try:
            self.preprocess_queue.put_nowait((frame_id, image))
            self.stats["frames_captured"] += 1
        except queue.Full:
            self.stats["frames_dropped"] += 1

    def get_result(self, timeout=1.0):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=2.0)
        logger.info("推理流水线已停止")

    def get_stats(self):
        total = max(self.stats["frames_captured"], 1)
        return {**self.stats, "drop_rate": round(self.stats["frames_dropped"] / total, 4)}
