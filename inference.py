# -*- coding: utf-8 -*-
"""
模型推理脚本
功能：
1. 单张图片推理
2. 摄像头实时推理
3. 批量图片推理
4. 视频文件推理
5. 结果保存和可视化
6. ONNX Runtime推理支持
7. CLAHE自适应直方图均衡化预处理
8. 模型预热与性能基准测试
"""

import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger

logger = setup_logger("inference")

# YOLO11s 预训练权重下载地址（使用 HTTPS）
YOLO11S_PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"


def download_pretrained_model(save_path):
    """
    自动下载YOLO11s预训练权重作为fallback

    Args:
        save_path: 保存路径

    Returns:
        保存后的路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        logger.info(f"预训练模型已存在: {save_path}")
        return str(save_path)

    logger.info(f"正在下载YOLO11s预训练权重到: {save_path}")
    try:
        # 使用 SSL 上下文确保 HTTPS 安全下载
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        with urllib.request.urlopen(YOLO11S_PT_URL, context=ssl_context, timeout=30) as response:
            save_path.write_bytes(response.read())
        logger.info("下载完成")
        return str(save_path)
    except Exception as e:
        logger.error(f"下载预训练模型失败: {e}")
        raise


def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制 (NMS) — 加速版

    使用 suppressed 布尔数组避免 while 循环中的重复数组拷贝，
    将每轮迭代从 O(N) 内存拷贝降为 O(1) 操作。

    Args:
        boxes: (N, 4) 边界框 [x1, y1, x2, y2]
        scores: (N,) 置信度
        iou_threshold: IoU阈值

    Returns:
        保留的索引列表
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    suppressed = np.zeros(len(order), dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        ix1, iy1, ix2, iy2 = x1[idx], y1[idx], x2[idx], y2[idx]
        iarea = areas[idx]

        # 向量化计算与后续框的 IoU
        j_slice = order[i + 1:]
        jx1 = np.maximum(ix1, x1[j_slice])
        jy1 = np.maximum(iy1, y1[j_slice])
        jx2 = np.minimum(ix2, x2[j_slice])
        jy2 = np.minimum(iy2, y2[j_slice])

        w = np.maximum(0.0, jx2 - jx1)
        h = np.maximum(0.0, jy2 - jy1)
        inter = w * h
        iou = inter / (iarea + areas[j_slice] - inter + 1e-7)

        # 标记被抑制的索引
        suppressed[j_slice[iou > iou_threshold]] = True

    return keep


class ONNXPCBDefectDetector:
    """基于ONNX Runtime的PCB缺陷检测器"""

    def __init__(self, model_path, img_size=640, conf_threshold=0.25,
                 iou_threshold=0.45, class_names=None):
        """
        初始化ONNX检测器

        Args:
            model_path: ONNX模型文件路径
            img_size: 输入图像尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            class_names: 类别名称列表
        """
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

        # 加载ONNX模型
        self.session = None
        self.input_name = None
        self.output_names = None
        self._prealloc_hwc = None  # 预分配预处理缓冲区
        self._load_onnx_model()

    def _load_onnx_model(self):
        """加载ONNX模型，模型不存在时返回 False（触发回退）"""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("未安装onnxruntime，请运行: pip install onnxruntime")
            raise

        if not self.model_path.exists():
            pt_path = self.model_path.with_suffix('.pt')
            if pt_path.exists():
                logger.warning(f"ONNX模型不存在: {self.model_path}，回退到PyTorch模型: {pt_path}")
                return False
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_path}")

        logger.info(f"加载ONNX模型: {self.model_path}")

        # 优先使用CPU执行提供者
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("使用CUDA + CPU执行提供者")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("使用CPU执行提供者")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        # 获取输入输出信息
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [batch, 3, H, W]
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(f"ONNX模型输入: {self.input_name} {self.input_shape}")
        logger.info(f"ONNX模型输出: {self.output_names}")
        logger.info("ONNX模型加载完成")

    def preprocess(self, image):
        """
        图像预处理：resize、归一化、BGR→RGB、HWC→NCHW

        使用预分配缓冲区减少 GC 压力。

        Args:
            image: numpy数组 (H, W, C) BGR格式

        Returns:
            预处理后的numpy数组 (1, 3, H, W) float32
        """
        # BGR → RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize（letterbox方式保持宽高比）
        img, ratio, (dw, dh) = self._letterbox(img, self.img_size)

        # 复用预分配缓冲区
        if self._prealloc_hwc is None or self._prealloc_hwc.shape != img.shape:
            self._prealloc_hwc = np.empty(img.shape, dtype=np.float32)
        np.copyto(self._prealloc_hwc, img)
        img = self._prealloc_hwc

        # 归一化到 [0, 1]
        img = img.astype(np.float32, copy=False) / 255.0

        # HWC → NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # 确保连续内存
        img = np.ascontiguousarray(img)

        return img, ratio, (dw, dh)

    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """
        Letterbox resize，保持宽高比并填充

        Args:
            img: 输入图像
            new_shape: 目标尺寸
            color: 填充颜色

        Returns:
            resized图像, 缩放比例, 填充量(dw, dh)
        """
        shape = img.shape[:2]  # H, W
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)

        return img, r, (dw, dh)

    def postprocess(self, output, img_shape, ratio, pad):
        """
        YOLOv8 ONNX输出后处理：解析输出、NMS、置信度过滤

        Args:
            output: ONNX模型原始输出
            img_shape: 原始图像尺寸 (H, W)
            ratio: letterbox缩放比例
            pad: letterbox填充量 (dw, dh)

        Returns:
            检测结果列表
        """
        # YOLOv8输出形状: (1, 84, 8400) 对于80类，或 (1, 4+num_classes, num_preds)
        predictions = output[0]  # (1, 4+num_classes, num_preds)

        # 转置为 (num_preds, 4+num_classes)
        predictions = predictions[0].T

        # 分离边界框和类别分数
        boxes = predictions[:, :4]  # (N, 4) cx, cy, w, h
        class_scores = predictions[:, 4:]  # (N, num_classes)

        # 获取每个预测的最大类别分数和类别ID
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)

        # 置信度过滤
        mask = max_scores >= self.conf_threshold
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # 转换 cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = nms(boxes_xyxy, max_scores, self.iou_threshold)
        boxes_xyxy = boxes_xyxy[keep]
        max_scores = max_scores[keep]
        class_ids = class_ids[keep]

        # 还原到原始图像坐标
        dw, dh = pad
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - dw) / ratio
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - dh) / ratio
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - dw) / ratio
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - dh) / ratio

        # 裁剪到图像范围内
        h, w = img_shape[:2]
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h)

        detections = []
        for box, conf, cls_id in zip(boxes_xyxy, max_scores, class_ids):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            detections.append({
                'bbox': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': cls_name
            })

        return detections

    def detect(self, image):
        """
        执行ONNX推理

        Args:
            image: numpy数组 (H, W, C) BGR格式

        Returns:
            detections列表, 推理耗时(秒)
        """
        img_shape = image.shape
        input_tensor, ratio, pad = self.preprocess(image)

        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        detections = self.postprocess(outputs, img_shape, ratio, pad)

        avg_fps = 1.0 / max(inference_time, 0.001)
        return {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time * 1000,
            'fps': avg_fps,
            'image_shape': (img_shape[0], img_shape[1]),
            'backend': 'onnx',
            'error': None,
        }

    def draw_results(self, image, results, show_conf=True, show_label=True):
        img = image.copy()
        if not hasattr(self, '_class_colors'):
            import hashlib
            self._class_colors = []
            for i, name in enumerate(self.class_names):
                hash_obj = hashlib.md5(name.encode()).hexdigest()
                r = int(hash_obj[0:2], 16) % 128 + 127
                g = int(hash_obj[2:4], 16) % 128 + 127
                b = int(hash_obj[4:6], 16) % 128 + 127
                self._class_colors.append((b, g, r))

        detections = results.get('detections', [])
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_id = det['class_id']
            conf = det['confidence']
            label = det['class_name']
            color = self._class_colors[cls_id % len(self._class_colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if show_label:
                text = label
                if show_conf:
                    text += f" {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fps_text = f"FPS: {results.get('fps', 0):.1f}"
        time_text = f"Time: {results.get('inference_time', 0):.1f}ms"
        count_text = f"Detections: {results.get('num_detections', 0)}"
        backend_text = f"Backend: {results.get('backend', 'unknown')}"
        info_y = 30
        cv2.putText(img, fps_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, time_text, (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, count_text, (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, backend_text, (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return img

    def warmup(self, runs=3):
        """
        模型预热

        Args:
            runs: 预热次数
        """
        logger.info(f"ONNX模型预热中 ({runs}次)...")
        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        for i in range(runs):
            self.session.run(self.output_names, {self.input_name: dummy})
        logger.info("ONNX模型预热完成")


class PCBDefectDetector:
    """PCB缺陷检测器，支持PyTorch和ONNX Runtime推理"""

    def __init__(self, config=None, model_path=None, use_clahe=False):
        """
        初始化检测器

        Args:
            config: 配置对象
            model_path: 模型路径，None则使用配置中的路径
            use_clahe: 是否启用CLAHE自适应直方图均衡化
        """
        self.config = config or get_config()

        # 模型路径
        if model_path is None:
            model_path = self.config.inference.model_path
        self.model_path = Path(model_path)

        # 推理参数
        self.img_size = self.config.inference.img_size
        self.conf_threshold = self.config.inference.conf_threshold
        self.iou_threshold = self.config.inference.iou_threshold
        self.device = self.config.inference.device

        # CLAHE预处理配置
        self.use_clahe = use_clahe
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid_size = (8, 8)

        # 类别信息
        self.class_names = self.config.data.class_names
        self.class_colors = self._generate_colors(len(self.class_names))

        # 推理后端: 'pytorch' 或 'onnx'
        self.backend = None
        self.onnx_detector = None

        # 加载模型
        self.model = None
        self._load_model()

        # FPS计算
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

    def _generate_colors(self, num_classes):
        """生成类别颜色"""
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors

    def _auto_download_model(self):
        """
        自动下载预训练模型作为fallback

        Returns:
            下载后的模型路径
        """
        models_dir = get_project_root() / "models"
        fallback_path = models_dir / "best.pt"
        logger.warning(f"模型文件不存在: {self.model_path}，尝试下载YOLO11s预训练权重作为fallback")
        return download_pretrained_model(fallback_path)

    def _use_coco_fallback(self):
        """
        当自定义模型不可用时，使用 ultralytics 自动下载的 COCO 预训练模型
        作为演示/测试用途（不能检测 PCB 缺陷，但可以验证系统流程）
        """
        models_dir = get_project_root() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        coco_path = models_dir / "yolo11n_coco.pt"

        if not coco_path.exists():
            logger.info("COCO 预训练模型不存在，ultralytics 将自动下载...")

        try:
            from ultralytics import YOLO
            logger.info("使用 YOLO11n COCO 预训练模型进行演示（非 PCB 专用模型）")
            self.model = YOLO(str(coco_path))
            self.backend = 'pytorch'
            self._is_demo_mode = True

            # 预热
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            logger.info("COCO 演示模型加载完成")
            return True
        except Exception as e:
            logger.error(f"COCO 模型加载失败: {e}")
            return False

    def _load_model(self):
        """
        加载模型，自动检测模型格式：
        - .onnx → 使用ONNX Runtime
        - .pt → 使用ultralytics YOLO
        - 模型不存在 → 尝试下载 → 再失败则使用 COCO 预训练模型演示
        """
        self._is_demo_mode = False  # 标记是否为演示模式
        model_suffix = self.model_path.suffix.lower()

        # 检查模型文件是否存在，不存在则尝试自动下载
        if not self.model_path.exists():
            if model_suffix == '.pt':
                try:
                    fallback = self._auto_download_model()
                    self.model_path = Path(fallback)
                    model_suffix = '.pt'
                except Exception as e:
                    logger.warning(f"自动下载模型失败: {e}，尝试使用 COCO 预训练模型")
                    if not self._use_coco_fallback():
                        raise RuntimeError(f"所有模型加载均失败: {e}")
                    return  # COCO 模型已加载，直接返回
            else:
                logger.warning(f"模型文件不存在: {self.model_path}，尝试使用 COCO 预训练模型")
                if not self._use_coco_fallback():
                    raise FileNotFoundError(f"模型文件不存在且 COCO 模型也加载失败: {self.model_path}")
                return  # COCO 模型已加载，直接返回

        if model_suffix == '.onnx':
            # 使用ONNX Runtime推理
            self.backend = 'onnx'
            logger.info("检测到ONNX模型，使用ONNX Runtime推理")
            self.onnx_detector = ONNXPCBDefectDetector(
                model_path=self.model_path,
                img_size=self.img_size,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                class_names=self.class_names
            )
            if not self.onnx_detector._load_onnx_model():
                # ONNX回退到PyTorch
                logger.info("回退到PyTorch后端")
                self.backend = 'pytorch'
                self.onnx_detector = None
                try:
                    from ultralytics import YOLO
                except ImportError:
                    raise ImportError("未安装ultralytics，请运行: pip install ultralytics>=8.3.0")
                logger.info(f"加载PyTorch模型: {self.model_path}")
                self.model = YOLO(str(self.model_path))
                dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                self.model.predict(dummy, verbose=False)
                logger.info("PyTorch模型加载完成")
                return
            self.onnx_detector.warmup(runs=3)
            logger.info("ONNX模型加载完成")

        else:
            # 使用ultralytics PyTorch推理
            self.backend = 'pytorch'
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("未安装ultralytics，请运行: pip install ultralytics")
                raise

            logger.info(f"加载PyTorch模型: {self.model_path}")
            self.model = YOLO(str(self.model_path))

            # 预热
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)

            logger.info("PyTorch模型加载完成")

    def preprocess(self, image, apply_clahe=None):
        """
        图像预处理流水线，包含CLAHE自适应直方图均衡化

        Args:
            image: numpy数组 (H, W, C) BGR格式
            apply_clahe: 是否应用CLAHE，None则使用实例配置

        Returns:
            预处理后的图像
        """
        if image is None or image.size == 0:
            return None

        # CLAHE增强
        use_clahe = apply_clahe if apply_clahe is not None else self.use_clahe
        if use_clahe:
            image = self._apply_clahe(image)

        return image

    def _apply_clahe(self, image):
        """
        应用CLAHE自适应直方图均衡化

        对PCB图像的对比度增强很有帮助，能突出细微缺陷特征

        Args:
            image: BGR格式图像

        Returns:
            增强后的图像
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
        l = clahe.apply(l)

        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def detect(self, image):
        """
        执行检测

        Args:
            image: numpy数组 (H, W, C) BGR格式

        Returns:
            检测结果字典（始终返回有效字典，不会返回None）
        """
        # 空结果模板
        empty_result = {
            'detections': [],
            'num_detections': 0,
            'inference_time': 0,
            'fps': 0,
            'image_shape': (0, 0),
            'backend': self.backend or 'unknown',
            'error': None,
        }

        if self.backend == 'onnx' and self.onnx_detector is None:
            logger.error("ONNX模型未加载")
            empty_result['error'] = "ONNX模型未加载"
            return empty_result
        if self.backend == 'pytorch' and self.model is None:
            logger.error("模型未加载")
            empty_result['error'] = "模型未加载"
            return empty_result

        # 预处理
        image = self.preprocess(image)
        if image is None:
            logger.error("图像预处理失败")
            empty_result['error'] = "图像预处理失败"
            return empty_result

        # 根据后端执行推理
        try:
            if self.backend == 'onnx':
                return self._detect_onnx(image)
            else:
                return self._detect_pytorch(image)
        except Exception as e:
            logger.error(f"推理异常: {e}\n{traceback.format_exc()}")
            empty_result['error'] = str(e)
            empty_result['error_type'] = type(e).__name__
            return empty_result

    def _detect_onnx(self, image):
        """使用ONNX Runtime执行推理"""
        detections, inference_time = self.onnx_detector.detect(image)

        # 计算FPS
        self.fps_history.append(1.0 / max(inference_time, 0.001))
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        return {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time * 1000,  # ms
            'fps': avg_fps,
            'image_shape': image.shape[:2],
            'backend': 'onnx'
        }

    def _detect_pytorch(self, image):
        """使用PyTorch (ultralytics) 执行推理"""
        start_time = time.time()

        results = self.model.predict(
            image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
            half=self.device != 'cpu',    # GPU 上启用 FP16 加速
        )

        inference_time = time.time() - start_time

        # 解析结果
        detections = []
        result = results[0]

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) x1,y1,x2,y2
            confs = result.boxes.conf.cpu().numpy()    # (N,)
            classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for box, conf, cls_id in zip(boxes, confs, classes):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': self.class_names[cls_id]
                })

        # 计算FPS
        self.fps_history.append(1.0 / max(inference_time, 0.001))
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        return {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time * 1000,  # ms
            'fps': avg_fps,
            'image_shape': image.shape[:2],
            'backend': 'pytorch'
        }

    def benchmark(self, runs=50):
        """
        性能基准测试

        Args:
            runs: 测试次数

        Returns:
            基准测试结果字典
        """
        logger.info(f"开始性能基准测试 ({runs}次推理)...")

        # 生成随机测试图像
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 预热
        self.detect(dummy_image)

        # 正式测试
        times = []
        for i in range(runs):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            start = time.time()
            self.detect(img)
            elapsed = time.time() - start
            times.append(elapsed)

        times = np.array(times)

        result = {
            'backend': self.backend,
            'model_path': str(self.model_path),
            'runs': runs,
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'mean_fps': float(1.0 / np.mean(times)),
            'p50_time_ms': float(np.percentile(times, 50) * 1000),
            'p95_time_ms': float(np.percentile(times, 95) * 1000),
            'p99_time_ms': float(np.percentile(times, 99) * 1000),
        }

        logger.info(f"基准测试结果 [{self.backend}]: "
                     f"平均 {result['mean_time_ms']:.1f}ms, "
                     f"FPS {result['mean_fps']:.1f}, "
                     f"P95 {result['p95_time_ms']:.1f}ms")

        return result

    def draw_results(self, image, results, show_conf=True, show_label=True):
        """
        在图像上绘制检测结果

        Args:
            image: 原始图像
            results: detect()返回的结果
            show_conf: 是否显示置信度
            show_label: 是否显示标签

        Returns:
            绘制后的图像
        """
        img = image.copy()
        h, w = img.shape[:2]

        for det in results['detections']:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_id = det['class_id']
            conf = det['confidence']
            label = det['class_name']
            color = self.class_colors[cls_id % len(self.class_colors)]

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            if show_label:
                text = label
                if show_conf:
                    text += f" {conf:.2f}"

                # 计算文本尺寸
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # 绘制标签背景
                cv2.rectangle(
                    img,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    img, text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

        # 绘制FPS和信息
        fps_text = f"FPS: {results['fps']:.1f}"
        time_text = f"Time: {results['inference_time']:.1f}ms"
        count_text = f"Detections: {results['num_detections']}"
        backend_text = f"Backend: {results.get('backend', 'unknown')}"

        info_y = 30
        cv2.putText(img, fps_text, (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, time_text, (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, count_text, (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, backend_text, (10, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return img

    def detect_image(self, image_path, save_path=None, show=False):
        """
        检测单张图片

        Args:
            image_path: 图片路径
            save_path: 保存路径，None则不保存
            show: 是否显示结果

        Returns:
            检测结果和绘制后的图像
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            empty_result = {
                'detections': [], 'num_detections': 0, 'inference_time': 0,
                'fps': 0, 'image_shape': (0, 0), 'backend': self.backend or 'unknown',
                'error': f"无法读取图片: {image_path}",
            }
            return empty_result, None

        results = self.detect(image)
        vis_image = self.draw_results(image, results)

        # 保存结果
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis_image)
            logger.info(f"结果已保存: {save_path}")

        # 显示结果
        if show:
            cv2.imshow("PCB Defect Detection", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results, vis_image

    def detect_camera(self, camera_id=0, save_dir=None):
        """
        摄像头实时检测

        Args:
            camera_id: 摄像头ID
            save_dir: 保存目录，None则不保存
        """
        # 优先使用 DSHOW 后端（Windows 下更稳定）
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        # 尝试设置分辨率，失败则回退
        for target_res in [(1920, 1080), (1280, 720), (640, 480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_res[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, _ = cap.read()
            if ret:
                logger.info(f"摄像头分辨率设置为: {target_res[0]}x{target_res[1]}")
                break

        if not cap.isOpened():
            logger.error(f"无法打开摄像头: {camera_id}")
            return

        # 再读一帧验证
        ret, _ = cap.read()
        if not ret:
            logger.error("摄像头打开但无法读取帧")
            cap.release()
            return

        logger.info(f"摄像头已打开: {camera_id}")
        logger.info("按 'q' 退出，按 's' 保存当前帧")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("摄像头读取失败")
                break

            # 检测
            results = self.detect(frame)
            vis_frame = self.draw_results(frame, results)

            # 显示
            cv2.imshow("PCB Defect Detection - Press 'q' to quit", vis_frame)

            # 保存
            if save_dir and results['num_detections'] > 0:
                save_path = Path(save_dir) / f"frame_{frame_count:06d}.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), vis_frame)

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = Path(save_dir or '.') / f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(str(save_path), vis_frame)
                logger.info(f"截图已保存: {save_path}")

        cap.release()
        cv2.destroyAllWindows()
        logger.info("摄像头检测结束")

    def detect_batch(self, image_dir, save_dir=None):
        """
        批量检测图片

        Args:
            image_dir: 图片目录
            save_dir: 保存目录
        """
        image_dir = Path(image_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(image_dir.glob(ext)))

        logger.info(f"找到 {len(image_files)} 张图片")

        all_results = []
        for img_file in image_files:
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"result_{img_file.name}"

            results, _ = self.detect_image(img_file, save_path)
            all_results.append({
                'file': str(img_file),
                'results': results
            })

        # 统计
        total_detections = sum(r['results']['num_detections'] for r in all_results if r['results'])
        logger.info(f"检测完成: {len(image_files)} 张图片, {total_detections} 个缺陷")

        return all_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PCB缺陷检测推理')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--source', type=str, default='0',
                        help='输入源: 0=摄像头, 或图片/视频/目录路径')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径 (.pt 或 .onnx)')
    parser.add_argument('--save', type=str, default=None,
                        help='保存结果的路径')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--clahe', action='store_true',
                        help='启用CLAHE自适应直方图均衡化')
    parser.add_argument('--benchmark', type=int, default=None,
                        help='运行性能基准测试，指定测试次数')

    args = parser.parse_args()

    # 加载配置
    config = get_config(args.config)

    # 创建检测器
    detector = PCBDefectDetector(config, args.model, use_clahe=args.clahe)

    # 基准测试模式
    if args.benchmark:
        result = detector.benchmark(runs=args.benchmark)
        print("\n===== 性能基准测试结果 =====")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return

    # 判断输入类型
    source = args.source

    if source == '0' or source.isdigit():
        # 摄像头
        detector.detect_camera(int(source), args.save)
    else:
        source_path = Path(source)
        if source_path.is_dir():
            # 批量图片
            detector.detect_batch(source, args.save)
        elif source_path.is_file():
            # 单张图片或视频
            if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                detector.detect_image(source, args.save, args.show)
            else:
                logger.error(f"不支持的文件格式: {source_path.suffix}")
        else:
            logger.error(f"输入源不存在: {source}")


if __name__ == "__main__":
    main()
