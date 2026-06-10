
# ============================================================
# 文件: main.py
# 说明: 主程序入口 — 系统初始化、运行模式管理
# 行数: 523
# ============================================================
# -*- coding: utf-8 -*-
"""
PCB缺陷检测与智能分拣系统 — 主程序入口
2026 睿抗机器人开发者大赛 · CAIR强体赛道 · AI视觉应用

功能：
1. 整合图像采集、模型推理、硬件控制、UI显示
2. 实现完整的"采集→推理→决策→执行→反馈"闭环
3. 支持自动模式和手动模式
4. 一键换线、误报屏蔽、缺陷追溯
5. 声光报警、设备健康度监控
6. 权限管理、生产报表
7. 异常处理和日志记录
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_loader import get_config
from utils.logger import setup_logger
from utils.serial_control import create_controller
from utils.production_manager import ProductionManager
from utils.false_alarm_suppressor import FalseAlarmSuppressor
from utils.traceability import TraceabilityDB
from utils.alarm import AlarmManager, AlarmLevel
from utils.equipment_monitor import EquipmentMonitor
from utils.auth import AuthManager

logger = setup_logger("main")


class PCBDefectSystem:
    """PCB缺陷检测与智能分拣系统主类"""

    def __init__(self, config_path="configs/config.yaml", mock_hardware=False):
        self.config = get_config(config_path)
        self.mock_hardware = mock_hardware
        self.running = False

        # 初始化组件
        self.detector = None
        self.controller = None
        self.camera = None

        # 初始化子系统
        self.production_manager = ProductionManager(
            profiles_dir=getattr(self.config, 'production', None) and self.config.production.profiles_dir or None
        )
        self.suppressor = FalseAlarmSuppressor(
            rules_dir=getattr(self.config, 'suppression', None) and self.config.suppression.rules_dir or None
        )
        self.trace_db = TraceabilityDB(
            db_path=getattr(self.config, 'traceability', None) and self.config.traceability.db_path or None
        )
        self.alarm_manager = AlarmManager()
        self.equipment_monitor = EquipmentMonitor(self.config)
        self.auth_manager = AuthManager(
            db_path=getattr(self.config, 'auth', None) and self.config.auth.db_path or None
        )

        # 统计
        self.total_processed = 0
        self.ok_count = 0
        self.ng_count = 0
        self.start_time = None

        # SN码
        self._current_sn = ""

        logger.info("=" * 60)
        logger.info("PCB缺陷检测与智能分拣系统")
        logger.info("2026 睿抗机器人开发者大赛 · CAIR强体赛道")
        logger.info("=" * 60)

    def initialize(self):
        """初始化所有组件"""
        logger.info("正在初始化系统组件...")

        # 1. 加载检测模型
        try:
            from inference import PCBDefectDetector
            self.detector = PCBDefectDetector(self.config)
            logger.info("检测模型加载成功")
        except Exception as e:
            logger.error(f"检测模型加载失败: {e}")
            return False

        # 2. 连接Arduino控制器
        try:
            self.controller = create_controller(self.config, mock=self.mock_hardware)
            if self.controller.connect():
                logger.info("Arduino控制器连接成功")
            else:
                logger.warning("Arduino控制器连接失败，切换到模拟模式")
                self.controller = create_controller(self.config, mock=True)
                self.controller.connect()
        except Exception as e:
            logger.error(f"控制器初始化失败: {e}")
            self.controller = create_controller(self.config, mock=True)
            self.controller.connect()

        # 设置报警管理器的控制器
        self.alarm_manager = AlarmManager(controller=self.controller)

        # 3. 打开摄像头
        try:
            self.camera = cv2.VideoCapture(self.config.hardware.camera.index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.hardware.camera.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.hardware.camera.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.hardware.camera.fps)

            if self.camera.isOpened():
                logger.info("摄像头已打开")
            else:
                logger.error("摄像头打开失败")
                return False
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False

        # 4. 创建结果保存目录
        save_dir = Path(self.config.system.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 5. 注册换线回调
        self.production_manager.on_line_switched(self._on_line_switched)

        logger.info("系统初始化完成")
        return True

    def _on_line_switched(self, profile):
        """换线回调"""
        if self.detector:
            try:
                self.detector.conf_threshold = profile.conf_threshold
                self.detector.iou_threshold = profile.iou_threshold
                logger.info(f"检测器参数已更新: conf={profile.conf_threshold}, iou={profile.iou_threshold}")
            except Exception as e:
                logger.error(f"更新检测器参数失败: {e}")
        self.suppressor.set_profile(profile.name)

    def set_sn(self, sn: str):
        """设置当前产品SN码"""
        self._current_sn = sn
        logger.info(f"SN码已设置: {sn}")

    def run_auto_mode(self):
        """
        自动运行模式
        工作流程：
        1. 传送带持续运行
        2. 红外传感器检测到PCB到位
        3. 停止传送带，采集图像
        4. 模型推理检测
        5. 误报屏蔽过滤
        6. 根据结果控制舵机分拣
        7. 记录追溯数据
        8. 传送带继续运行
        """
        logger.info("=" * 60)
        logger.info("启动自动运行模式")
        logger.info("=" * 60)
        logger.info("按 Ctrl+C 停止")

        self.running = True
        self.start_time = time.time()

        # 启动传送带
        self.controller.motor_start()

        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("摄像头读取失败")
                    time.sleep(0.1)
                    continue

                if self.config.inference.show_preview:
                    preview = cv2.resize(frame, (640, 360))
                    cv2.imshow("System Preview - Press 'q' to quit", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # 模拟传送带节拍（实际应通过红外传感器触发）
                interval = getattr(self.config.hardware, 'interval', 2.0)
                time.sleep(interval)

                if not self.running:
                    break

                self._process_single_pcb()

        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self.shutdown()

    def _process_single_pcb(self):
        """处理单个PCB板的完整流程"""
        logger.info("-" * 40)
        logger.info(f"开始处理 PCB #{self.total_processed + 1}")

        # 1. 停止传送带
        self.controller.motor_stop()
        time.sleep(0.2)

        # 2. 采集图像
        ret, frame = self.camera.read()
        if not ret:
            logger.error("图像采集失败")
            self.controller.motor_start()
            return

        # 保存原始图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = Path(self.config.system.save_dir) / f"raw_{timestamp}.jpg"
        cv2.imwrite(str(raw_path), frame)

        # 3. 模型推理
        logger.info("正在推理...")
        results = self.detector.detect(frame)

        # 检查推理是否出错
        if results.get('error'):
            logger.error(f"推理出错: {results['error']}")
            self.controller.motor_start()
            return

        inference_time = results.get('inference_time', 0)
        num_detections = results.get('num_detections', 0)

        logger.info(f"推理完成: {inference_time:.1f}ms, 检测到 {num_detections} 个缺陷")

        # 4. 误报屏蔽过滤
        if results.get('detections'):
            valid, suppressed = self.suppressor.filter_detections(results['detections'])
            results['detections'] = valid
            results['num_detections'] = len(valid)
            num_detections = len(valid)
            if suppressed:
                logger.info(f"屏蔽了 {len(suppressed)} 个误报")

        # 5. 设备健康度记录
        self.equipment_monitor.record_frame(frame, inference_time)

        # 6. 判定结果
        is_defective = num_detections > 0

        # 7. 声光报警
        if is_defective:
            level = AlarmLevel.CRITICAL if num_detections >= 3 else AlarmLevel.WARNING
            self.alarm_manager.trigger(level, num_detections, f"检出{num_detections}个缺陷")

        # 8. 绘制结果
        vis_frame = self.detector.draw_results(frame, results)
        result_path = Path(self.config.system.save_dir) / f"result_{timestamp}.jpg"
        cv2.imwrite(str(result_path), vis_frame)

        # 9. 控制舵机分拣
        if is_defective:
            logger.info("判定: 不合格品 → 分拣到NG通道")
            self.controller.sort_pcb(is_ok=False, delay_ms=self.config.hardware.servo.delay_ms)
            self.ng_count += 1
        else:
            logger.info("判定: 合格品 → 分拣到OK通道")
            self.controller.sort_pcb(is_ok=True, delay_ms=self.config.hardware.servo.delay_ms)
            self.ok_count += 1

        # 10. 追溯记录
        result_status = "NG" if is_defective else "OK"
        shift = self.production_manager.shift_info
        self.trace_db.add_record(
            product_sn=self._current_sn,
            operator_id=shift.get('operator_id', ''),
            equipment_id=shift.get('equipment_id', ''),
            line_profile=self.production_manager.current_profile_name,
            result=result_status,
            num_defects=num_detections,
            inference_time_ms=inference_time,
            image_path=str(raw_path),
            result_image_path=str(result_path),
            defects=results.get('detections', []),
        )

        # SN码消费后清空
        if self._current_sn:
            self._current_sn = ""

        # 11. 记录统计
        self.total_processed += 1

        # 12. 重新启动传送带
        self.controller.motor_start()

        logger.info(f"处理完成: OK={self.ok_count}, NG={self.ng_count}, Total={self.total_processed}")

    def run_manual_mode(self):
        """
        手动运行模式
        通过键盘控制：
        - 空格: 触发一次检测
        - s: 保存当前帧
        - n: 输入SN码
        - q: 退出
        """
        logger.info("=" * 60)
        logger.info("启动手动运行模式")
        logger.info("控制键: [空格]=检测, [s]=保存, [n]=SN码, [q]=退出")
        logger.info("=" * 60)

        self.running = True

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            display = cv2.resize(frame, (1280, 720))
            cv2.putText(display, "Manual Mode - SPACE:detect  S:save  N:SN  Q:quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("PCB Defect Detection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                self._process_single_pcb()
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(self.config.system.save_dir) / f"manual_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                logger.info(f"截图已保存: {save_path}")
            elif key == ord('n'):
                sn = input("请输入SN码: ").strip()
                if sn:
                    self.set_sn(sn)
            elif key == ord('q'):
                break

        self.shutdown()

    def run_demo_mode(self):
        """
        演示模式（选拔赛/总决赛用）
        对20张样本进行连续检测，生成评估报告
        """
        logger.info("=" * 60)
        logger.info("启动演示模式（竞赛评估）")
        logger.info("=" * 60)

        self.running = True
        demo_results = []

        logger.info("准备检测20张样本...")

        for i in range(20):
            if not self.running:
                break

            logger.info(f"\n样本 {i+1}/20")

            ret, frame = self.camera.read()
            if not ret:
                logger.error("图像采集失败")
                continue

            start = time.time()
            results = self.detector.detect(frame)
            elapsed = (time.time() - start) * 1000

            # 误报屏蔽
            if results.get('detections'):
                valid, _ = self.suppressor.filter_detections(results['detections'])
                results['detections'] = valid
                results['num_detections'] = len(valid)

            vis_frame = self.detector.draw_results(frame, results)
            save_path = Path(self.config.system.save_dir) / f"demo_{i+1:02d}.jpg"
            cv2.imwrite(str(save_path), vis_frame)

            demo_results.append({
                'index': i + 1,
                'detections': results.get('num_detections', 0),
                'inference_time': elapsed,
                'image_path': str(save_path)
            })

            logger.info(f"  检测数: {results.get('num_detections', 0)}, 耗时: {elapsed:.1f}ms")

            cv2.imshow("Demo Mode - Press 'q' to quit", vis_frame)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        self._generate_demo_report(demo_results)
        self.shutdown()

    def _generate_demo_report(self, results):
        """生成演示评估报告"""
        logger.info("=" * 60)
        logger.info("演示评估报告")
        logger.info("=" * 60)

        total_time = sum(r['inference_time'] for r in results)
        avg_time = total_time / len(results) if results else 0
        max_time = max((r['inference_time'] for r in results), default=0)

        logger.info(f"总样本数: {len(results)}")
        logger.info(f"总推理时间: {total_time:.1f}ms")
        logger.info(f"平均推理时间: {avg_time:.1f}ms")
        logger.info(f"最大推理时间: {max_time:.1f}ms")
        logger.info(f"平均FPS: {1000/max(avg_time,1):.1f}")

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(results),
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'results': results
        }

        report_path = Path(self.config.system.save_dir) / 'demo_report.json'
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"报告已保存: {report_path}")

    def shutdown(self):
        """关闭系统"""
        logger.info("=" * 60)
        logger.info("正在关闭系统...")
        logger.info("=" * 60)

        self.running = False

        if self.controller:
            self.controller.motor_stop()
            self.controller.disconnect()

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"运行时间: {elapsed:.1f}秒")

        logger.info(f"总处理: {self.total_processed}")
        logger.info(f"合格品: {self.ok_count}")
        logger.info(f"不合格品: {self.ng_count}")

        # 关闭追溯数据库
        self.trace_db.close()

        logger.info("系统已关闭")

    def signal_handler(self, signum, frame):
        """信号处理"""
        logger.info("收到终止信号")
        self.shutdown()
        sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PCB缺陷检测与智能分拣系统')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'manual', 'demo', 'gui'],
                        help='运行模式: auto=自动, manual=手动, demo=演示, gui=图形界面')
    parser.add_argument('--mock', action='store_true',
                        help='使用模拟硬件（无Arduino调试模式）')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--sn', type=str, default=None,
                        help='产品SN码')

    args = parser.parse_args()

    # 创建系统
    system = PCBDefectSystem(args.config, mock_hardware=args.mock)

    # 设置SN码
    if args.sn:
        system.set_sn(args.sn)

    # 注册信号处理
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)

    # 初始化
    if not system.initialize():
        logger.error("系统初始化失败")
        sys.exit(1)

    # 运行模式
    if args.mode == 'auto':
        system.run_auto_mode()
    elif args.mode == 'manual':
        system.run_manual_mode()
    elif args.mode == 'demo':
        system.run_demo_mode()
    elif args.mode == 'gui':
        from ui.gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()

# ============================================================
# 文件: inference.py
# 说明: 模型推理 — YOLO11检测器、NMS、CLAHE预处理
# 行数: 988
# ============================================================
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
from utils.config_loader import get_config
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

        with urllib.request.urlopen(YOLO11S_PT_URL, context=ssl_context, timeout=60) as response:
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

        return detections, inference_time

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
        models_dir = Path(__file__).parent / "models"
        fallback_path = models_dir / "yolo11s.pt"
        logger.warning(f"模型文件不存在: {self.model_path}，尝试下载YOLO11s预训练权重作为fallback")
        return download_pretrained_model(fallback_path)

    def _load_model(self):
        """
        加载模型，自动检测模型格式：
        - .onnx → 使用ONNX Runtime
        - .pt → 使用ultralytics YOLO
        """
        model_suffix = self.model_path.suffix.lower()

        # 检查模型文件是否存在，不存在则尝试自动下载
        if not self.model_path.exists():
            if model_suffix == '.pt':
                try:
                    fallback = self._auto_download_model()
                    self.model_path = Path(fallback)
                    model_suffix = '.pt'
                except Exception as e:
                    logger.error(f"自动下载模型失败: {e}")
                    raise FileNotFoundError(f"模型文件不存在且自动下载失败: {self.model_path}")
            else:
                logger.error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

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
                logger.info(f"加载PyTorch模型: {pt_path}")
                self.model = YOLO(str(pt_path))
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
            img_augment=False,            # 推理时禁用数据增强
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
            color = self.class_colors[cls_id]

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
        cap = cv2.VideoCapture(camera_id)

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            logger.error(f"无法打开摄像头: {camera_id}")
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

# ============================================================
# 文件: train.py
# 说明: 模型训练 — YOLO11训练、DAGE模块注册
# 行数: 553
# ============================================================
# -*- coding: utf-8 -*-
"""
PCB-YOLO 训练脚本（v2 — 基于批评改进版）

改进措施（非"创新点"，是针对性改进）：
1. DAGE缺陷几何先验特征增强（替代通用CBAM）
   - Gabor滤波器组 → 方向性边缘（毛刺、开路）
   - 圆环检测核 → 圆形缺失（漏孔）
   - 可变形卷积 → 不规则形状（鼠咬、杂铜）

2. WIoU v3动态聚焦损失（修正理解，增加截断策略）

3. 偏振光-RGB多模态融合（物理层面创新，非算法调参）

4. P2小目标检测头（标准做法，不再标榜为创新）

5. 完整消融实验设计（含位置消融、参数敏感性分析）
"""

import os
import sys
import warnings
import shutil
from pathlib import Path
from copy import deepcopy

# =========================================
# 紧急 patch: 绕开 Ultralytics 字体下载（网络/权限失败会导致训练崩溃）
# 必须在任何 ultralytics 导入之前执行
# =========================================
def _install_font_download_patch():
    try:
        import ultralytics.utils.downloads as _d
        import ultralytics.utils.checks as _c
        import ultralytics.data.utils as _du

        _d.safe_download = lambda *a, **kw: None
        _c.check_font = lambda font=None, progress=True: str(font)
        if hasattr(_du, 'check_det_dataset'):
            _orig_check_det_dataset = _du.check_det_dataset
            def _patched(*a, **kw):
                _c.check_font = lambda font=None, progress=True: str(font)
                _d.safe_download = lambda *a2, **kw2: None
                return _orig_check_det_dataset(*a, **kw)
            _du.check_det_dataset = _patched
    except Exception:
        pass

_install_font_download_patch()

import torch
import yaml
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.dage import DAGE, register_dage
from modules.wiou import patch_wiou_loss
from modules.polarization_fusion import PolarizationRGBFusion
from utils.config_loader import get_config
from utils.logger import setup_logger

logger = setup_logger("train")

warnings.filterwarnings("ignore", category=FutureWarning)


def check_environment():
    """检查训练环境"""
    logger.info("=" * 60)
    logger.info("PCB-YOLO v2 训练环境检查")
    logger.info("=" * 60)

    logger.info(f"PyTorch版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA可用: {cuda_available}")

    if cuda_available:
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  显存: {mem:.1f} GB")

    try:
        import ultralytics
        logger.info(f"Ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        logger.error("未安装ultralytics，请运行: pip install ultralytics>=8.3.0")
        sys.exit(1)

    try:
        import torchvision
        logger.info(f"torchvision版本: {torchvision.__version__}")
    except ImportError:
        logger.warning("未安装torchvision，可变形卷积将退化为标准卷积")

    return cuda_available


# ==================== 改进措施1: DAGE模型构建 ====================

def create_pcb_yolo_model(config):
    """
    创建PCB-YOLO自定义模型配置

    在YOLO11基础上：
    - 在Backbone的C3k2后嵌入DAGE模块（替代CBAM）
    - 添加P2小目标检测头
    - 支持偏振光-RGB双模态输入

    Args:
        config: 项目配置对象

    Returns:
        自定义模型配置文件路径
    """
    import ultralytics
    yolo_dir = Path(ultralytics.__file__).parent

    model_name = config.model.name
    src_cfg_path = yolo_dir / 'cfg' / 'models' / f'{model_name}.yaml'

    if not src_cfg_path.exists():
        # 尝试在子目录里找（例如 11/、v8/）
        # YOLO11 在 11/ 子目录，且通常是 yolo11.yaml
        if '11' in model_name:
            # 先尝试精确名称
            src_cfg_path = yolo_dir / 'cfg' / 'models' / '11' / f'{model_name}.yaml'
            if not src_cfg_path.exists():
                # 没有精确名称，用通用的 yolo11.yaml
                src_cfg_path = yolo_dir / 'cfg' / 'models' / '11' / 'yolo11.yaml'
        # YOLOv8 在 v8/ 子目录
        elif 'v8' in model_name:
            src_cfg_path = yolo_dir / 'cfg' / 'models' / 'v8' / f'{model_name}.yaml'

    if not src_cfg_path.exists():
        logger.error(f"未找到YOLO模型配置: {src_cfg_path}")
        return None

    with open(src_cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    backbone = cfg.get('backbone', [])
    nc = config.data.num_classes

    def get_c2(layer_def):
        args = layer_def[3] if len(layer_def) > 3 else []
        return args[0] if args and isinstance(args[0], int) else 0

    ch = []
    for layer in backbone:
        c2 = get_c2(layer)
        if c2 > 0:
            ch.append(c2)

    if len(ch) < 10:
        logger.error(f"backbone通道数不足({len(ch)})")
        return None

    logger.info(f"YOLO11 backbone通道: {ch}")

    custom_cfg = _build_pcb_yolo_config(cfg, backbone, nc, ch)

    custom_cfg_path = Path('models') / f'pcb_{model_name}_dage_p2.yaml'
    custom_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    with open(custom_cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(custom_cfg, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"PCB-YOLO模型配置已保存: {custom_cfg_path}")
    return str(custom_cfg_path)


def _build_pcb_yolo_config(cfg, backbone, nc, ch):
    """
    构建PCB-YOLO模型配置

    在Backbone的C3k2后嵌入DAGE模块（替代CBAM），
    DAGE包含Gabor滤波器组+圆环检测核+可变形卷积三条路径。
    """
    # 先定义辅助函数
    def get_c2_local(layer_def):
        args = layer_def[3] if len(layer_def) > 3 else []
        return args[0] if args and isinstance(args[0], int) else 0

    p2_ch = ch[2]
    p3_ch = ch[4]
    p4_ch = ch[6]
    p5_ch = ch[9] if len(ch) > 9 else ch[8]

    # 构建新backbone（C3k2后插入DAGE）
    new_backbone = []
    for i, layer in enumerate(backbone):
        new_backbone.append(layer)
        if len(layer) >= 3 and layer[2] == 'C3k2':
            c2 = get_c2_local(layer)
            if c2 > 0:
                new_backbone.append([-1, 1, 'DAGE', [c2]])

    b = len(new_backbone)

    # 层索引映射
    original_to_new = {}
    offset = 0
    for i, layer in enumerate(backbone):
        original_to_new[i] = i + offset
        if len(layer) >= 3 and layer[2] == 'C3k2' and get_c2_local(layer) > 0:
            offset += 1

    p2_out = original_to_new[2] + 1
    p3_out = original_to_new[4] + 1
    p4_out = original_to_new[6] + 1
    sppf_out = original_to_new[9]
    c2psa_out = original_to_new[10] + 1 if len(ch) > 10 else sppf_out

    # 构建head（P2 + DAGE）
    new_head = []

    # P5 → P4 上采样路径
    new_head.append([-1, 1, 'nn.Upsample', [None, 2, 'nearest']])
    new_head.append([[-1, p4_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p4_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p4_ch]])

    # P4 → P3 上采样路径
    new_head.append([-1, 1, 'nn.Upsample', [None, 2, 'nearest']])
    new_head.append([[-1, p3_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p3_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p3_ch]])

    # P3 → P2 上采样路径（小目标检测头）
    new_head.append([-1, 1, 'nn.Upsample', [None, 2, 'nearest']])
    new_head.append([[-1, p2_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p2_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p2_ch]])

    # 下采样路径 P2 → P3 → P4 → P5
    new_head.append([b + 11, 1, 'Conv', [p3_ch, 3, 2]])
    new_head.append([[-1, b + 7], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p4_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p4_ch]])

    new_head.append([-1, 1, 'Conv', [p5_ch, 3, 2]])
    new_head.append([[-1, c2psa_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p5_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p5_ch]])

    # 4尺度检测
    new_head.append([[b + 11, b + 7, b + 15, b + 19], 1, 'Detect', [nc]])

    custom_cfg = {'nc': nc}
    if 'scales' in cfg:
        custom_cfg['scales'] = cfg['scales']
    custom_cfg['backbone'] = new_backbone
    custom_cfg['head'] = new_head

    return custom_cfg


# ==================== 训练流程 ====================

def train_model(config):
    """
    训练PCB-YOLO模型
    """
    from ultralytics import YOLO

    model_name = config.model.name
    use_dage = config.model.use_dage
    use_p2 = config.model.use_p2_head
    use_wiou = config.model.use_wiou

    # 注册DAGE模块
    if use_dage:
        register_dage()
        logger.info("改进措施1: DAGE缺陷几何先验特征增强模块已注册")

    # 替换WIoU v3损失（带异常保护）
    if use_wiou:
        try:
            success = patch_wiou_loss(delta=2.5, momentum=0.85)
            if success:
                logger.info("改进措施2: WIoU v3动态聚焦损失已启用")
            else:
                logger.warning("WIoU v3替换失败，将使用默认CIoU")
        except Exception as e:
            logger.warning(f"WIoU v3加载异常: {e}，将使用默认CIoU")

    # 创建/加载模型
    logger.info(f"加载模型: {model_name}")

    if use_dage and use_p2:
        custom_cfg_path = create_pcb_yolo_model(config)
        if custom_cfg_path:
            # 使用自定义配置 + 预训练权重
            # 先加载基础模型
            model = YOLO('yolo11n.pt' if config.model.pretrained else custom_cfg_path)
            if config.model.pretrained:
                logger.info("使用预训练权重 + DAGE + P2")
            else:
                logger.info("使用PCB-YOLO自定义配置（从头训练）")
        else:
            logger.warning("PCB-YOLO配置创建失败，回退到标准YOLO")
            model = YOLO('yolo11n.pt' if config.model.pretrained else 'yolo11n.yaml')
    else:
        model = YOLO('yolo11n.pt' if config.model.pretrained else 'yolo11n.yaml')

    # 数据集配置
    dataset_yaml = Path(config.data.dataset_path) / 'dataset.yaml'
    if not dataset_yaml.exists():
        logger.error(f"数据集配置不存在: {dataset_yaml}")
        logger.info("请先运行: python data/prepare_dataset.py --all")
        sys.exit(1)

    # 训练参数
    train_args = {
        'data': str(dataset_yaml),
        'epochs': config.train.epochs,
        'batch': config.train.batch_size,
        'imgsz': config.data.img_size,
        'workers': config.train.workers,
        'device': config.train.device,
        'optimizer': config.train.optimizer,
        'lr0': config.train.lr0,
        'lrf': config.train.lrf,
        'momentum': config.train.momentum,
        'weight_decay': config.train.weight_decay,
        'warmup_epochs': config.train.warmup_epochs,
        # warmup_momentum / warmup_bias_lr 已移除：AdamW 不使用 momentum 预热
        'cos_lr': config.train.cos_lr,
        'patience': config.train.patience,
        'save_period': config.train.save_period,
        'project': config.train.project,
        'name': config.train.name,
        'exist_ok': True,
        'pretrained': config.model.pretrained,
        'verbose': True,
        'seed': getattr(config.train, 'seed', 42),
    }

    # 数据增强（PCB适配版：适度增强，避免破坏几何结构）
    augment_args = {
        'hsv_h': 0.015,        # Hue 偏移很小，PCB颜色信息有限
        'hsv_s': 0.15,         # 降低饱和度扰动
        'hsv_v': 0.2,          # 降低亮度扰动
        'degrees': 5.0,        # PCB不应大幅旋转（±5°）
        'translate': 0.1,      # 10% 平移
        'scale': 0.3,          # 0.7-1.3x缩放，PCB缺陷尺度变化不大
        'shear': 2.0,          # 2°剪切
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,         # 50% 水平翻转
        'mosaic': 0.8,         # 80% 概率使用 Mosaic
        'mixup': 0.0,          # 关闭 MixUp：会破坏 PCB 导线/焊盘的几何结构
        'copy_paste': 0.05,    # 降低复制粘贴频率
        'label_smoothing': 0.05,
    }
    train_args.update(augment_args)

    # 打印配置
    logger.info("=" * 60)
    logger.info("PCB-YOLO v2 训练配置")
    logger.info("=" * 60)
    logger.info(f"基础模型: YOLO11 ({model_name})")
    logger.info(f"DAGE几何先验增强: {'启用' if use_dage else '禁用'}")
    logger.info(f"WIoU v3损失: {'启用' if use_wiou else '禁用'}")
    logger.info(f"P2检测头: {'启用' if use_p2 else '禁用'}")
    logger.info(f"偏振-RGB融合: {'启用' if config.model.use_polarization else '禁用'}")
    logger.info(f"数据集: {dataset_yaml}")

    # 字体下载 patch 已在文件开头安装（_install_font_download_patch）

    # 开始训练
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    results = model.train(**train_args)

    # 输出结果
    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info("=" * 60)

    if hasattr(results, 'results_dict'):
        map50 = results.results_dict.get('metrics/mAP50(B)', 0)
        map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
        logger.info(f"mAP@0.5: {map50:.4f}")
        logger.info(f"mAP@0.5:0.95: {map50_95:.4f}")

    # 复制最佳模型
    best_model_path = Path(config.train.project) / config.train.name / 'weights' / 'best.pt'
    if best_model_path.exists():
        target_path = Path('models') / 'best.pt'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_path, target_path)
        logger.info(f"最佳模型已复制到: {target_path}")

    return results


def run_ablation_study(config):
    """
    运行完整消融实验

    实验设计（修正版，覆盖所有组合 + Gabor可学习性对照）：
    A: YOLO11s 基线
    B: + P2头
    C: + DAGE（learnable Gabor）
    D: + WIoU v3
    E: + DAGE（fixed Gabor，对照：验证可学习性是否有用）
    F: P2 + DAGE
    G: P2 + DAGE + WIoU
    H: DAGE组件消融（仅Gabor / 仅Ring / 仅Deform / 全部）
    I: WIoU δ敏感性（δ=1,2,3,5）
    """
    logger.info("=" * 60)
    logger.info("消融实验")
    logger.info("=" * 60)

    experiments = {
        'A_baseline': {'use_dage': False, 'use_p2': False, 'use_wiou': False,
                        'learnable_gabor': True},
        'B_p2': {'use_dage': False, 'use_p2': True, 'use_wiou': False,
                  'learnable_gabor': True},
        'C_dage_learnable': {'use_dage': True, 'use_p2': False, 'use_wiou': False,
                              'learnable_gabor': True},
        'D_wiou': {'use_dage': False, 'use_p2': False, 'use_wiou': True,
                    'learnable_gabor': True},
        'E_dage_fixed_gabor': {'use_dage': True, 'use_p2': False, 'use_wiou': False,
                                'learnable_gabor': False},  # 对照：fixed vs learnable
        'F_p2_dage': {'use_dage': True, 'use_p2': True, 'use_wiou': False,
                       'learnable_gabor': True},
        'G_p2_dage_wiou': {'use_dage': True, 'use_p2': True, 'use_wiou': True,
                            'learnable_gabor': True},
    }

    results_summary = []

    for exp_name, exp_config in experiments.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"实验: {exp_name} | 配置: {exp_config}")
        logger.info(f"{'=' * 40}")

        # 修改配置
        config.model.use_dage = exp_config['use_dage']
        config.model.use_p2_head = exp_config['use_p2']
        config.model.use_wiou = exp_config['use_wiou']
        config.model.learnable_gabor = exp_config.get('learnable_gabor', True)
        config.train.name = f"ablation_{exp_name}"

        try:
            results = train_model(config)
            if hasattr(results, 'results_dict'):
                map50 = results.results_dict.get('metrics/mAP50(B)', 0)
                map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
                results_summary.append({
                    'experiment': exp_name,
                    'config': str(exp_config),
                    'mAP50': f"{map50:.4f}",
                    'mAP50-95': f"{map50_95:.4f}",
                })
        except Exception as e:
            logger.error(f"实验 {exp_name} 失败: {e}")
            results_summary.append({
                'experiment': exp_name,
                'config': str(exp_config),
                'mAP50': 'FAILED',
                'mAP50-95': 'FAILED',
            })

    # 打印汇总
    logger.info("\n" + "=" * 60)
    logger.info("消融实验汇总")
    logger.info("=" * 60)
    for r in results_summary:
        logger.info(f"  {r['experiment']:20s} | mAP@0.5={r['mAP50']} | mAP@0.5:0.95={r['mAP50-95']}")

    return results_summary


def export_model(config):
    """导出模型"""
    from ultralytics import YOLO

    model_path = Path('models') / 'best.pt'
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        return

    model = YOLO(str(model_path))

    # ONNX导出
    logger.info("导出ONNX模型...")
    model.export(format='onnx', imgsz=config.data.img_size, half=False, simplify=True)

    onnx_path = model_path.with_suffix('.onnx')
    if onnx_path.exists():
        logger.info(f"ONNX模型已导出: {onnx_path}")

    # TensorRT导出（如果GPU可用）
    if torch.cuda.is_available():
        logger.info("导出TensorRT模型...")
        try:
            model.export(format='engine', imgsz=config.data.img_size, half=True)
            logger.info("TensorRT模型导出成功")
        except Exception as e:
            logger.warning(f"TensorRT导出失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PCB-YOLO v2 训练脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--export', action='store_true', help='训练后导出模型')
    parser.add_argument('--ablation', action='store_true', help='运行消融实验')
    parser.add_argument('--no-dage', action='store_true', help='禁用DAGE')
    parser.add_argument('--no-p2', action='store_true', help='禁用P2检测头')
    parser.add_argument('--no-wiou', action='store_true', help='禁用WIoU v3')
    parser.add_argument('--model', type=str, default=None, help='覆盖模型名称')

    args = parser.parse_args()

    check_environment()

    config = get_config(args.config)

    if args.no_dage:
        config.model.use_dage = False
    if args.no_p2:
        config.model.use_p2_head = False
    if args.no_wiou:
        config.model.use_wiou = False
    if args.model:
        config.model.name = args.model

    if args.ablation:
        run_ablation_study(config)
    else:
        results = train_model(config)

    if args.export:
        export_model(config)

    logger.info("全部完成！")


if __name__ == "__main__":
    main()

# ============================================================
# 文件: utils/config_loader.py
# 说明: 配置加载
# 行数: 109
# ============================================================
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

# ============================================================
# 文件: utils/logger.py
# 说明: 日志系统
# 行数: 87
# ============================================================
# -*- coding: utf-8 -*-
"""
日志模块
统一日志格式和输出
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "pcb_defect",
                 log_level: str = "INFO",
                 log_file: str = None,
                 console_output: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径，None则不写入文件
        console_output: 是否输出到控制台
        
    Returns:
        logging.Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TqdmLoggingHandler(logging.Handler):
    """兼容tqdm进度条的日志处理器"""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def get_logger(name: str = "pcb_defect") -> logging.Logger:
    """获取已配置的日志器"""
    return logging.getLogger(name)


if __name__ == "__main__":
    logger = setup_logger(log_file="../results/test.log")
    logger.info("日志模块测试")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")

# ============================================================
# 文件: utils/auth.py
# 说明: 权限管理 — 登录认证、角色控制
# 行数: 448
# ============================================================
# -*- coding: utf-8 -*-
"""
权限管理模块
功能：
1. 角色权限控制（操作员/工程师/管理员）
2. 操作审计日志
3. 参数修改权限校验
"""

import json
import hashlib
import secrets
import time
import threading
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from utils.logger import setup_logger

logger = setup_logger("auth")

# 用户数据存储路径
AUTH_DB_PATH = Path("./configs/users.json")
AUDIT_LOG_PATH = Path("./results/audit.log")

# 密码最小长度要求
PASSWORD_MIN_LENGTH = 8
# 登录失败锁定次数
LOGIN_MAX_ATTEMPTS = 5
# 锁定时间（秒）
LOGIN_LOCKOUT_DURATION = 300


class Role:
    """角色定义"""
    OPERATOR = "operator"       # 操作员：只能查看和基本操作
    ENGINEER = "engineer"       # 工程师：可以调整参数
    ADMIN = "admin"             # 管理员：全部权限


# 权限定义
PERMISSIONS = {
    Role.OPERATOR: {
        'view_detection',       # 查看检测结果
        'start_stop',           # 启动/停止检测
        'capture',              # 截图
        'export_csv',           # 导出报表
        'review',               # 复核
        'switch_line',          # 一键换线
        'add_suppression',      # 添加误报屏蔽
    },
    Role.ENGINEER: {
        'view_detection', 'start_stop', 'capture', 'export_csv', 'review',
        'switch_line', 'add_suppression',
        'adjust_threshold',     # 调整置信度/IOU阈值
        'adjust_camera',        # 调整摄像头参数
        'adjust_hardware',      # 调整硬件参数
        'view_debug_info',      # 查看调试信息（Grad-CAM等）
        'manage_suppression',   # 管理屏蔽规则
        'view_health',          # 查看设备健康度
    },
    Role.ADMIN: {
        'view_detection', 'start_stop', 'capture', 'export_csv', 'review',
        'switch_line', 'add_suppression',
        'adjust_threshold', 'adjust_camera', 'adjust_hardware',
        'view_debug_info', 'manage_suppression', 'view_health',
        'manage_users',         # 用户管理
        'manage_profiles',      # 产线配置管理
        'system_config',        # 系统配置
        'view_audit_log',       # 查看审计日志
    },
}


class User:
    """用户"""

    def __init__(self, username: str, role: str, password_hash: str = "",
                 display_name: str = "", employee_id: str = ""):
        self.username = username
        self.role = role
        self.password_hash = password_hash
        self.display_name = display_name or username
        self.employee_id = employee_id
        self.last_login = ""
        self.password_changed_at = ""

    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in PERMISSIONS.get(self.role, set())

    def to_dict(self) -> dict:
        return {
            'username': self.username,
            'role': self.role,
            'password_hash': self.password_hash,
            'display_name': self.display_name,
            'employee_id': self.employee_id,
            'last_login': self.last_login,
            'password_changed_at': self.password_changed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        user = cls(
            username=data['username'],
            role=data.get('role', Role.OPERATOR),
            password_hash=data.get('password_hash', ''),
            display_name=data.get('display_name', ''),
            employee_id=data.get('employee_id', ''),
        )
        user.last_login = data.get('last_login', '')
        user.password_changed_at = data.get('password_changed_at', '')
        return user


class AuthManager:
    """权限管理器"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else AUTH_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._users: Dict[str, User] = {}
        self._current_user: Optional[User] = None
        self._lock = threading.Lock()
        # 登录失败追踪
        self._login_attempts: Dict[str, int] = {}
        self._lockout_until: Dict[str, float] = {}
        self._load_users()

    def _load_users(self):
        """加载用户数据"""
        if not self.db_path.exists():
            # 创建默认用户
            self._create_default_users()
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for user_data in data.get('users', []):
                user = User.from_dict(user_data)
                self._users[user.username] = user
            logger.info(f"已加载 {len(self._users)} 个用户")
        except Exception as e:
            logger.error(f"加载用户数据失败: {e}")
            self._create_default_users()

    def _generate_salt(self) -> str:
        """生成安全随机盐"""
        return secrets.token_hex(32)

    def _hash_password(self, password: str, salt: str = "") -> str:
        """
        密码哈希 - 使用 PBKDF2-SHA256 带盐
        安全改进：使用更安全的哈希方法
        """
        if not salt:
            salt = self._generate_salt()
        # PBKDF2 with SHA256, 100000 iterations
        import hmac
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${key.hex()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """验证密码"""
        try:
            if '$' not in stored_hash:
                # 旧格式兼容：纯SHA256
                return hashlib.sha256(password.encode('utf-8')).hexdigest() == stored_hash
            salt, key_hex = stored_hash.split('$', 1)
            computed_key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return secrets.compare_digest(computed_key.hex(), key_hex)
        except Exception:
            return False

    def _create_default_users(self):
        """创建默认用户（强制首次登录修改密码）"""
        # 生成随机初始密码而非硬编码
        admin_pwd = secrets.token_urlsafe(12)
        engineer_pwd = secrets.token_urlsafe(12)
        operator_pwd = secrets.token_urlsafe(12)

        default_users = [
            User("admin", Role.ADMIN, self._hash_password(admin_pwd), "管理员", "A001"),
            User("engineer", Role.ENGINEER, self._hash_password(engineer_pwd), "工程师", "E001"),
            User("operator", Role.OPERATOR, self._hash_password(operator_pwd), "操作员", "O001"),
        ]
        for user in default_users:
            self._users[user.username] = user

        # 保存默认密码到单独文件（仅首次需要）
        default_pwds_file = self.db_path.parent / "default_passwords.json"
        try:
            with open(default_pwds_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'admin': admin_pwd,
                    'engineer': engineer_pwd,
                    'operator': operator_pwd,
                    'created_at': datetime.now().isoformat(),
                    'warning': '首次登录后请立即修改密码'
                }, f, ensure_ascii=False, indent=2)
            logger.warning(f"默认密码已生成，请查看: {default_pwds_file}")
        except Exception as e:
            logger.error(f"保存默认密码失败: {e}")

        self._save_users()
        logger.info("已创建默认用户（请查看 default_passwords.json 获取初始密码）")

    def _is_locked_out(self, username: str) -> bool:
        """检查是否被锁定"""
        if username not in self._lockout_until:
            return False
        if time.time() < self._lockout_until[username]:
            return True
        # 锁定已过期，清除状态
        del self._lockout_until[username]
        if username in self._login_attempts:
            self._login_attempts[username] = 0
        return False

    def _record_failed_attempt(self, username: str):
        """记录失败尝试"""
        self._login_attempts[username] = self._login_attempts.get(username, 0) + 1
        if self._login_attempts[username] >= LOGIN_MAX_ATTEMPTS:
            self._lockout_until[username] = time.time() + LOGIN_LOCKOUT_DURATION
            logger.warning(f"用户 {username} 因多次登录失败被锁定 {LOGIN_LOCKOUT_DURATION} 秒")
            self._audit_log("ACCOUNT_LOCKED", username, f"登录失败{LOGIN_MAX_ATTEMPTS}次")

    def _clear_failed_attempts(self, username: str):
        """清除失败尝试记录"""
        self._login_attempts.pop(username, None)
        self._lockout_until.pop(username, None)

    def _save_users(self):
        """保存用户数据"""
        try:
            data = {
                'users': [u.to_dict() for u in self._users.values()],
            }
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存用户数据失败: {e}")

    @staticmethod
    def _hash_password(password: str) -> str:
        """密码哈希 - 使用 PBKDF2-SHA256 带盐"""
        import hmac
        salt = secrets.token_hex(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${key.hex()}"

    def login(self, username: str, password: str) -> bool:
        """
        用户登录

        Returns:
            是否登录成功
        """
        with self._lock:
            # 检查是否被锁定
            if self._is_locked_out(username):
                remaining = int(self._lockout_until.get(username, 0) - time.time())
                logger.warning(f"登录失败: 用户 {username} 已被锁定，剩余 {remaining} 秒")
                self._audit_log("LOGIN_BLOCKED", username, f"账户被锁定")
                return False

            user = self._users.get(username)
            if not user:
                logger.warning(f"登录失败: 用户不存在 [{username}]")
                self._audit_log("LOGIN_FAIL", username, "用户不存在")
                return False

            if not self._verify_password(password, user.password_hash):
                self._record_failed_attempt(username)
                attempts = self._login_attempts.get(username, 0)
                logger.warning(f"登录失败: 密码错误 [{username}]，剩余尝试次数 {LOGIN_MAX_ATTEMPTS - attempts}")
                self._audit_log("LOGIN_FAIL", username, f"密码错误，剩余尝试次数 {LOGIN_MAX_ATTEMPTS - attempts}")
                return False

            self._clear_failed_attempts(username)
            self._current_user = user
            user.last_login = datetime.now().isoformat()
            self._save_users()
            logger.info(f"用户登录: {username} (角色={user.role})")
            self._audit_log("LOGIN", username, f"角色={user.role}")
            return True

    def logout(self):
        """用户登出"""
        if self._current_user:
            self._audit_log("LOGOUT", self._current_user.username, "")
            logger.info(f"用户登出: {self._current_user.username}")
        self._current_user = None

    @property
    def current_user(self) -> Optional[User]:
        return self._current_user

    @property
    def is_logged_in(self) -> bool:
        return self._current_user is not None

    def check_permission(self, permission: str) -> bool:
        """
        检查当前用户是否有指定权限

        Args:
            permission: 权限名称

        Returns:
            是否有权限
        """
        if not self._current_user:
            return False
        return self._current_user.has_permission(permission)

    def require_permission(self, permission: str) -> bool:
        """
        要求权限，无权限时记录审计日志

        Returns:
            是否有权限
        """
        if self.check_permission(permission):
            return True
        username = self._current_user.username if self._current_user else "(未登录)"
        logger.warning(f"权限不足: {username} 缺少 {permission}")
        self._audit_log("PERMISSION_DENIED", username, f"需要 {permission}")
        return False

    def _validate_password_strength(self, password: str) -> tuple:
        """
        验证密码强度

        Returns:
            (是否有效, 错误信息)
        """
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"密码长度至少{PASSWORD_MIN_LENGTH}位"
        # 检查是否包含字母和数字
        if not re.search(r'[A-Za-z]', password):
            return False, "密码必须包含字母"
        if not re.search(r'[0-9]', password):
            return False, "密码必须包含数字"
        return True, ""

    def add_user(self, username: str, role: str, password: str,
                 display_name: str = "", employee_id: str = "") -> bool:
        """添加用户"""
        if not self.require_permission('manage_users'):
            return False

        # 验证密码强度
        valid, msg = self._validate_password_strength(password)
        if not valid:
            logger.error(f"添加用户失败: {msg}")
            return False

        with self._lock:
            if username in self._users:
                logger.error(f"用户已存在: {username}")
                return False

            user = User(username, role, self._hash_password(password),
                        display_name, employee_id)
            user.password_changed_at = datetime.now().isoformat()
            self._users[username] = user
            self._save_users()
            self._audit_log("ADD_USER", self._current_user.username,
                            f"添加用户 {username} 角色={role}")
            logger.info(f"添加用户: {username} (角色={role})")
            return True

    def remove_user(self, username: str) -> bool:
        """删除用户"""
        if not self.require_permission('manage_users'):
            return False

        with self._lock:
            if username in self._users:
                del self._users[username]
                self._save_users()
                self._audit_log("REMOVE_USER", self._current_user.username,
                                f"删除用户 {username}")
                return True
        return False

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """修改密码"""
        # 验证新密码强度
        valid, msg = self._validate_password_strength(new_password)
        if not valid:
            logger.error(f"修改密码失败: {msg}")
            return False

        with self._lock:
            user = self._users.get(username)
            if not user:
                return False
            if not self._verify_password(old_password, user.password_hash):
                return False
            user.password_hash = self._hash_password(new_password)
            user.password_changed_at = datetime.now().isoformat()
            self._save_users()
            self._audit_log("CHANGE_PASSWORD", username, "")
            return True

    def list_users(self) -> List[dict]:
        """列出所有用户（不含密码）"""
        return [
            {
                'username': u.username,
                'role': u.role,
                'display_name': u.display_name,
                'employee_id': u.employee_id,
                'last_login': u.last_login,
            }
            for u in self._users.values()
        ]

    def _audit_log(self, action: str, username: str, detail: str):
        """审计日志"""
        try:
            AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {action} user={username} detail={detail}\n")
        except Exception as e:
            logger.error(f"写入审计日志失败: {e}")

# ============================================================
# 文件: utils/traceability.py
# 说明: 追溯数据库 — SQLite记录
# 行数: 364
# ============================================================
# -*- coding: utf-8 -*-
"""
缺陷追溯模块
功能：
1. 产品SN码关联
2. 操作员工号记录
3. 设备编号记录
4. 完整检测历史（SQLite持久化，不限1000条）
5. 质量事故追溯查询
"""

import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from utils.logger import setup_logger

logger = setup_logger("traceability")

# 默认数据库路径
DEFAULT_DB_PATH = "./results/traceability.db"


class TraceabilityDB:
    """缺陷追溯数据库"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = None
        self._connect()
        self._init_tables()

    def _connect(self):
        """连接数据库"""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        logger.info(f"追溯数据库已连接: {self.db_path}")

    def _init_tables(self):
        """初始化数据表"""
        with self._lock:
            cursor = self._conn.cursor()

            # 检测记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    product_sn TEXT DEFAULT '',
                    operator_id TEXT DEFAULT '',
                    equipment_id TEXT DEFAULT '',
                    line_profile TEXT DEFAULT '',
                    result TEXT NOT NULL DEFAULT 'OK',
                    num_defects INTEGER DEFAULT 0,
                    inference_time_ms REAL DEFAULT 0,
                    image_path TEXT DEFAULT '',
                    result_image_path TEXT DEFAULT '',
                    reviewed INTEGER DEFAULT 0,
                    review_result TEXT DEFAULT '',
                    reviewer_id TEXT DEFAULT '',
                    review_time TEXT DEFAULT '',
                    extra_data TEXT DEFAULT '{}'
                )
            """)

            # 缺陷明细表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    bbox TEXT DEFAULT '[]',
                    suppressed INTEGER DEFAULT 0,
                    suppressed_by TEXT DEFAULT '',
                    FOREIGN KEY (record_id) REFERENCES detection_records(id)
                )
            """)

            # 班次统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shift_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shift_date TEXT NOT NULL,
                    shift_name TEXT NOT NULL,
                    operator_id TEXT DEFAULT '',
                    equipment_id TEXT DEFAULT '',
                    line_profile TEXT DEFAULT '',
                    total_count INTEGER DEFAULT 0,
                    ok_count INTEGER DEFAULT 0,
                    ng_count INTEGER DEFAULT 0,
                    defect_class_stats TEXT DEFAULT '{}',
                    avg_inference_ms REAL DEFAULT 0,
                    start_time TEXT DEFAULT '',
                    end_time TEXT DEFAULT ''
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sn ON detection_records(product_sn)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_operator ON detection_records(operator_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equipment ON detection_records(equipment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_result ON detection_records(result)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviewed ON detection_records(reviewed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_shift_date ON shift_stats(shift_date)")

            self._conn.commit()

    def add_record(self, product_sn: str = "", operator_id: str = "",
                   equipment_id: str = "", line_profile: str = "",
                   result: str = "OK", num_defects: int = 0,
                   inference_time_ms: float = 0, image_path: str = "",
                   result_image_path: str = "", defects: List[dict] = None,
                   extra_data: dict = None) -> int:
        """
        添加检测记录

        Returns:
            记录ID
        """
        with self._lock:
            cursor = self._conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO detection_records
                (timestamp, product_sn, operator_id, equipment_id, line_profile,
                 result, num_defects, inference_time_ms, image_path, result_image_path, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, product_sn, operator_id, equipment_id, line_profile,
                result, num_defects, inference_time_ms, image_path, result_image_path,
                json.dumps(extra_data or {}, ensure_ascii=False)
            ))

            record_id = cursor.lastrowid

            # 插入缺陷明细
            if defects:
                for det in defects:
                    cursor.execute("""
                        INSERT INTO defect_details
                        (record_id, class_name, confidence, bbox, suppressed, suppressed_by)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        record_id,
                        det.get('class_name', ''),
                        det.get('confidence', 0),
                        json.dumps(det.get('bbox', [])),
                        1 if det.get('suppressed_by') else 0,
                        det.get('suppressed_by', ''),
                    ))

            self._conn.commit()
            return record_id

    def review_record(self, record_id: int, review_result: str, reviewer_id: str = ""):
        """复核记录"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                UPDATE detection_records
                SET reviewed = 1, review_result = ?, reviewer_id = ?, review_time = ?
                WHERE id = ?
            """, (review_result, reviewer_id, datetime.now().isoformat(), record_id))
            self._conn.commit()

    def batch_review(self, record_ids: List[int], review_result: str, reviewer_id: str = ""):
        """批量复核"""
        with self._lock:
            cursor = self._conn.cursor()
            now = datetime.now().isoformat()
            for rid in record_ids:
                cursor.execute("""
                    UPDATE detection_records
                    SET reviewed = 1, review_result = ?, reviewer_id = ?, review_time = ?
                    WHERE id = ?
                """, (review_result, reviewer_id, now, rid))
            self._conn.commit()
        logger.info(f"批量复核: {len(record_ids)} 条记录, 结果={review_result}")

    def get_unreviewed(self, limit: int = 50) -> List[dict]:
        """获取未复核记录"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_records
                WHERE reviewed = 0 AND result = 'NG'
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_sn(self, product_sn: str) -> List[dict]:
        """按产品SN码查询"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_records
                WHERE product_sn = ?
                ORDER BY timestamp DESC
            """, (product_sn,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_operator(self, operator_id: str, date: str = None) -> List[dict]:
        """按操作员查询"""
        with self._lock:
            cursor = self._conn.cursor()
            if date:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE operator_id = ? AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC
                """, (operator_id, date))
            else:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE operator_id = ?
                    ORDER BY timestamp DESC LIMIT 100
                """, (operator_id,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_equipment(self, equipment_id: str, date: str = None) -> List[dict]:
        """按设备编号查询"""
        with self._lock:
            cursor = self._conn.cursor()
            if date:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE equipment_id = ? AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC
                """, (equipment_id, date))
            else:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE equipment_id = ?
                    ORDER BY timestamp DESC LIMIT 100
                """, (equipment_id,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_timerange(self, start: str, end: str) -> List[dict]:
        """按时间范围查询"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
