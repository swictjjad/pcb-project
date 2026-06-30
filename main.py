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

        # 1. 加载检测模型（优先ONNX）
        try:
            from inference import PCBDefectDetector, ONNXPCBDefectDetector
            model_path = Path(self.config.inference.model_path)
            onnx_path = model_path.with_suffix(".onnx")
            use_onnx = onnx_path.exists()
            if use_onnx:
                logger.info(f"检测到ONNX模型: {onnx_path}，使用ONNX推理后端")
                self.detector = ONNXPCBDefectDetector(
                    str(onnx_path),
                    img_size=self.config.inference.img_size,
                    conf_threshold=self.config.inference.conf_threshold,
                    iou_threshold=self.config.inference.iou_threshold,
                    class_names=self.config.data.class_names,
                )
                self.detector.warmup(runs=3)
            else:
                logger.info(f"未找到ONNX模型，使用PyTorch后端: {model_path}")
                self.detector = PCBDefectDetector(self.config)
            logger.info("检测模型加载成功")
        except Exception as e:
            logger.error(f"检测模型加载失败: {e}")
            return False
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
            # 优先使用 DSHOW 后端（Windows 下更稳定）
            self.camera = cv2.VideoCapture(self.config.hardware.camera.index, cv2.CAP_DSHOW)

            # 尝试多种分辨率，找到摄像头支持的最大分辨率
            camera_opened = False
            for target_res in [(self.config.hardware.camera.width, self.config.hardware.camera.height),
                               (1280, 720), (640, 480)]:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, target_res[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, target_res[1])
                self.camera.set(cv2.CAP_PROP_FPS, self.config.hardware.camera.fps)

                if self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        # 成功打开并读取到帧
                        h, w = frame.shape[:2]
                        self.config.hardware.camera.width = w
                        self.config.hardware.camera.height = h
                        logger.info(f"摄像头已打开，分辨率: {w}x{h}")
                        camera_opened = True
                        break
                    else:
                        self.camera.release()

            if not camera_opened:
                # 降级：不使用 DSHOW 后端重试
                logger.warning("DSHOW 后端打开失败，尝试默认后端...")
                self.camera.release()
                self.camera = cv2.VideoCapture(self.config.hardware.camera.index)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 30)

                if self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        self.config.hardware.camera.width = w
                        self.config.hardware.camera.height = h
                        logger.info(f"摄像头已打开（默认后端），分辨率: {w}x{h}")
                    else:
                        logger.error("摄像头打开但无法读取帧")
                        self.camera.release()
                        return False
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

                # NOTE: cv2.imshow conflicts with Qt event loop in GUI mode.
                # The GUI (ui/gui.py) handles video preview via QImage/QPixmap.
                # show_preview is kept for headless/non-GUI debugging only.
                if self.config.inference.show_preview and not getattr(self, '_gui_mode', False):
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

            # Headless demo preview (conflicts with Qt GUI)
            if not getattr(self, '_gui_mode', False):
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
                        choices=['auto', 'manual', 'demo', 'gui', 'kanban'],
                        help='运行模式: auto=自动, manual=手动, demo=演示, gui=图形界面, kanban=产线看板')
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
    elif args.mode == 'kanban':
        system._gui_mode = True  # 禁用 cv2.imshow
        from ui.kanban import run_kanban
        run_kanban(system)


if __name__ == "__main__":
    main()
