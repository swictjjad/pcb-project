# -*- coding: utf-8 -*-
"""
产线看板模块
功能：
1. 全屏车间大屏展示
2. 实时统计、缺陷分布、最近事件
3. 定时刷新数据
"""

import sys
import time
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

from ui.theme import ProductionKanban
from utils.logger import setup_logger

logger = setup_logger("kanban")


def run_kanban(system):
    """
    运行产线看板模式

    Args:
        system: PCBDefectSystem 实例
    """
    logger.info("=" * 60)
    logger.info("启动产线看板模式")
    logger.info("=" * 60)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    kanban = ProductionKanban()

    # 初始数据
    kanban.update_stats(0, 0, 0, 0)
    kanban.add_event("看板已启动")

    # 定时刷新统计
    def refresh_stats():
        total = system.total_processed
        ok = system.ok_count
        ng = system.ng_count
        rate = (ok / total * 100) if total > 0 else 0

        kanban.update_stats(total, ok, ng, rate)

        # 缺陷分布
        if system.detector and hasattr(system, '_last_results'):
            defs = {}
            for det in system._last_results.get('detections', []):
                name = det.get('class_name', 'unknown')
                defs[name] = defs.get(name, 0) + 1
            kanban.update_defect_distribution(defs)

        # 产线名称
        kanban.set_line_name(system.production_manager.current_profile_name)

        # 班次信息
        kanban.set_shift_info(system.production_manager.shift_info)

    # 每 2 秒刷新一次
    refresh_timer = QTimer()
    refresh_timer.timeout.connect(refresh_stats)
    refresh_timer.start(2000)

    # 监听检测完成事件
    def on_detection_complete(results):
        num = results.get('num_detections', 0)
        status = "NG" if num > 0 else "OK"
        kanban.add_event(f"{'合格' if status == 'OK' else '不良'} - 检测数:{num}")

        # 更新缺陷分布
        defs = {}
        for det in results.get('detections', []):
            name = det.get('class_name', 'unknown')
            defs[name] = defs.get(name, 0) + 1
        kanban.update_defect_distribution(defs)

    # 通过定时器轮询检测器最新结果
    def poll_detection():
        if hasattr(system, '_last_results') and system._last_results:
            on_detection_complete(system._last_results)

    poll_timer = QTimer()
    poll_timer.timeout.connect(poll_detection)
    poll_timer.start(1000)

    logger.info("看板运行中 - 按 Ctrl+C 退出")

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        logger.info("看板已退出")
