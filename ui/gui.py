# -*- coding: utf-8 -*-
"""
PyQt5 图形用户界面
功能：
1. 实时视频预览
2. 检测结果显示（带边界框）
3. 专家评分模式（评委逐张判分）
4. 历史记录统计
5. 系统状态监控
6. 急停按钮
7. 统计仪表盘（合格率、缺陷类别分布、平均耗时）
8. CSV报表导出
9. 摄像头断开重连机制
10. 一键换线（产线配置热切换）
11. 误报屏蔽（区域忽略规则）
12. 缺陷追溯（SN码/工号/设备号）
13. 产线/调试模式切换
14. 批量复核
15. 设备健康度监控
16. 声光报警
17. 生产报表
18. 权限管理
"""

import os
import sys
import time
import csv
import json
import re
import random
from pathlib import Path
from datetime import datetime
from collections import deque, Counter

# ⚠️ 必须在 PyQt5 之前导入 PyTorch/ultralytics
# Windows 上 PyQt5 的 QApplication 会改变 DLL 加载行为
# 导致 torch 的 c10.dll 加载失败 (WinError 1114)
try:
    import torch
    import ultralytics
except Exception as e:
    # 允许导入失败（可能环境不支持），后续检测器创建时会捕获
    torch = None  # noqa: F841
    ultralytics = None  # noqa: F841

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QGroupBox, QSplitter,
    QStatusBar, QProgressBar, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QTextEdit, QTabWidget,
    QLineEdit, QDialog, QFormLayout, QDialogButtonBox,
    QListWidget, QListWidgetItem, QMenu, QAction, QInputDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QPainter, QPen, QBrush, QIcon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import get_config
from utils.logger import setup_logger
from utils.production_manager import ProductionManager, LineProfile
from utils.false_alarm_suppressor import FalseAlarmSuppressor, SuppressionRule
from utils.traceability import TraceabilityDB
from utils.alarm import AlarmManager, AlarmConfig, AlarmLevel
from utils.equipment_monitor import EquipmentMonitor, HealthStatus
from utils.auth import AuthManager, Role
from utils.production_report import ProductionReportGenerator
from ui.theme import (
    GLOBAL_STYLESHEET, Colors, BrandBar, StatCardRow, PanelSection,
    BTN_PRIMARY, BTN_SUCCESS, BTN_DANGER, BTN_OUTLINE, BTN_GHOST,
    EMERGENCY_STOP_STYLE,
)

logger = setup_logger("gui")


class CameraThread(QThread):
    """摄像头采集线程 - 带断开重连机制"""

    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    connection_lost = pyqtSignal()
    connection_restored = pyqtSignal()

    def __init__(self, camera_id=0, width=1920, height=1080, fps=30):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.cap = None
        self._mutex = QMutex()
        self._reconnect_interval = 3.0
        self._max_reconnect_attempts = 0
        self._connected = False
        # 指定 DSHOW 后端（Windows DirectShow），避免 MSMF 后端导致流选择失败
        self._backend = cv2.CAP_DSHOW

    def run(self):
        self.running = True

        while self.running:
            if not self._open_camera():
                self.error_occurred.emit(f"无法打开摄像头: {self.camera_id}")
                self._connected = False
                self.connection_lost.emit()

                attempt = 0
                while self.running:
                    attempt += 1
                    logger.info(f"摄像头重连尝试 #{attempt}...")
                    time.sleep(self._reconnect_interval)

                    if not self.running:
                        break

                    if self._open_camera():
                        self._connected = True
                        self.connection_restored.emit()
                        logger.info(f"摄像头重连成功 (第{attempt}次尝试)")
                        break

                    if self._max_reconnect_attempts > 0 and attempt >= self._max_reconnect_attempts:
                        self.error_occurred.emit(f"摄像头重连失败，已达最大尝试次数({self._max_reconnect_attempts})")
                        return
                continue

            self._connected = True
            self.connection_restored.emit()

            consecutive_failures = 0
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    consecutive_failures = 0
                    self.frame_ready.emit(frame)
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        logger.warning("摄像头连续读取失败，判定为断开连接")
                        self._connected = False
                        self.connection_lost.emit()
                        self._release_camera()
                        break

                time.sleep(1.0 / self.fps)

        self._release_camera()

    def _open_camera(self):
        self._release_camera()
        self.cap = cv2.VideoCapture(self.camera_id, self._backend)
        if not self.cap.isOpened():
            return False

        # 尝试设置分辨率，如果失败则回退到摄像头默认分辨率
        for target_res in [(self.width, self.height), (1280, 720), (640, 480)]:
            w_ok = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_res[0])
            h_ok = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_res[1])
            fps_ok = self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            if w_ok and h_ok:
                self.width = target_res[0]
                self.height = target_res[1]
                logger.info(f"摄像头分辨率设置为: {self.width}x{self.height}")
                break
            elif not w_ok and not h_ok and target_res == (self.width, self.height):
                # 目标分辨率不支持，回退到默认 720p
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.width = 1280
                self.height = 720
                logger.warning(f"分辨率 {self.width}x{self.height} 不支持，回退到 1280x720")

        # 尝试读取一帧验证摄像头正常工作
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._release_camera()
            return False
        # 回退放回
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def _release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_connected(self):
        return self._connected

    def stop(self):
        self._mutex.lock()
        self.running = False
        self._mutex.unlock()
        self.wait(3000)


class DetectionThread(QThread):
    """检测推理线程"""

    result_ready = pyqtSignal(dict, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self._frame = None
        self._has_new_frame = False
        self.running = False
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()

    def set_frame(self, frame):
        self._mutex.lock()
        self._frame = frame.copy() if frame is not None else None
        self._has_new_frame = True
        self._mutex.unlock()
        self._wait_condition.wakeOne()

    def run(self):
        self.running = True

        while self.running:
            self._mutex.lock()
            while not self._has_new_frame and self.running:
                self._wait_condition.wait(self._mutex, 100)

            frame = self._frame
            self._frame = None
            self._has_new_frame = False
            self._mutex.unlock()

            if frame is not None and self.running:
                try:
                    results = self.detector.detect(frame)
                    vis_frame = self.detector.draw_results(frame, results)
                    self.result_ready.emit(results, vis_frame)
                except Exception as e:
                    self.error_occurred.emit(str(e))

    def stop(self):
        self._mutex.lock()
        self.running = False
        self._mutex.unlock()
        self._wait_condition.wakeAll()
        self.wait(3000)


class PassRateWidget(QWidget):
    """合格率显示组件 - 圆形进度条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rate = 0.0
        self.setMinimumSize(120, 120)

    def set_rate(self, rate):
        self._rate = max(0.0, min(100.0, rate))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        painter.setViewport((self.width() - side) // 2, (self.height() - side) // 2, side, side)
        painter.setWindow(0, 0, 120, 120)

        pen = QPen(QColor(Colors.BG_INPUT), 8)
        painter.setPen(pen)
        painter.drawArc(15, 15, 90, 90, 0, 360 * 16)

        if self._rate > 0:
            if self._rate >= 90:
                color = QColor(Colors.SUCCESS)
            elif self._rate >= 70:
                color = QColor(Colors.WARNING)
            else:
                color = QColor(Colors.DANGER)

            pen = QPen(color, 8)
            painter.setPen(pen)
            span_angle = int(self._rate / 100.0 * 360 * 16)
            painter.drawArc(15, 15, 90, 90, 90 * 16, -span_angle)

        painter.setPen(QColor(Colors.TEXT_PRIMARY))
        font = QFont("Microsoft YaHei", 16, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRect(0, 0, 120, 120), Qt.AlignCenter, f"{self._rate:.1f}%")

        painter.end()


class DefectBarWidget(QWidget):
    """缺陷类别分布柱状图组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = {}
        self.setMinimumHeight(160)

    def set_data(self, data):
        self._data = data
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin_left = 40
        margin_bottom = 30
        margin_top = 10
        margin_right = 10

        chart_w = w - margin_left - margin_right
        chart_h = h - margin_bottom - margin_top

        categories = list(self._data.keys())
        values = list(self._data.values())
        max_val = max(values) if values else 1

        bar_width = max(10, min(40, chart_w // len(categories) - 10))
        spacing = (chart_w - bar_width * len(categories)) / (len(categories) + 1)

        colors = [
            QColor(Colors.DANGER), QColor(Colors.WARNING), QColor(Colors.ACCENT),
            QColor(Colors.SUCCESS), QColor("#8B5CF6"), QColor("#EC4899"),
            QColor("#14B8A6"), QColor("#F97316"),
        ]

        painter.setPen(QPen(QColor(Colors.BORDER_LIGHT), 1))
        painter.drawLine(margin_left, margin_top, margin_left, h - margin_bottom)
        painter.drawLine(margin_left, h - margin_bottom, w - margin_right, h - margin_bottom)

        font = QFont("Microsoft YaHei", 8)
        painter.setFont(font)

        for i, (cat, val) in enumerate(self._data.items()):
            x = margin_left + spacing + i * (bar_width + spacing)
            bar_h = (val / max_val) * chart_h if max_val > 0 else 0
            y = h - margin_bottom - bar_h

            color = colors[i % len(colors)]
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(x), int(y), bar_width, int(bar_h))

            painter.setPen(QColor(Colors.TEXT_MUTED))
            painter.drawText(int(x - 5), h - margin_bottom + 15, bar_width + 10, 15,
                             Qt.AlignCenter, cat[:4])
            painter.setPen(QColor(Colors.TEXT_PRIMARY))
            painter.drawText(int(x - 5), int(y) - 15, bar_width + 10, 15,
                             Qt.AlignCenter, str(val))

        painter.end()


class LoginDialog(QDialog):
    """登录对话框"""

    def __init__(self, auth_manager, parent=None):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.setWindowTitle("系统登录")
        self.setFixedSize(350, 250)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
            }}
        """)

        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("PCB 缺陷检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {Colors.ACCENT}; padding: 8px;")
        layout.addRow(title)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("请输入用户名")
        layout.addRow("用户名:", self.username_edit)

        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("请输入密码")
        self.password_edit.setEchoMode(QLineEdit.Password)
        layout.addRow("密码:", self.password_edit)

        self.operator_id_edit = QLineEdit()
        self.operator_id_edit.setPlaceholderText("请输入工号（可选）")
        layout.addRow("工号:", self.operator_id_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 24px;
                border-radius: 6px;
                font-weight: bold;
            }}
        """)
        buttons.accepted.connect(self._try_login)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.logged_in = False

    def _try_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        operator_id = self.operator_id_edit.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "提示", "请输入用户名和密码")
            return

        if self.auth_manager.login(username, password):
            self.logged_in = True
            # 设置操作员工号
            if operator_id:
                self.auth_manager.current_user.employee_id = operator_id
            self.accept()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误")


class LineSwitchDialog(QDialog):
    """一键换线对话框"""

    def __init__(self, production_manager, parent=None):
        super().__init__(parent)
        self.pm = production_manager
        self.setWindowTitle("一键换线")
        self.setFixedSize(400, 350)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 当前产线
        current_label = QLabel(f"当前产线: {self.pm.current_profile_name}")
        current_label.setStyleSheet(f"font-size: 14px; font-weight: bold; padding: 5px; color: {Colors.ACCENT};")
        layout.addWidget(current_label)

        # 产线列表
        list_label = QLabel("选择产线:")
        list_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(list_label)

        self.profile_list = QListWidget()
        for name in self.pm.list_profiles():
            self.profile_list.addItem(name)
        self.profile_list.itemDoubleClicked.connect(self._switch)
        self.profile_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                color: {Colors.TEXT_PRIMARY};
                padding: 4px;
            }}
            QListWidget::item:hover {{
                background-color: {Colors.BG_HEADER};
            }}
            QListWidget::item:selected {{
                background-color: {Colors.ACCENT};
                color: white;
            }}
        """)
        layout.addWidget(self.profile_list)

        # 按钮行
        btn_layout = QHBoxLayout()

        switch_btn = QPushButton("切换")
        switch_btn.setProperty("variant", "success")
        switch_btn.clicked.connect(self._switch)
        btn_layout.addWidget(switch_btn)

        new_btn = QPushButton("新建产线")
        new_btn.setProperty("variant", "outline")
        new_btn.clicked.connect(self._new_profile)
        btn_layout.addWidget(new_btn)

        delete_btn = QPushButton("删除")
        delete_btn.setProperty("variant", "danger")
        delete_btn.clicked.connect(self._delete_profile)
        btn_layout.addWidget(delete_btn)

        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "ghost")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        self.switched = False

    def _switch(self):
        item = self.profile_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择一个产线配置")
            return

        name = item.text()
        reply = QMessageBox.question(self, "确认换线",
                                     f"确认切换到产线 [{name}]？\n系统将自动更新模型和参数。",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.pm.switch_line(name):
                QMessageBox.information(self, "换线成功", f"已切换到产线: {name}")
                self.switched = True
                self.accept()
            else:
                QMessageBox.critical(self, "换线失败", f"切换产线 [{name}] 失败")

    def _new_profile(self):
        name, ok = QInputDialog.getText(self, "新建产线", "产线名称:")
        if ok and name.strip():
            profile = self.pm.create_profile_from_current(name.strip())
            self.pm.save_profile(profile)
            self.profile_list.addItem(name.strip())

    def _delete_profile(self):
        item = self.profile_list.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(self, "确认删除", f"确认删除产线配置 [{name}]？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.pm.delete_profile(name)
            self.profile_list.takeItem(self.profile_list.row(item))


class SuppressionDialog(QDialog):
    """误报屏蔽管理对话框"""

    def __init__(self, suppressor, parent=None):
        super().__init__(parent)
        self.suppressor = suppressor
        self.setWindowTitle("误报屏蔽管理")
        self.setFixedSize(500, 400)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 规则列表
        self.rule_table = QTableWidget()
        self.rule_table.setColumnCount(5)
        self.rule_table.setHorizontalHeaderLabels(["名称", "类型", "区域", "启用", "命中次数"])
        self.rule_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rule_table.setAlternatingRowColors(True)
        self._refresh_rules()
        layout.addWidget(self.rule_table)

        # 按钮行
        btn_layout = QHBoxLayout()

        add_btn = QPushButton("+ 矩形区域")
        add_btn.setProperty("variant", "success")
        add_btn.clicked.connect(self._add_rect_rule)
        btn_layout.addWidget(add_btn)

        toggle_btn = QPushButton("启用/禁用")
        toggle_btn.setProperty("variant", "outline")
        toggle_btn.clicked.connect(self._toggle_rule)
        btn_layout.addWidget(toggle_btn)

        delete_btn = QPushButton("删除")
        delete_btn.setProperty("variant", "danger")
        delete_btn.clicked.connect(self._delete_rule)
        btn_layout.addWidget(delete_btn)

        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "ghost")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _refresh_rules(self):
        rules = self.suppressor.list_rules()
        self.rule_table.setRowCount(len(rules))
        for i, rule in enumerate(rules):
            self.rule_table.setItem(i, 0, QTableWidgetItem(rule.name))
            self.rule_table.setItem(i, 1, QTableWidgetItem(rule.rule_type))
            if rule.rule_type == "rect":
                r = rule.region
                region_str = f"({r.get('x1',0)},{r.get('y1',0)})-({r.get('x2',0)},{r.get('y2',0)})"
            else:
                region_str = f"{len(rule.region.get('points',[]))}点"
            self.rule_table.setItem(i, 2, QTableWidgetItem(region_str))
            self.rule_table.setItem(i, 3, QTableWidgetItem("是" if rule.enabled else "否"))
            self.rule_table.setItem(i, 4, QTableWidgetItem(str(rule.hit_count)))

    def _add_rect_rule(self):
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "添加屏蔽区域", "规则名称:")
        if not ok or not name.strip():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("设置屏蔽区域坐标")
        dialog.setFixedSize(300, 200)
        form = QFormLayout(dialog)

        x1_spin = QSpinBox(); x1_spin.setRange(0, 9999); x1_spin.setValue(0)
        y1_spin = QSpinBox(); y1_spin.setRange(0, 9999); y1_spin.setValue(0)
        x2_spin = QSpinBox(); x2_spin.setRange(0, 9999); x2_spin.setValue(100)
        y2_spin = QSpinBox(); y2_spin.setRange(0, 9999); y2_spin.setValue(100)

        form.addRow("左上X:", x1_spin)
        form.addRow("左上Y:", y1_spin)
        form.addRow("右下X:", x2_spin)
        form.addRow("右下Y:", y2_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() == QDialog.Accepted:
            self.suppressor.add_rect_rule(
                name=name.strip(),
                x1=x1_spin.value(), y1=y1_spin.value(),
                x2=x2_spin.value(), y2=y2_spin.value(),
            )
            self._refresh_rules()

    def _toggle_rule(self):
        row = self.rule_table.currentRow()
        if row < 0:
            return
        name = self.rule_table.item(row, 0).text()
        self.suppressor.toggle_rule(name)
        self._refresh_rules()

    def _delete_rule(self):
        row = self.rule_table.currentRow()
        if row < 0:
            return
        name = self.rule_table.item(row, 0).text()
        self.suppressor.remove_rule(name)
        self._refresh_rules()


class BatchReviewDialog(QDialog):
    """批量复核对话框"""

    def __init__(self, db, auth_manager, parent=None):
        super().__init__(parent)
        self.db = db
        self.auth = auth_manager
        self.setWindowTitle("批量复核")
        self.setFixedSize(800, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 未复核记录列表
        list_label = QLabel("未复核的NG记录:")
        list_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(list_label)

        self.record_table = QTableWidget()
        self.record_table.setColumnCount(7)
        self.record_table.setHorizontalHeaderLabels(
            ["选择", "ID", "时间", "SN码", "操作员", "缺陷数", "设备"])
        self.record_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.record_table.setAlternatingRowColors(True)
        layout.addWidget(self.record_table)

        self._load_records()

        # 按钮行
        btn_layout = QHBoxLayout()

        select_all_btn = QPushButton("全选")
        select_all_btn.setProperty("variant", "outline")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        confirm_btn = QPushButton("✓ 批量确认")
        confirm_btn.setProperty("variant", "success")
        confirm_btn.clicked.connect(lambda: self._batch_review("confirmed"))
        btn_layout.addWidget(confirm_btn)

        false_alarm_btn = QPushButton("⚠ 标记误报")
        false_alarm_btn.setProperty("variant", "warning")
        false_alarm_btn.clicked.connect(lambda: self._batch_review("false_alarm"))
        btn_layout.addWidget(false_alarm_btn)

        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "ghost")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _load_records(self):
        records = self.db.get_unreviewed(limit=50)
        self.record_table.setRowCount(len(records))
        self._checkboxes = []

        for i, rec in enumerate(records):
            cb = QCheckBox()
            self._checkboxes.append(cb)
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)

            self.record_table.setCellWidget(i, 0, cb_widget)
            self.record_table.setItem(i, 1, QTableWidgetItem(str(rec.get('id', ''))))
            self.record_table.setItem(i, 2, QTableWidgetItem(rec.get('timestamp', '')[:19]))
            self.record_table.setItem(i, 3, QTableWidgetItem(rec.get('product_sn', '')))
            self.record_table.setItem(i, 4, QTableWidgetItem(rec.get('operator_id', '')))
            self.record_table.setItem(i, 5, QTableWidgetItem(str(rec.get('num_defects', 0))))
            self.record_table.setItem(i, 6, QTableWidgetItem(rec.get('equipment_id', '')))

    def _select_all(self):
        for cb in self._checkboxes:
            cb.setChecked(not cb.isChecked())

    def _batch_review(self, result):
        ids = []
        for i, cb in enumerate(self._checkboxes):
            if cb.isChecked():
                id_item = self.record_table.item(i, 1)
                if id_item:
                    ids.append(int(id_item.text()))

        if not ids:
            QMessageBox.warning(self, "提示", "请先选择要复核的记录")
            return

        reviewer = self.auth.current_user.username if self.auth.current_user else "unknown"
        self.db.batch_review(ids, result, reviewer)
        QMessageBox.information(self, "完成", f"已复核 {len(ids)} 条记录")
        self._load_records()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, config=None, detector=None, controller=None):
        super().__init__()

        self.config = config or get_config()
        self.detector = detector
        self.controller = controller

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
        self.alarm_manager = AlarmManager(controller=self.controller)
        self.equipment_monitor = EquipmentMonitor(config=self.config)
        self.auth_manager = AuthManager(
            db_path=getattr(self.config, 'auth', None) and self.config.auth.db_path or None
        )
        self.report_generator = ProductionReportGenerator(self.trace_db)

        # 状态
        self.is_running = False
        self.is_emergency_stopped = False
        self.is_expert_mode = self.config.gui.expert_mode
        self.current_frame = None
        self.detection_results = None
        self._use_camera = True  # 是否使用真实摄像头（False=模拟模式）

        # 当前模式
        default_mode = getattr(self.config.gui, 'default_mode', 'production')
        self.production_manager.set_mode(default_mode)

        # 历史记录
        self.history = deque(maxlen=self.config.system.max_history)
        self.expert_scores = []
        self.total_images = 0
        self.correct_images = 0

        # 统计数据
        self._defect_counter = Counter()
        self._total_detections = 0
        self._total_pass = 0
        self._total_fail = 0
        self._inference_times = deque(maxlen=100)

        # SN码输入
        self._current_sn = ""

        # 初始化UI
        self.init_ui()
        self.init_timers()

        # 注册换线回调
        self.production_manager.on_line_switched(self._on_line_switched)
        self.production_manager.on_mode_changed(self._on_mode_changed)

        # 注册报警回调
        self.alarm_manager.on_alarm(self._on_alarm_triggered)

        logger.info("GUI初始化完成")

    def init_ui(self):
        """初始化用户界面 - 企业级风格"""
        self.setWindowTitle(self.config.gui.window_title)
        self.setGeometry(100, 100, self.config.gui.window_width, self.config.gui.window_height)

        # 应用全局样式表
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ===== 顶部品牌栏 =====
        self.brand_bar = BrandBar(
            title=self.config.gui.window_title,
            subtitle="PCB智能缺陷检测与分拣系统",
            parent=self,
        )
        main_layout.addWidget(self.brand_bar)

        # ===== 主体内容（左右分割） =====
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(3)

        # ---- 左侧面板 ----
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # 工具栏
        toolbar_frame = PanelSection("控制面板", collapsed=False)
        toolbar_frame.content_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_frame.content_layout.setSpacing(6)

        toolbar_layout = QHBoxLayout()

        self.mode_btn = QPushButton("产线模式")
        self.mode_btn.setProperty("variant", "outline")
        self.mode_btn.setFixedWidth(90)
        self.mode_btn.clicked.connect(self.toggle_mode)
        toolbar_layout.addWidget(self.mode_btn)

        self.line_switch_btn = QPushButton("一键换线")
        self.line_switch_btn.setProperty("variant", "outline")
        self.line_switch_btn.clicked.connect(self.show_line_switch)
        toolbar_layout.addWidget(self.line_switch_btn)

        self.line_label = QLabel(f"产线: {self.production_manager.current_profile_name}")
        self.line_label.setProperty("variant", "muted")
        toolbar_layout.addWidget(self.line_label)

        toolbar_layout.addWidget(QLabel("SN:"))
        self.sn_input = QLineEdit()
        self.sn_input.setPlaceholderText("扫码/输入SN码")
        self.sn_input.setMaximumWidth(160)
        self.sn_input.returnPressed.connect(self._on_sn_entered)
        toolbar_layout.addWidget(self.sn_input)

        self.suppression_btn = QPushButton("屏蔽区域")
        self.suppression_btn.setProperty("variant", "outline")
        self.suppression_btn.clicked.connect(self.show_suppression_dialog)
        toolbar_layout.addWidget(self.suppression_btn)

        self.kanban_btn = QPushButton("📺 看板")
        self.kanban_btn.setProperty("variant", "outline")
        self.kanban_btn.clicked.connect(self.show_kanban)
        toolbar_layout.addWidget(self.kanban_btn)

        toolbar_layout.addStretch()
        toolbar_frame.content_layout.addLayout(toolbar_layout)
        left_layout.addWidget(toolbar_frame)

        # 视频显示
        video_frame = PanelSection("实时画面")
        video_frame.content_layout.setContentsMargins(0, 0, 0, 0)
        video_frame.content_layout.setSpacing(0)

        self.video_label = QLabel("等待启动...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet(
            f"background-color: {Colors.BG_HEADER}; border-radius: 4px; "
            f"color: {Colors.TEXT_MUTED}; font-size: 14px;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_frame.content_layout.addWidget(self.video_label, 1)
        left_layout.addWidget(video_frame, 1)

        # 控制按钮
        controls_frame = PanelSection("操作控制")
        controls_frame.content_layout.setContentsMargins(8, 8, 8, 8)
        controls_frame.content_layout.setSpacing(8)

        # 第一行：启动 / 截图 / 加载图片
        btn_row1 = QHBoxLayout()

        self.start_btn = QPushButton("▶ 启动检测")
        self.start_btn.setProperty("variant", "success")
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self.toggle_detection)
        btn_row1.addWidget(self.start_btn)

        self.capture_btn = QPushButton("📷 截图")
        self.capture_btn.setProperty("variant", "outline")
        self.capture_btn.setFixedHeight(40)
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_image)
        btn_row1.addWidget(self.capture_btn)

        self.load_btn = QPushButton("📂 加载图片")
        self.load_btn.setProperty("variant", "outline")
        self.load_btn.setFixedHeight(40)
        self.load_btn.clicked.connect(self.load_image)
        btn_row1.addWidget(self.load_btn)

        controls_frame.content_layout.addLayout(btn_row1)

        # 第二行：急停 + 报表
        btn_row2 = QHBoxLayout()

        self.emergency_stop_btn = QPushButton("⛔ 急 停")
        self.emergency_stop_btn.setObjectName("emergencyBtn")
        self.emergency_stop_btn.setFixedHeight(46)
        self.emergency_stop_btn.clicked.connect(self.emergency_stop)
        btn_row2.addWidget(self.emergency_stop_btn, 1)

        self.export_csv_btn = QPushButton("📊 导出报表")
        self.export_csv_btn.setProperty("variant", "outline")
        self.export_csv_btn.setFixedHeight(40)
        self.export_csv_btn.clicked.connect(self.export_csv)
        btn_row2.addWidget(self.export_csv_btn)

        self.batch_review_btn = QPushButton("🔄 批量复核")
        self.batch_review_btn.setProperty("variant", "outline")
        self.batch_review_btn.setFixedHeight(40)
        self.batch_review_btn.clicked.connect(self.show_batch_review)
        btn_row2.addWidget(self.batch_review_btn)

        self.production_report_btn = QPushButton("📋 生产报表")
        self.production_report_btn.setProperty("variant", "outline")
        self.production_report_btn.setFixedHeight(40)
        self.production_report_btn.clicked.connect(self.show_production_report)
        btn_row2.addWidget(self.production_report_btn)

        controls_frame.content_layout.addLayout(btn_row2)
        left_layout.addWidget(controls_frame)

        # 底部状态栏
        status_frame = PanelSection("系统状态", collapsed=True)
        status_frame.content_layout.setContentsMargins(8, 4, 8, 4)
        status_frame.content_layout.setSpacing(4)

        status_layout = QHBoxLayout()

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setProperty("variant", "muted")
        status_layout.addWidget(self.status_label)

        self.camera_status_label = QLabel("摄像头: 未连接")
        self.camera_status_label.setProperty("variant", "muted")
        status_layout.addWidget(self.camera_status_label)

        self.health_label = QLabel("设备: 正常")
        self.health_label.setProperty("variant", "success")
        status_layout.addWidget(self.health_label)

        self.alarm_label = QLabel("")
        status_layout.addStretch()
        status_layout.addWidget(self.alarm_label)

        status_frame.content_layout.addLayout(status_layout)
        left_layout.addWidget(status_frame)

        content_splitter.addWidget(left_panel)

        # ---- 右侧面板 ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # 标签页
        self.tab_widget = QTabWidget()

        # === 检测结果页 ===
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(8)

        self.info_group = PanelSection("检测信息")
        info_layout = QGridLayout()
        info_layout.setSpacing(6)

        self.fps_label = QLabel("--")
        self.time_label = QLabel("-- ms")
        self.count_label = QLabel("--")
        self.conf_label = QLabel("--")

        info_layout.addWidget(QLabel("帧率:"), 0, 0)
        info_layout.addWidget(self.fps_label, 0, 1)
        info_layout.addWidget(QLabel("推理时间:"), 1, 0)
        info_layout.addWidget(self.time_label, 1, 1)
        info_layout.addWidget(QLabel("检测数量:"), 2, 0)
        info_layout.addWidget(self.count_label, 2, 1)
        info_layout.addWidget(QLabel("平均置信度:"), 3, 0)
        info_layout.addWidget(self.conf_label, 3, 1)
        self.info_group.content_layout.addLayout(info_layout)
        result_layout.addWidget(self.info_group)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度", "位置", "操作"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setAlternatingRowColors(True)
        result_layout.addWidget(self.result_table)
        self.tab_widget.addTab(result_tab, "🔍 检测结果")

        # === 专家评分页 ===
        if self.is_expert_mode:
            expert_tab = QWidget()
            expert_layout = QVBoxLayout(expert_tab)
            expert_layout.setSpacing(8)

            self.score_label = QLabel("评分统计: 0/0 (0%)")
            self.score_label.setProperty("variant", "accent")
            self.score_label.setStyleSheet("font-size: 15px; padding: 8px;")
            expert_layout.addWidget(self.score_label)

            score_btn_layout = QHBoxLayout()

            self.correct_btn = QPushButton("✓ 正确")
            self.correct_btn.setProperty("variant", "success")
            self.correct_btn.setFixedHeight(40)
            self.correct_btn.clicked.connect(lambda: self.score_detection(True))
            score_btn_layout.addWidget(self.correct_btn)

            self.wrong_btn = QPushButton("✗ 错误")
            self.wrong_btn.setProperty("variant", "danger")
            self.wrong_btn.setFixedHeight(40)
            self.wrong_btn.clicked.connect(lambda: self.score_detection(False))
            score_btn_layout.addWidget(self.wrong_btn)

            expert_layout.addLayout(score_btn_layout)

            self.score_table = QTableWidget()
            self.score_table.setColumnCount(4)
            self.score_table.setHorizontalHeaderLabels(["序号", "时间", "结果", "判定"])
            self.score_table.setAlternatingRowColors(True)
            expert_layout.addWidget(self.score_table)

            self.tab_widget.addTab(expert_tab, "⭐ 专家评分")

        # === 统计仪表盘页 ===
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setSpacing(8)

        # 合格率 + 缺陷分布并排
        top_stats = QHBoxLayout()

        pass_rate_frame = PanelSection("合格率")
        pass_rate_inner = QHBoxLayout()
        self.pass_rate_widget = PassRateWidget()
        pass_rate_inner.addWidget(self.pass_rate_widget)
        self.pass_rate_detail_label = QLabel("合格: 0 | 缺陷: 0 | 总计: 0")
        self.pass_rate_detail_label.setProperty("variant", "muted")
        pass_rate_inner.addWidget(self.pass_rate_detail_label)
        pass_rate_frame.content_layout.addLayout(pass_rate_inner)
        top_stats.addWidget(pass_rate_frame, 1)

        defect_frame = PanelSection("缺陷类别分布")
        self.defect_bar_widget = DefectBarWidget()
        defect_frame.content_layout.addWidget(self.defect_bar_widget)
        top_stats.addWidget(defect_frame, 1)

        stats_layout.addLayout(top_stats)

        # 性能统计 + 设备健康度
        bottom_stats = QHBoxLayout()

        perf_frame = PanelSection("性能统计")
        perf_layout = QGridLayout()
        perf_layout.setSpacing(6)

        self.avg_inference_label = QLabel("平均: -- ms")
        self.max_inference_label = QLabel("最大: -- ms")
        self.min_inference_label = QLabel("最小: -- ms")
        self.total_detected_label = QLabel("累计: 0 帧")

        for row, (label, val) in enumerate([
            ("推理耗时", self.avg_inference_label),
            ("", self.max_inference_label),
            ("", self.min_inference_label),
            ("累计检测", self.total_detected_label),
        ]):
            if label:
                lbl = QLabel(label + ":")
                lbl.setProperty("variant", "muted")
                perf_layout.addWidget(lbl, row, 0)
            perf_layout.addWidget(val, row, 1)

        perf_frame.content_layout.addLayout(perf_layout)
        bottom_stats.addWidget(perf_frame, 1)

        health_frame = PanelSection("设备健康度")
        health_layout = QGridLayout()
        health_layout.setSpacing(6)

        self.light_health_label = QLabel("光源: 正常")
        self.camera_health_label = QLabel("摄像头: 正常")
        self.inference_health_label = QLabel("推理: 正常")
        self.brightness_label = QLabel("亮度: --")

        health_layout.addWidget(self.light_health_label, 0, 0)
        health_layout.addWidget(self.camera_health_label, 0, 1)
        health_layout.addWidget(self.inference_health_label, 1, 0)
        health_layout.addWidget(self.brightness_label, 1, 1)
        health_frame.content_layout.addLayout(health_layout)
        bottom_stats.addWidget(health_frame, 1)

        stats_layout.addLayout(bottom_stats)
        stats_layout.addStretch()
        self.tab_widget.addTab(stats_tab, "📈 统计仪表盘")

        # === 历史记录页 ===
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setSpacing(8)

        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("SN码:"))
        self.query_sn_edit = QLineEdit()
        self.query_sn_edit.setPlaceholderText("输入SN码查询（留空显示全部）")
        query_layout.addWidget(self.query_sn_edit)
        query_btn = QPushButton("查询")
        query_btn.setProperty("variant", "outline")
        query_btn.clicked.connect(self._query_by_sn)
        query_layout.addWidget(query_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setProperty("variant", "outline")
        refresh_btn.clicked.connect(self._load_all_history)
        query_layout.addWidget(refresh_btn)

        load_more_btn = QPushButton("加载更多")
        load_more_btn.setProperty("variant", "outline")
        load_more_btn.clicked.connect(self._load_all_history)
        query_layout.addWidget(load_more_btn)
        query_layout.addStretch()
        history_layout.addLayout(query_layout)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "SN码", "结果", "缺陷数", "操作员", "设备", "产线", "推理时间"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        history_layout.addWidget(self.history_table)

        # 保留 QTextEdit 作为详细查看
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(120)
        history_layout.addWidget(QLabel("详情:"))
        history_layout.addWidget(self.history_text)

        self.tab_widget.addTab(history_tab, "📜 历史记录")

        # 默认加载历史记录
        self._load_all_history()

        right_layout.addWidget(self.tab_widget)

        # 系统日志
        log_frame = PanelSection("系统日志", collapsed=True)
        log_frame.content_layout.setContentsMargins(4, 4, 4, 4)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet(f"""
            background-color: {Colors.PRIMARY_DARK};
            color: {Colors.TEXT_SECONDARY};
            font-family: Consolas, "Courier New", monospace;
            font-size: 11px;
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
            padding: 6px;
        """)
        log_frame.content_layout.addWidget(self.log_text)
        right_layout.addWidget(log_frame)

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([900, 500])

        main_layout.addWidget(content_splitter, 1)

        # 底部统计卡片行
        self._init_stat_cards()
        main_layout.addWidget(self.stat_card_row)

        # 设置急停按钮样式（唯一一处）
        self.emergency_stop_btn.setStyleSheet(EMERGENCY_STOP_STYLE)

    def init_timers(self):
        """初始化定时器"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_display)
        self.refresh_timer.start(self.config.gui.refresh_interval)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

        # 设备健康度更新
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.update_health_display)
        self.health_timer.start(5000)

    def _init_stat_cards(self):
        """初始化底部统计卡片行"""
        self._stat_cards = []

        def make_card(icon, label, color):
            card = StatCard(icon, label, "0", color)
            self._stat_cards.append(card)
            return card

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(4, 4, 4, 4)
        row_layout.setSpacing(8)

        cards_data = [
            ("📊", "总产量", Colors.ACCENT),
            ("✅", "合格品", Colors.SUCCESS),
            ("❌", "不良品", Colors.DANGER),
            ("📈", "合格率", Colors.WARNING),
            ("⚡", "推理耗时", Colors.PRIMARY_LIGHT),
        ]

        self._stat_values = []
        for icon, label, color in cards_data:
            card = make_card(icon, label, color)
            row_layout.addWidget(card, 1)
            self._stat_values.append(card.value_label)

        self.stat_card_row = QWidget()
        self.stat_card_row.setLayout(row_layout)
        self.stat_card_row.setStyleSheet(f"background-color: {Colors.BG_PAGE}; border-radius: 4px; padding: 4px;")

    def _calc_pass_rate(self) -> float:
        total = self._total_pass + self._total_fail
        return (self._total_pass / total * 100) if total > 0 else 0.0

    def _avg_inference(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    # ==================== 模式切换 ====================

    def toggle_mode(self):
        """切换产线/调试模式"""
        if self.production_manager.is_production_mode:
            self.production_manager.set_mode(ProductionManager.MODE_DEBUG)
        else:
            self.production_manager.set_mode(ProductionManager.MODE_PRODUCTION)

    def _on_mode_changed(self, mode):
        """模式切换回调"""
        if mode == ProductionManager.MODE_PRODUCTION:
            self.mode_btn.setText("产线模式")
            self.mode_btn.setProperty("variant", "outline")
            self.brand_bar.set_mode("production")
            if hasattr(self, 'info_group'):
                self.info_group.setTitle("检测信息")
        else:
            self.mode_btn.setText("调试模式")
            self.mode_btn.setProperty("variant", "warning")
            self.brand_bar.set_mode("debug")
            if hasattr(self, 'info_group'):
                self.info_group.setTitle("检测信息 (调试)")

    # ==================== 一键换线 ====================

    def show_line_switch(self):
        """显示一键换线对话框"""
        dialog = LineSwitchDialog(self.production_manager, self)
        dialog.exec_()
        self.line_label.setText(f"产线: {self.production_manager.current_profile_name}")

    def _on_line_switched(self, profile):
        """换线回调 - 更新检测器参数"""
        if self.detector:
            try:
                self.detector.conf_threshold = profile.conf_threshold
                self.detector.iou_threshold = profile.iou_threshold
                logger.info(f"检测器参数已更新: conf={profile.conf_threshold}, iou={profile.iou_threshold}")
            except Exception as e:
                logger.error(f"更新检测器参数失败: {e}")

        # 切换误报屏蔽规则
        self.suppressor.set_profile(profile.name)

    # ==================== 误报屏蔽 ====================

    def show_suppression_dialog(self):
        """显示误报屏蔽管理对话框"""
        dialog = SuppressionDialog(self.suppressor, self)
        dialog.exec_()

    def show_kanban(self):
        """打开工控看板窗口"""
        from PyQt5.QtWidgets import QMessageBox
        try:
            from ui.kanban import run_kanban
            # 在子线程中运行看板，避免阻塞主GUI
            import threading
            threading.Thread(target=run_kanban, args=(self, ), daemon=True).start()
        except Exception as e:
            logger.error(f"看板启动失败: {e}")
            QMessageBox.critical(self, "看板错误", f"看板启动失败:\n{str(e)}")

    # ==================== SN码 ====================

    # SN码最大长度限制
    SN_MAX_LENGTH = 64
    # SN码正则：只允许字母、数字、连字符、下划线
    SN_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')

    def _on_sn_entered(self):
        """SN码输入回调"""
        sn = self.sn_input.text().strip()
        if not sn:
            return
        # 验证SN码格式
        if len(sn) > self.SN_MAX_LENGTH:
            self.log_text.append(f"[WARNING] SN码长度超过限制({self.SN_MAX_LENGTH}字符)")
            sn = sn[:self.SN_MAX_LENGTH]
        if not self.SN_PATTERN.match(sn):
            self.log_text.append(f"[WARNING] SN码包含非法字符，只允许字母、数字、连字符、下划线")
            # 过滤掉非法字符
            sn = ''.join(c for c in sn if c.isalnum() or c in '_-')
            if not sn:
                return
        self._current_sn = sn
        self.log_text.append(f"[INFO] SN码已设置: {self._current_sn}")
        self.sn_input.clear()

    # ==================== 批量复核 ====================

    def show_batch_review(self):
        """显示批量复核对话框"""
        dialog = BatchReviewDialog(self.trace_db, self.auth_manager, self)
        dialog.exec_()

    # ==================== 生产报表 ====================

    def show_production_report(self):
        """显示生产报表"""
        from PyQt5.QtWidgets import QInputDialog
        modes = ["班次报表", "缺陷趋势(7天)", "操作员绩效"]
        mode, ok = QInputDialog.getItem(self, "生产报表", "选择报表类型:", modes, 0, False)
        if not ok:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存报表",
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv);;JSON文件 (*.json)"
        )
        if not file_path:
            return

        try:
            if mode == "班次报表":
                report = self.report_generator.generate_shift_report()
            elif mode == "缺陷趋势(7天)":
                report = self.report_generator.generate_defect_trend(days=7)
            else:
                report = self.report_generator.generate_operator_report()

            if file_path.endswith('.json'):
                self.report_generator.export_json(report, file_path)
            else:
                self.report_generator.export_csv(report, file_path)

            QMessageBox.information(self, "导出成功", f"报表已保存至:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"报表生成失败:\n{str(e)}")

    # ==================== 追溯查询 ====================

    def _query_by_sn(self):
        """按SN码查询"""
        sn = self.query_sn_edit.text().strip()
        if not sn:
            self._load_all_history()
            return

        records = self.trace_db.query_by_sn(sn)
        if not records:
            QMessageBox.information(self, "查询结果", f"未找到SN码 [{sn}] 的记录")
            self.history_table.setRowCount(0)
            self.history_text.clear()
            return

        self._populate_history_table(records)

    def _load_all_history(self):
        """加载全部历史记录"""
        records = self.trace_db.get_recent_records(limit=200)
        self._populate_history_table(records)

    def _populate_history_table(self, records):
        """将记录填充到历史表格"""
        self.history_table.setRowCount(len(records))
        for i, rec in enumerate(records):
            ts = rec.get('timestamp', '')[:19]
            sn = rec.get('product_sn', '-')
            result = rec.get('result', '-')
            defects = str(rec.get('num_defects', 0))
            operator = rec.get('operator_id', '-')
            equipment = rec.get('equipment_id', '-')
            line = rec.get('line_profile', '-')
            infer_time = f"{rec.get('inference_time_ms', 0):.0f}ms"

            self.history_table.setItem(i, 0, QTableWidgetItem(ts))
            self.history_table.setItem(i, 1, QTableWidgetItem(sn))
            self.history_table.setItem(i, 2, QTableWidgetItem(result))
            self.history_table.setItem(i, 3, QTableWidgetItem(defects))
            self.history_table.setItem(i, 4, QTableWidgetItem(operator))
            self.history_table.setItem(i, 5, QTableWidgetItem(equipment))
            self.history_table.setItem(i, 6, QTableWidgetItem(line))
            self.history_table.setItem(i, 7, QTableWidgetItem(infer_time))

        # 高亮NG行
        ng_color = QColor(Colors.DANGER)
        for i in range(len(records)):
            item = self.history_table.item(i, 2)
            if item and item.text() == 'NG':
                item.setForeground(ng_color)
                self.history_table.item(i, 3).setForeground(ng_color)

    # ==================== 报警 ====================

    def _on_alarm_triggered(self, level, defect_count, message):
        """报警回调 - 更新UI"""
        level_names = {AlarmLevel.INFO: "提示", AlarmLevel.WARNING: "警告", AlarmLevel.CRITICAL: "严重"}
        colors_map = {AlarmLevel.INFO: Colors.ACCENT, AlarmLevel.WARNING: Colors.WARNING, AlarmLevel.CRITICAL: Colors.DANGER}

        self.alarm_label.setText(f"报警: [{level_names.get(level, level)}] {message}")
        self.alarm_label.setProperty("variant", "danger")
        self.alarm_label.setStyleSheet(f"color: {colors_map.get(level, Colors.DANGER)}; font-weight: bold;")

        # 3秒后清除报警显示
        QTimer.singleShot(3000, lambda: self.alarm_label.setText(""))

    # ==================== 设备健康度 ====================

    def update_health_display(self):
        """更新设备健康度显示"""
        details = self.equipment_monitor.get_details()
        status_colors = {
            HealthStatus.GOOD: "#10B981",
            HealthStatus.WARNING: "#F59E0B",
            HealthStatus.CRITICAL: "#EF4444",
        }
        status_names = {HealthStatus.GOOD: "正常", HealthStatus.WARNING: "注意", HealthStatus.CRITICAL: "异常"}

        light = details.get('light_source', {})
        camera = details.get('camera', {})
        inference = details.get('inference', {})

        self.light_health_label.setText(
            f"光源: {status_names.get(light.get('status', 'good'), '正常')}"
        )
        self.light_health_label.setStyleSheet(
            f"color: {status_colors.get(light.get('status', 'good'), Colors.SUCCESS)};"
        )

        self.camera_health_label.setText(
            f"摄像头: {status_names.get(camera.get('status', 'good'), '正常')} ({camera.get('avg_fps', 0):.0f}fps)"
        )
        self.camera_health_label.setStyleSheet(
            f"color: {status_colors.get(camera.get('status', 'good'), Colors.SUCCESS)};"
        )

        self.inference_health_label.setText(
            f"推理: {status_names.get(inference.get('status', 'good'), '正常')} ({inference.get('avg_time_ms', 0):.0f}ms)"
        )
        self.inference_health_label.setStyleSheet(
            f"color: {status_colors.get(inference.get('status', 'good'), Colors.SUCCESS)};"
        )

        self.brightness_label.setText(f"亮度: {light.get('avg_brightness', 0):.0f}")

        # 更新状态栏
        overall = self.equipment_monitor.overall_status
        self.health_label.setText(f"设备: {status_names.get(overall, '正常')}")
        self.health_label.setStyleSheet(
            f"color: {status_colors.get(overall, Colors.SUCCESS)};"
        )

    # ==================== 检测控制 ====================

    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "检测器未初始化")
            return

        self.is_emergency_stopped = False
        self.emergency_stop_btn.setStyleSheet(EMERGENCY_STOP_STYLE)

        # 尝试打开摄像头，如果失败则切换到图片模拟模式
        self._use_camera = True
        self.camera_thread = CameraThread(
            camera_id=self.config.hardware.camera.index,
            width=self.config.hardware.camera.width,
            height=self.config.hardware.camera.height,
            fps=self.config.hardware.camera.fps
        )
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.error_occurred.connect(self.on_error)
        self.camera_thread.connection_lost.connect(self.on_camera_lost)
        self.camera_thread.connection_restored.connect(self.on_camera_restored)
        self.camera_thread.start()

        # 摄像头状态标签由 CameraThread 的信号更新
        # 但由于 Qt 信号槽队列排队机制，可能需要刷新事件循环
        # 这里用 QTimer 主动同步一次状态
        QTimer.singleShot(1500, self._check_camera_and_fallback)

        self.detection_thread = DetectionThread(self.detector)
        self.detection_thread.result_ready.connect(self.on_detection_result)
        self.detection_thread.error_occurred.connect(self.on_error)
        self.detection_thread.start()

        self.is_running = True
        self.start_btn.setText("⏹ 停止检测")
        self.start_btn.setProperty("variant", "danger")
        self.capture_btn.setEnabled(True)

        logger.info("检测已启动")

    def _check_camera_and_fallback(self):
        """如果摄像头未连接，自动切换到模拟图片模式"""
        if not self._use_camera:
            return  # 已经手动切换到模拟模式

        if hasattr(self, 'camera_thread') and self.camera_thread.is_connected():
            return  # 摄像头正常

        # 摄像头不可用，切换到模拟模式
        logger.warning("摄像头不可用，切换到模拟图片模式")
        self._use_camera = False
        self.camera_status_label.setText("摄像头: 模拟模式")
        self.camera_status_label.setStyleSheet(f"color: {Colors.WARNING}; background-color: {Colors.PRIMARY_DARK};")
        self.log_text.append("[INFO] 摄像头不可用，已切换到模拟图片模式。请使用'加载图片'按钮进行测试。")

        # 自动生成一张模拟 PCB 图像用于演示
        self._generate_mock_frame()

    def _generate_mock_frame(self):
        """生成一张模拟 PCB 图像用于演示（无摄像头时的 fallback）"""
        try:
            from demo_simulate import generate_sample_image
            img, gt = generate_sample_image([random.randint(0, 5)])
            self.current_frame = img
            # 立即进行一次检测
            results = self.detector.detect(img)
            vis_frame = self.detector.draw_results(img, results)
            self.on_detection_result(results, vis_frame)
        except Exception as e:
            logger.error(f"生成模拟图像失败: {e}")

    def stop_detection(self):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        self.is_running = False
        self.start_btn.setText("▶ 启动检测")
        self.start_btn.setProperty("variant", "success")
        self.capture_btn.setEnabled(False)
        if not getattr(self, '_use_camera', True):
            self.camera_status_label.setText("摄像头: 模拟模式")
            self.camera_status_label.setStyleSheet(f"color: {Colors.WARNING}; background-color: {Colors.PRIMARY_DARK};")
        else:
            self.camera_status_label.setText("摄像头: 未连接")
            self.camera_status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background-color: {Colors.PRIMARY_DARK};")

        logger.info("检测已停止")

    def emergency_stop(self):
        if self.is_emergency_stopped:
            self.is_emergency_stopped = False
            self.emergency_stop_btn.setStyleSheet(EMERGENCY_STOP_STYLE)
            self.emergency_stop_btn.setText("⛔ 急 停")
            self.log_text.append("[INFO] 急停已解除，可重新启动检测")
            return

        self.is_emergency_stopped = True
        self.emergency_stop_btn.setStyleSheet(
            f"background-color: {Colors.PRIMARY_DARK}; color: {Colors.DANGER_LIGHT}; "
            f"font-size: 15px; font-weight: bold; padding: 10px 20px; "
            f"border: 2px solid {Colors.DANGER_LIGHT}; border-radius: 8px;"
        )
        self.emergency_stop_btn.setText("⛔ 已急停 - 点击解除")

        if self.controller is not None:
            try:
                self.controller.emergency_stop()
            except Exception as e:
                logger.error(f"硬件急停执行失败: {e}")

        self.stop_detection()
        self.log_text.append("[WARNING] 急停已触发！传送带和舵机已停止！")

    # ==================== 检测结果处理 ====================

    def on_frame_ready(self, frame):
        self.current_frame = frame
        if hasattr(self, 'detection_thread') and not self.is_emergency_stopped:
            self.detection_thread.set_frame(frame)

    def on_detection_result(self, results, vis_frame):
        self.detection_results = results

        # 误报屏蔽过滤
        if results.get('detections'):
            valid, suppressed = self.suppressor.filter_detections(results['detections'])
            results['detections'] = valid
            results['num_detections'] = len(valid)
            if suppressed:
                self.log_text.append(f"[INFO] 屏蔽了 {len(suppressed)} 个误报")

        # 设备健康度记录
        self.equipment_monitor.record_frame(vis_frame, results.get('inference_time', 0))

        # 声光报警
        num_defects = results.get('num_detections', 0)
        if num_defects > 0:
            level = AlarmLevel.CRITICAL if num_defects >= 3 else AlarmLevel.WARNING
            self.alarm_manager.trigger(level, num_defects, f"检出{num_defects}个缺陷")

        # 追溯记录（仅 NG 帧写入数据库，避免 OK 帧高频写入导致 I/O 竞争）
        result_status = "NG" if num_defects > 0 else "OK"
        if num_defects > 0:
            shift = self.production_manager.shift_info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(self.config.system.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(save_dir / f"gui_raw_{timestamp}.jpg")
            result_image_path = str(save_dir / f"gui_result_{timestamp}.jpg")
            cv2.imwrite(image_path, vis_frame)  # 保存带标注的图
            self.trace_db.add_record(
                product_sn=self._current_sn,
                operator_id=shift.get('operator_id', ''),
                equipment_id=shift.get('equipment_id', ''),
                line_profile=self.production_manager.current_profile_name,
                result=result_status,
                num_defects=num_defects,
                inference_time_ms=results.get('inference_time', 0),
                image_path=image_path,
                result_image_path=result_image_path,
                defects=results.get('detections', []),
            )

        # SN码消费后清空
        if self._current_sn:
            self._current_sn = ""

        self.display_frame(vis_frame)
        self.update_result_info(results)
        self.update_statistics(results)

    def on_error(self, error_msg):
        logger.error(error_msg)
        self.log_text.append(f"[ERROR] {error_msg}")

    def on_camera_lost(self):
        self.camera_status_label.setText("摄像头: 已断开 (尝试重连中...)")
        self.camera_status_label.setStyleSheet(f"color: {Colors.DANGER}; background-color: {Colors.PRIMARY_DARK};")
        self.log_text.append("[WARNING] 摄像头连接已断开，正在尝试重连...")

    def on_camera_restored(self):
        self.camera_status_label.setText("摄像头: 已连接")
        self.camera_status_label.setStyleSheet(f"color: {Colors.SUCCESS}; background-color: {Colors.PRIMARY_DARK};")
        self.log_text.append("[INFO] 摄像头连接已恢复")

    def _sync_camera_status(self):
        """主动同步摄像头状态到UI标签（修复Qt信号队列延迟问题）"""
        if hasattr(self, 'camera_thread') and self.camera_thread.is_connected():
            current = self.camera_status_label.text()
            if "未连接" in current or "已断开" in current:
                self.camera_status_label.setText("摄像头: 已连接")
                self.camera_status_label.setStyleSheet(f"color: {Colors.SUCCESS}; background-color: {Colors.PRIMARY_DARK};")

    def update_statistics(self, results):
        self._total_detections += 1
        num_defects = results.get('num_detections', 0)

        if num_defects == 0:
            self._total_pass += 1
        else:
            self._total_fail += 1

        for det in results.get('detections', []):
            class_name = det.get('class_name', 'unknown')
            self._defect_counter[class_name] += 1

        inference_time = results.get('inference_time', 0)
        if inference_time > 0:
            self._inference_times.append(inference_time)

        total = self._total_pass + self._total_fail
        pass_rate = (self._total_pass / total * 100) if total > 0 else 0
        self.pass_rate_widget.set_rate(pass_rate)
        self.pass_rate_detail_label.setText(
            f"合格: {self._total_pass} | 缺陷: {self._total_fail} | 总计: {total}"
        )

        self.defect_bar_widget.set_data(dict(self._defect_counter))

        if self._inference_times:
            avg_time = sum(self._inference_times) / len(self._inference_times)
            max_time = max(self._inference_times)
            min_time = min(self._inference_times)
            self.avg_inference_label.setText(f"平均推理耗时: {avg_time:.1f} ms")
            self.max_inference_label.setText(f"最大推理耗时: {max_time:.1f} ms")
            self.min_inference_label.setText(f"最小推理耗时: {min_time:.1f} ms")

        self.total_detected_label.setText(f"累计检测帧数: {self._total_detections}")

        # 更新底部统计卡片
        if hasattr(self, '_stat_values'):
            avg_inf = self._avg_inference()
            self._stat_values[0].setText(str(self._total_detections))
            self._stat_values[1].setText(str(self._total_pass))
            self._stat_values[2].setText(str(self._total_fail))
            self._stat_values[3].setText(f"{pass_rate:.1f}%")
            self._stat_values[4].setText(f"{avg_inf:.0f} ms")

    def export_csv(self):
        """导出CSV报表"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV报表", f"pcb_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                writer.writerow(["PCB缺陷检测系统 - 检测报表"])
                writer.writerow(["导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(["产线", self.production_manager.current_profile_name])
                writer.writerow([])

                writer.writerow(["=== 统计概要 ==="])
                total = self._total_pass + self._total_fail
                pass_rate = (self._total_pass / total * 100) if total > 0 else 0
                writer.writerow(["总检测帧数", self._total_detections])
                writer.writerow(["合格帧数", self._total_pass])
                writer.writerow(["缺陷帧数", self._total_fail])
                writer.writerow(["合格率", f"{pass_rate:.1f}%"])

                if self._inference_times:
                    avg_time = sum(self._inference_times) / len(self._inference_times)
                    writer.writerow(["平均推理耗时(ms)", f"{avg_time:.1f}"])
                    writer.writerow(["最大推理耗时(ms)", f"{max(self._inference_times):.1f}"])
                    writer.writerow(["最小推理耗时(ms)", f"{min(self._inference_times):.1f}"])
                writer.writerow([])

                writer.writerow(["=== 缺陷类别分布 ==="])
                writer.writerow(["类别", "数量", "占比"])
                total_defects = sum(self._defect_counter.values())
                for cat, count in self._defect_counter.most_common():
                    ratio = f"{count / total_defects * 100:.1f}%" if total_defects > 0 else "0%"
                    writer.writerow([cat, count, ratio])
                writer.writerow([])

                if self.expert_scores:
                    writer.writerow(["=== 专家评分记录 ==="])
                    writer.writerow(["序号", "时间", "检测数量", "评分"])
                    for record in self.expert_scores:
                        writer.writerow([
                            record['index'], record['time'],
                            record['detections'], "正确" if record['correct'] else "错误"
                        ])

            QMessageBox.information(self, "导出成功", f"报表已保存至:\n{file_path}")
            logger.info(f"CSV报表已导出: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出CSV失败:\n{str(e)}")

    def display_frame(self, frame):
        if frame is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def update_result_info(self, results):
        self.fps_label.setText(f"{results.get('fps', 0):.1f}")
        self.time_label.setText(f"{results.get('inference_time', 0):.1f} ms")
        self.count_label.setText(f"{results.get('num_detections', 0)}")

        detections = results.get('detections', [])
        if detections:
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            self.conf_label.setText(f"{avg_conf:.3f}")
        else:
            self.conf_label.setText("--")

        self.result_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            self.result_table.setItem(i, 0, QTableWidgetItem(det.get('class_name', '')))
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{det.get('confidence', 0):.3f}"))
            bbox = det.get('bbox', [0, 0, 0, 0])
            bbox_str = f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            self.result_table.setItem(i, 2, QTableWidgetItem(bbox_str))

            # 添加"屏蔽此区域"按钮
            suppress_btn = QPushButton("屏蔽")
            suppress_btn.clicked.connect(lambda checked, b=bbox: self._quick_suppress(b))
            self.result_table.setCellWidget(i, 3, suppress_btn)

    def _quick_suppress(self, bbox):
        """快速屏蔽某个检测框区域"""
        x1, y1, x2, y2 = map(int, bbox)
        margin = 20  # 扩展一点区域
        name = f"auto_{datetime.now().strftime('%H%M%S')}"
        self.suppressor.add_rect_rule(
            name=name,
            x1=max(0, x1 - margin), y1=max(0, y1 - margin),
            x2=x2 + margin, y2=y2 + margin,
            note=f"自动屏蔽于{datetime.now().strftime('%H:%M:%S')}"
        )
        self.log_text.append(f"[INFO] 已添加屏蔽区域: {name}")

    def update_display(self):
        pass

    def update_status(self):
        if self.is_emergency_stopped:
            status = "⛔ 急停"
        elif self.is_running:
            status = "运行中"
        else:
            status = "就绪"

        user_info = ""
        if self.auth_manager.current_user:
            user_info = f" | {self.auth_manager.current_user.display_name}"

        self.status_label.setText(
            f"状态: {status} | 历史记录: {len(self.history)}{user_info}"
        )

    def capture_image(self):
        if self.current_frame is None:
            return

        save_dir = Path(self.config.system.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"capture_{timestamp}.jpg"

        cv2.imwrite(str(save_path), self.current_frame)
        logger.info(f"截图已保存: {save_path}")
        self.log_text.append(f"[INFO] 截图已保存: {save_path}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            if self.detector:
                results, vis_image = self.detector.detect_image(file_path, show=False)
                if vis_image is not None:
                    self.display_frame(vis_image)
                    self.update_result_info(results)
                    self.update_statistics(results)
                    self.current_frame = vis_image
                    logger.info(f"已加载图片: {file_path}")

    def score_detection(self, is_correct):
        if self.detection_results is None:
            return

        self.total_images += 1
        if is_correct:
            self.correct_images += 1

        accuracy = self.correct_images / max(self.total_images, 1) * 100
        self.score_label.setText(f"评分统计: {self.correct_images}/{self.total_images} ({accuracy:.1f}%)")

        score_record = {
            'index': self.total_images,
            'time': datetime.now().strftime("%H:%M:%S"),
            'detections': self.detection_results.get('num_detections', 0),
            'correct': is_correct
        }
        self.expert_scores.append(score_record)

        row = self.score_table.rowCount()
        self.score_table.insertRow(row)
        self.score_table.setItem(row, 0, QTableWidgetItem(str(score_record['index'])))
        self.score_table.setItem(row, 1, QTableWidgetItem(score_record['time']))
        self.score_table.setItem(row, 2, QTableWidgetItem(str(score_record['detections'])))
        self.score_table.setItem(row, 3, QTableWidgetItem("✓" if is_correct else "✗"))

    def closeEvent(self, event):
        self.stop_detection()
        self.trace_db.close()
        event.accept()


def main(mock=False):
    """主函数
    
    Args:
        mock: 是否使用模拟检测器（由 main.py 传入）
    """
    import argparse as _argparse

    _parsed_mock = mock
    if not _parsed_mock:
        _remaining = [a for a in sys.argv[1:] if a.startswith("--")]
        if "--mock" in _remaining:
            _parsed_mock = True

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon())  # 使用系统默认图标

    config = get_config()

    # 权限管理 - 登录
    auth_manager = AuthManager(
        db_path=getattr(config, 'auth', None) and config.auth.db_path or None
    )

    if getattr(config, 'auth', None) and config.auth.enabled:
        login_dialog = LoginDialog(auth_manager)
        if login_dialog.exec_() != QDialog.Accepted:
            sys.exit(0)

    # 创建检测器
    detector = None
    try:
        if _parsed_mock:
            logger.info("使用模拟检测器模式")
            from demo_simulate import MockDetector
            detector = MockDetector(config)
        else:
            from inference import PCBDefectDetector
            detector = PCBDefectDetector(config)
    except Exception as e:
        logger.warning(f"检测器初始化失败: {e}")
        logger.warning("将使用模拟检测器模式作为后备")
        try:
            from demo_simulate import MockDetector
            detector = MockDetector(config)
            logger.info("已回退到模拟检测器模式")
        except Exception as e2:
            logger.error(f"模拟检测器也失败: {e2}")

    # 创建控制器
    from utils.serial_control import create_controller
    controller = create_controller(config, mock=True)
    controller.connect()

    # 创建主窗口
    window = MainWindow(config, detector, controller)
    window.auth_manager = auth_manager
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


