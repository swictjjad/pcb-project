# -*- coding: utf-8 -*-
"""
企业级主题模块
功能：
1. 统一颜色调色板
2. 全局样式表（QSS）
3. 预制控件样式常量
4. 品牌栏、统计卡片等复合控件
"""

from datetime import datetime
from collections import deque
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QVBoxLayout, QFrame,
    QMainWindow, QListWidget, QListWidgetItem, QTableWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QRadialGradient


# ==================== 颜色调色板 ====================

class Colors:
    """企业级颜色调色板（白/灰风格）"""
    # 主色 - 深蓝（仅用于品牌栏和强调）
    PRIMARY = "#1E3A5F"
    PRIMARY_LIGHT = "#2D5F8A"
    PRIMARY_DARK = "#0F2440"
    # 强调色 - 亮蓝
    ACCENT = "#3B82F6"
    ACCENT_LIGHT = "#60A5FA"
    ACCENT_DARK = "#2563EB"
    # 成功色
    SUCCESS = "#10B981"
    SUCCESS_LIGHT = "#34D399"
    # 警告色
    WARNING = "#F59E0B"
    WARNING_LIGHT = "#FBBF24"
    # 危险色
    DANGER = "#EF4444"
    DANGER_LIGHT = "#F87171"
    # 中性色 - 白色/浅灰
    BG_BASE = "#FFFFFF"           # 主背景
    BG_PAGE = "#F8F9FA"           # 页面背景
    BG_CARD = "#FFFFFF"           # 卡片背景
    BG_HEADER = "#F1F3F5"         # 头部/栏位背景
    BG_INPUT = "#FFFFFF"          # 输入框背景
    BORDER = "#E5E7EB"            # 边框
    BORDER_FOCUS = "#CBD5E1"      # 聚焦边框
    BORDER_LIGHT = "#D1D5DB"      # 浅色边框（兼容）
    TEXT_PRIMARY = "#111827"      # 主文本
    TEXT_SECONDARY = "#6B7280"    # 次要文本
    TEXT_MUTED = "#9CA3AF"        # 淡化文本
    # 渐变
    GRADIENT_PRIMARY = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E3A5F, stop:1 #2D5F8A)"
    GRADIENT_ACCENT = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3B82F6, stop:1 #60A5FA)"
    GRADIENT_SUCCESS = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #34D399)"
    GRADIENT_DANGER = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EF4444, stop:1 #F87171)"


# ==================== 按钮样式常量 ====================

BTN_BASE = (
    "border-radius: 6px; padding: 8px 16px; font-size: 13px; font-weight: 500;"
    "outline: none; border: 1px solid transparent;"
)

BTN_PRIMARY = f"background-color: {Colors.ACCENT}; color: white; {BTN_BASE}" \
              f"hover:background-color: {Colors.ACCENT_DARK};" \
              f"pressed:background-color: {Colors.PRIMARY};"

BTN_SUCCESS = f"background-color: {Colors.SUCCESS}; color: white; {BTN_BASE}" \
              f"hover:background-color: {Colors.SUCCESS_LIGHT};"

BTN_DANGER = f"background-color: {Colors.DANGER}; color: white; {BTN_BASE}" \
             f"hover:background-color: {Colors.DANGER_LIGHT};"

BTN_WARNING = f"background-color: {Colors.WARNING}; color: white; {BTN_BASE}" \
              f"hover:background-color: {Colors.WARNING_LIGHT};"

BTN_OUTLINE = (f"background-color: transparent; color: {Colors.TEXT_PRIMARY}; {BTN_BASE}"
               f"border-color: {Colors.BORDER_LIGHT};"
               f"hover:background-color: {Colors.BG_INPUT};"
               f"pressed:background-color: {Colors.BORDER};")

BTN_GHOST = f"background-color: transparent; color: {Colors.TEXT_SECONDARY}; {BTN_BASE}" \
            f"hover:background-color: {Colors.BG_INPUT};"

EMERGENCY_STOP_STYLE = (
    f"background-color: {Colors.DANGER}; color: white; font-size: 15px; font-weight: bold;"
    f"padding: 10px 20px; border-radius: 8px;"
    f"border: 2px solid {Colors.DANGER_LIGHT};"
    f"animation: pulse 2s infinite;"
)


# ==================== 全局样式表 ====================

GLOBAL_STYLESHEET = f"""
/* ===== 全局 ===== */
QWidget {{
    background-color: {Colors.BG_PAGE};
    color: {Colors.TEXT_PRIMARY};
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 13px;
}}

/* ===== 窗口/框架 ===== */
QFrame {{
    background-color: {Colors.BG_CARD};
    border-radius: 8px;
    border: 1px solid {Colors.BORDER};
}}

QMainWindow {{
    background-color: {Colors.BG_PAGE};
}}

/* ===== 标签 ===== */
QLabel {{
    color: {Colors.TEXT_PRIMARY};
    background-color: transparent;
}}

QLabel[variant="subtitle"] {{
    color: {Colors.TEXT_SECONDARY};
    font-size: 12px;
}}

QLabel[variant="muted"] {{
    color: {Colors.TEXT_MUTED};
}}

QLabel[variant="accent"] {{
    color: {Colors.ACCENT};
    font-weight: bold;
}}

QLabel[variant="success"] {{
    color: {Colors.SUCCESS};
    font-weight: bold;
}}

QLabel[variant="warning"] {{
    color: {Colors.WARNING};
    font-weight: bold;
}}

QLabel[variant="danger"] {{
    color: {Colors.DANGER};
    font-weight: bold;
}}

/* ===== 按钮 ===== */
QPushButton {{
    {BTN_BASE}
    background-color: {Colors.ACCENT};
    color: white;
    border: none;
}}

QPushButton:hover {{
    background-color: {Colors.ACCENT_DARK};
}}

QPushButton:pressed {{
    background-color: {Colors.PRIMARY};
}}

QPushButton:disabled {{
    background-color: {Colors.BG_HEADER};
    color: {Colors.TEXT_MUTED};
}}

QPushButton[variant="success"] {{
    background-color: {Colors.SUCCESS};
}}
QPushButton[variant="success"]:hover {{
    background-color: {Colors.SUCCESS_LIGHT};
}}

QPushButton[variant="danger"] {{
    background-color: {Colors.DANGER};
}}
QPushButton[variant="danger"]:hover {{
    background-color: {Colors.DANGER_LIGHT};
}}

QPushButton[variant="warning"] {{
    background-color: {Colors.WARNING};
}}
QPushButton[variant="warning"]:hover {{
    background-color: {Colors.WARNING_LIGHT};
}}

QPushButton[variant="outline"] {{
    background-color: transparent;
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
}}
QPushButton[variant="outline"]:hover {{
    background-color: {Colors.BG_HEADER};
}}

QPushButton[variant="ghost"] {{
    background-color: transparent;
    color: {Colors.TEXT_SECONDARY};
}}
QPushButton[variant="ghost"]:hover {{
    background-color: {Colors.BG_HEADER};
}}

/* ===== 输入框 ===== */
QLineEdit {{
    background-color: {Colors.BG_INPUT};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    color: {Colors.TEXT_PRIMARY};
    selection-background-color: {Colors.ACCENT};
}}

QLineEdit:focus {{
    border: 1px solid {Colors.ACCENT};
}}

QLineEdit:disabled {{
    background-color: {Colors.BG_HEADER};
    color: {Colors.TEXT_MUTED};
}}

/* ===== 下拉框 ===== */
QComboBox {{
    background-color: {Colors.BG_INPUT};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    color: {Colors.TEXT_PRIMARY};
    min-width: 100px;
}}

QComboBox:hover {{
    border-color: {Colors.ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {Colors.TEXT_SECONDARY};
    margin-right: 4px;
}}

QComboBox QAbstractItemView {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    selection-background-color: {Colors.ACCENT};
    border-radius: 4px;
    padding: 4px;
}}

/* ===== 分组框 ===== */
QGroupBox {{
    background-color: transparent;
    border: 1px solid {Colors.BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    font-size: 14px;
    color: {Colors.TEXT_PRIMARY};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 0px;
    padding: 0 8px;
    background-color: {Colors.BG_PAGE};
    color: {Colors.ACCENT};
}}

/* ===== 表格 ===== */
QTableWidget {{
    background-color: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER};
    border-radius: 8px;
    gridline-color: {Colors.BORDER};
    selection-background-color: {Colors.ACCENT};
    alternate-background-color: {Colors.BG_PAGE};
}}

QTableWidget::item {{
    padding: 6px 8px;
}}

QTableWidget::item:selected {{
    background-color: {Colors.ACCENT};
    color: white;
}}

QHeaderView::section {{
    background-color: {Colors.BG_HEADER};
    color: {Colors.TEXT_PRIMARY};
    padding: 8px;
    border: none;
    border-bottom: 2px solid {Colors.BORDER};
    font-weight: bold;
    font-size: 12px;
}}

QHeaderView::section:first {{
    border-top-left-radius: 8px;
}}

QHeaderView::section:last {{
    border-top-right-radius: 8px;
}}

QHeaderView::down-arrow {{
    subcontrol-position: right center;
    margin-right: 4px;
}}

QHeaderView::up-arrow {{
    subcontrol-position: right center;
    margin-right: 4px;
}}

/* ===== 滚动条 ===== */
QScrollBar:vertical {{
    background-color: {Colors.BG_PAGE};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {Colors.BORDER};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {Colors.BORDER_FOCUS};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {Colors.BG_PAGE};
    height: 8px;
    border-radius: 4px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {Colors.BORDER};
    border-radius: 4px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {Colors.BORDER_FOCUS};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ===== 标签页 ===== */
QTabWidget::pane {{
    border: 1px solid {Colors.BORDER};
    border-radius: 8px;
    background-color: {Colors.BG_CARD};
}}

QTabBar::tab {{
    background-color: {Colors.BG_HEADER};
    color: {Colors.TEXT_SECONDARY};
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-size: 13px;
}}

QTabBar::tab:selected {{
    background-color: {Colors.BG_CARD};
    color: {Colors.ACCENT};
    font-weight: bold;
}}

QTabBar::tab:hover {{
    background-color: #EEF0F2;
    color: {Colors.TEXT_PRIMARY};
}}

/* ===== 进度条 ===== */
QProgressBar {{
    background-color: {Colors.BG_HEADER};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    text-align: center;
    height: 8px;
}}

QProgressBar::chunk {{
    background-color: {Colors.ACCENT};
    border-radius: 5px;
}}

/* ===== 复选框 ===== */
QCheckBox {{
    color: {Colors.TEXT_PRIMARY};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    background-color: {Colors.BG_INPUT};
}}

QCheckBox::indicator:checked {{
    background-color: {Colors.ACCENT};
    border-color: {Colors.ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {Colors.ACCENT};
}}

/* ===== 对话框 ===== */
QDialog {{
    background-color: {Colors.BG_PAGE};
}}

/* ===== 消息框 ===== */
QMessageBox {{
    background-color: {Colors.BG_PAGE};
}}

QMessageBox QLabel {{
    color: {Colors.TEXT_PRIMARY};
}}

QMessageBox QPushButton {{
    {BTN_BASE}
}}

/* ===== 工具提示 ===== */
QToolTip {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 4px 8px;
}}

/* ===== 分割线 ===== */
QFrame[frameShape="4"] {{ /* HLine */
    background-color: {Colors.BORDER};
    border-radius: 1px;
}}
"""


# ==================== 复合控件 ====================

class BrandBar(QFrame):
    """顶部品牌栏"""

    def __init__(self, title="PCB Defect Detection", subtitle="智能缺陷检测与智能分拣系统",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("brandBar")
        self.setFixedHeight(56)
        self.setStyleSheet(f"""
            #brandBar {{
                background-color: {Colors.PRIMARY};
                border-bottom: 1px solid {Colors.ACCENT};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        # Logo / 图标区域
        logo_label = QLabel("◆")
        logo_label.setStyleSheet(f"color: {Colors.ACCENT_LIGHT}; font-size: 22px; font-weight: bold;")
        logo_label.setFixedWidth(28)
        layout.addWidget(logo_label)

        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: #ffffff; font-size: 16px; font-weight: bold;
            background-color: transparent;
        """)
        layout.addWidget(title_label)

        # 副标题
        sub_label = QLabel(subtitle)
        sub_label.setProperty("variant", "subtitle")
        sub_label.setStyleSheet(f"""
            color: #b0c4de; font-size: 12px;
            background-color: transparent;
        """)
        layout.addWidget(sub_label)

        # 弹性空间
        layout.addStretch()

        # 模式指示
        self.mode_label = QLabel("产线模式")
        self.mode_label.setProperty("variant", "accent")
        self.mode_label.setStyleSheet(f"""
            color: {Colors.ACCENT_LIGHT}; font-size: 12px; font-weight: 500;
            background-color: transparent;
            padding: 4px 12px;
            border: 1px solid {Colors.ACCENT_LIGHT};
            border-radius: 12px;
        """)
        layout.addWidget(self.mode_label)

        # 时间
        self.time_label = QLabel()
        self.time_label.setProperty("variant", "muted")
        self.time_label.setStyleSheet(f"color: #8899aa; font-size: 12px; background-color: transparent;")
        layout.addWidget(self.time_label)

    def set_mode(self, mode: str):
        """设置模式标签"""
        names = {"production": "产线模式", "debug": "调试模式"}
        self.mode_label.setText(names.get(mode, mode))


class StatCard(QFrame):
    """统计卡片 - 底部信息栏使用"""

    def __init__(self, icon: str, label: str, value: str, color: str = Colors.ACCENT,
                 parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                border-left: 4px solid {color};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)

        # 图标
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"color: {color}; font-size: 22px; background-color: transparent;")
        icon_label.setFixedWidth(32)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # 文字
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        label_widget = QLabel(label)
        label_widget.setProperty("variant", "muted")
        label_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px; background-color: transparent;")
        text_layout.addWidget(label_widget)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold; background-color: transparent;")
        text_layout.addWidget(self.value_label)

        layout.addLayout(text_layout)
        layout.addStretch()

    def set_value(self, value: str):
        self.value_label.setText(value)


class StatCardRow(QWidget):
    """统计卡片行 - 一排多个卡片"""

    def __init__(self, cards: list, parent=None):
        """
        cards: list of (icon, label, value, color)
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        for icon, label, value, color in cards:
            card = StatCard(icon, label, value, color)
            layout.addWidget(card, 1)


class PanelSection(QFrame):
    """可折叠面板区域（带标题栏）"""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("panelSection")
        self._collapsed = collapsed

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        header = QFrame()
        header.setObjectName("sectionHeader")
        header.setStyleSheet(f"""
            #sectionHeader {{
                background-color: {Colors.BG_HEADER};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid {Colors.BORDER};
                border-bottom: none;
            }}
        """)
        header.setFixedHeight(36)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 4, 12, 4)

        self.toggle_btn = QLabel(("▼" if not collapsed else "▶") + " " + title)
        self.toggle_btn.setStyleSheet(f"""
            color: {Colors.ACCENT}; font-size: 12px; font-weight: bold;
            background-color: transparent;
        """)
        header_layout.addWidget(self.toggle_btn)
        header_layout.addStretch()

        layout.addWidget(header)

        # 内容区
        self.content_frame = QFrame()
        self.content_frame.setObjectName("sectionContent")
        self.content_frame.setStyleSheet(f"""
            #sectionContent {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-top: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }}
        """)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(8)

        if collapsed:
            self.content_frame.setVisible(False)

        layout.addWidget(self.content_frame)

    def toggle(self):
        self._collapsed = not self._collapsed
        self.content_frame.setVisible(not self._collapsed)
        self.toggle_btn.setText(("▼" if not self._collapsed else "▶") + " " + self.toggle_btn.text()[2:])

    def add_widget(self, widget, stretch=0):
        self.content_layout.addWidget(widget, stretch)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


# ==================== 产线看板 ====================

class ProductionKanban(QMainWindow):
    """产线看板（车间大屏展示）— 全屏 Kiosk 模式"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("产线看板")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        # 全屏
        self.showFullScreen()

        # 数据
        self._total = 0
        self._ok = 0
        self._ng = 0
        self._rate = 0.0
        self._defect_counter = {}
        self._recent_events = deque(maxlen=20)
        self._line_name = "未选择"
        self._shift_info = {}

        # 创建根布局
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # 顶部标题栏
        header = QLabel("PCB 产线看板")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"""
            font-size: 28px; font-weight: bold; color: {Colors.ACCENT};
            background-color: {Colors.PRIMARY};
            padding: 12px;
            border-radius: 8px;
        """)
        main_layout.addWidget(header)

        # 产线信息行
        info_layout = QHBoxLayout()

        self.line_label = QLabel("产线: --")
        self.line_label.setStyleSheet(f"font-size: 16px; color: {Colors.TEXT_PRIMARY}; background-color: {Colors.BG_CARD}; padding: 8px 16px; border-radius: 6px; border: 1px solid {Colors.BORDER};")
        info_layout.addWidget(self.line_label)

        self.shift_label = QLabel("班次: --")
        self.shift_label.setStyleSheet(f"font-size: 16px; color: {Colors.TEXT_PRIMARY}; background-color: {Colors.BG_CARD}; padding: 8px 16px; border-radius: 6px; border: 1px solid {Colors.BORDER};")
        info_layout.addWidget(self.shift_label)

        self.time_label = QLabel()
        self.time_label.setStyleSheet(f"font-size: 16px; color: {Colors.TEXT_PRIMARY}; background-color: {Colors.BG_CARD}; padding: 8px 16px; border-radius: 6px; border: 1px solid {Colors.BORDER};")
        info_layout.addWidget(self.time_label)

        info_layout.addStretch()
        main_layout.addLayout(info_layout)

        # 核心指标大卡片区
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(16)

        # 总产量
        self.total_card = self._make_kpi_card("总产量", "0", Colors.ACCENT)
        kpi_layout.addWidget(self.total_card, 1)

        # 合格数
        self.ok_card = self._make_kpi_card("合格品", "0", Colors.SUCCESS)
        kpi_layout.addWidget(self.ok_card, 1)

        # 不良数
        self.ng_card = self._make_kpi_card("不良品", "0", Colors.DANGER)
        kpi_layout.addWidget(self.ng_card, 1)

        # 合格率
        self.rate_card = self._make_kpi_card("合格率", "0%", Colors.WARNING)
        kpi_layout.addWidget(self.rate_card, 1)

        main_layout.addLayout(kpi_layout)

        # 中部：缺陷分布 + 最近事件
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(16)

        # 缺陷分布表格
        defect_frame = QFrame()
        defect_frame.setStyleSheet(f"background-color: {Colors.BG_CARD}; border-radius: 8px; border: 1px solid {Colors.BORDER};")
        defect_inner = QVBoxLayout(defect_frame)
        defect_inner.setContentsMargins(16, 12, 16, 12)

        defect_title = QLabel("缺陷类别分布")
        defect_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Colors.ACCENT};")
        defect_inner.addWidget(defect_title)

        self.defect_table = QTableWidget()
        self.defect_table.setColumnCount(3)
        self.defect_table.setHorizontalHeaderLabels(["缺陷类型", "数量", "占比"])
        self.defect_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.defect_table.setStyleSheet(f"""
            QTableWidget {{ font-size: 14px; gridline-color: {Colors.BORDER}; }}
            QHeaderView::section {{ background-color: {Colors.BG_HEADER}; color: {Colors.TEXT_PRIMARY}; font-size: 13px; padding: 6px; }}
        """)
        defect_inner.addWidget(self.defect_table)
        mid_layout.addWidget(defect_frame, 2)

        # 最近事件
        event_frame = QFrame()
        event_frame.setStyleSheet(f"background-color: {Colors.BG_CARD}; border-radius: 8px; border: 1px solid {Colors.BORDER};")
        event_inner = QVBoxLayout(event_frame)
        event_inner.setContentsMargins(16, 12, 16, 12)

        event_title = QLabel("最近事件")
        event_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Colors.ACCENT};")
        event_inner.addWidget(event_title)

        self.event_list = QListWidget()
        self.event_list.setStyleSheet(f"""
            QListWidget {{ font-size: 13px; background-color: {Colors.BG_PAGE}; border: none; border-radius: 4px; }}
            QListWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {Colors.BORDER}; }}
            QListWidget::item:selected {{ background-color: {Colors.ACCENT}; }}
        """)
        event_inner.addWidget(self.event_list)
        mid_layout.addWidget(event_frame, 1)

        main_layout.addLayout(mid_layout)
        main_layout.addStretch()

        # 底部时钟
        self.bottom_clock = QLabel()
        self.bottom_clock.setAlignment(Qt.AlignCenter)
        self.bottom_clock.setStyleSheet(f"""
            font-size: 42px; font-weight: bold; color: {Colors.TEXT_PRIMARY};
            background-color: {Colors.BG_HEADER};
            padding: 16px;
            border-radius: 8px;
            border: 1px solid {Colors.BORDER};
        """)
        main_layout.addWidget(self.bottom_clock)

        # 刷新定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_clock)
        self._timer.start(1000)

    def _make_kpi_card(self, title: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_CARD};
                border: 2px solid {color};
                border-radius: 10px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(4)

        label = QLabel(title)
        label.setStyleSheet(f"font-size: 16px; color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(label)

        value_lbl = QLabel(value)
        value_lbl.setObjectName(f"kpiValue_{title}")
        value_lbl.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color};")
        value_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_lbl)

        return card

    def _update_clock(self):
        now = datetime.now()
        self.time_label.setText(f"时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.bottom_clock.setText(now.strftime("%H:%M:%S"))

    def update_stats(self, total: int, ok: int, ng: int, rate: float):
        self._total = total
        self._ok = ok
        self._ng = ng
        self._rate = rate

        # 更新 KPI 值
        for card in self.findChildren(QFrame):
            for lbl in card.findChildren(QLabel):
                text = lbl.text()
                if text == "总产量":
                    lbl_parent = card
                    for child in lbl_parent.layout().itemAt(1).widget().parentWidget().findChildren(QLabel):
                        if child.objectName() == "" and child.styleSheet().startswith("font-size: 36px"):
                            child.setText(str(total))
                    # 更简单的方式：直接按标签内容查找
                    pass

        # 直接设置
        self._set_kpi_value("总产量", str(total))
        self._set_kpi_value("合格品", str(ok))
        self._set_kpi_value("不良品", str(ng))
        self._set_kpi_value("合格率", f"{rate:.1f}%")

    def _set_kpi_value(self, title: str, value: str):
        """通过 objectName 找到对应的值标签"""
        obj_name = f"kpiValue_{title}"
        for child in self.findChildren(QLabel):
            if child.objectName() == obj_name:
                child.setText(value)
                return

    def update_defect_distribution(self, defects: dict):
        self._defect_counter = defects
        total = sum(defects.values()) if defects else 1
        self.defect_table.setRowCount(len(defects))
        for i, (name, count) in enumerate(sorted(defects.items(), key=lambda x: -x[1])):
            self.defect_table.setItem(i, 0, QTableWidgetItem(name))
            self.defect_table.setItem(i, 1, QTableWidgetItem(str(count)))
            ratio = f"{count / total * 100:.1f}%" if total > 0 else "0%"
            self.defect_table.setItem(i, 2, QTableWidgetItem(ratio))

    def add_event(self, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{now}] {message}")
        self.event_list.insertItem(0, item)

    def set_line_name(self, name: str):
        self._line_name = name
        self.line_label.setText(f"产线: {name}")

    def set_shift_info(self, info: dict):
        self._shift_info = info
        shift_name = info.get('shift_name', '--')
        operator = info.get('operator_id', '--')
        self.shift_label.setText(f"班次: {shift_name} | 操作员: {operator}")

