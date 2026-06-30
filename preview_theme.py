# -*- coding: utf-8 -*-
"""企业级风格预览 - 独立演示脚本"""
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTabWidget, QTextEdit,
    QLineEdit, QSplitter, QTableWidgetItem,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ui.theme import (
    GLOBAL_STYLESHEET, Colors, BrandBar, PanelSection,
    StatCard, EMERGENCY_STOP_STYLE,
)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(GLOBAL_STYLESHEET)

    win = QMainWindow()
    win.setWindowTitle('PCB缺陷检测系统 - 企业级风格预览')
    win.resize(1400, 800)

    central = QWidget()
    win.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    # 品牌栏
    brand = BrandBar('PCB Defect Detection', '智能缺陷检测与分拣系统')
    main_layout.addWidget(brand)

    # 主体分割
    splitter = QSplitter(Qt.Horizontal)

    # ===== 左侧 =====
    left = QWidget()
    ll = QVBoxLayout(left)
    ll.setContentsMargins(10, 10, 10, 10)
    ll.setSpacing(10)

    # 工具栏
    toolbar = PanelSection('控制面板')
    tl = QHBoxLayout()
    mb = QPushButton('产线模式')
    mb.setProperty('variant', 'outline')
    mb.setFixedWidth(90)
    lsb = QPushButton('一键换线')
    lsb.setProperty('variant', 'outline')
    ln = QLabel('产线: Line-A')
    ln.setProperty('variant', 'muted')
    sn = QLineEdit()
    sn.setPlaceholderText('扫码/输入SN码')
    sn.setMaximumWidth(160)
    sb = QPushButton('屏蔽区域')
    sb.setProperty('variant', 'outline')
    kb = QPushButton('看板')
    kb.setProperty('variant', 'outline')
    tl.addWidget(mb)
    tl.addWidget(lsb)
    tl.addWidget(ln)
    tl.addWidget(QLabel('SN:'))
    tl.addWidget(sn)
    tl.addWidget(sb)
    tl.addWidget(kb)
    tl.addStretch()
    toolbar.content_layout.addLayout(tl)
    ll.addWidget(toolbar)

    # 视频区域
    video = PanelSection('实时画面')
    vl = QLabel('视频预览 -- 等待启动')
    vl.setAlignment(Qt.AlignCenter)
    vl.setStyleSheet(
        f'background-color: {Colors.BG_HEADER}; color: {Colors.TEXT_MUTED}; '
        f'font-size: 16px; padding: 120px; border-radius: 4px;'
    )
    video.content_layout.addWidget(vl)
    ll.addWidget(video, 1)

    # 控制按钮
    ctrl = PanelSection('操作控制')
    cl1 = QHBoxLayout()
    start = QPushButton('▶ 启动检测')
    start.setProperty('variant', 'success')
    start.setFixedHeight(40)
    capture = QPushButton('\U0001f4f7 截图')
    capture.setProperty('variant', 'outline')
    capture.setFixedHeight(40)
    load = QPushButton('\U0001f4c2 加载图片')
    load.setProperty('variant', 'outline')
    load.setFixedHeight(40)
    cl1.addWidget(start)
    cl1.addWidget(capture)
    cl1.addWidget(load)
    ctrl.content_layout.addLayout(cl1)

    cl2 = QHBoxLayout()
    stop = QPushButton('⛔ 急 停')
    stop.setStyleSheet(EMERGENCY_STOP_STYLE)
    export = QPushButton('\U0001f4ca 导出报表')
    export.setProperty('variant', 'outline')
    review = QPushButton('\U0001f504 批量复核')
    review.setProperty('variant', 'outline')
    report = QPushButton('\U0001f4cb 生产报表')
    report.setProperty('variant', 'outline')
    cl2.addWidget(stop)
    cl2.addWidget(export)
    cl2.addWidget(review)
    cl2.addWidget(report)
    ctrl.content_layout.addLayout(cl2)
    ll.addWidget(ctrl)

    # 状态栏
    status = PanelSection('系统状态', collapsed=True)
    sl = QHBoxLayout()
    st_lbl = QLabel('状态: 就绪')
    st_lbl.setProperty('variant', 'muted')
    cam = QLabel('摄像头: 未连接')
    cam.setProperty('variant', 'muted')
    health = QLabel('设备: 正常')
    health.setProperty('variant', 'success')
    alarm = QLabel('')
    sl.addWidget(st_lbl)
    sl.addWidget(cam)
    sl.addWidget(health)
    sl.addStretch()
    sl.addWidget(alarm)
    status.content_layout.addLayout(sl)
    ll.addWidget(status)

    splitter.addWidget(left)

    # ===== 右侧 =====
    right = QWidget()
    rl = QVBoxLayout(right)
    rl.setContentsMargins(10, 10, 10, 10)
    rl.setSpacing(10)

    tabs = QTabWidget()

    # Tab 1: 检测结果
    rt = QWidget()
    rt_layout = QVBoxLayout(rt)
    rt_layout.setSpacing(8)

    info = PanelSection('检测信息')
    il = QHBoxLayout()
    il.addWidget(QLabel('帧率:'))
    il.addWidget(QLabel('30.2 FPS'))
    il.addWidget(QLabel('推理时间:'))
    il.addWidget(QLabel('45.3 ms'))
    il.addWidget(QLabel('检测数量:'))
    il.addWidget(QLabel('3'))
    il.addWidget(QLabel('平均置信度:'))
    il.addWidget(QLabel('0.851'))
    info.content_layout.addLayout(il)
    rt_layout.addWidget(info)

    rtable = QTableWidget()
    rtable.setColumnCount(4)
    rtable.setHorizontalHeaderLabels(['类别', '置信度', '位置', '操作'])
    rtable.setRowCount(3)
    for i, (name, conf) in enumerate([('划痕', '0.923'), ('虚焊', '0.876'), ('短路', '0.754')]):
        rtable.setItem(i, 0, QTableWidgetItem(name))
        rtable.setItem(i, 1, QTableWidgetItem(conf))
        rtable.setItem(i, 2, QTableWidgetItem(f'({i*100},{i*50},{i*200},{i*100})'))
        rtable.setCellWidget(i, 3, QPushButton('⛔ 屏蔽'))
    rt_layout.addWidget(rtable)
    tabs.addTab(rt, '检测结果')

    # Tab 2: 统计仪表盘
    stats_tab = QWidget()
    stats_layout = QVBoxLayout(stats_tab)
    stats_layout.setSpacing(8)

    top_s = QHBoxLayout()
    pr = PanelSection('合格率')
    prl = QHBoxLayout()
    prw = QLabel('\U0001f50d 87.5%')
    prw.setStyleSheet(
        f'font-size: 28px; font-weight: bold; color: {Colors.WARNING};'
    )
    prl.addWidget(prw)
    prl.addWidget(QLabel('合格: 175 | 缺陷: 25 | 总计: 200'))
    pr.content_layout.addLayout(prl)
    top_s.addWidget(pr, 1)

    db = PanelSection('缺陷类别分布')
    dbl = QLabel('[柱状图区域]')
    dbl.setAlignment(Qt.AlignCenter)
    dbl.setStyleSheet(f'color: {Colors.TEXT_MUTED}; padding: 40px;')
    db.content_layout.addWidget(dbl)
    top_s.addWidget(db, 1)
    stats_layout.addLayout(top_s)

    bot_s = QHBoxLayout()
    pf = PanelSection('性能统计')
    pfl = QHBoxLayout()
    pfl.addWidget(QLabel('平均: 42.3 ms'))
    pfl.addWidget(QLabel('最大: 89.1 ms'))
    pfl.addWidget(QLabel('最小: 18.7 ms'))
    pfl.addWidget(QLabel('累计: 200 帧'))
    pf.content_layout.addLayout(pfl)
    bot_s.addWidget(pf, 1)

    hf = PanelSection('设备健康度')
    hfl = QHBoxLayout()
    hfl.addWidget(QLabel('光源: ✅ 正常'))
    hfl.addWidget(QLabel('摄像头: ✅ 正常'))
    hfl.addWidget(QLabel('推理: ✅ 正常'))
    hfl.addWidget(QLabel('亮度: 120'))
    hf.content_layout.addLayout(hfl)
    bot_s.addWidget(hf, 1)
    stats_layout.addLayout(bot_s)
    stats_layout.addStretch()
    tabs.addTab(stats_tab, '统计仪表盘')

    # Tab 3: 历史记录
    hist_tab = QWidget()
    hist_layout = QVBoxLayout(hist_tab)
    hist_layout.setSpacing(8)

    ql = QHBoxLayout()
    ql.addWidget(QLabel('SN码:'))
    qne = QLineEdit()
    qne.setPlaceholderText('输入SN码查询（留空显示全部）')
    ql.addWidget(qne)
    qb = QPushButton('查询')
    qb.setProperty('variant', 'outline')
    ql.addWidget(qb)
    rb = QPushButton('刷新')
    rb.setProperty('variant', 'outline')
    ql.addWidget(rb)
    mlb = QPushButton('加载更多')
    mlb.setProperty('variant', 'outline')
    ql.addWidget(mlb)
    ql.addStretch()
    hist_layout.addLayout(ql)

    htable = QTableWidget()
    htable.setColumnCount(8)
    htable.setHorizontalHeaderLabels([
        '时间', 'SN码', '结果', '缺陷数', '操作员', '设备', '产线', '推理时间'
    ])
    htable.setRowCount(5)
    data = [
        ('2026-06-17 14:30:01', 'SN001', 'OK', '0', 'OP01', 'EQ01', 'Line-A', '42ms'),
        ('2026-06-17 14:30:05', 'SN002', 'NG', '2', 'OP01', 'EQ01', 'Line-A', '45ms'),
        ('2026-06-17 14:30:09', 'SN003', 'OK', '0', 'OP02', 'EQ01', 'Line-A', '38ms'),
        ('2026-06-17 14:30:13', 'SN004', 'NG', '1', 'OP01', 'EQ01', 'Line-A', '51ms'),
        ('2026-06-17 14:30:17', 'SN005', 'OK', '0', 'OP02', 'EQ01', 'Line-A', '40ms'),
    ]
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            item = QTableWidgetItem(val)
            if val == 'NG':
                item.setForeground(QColor(Colors.DANGER))
            htable.setItem(i, j, item)
    htable.setAlternatingRowColors(True)
    htable.setSelectionBehavior(QTableWidget.SelectRows)
    htable.setEditTriggers(QTableWidget.NoEditTriggers)
    hist_layout.addWidget(htable)

    dt = QTextEdit()
    dt.setReadOnly(True)
    dt.setMaximumHeight(80)
    dt.setStyleSheet(
        f'background-color: {Colors.BG_HEADER}; color: {Colors.TEXT_SECONDARY}; '
        f'font-family: Consolas, monospace; font-size: 11px; '
        f'border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 6px;'
    )
    hist_layout.addWidget(QLabel('详情:'))
    hist_layout.addWidget(dt)
    tabs.addTab(hist_tab, '历史记录')

    rl.addWidget(tabs)

    # 日志
    lf = PanelSection('系统日志', collapsed=True)
    lt = QTextEdit()
    lt.setReadOnly(True)
    lt.setMaximumHeight(100)
    lt.setStyleSheet(
        f'background-color: {Colors.BG_HEADER}; color: {Colors.TEXT_SECONDARY}; '
        f'font-family: Consolas, monospace; font-size: 11px; '
        f'border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 6px;'
    )
    lf.content_layout.addWidget(lt)
    rl.addWidget(lf)

    splitter.addWidget(right)
    splitter.setSizes([800, 500])
    main_layout.addWidget(splitter, 1)

    # 底部统计卡片行
    scr = QWidget()
    scr_l = QHBoxLayout(scr)
    scr_l.setContentsMargins(4, 4, 4, 4)
    scr_l.setSpacing(8)

    cards_data = [
        ('\U0001f4ca', '总产量', '200', Colors.ACCENT),
        ('✅', '合格品', '175', Colors.SUCCESS),
        ('❌', '不良品', '25', Colors.DANGER),
        ('\U0001f4c8', '合格率', '87.5%', Colors.WARNING),
        ('⚡', '推理耗时', '42 ms', Colors.PRIMARY_LIGHT),
    ]

    for icon, label, value, color in cards_data:
        card = StatCard(icon, label, value, color)
        scr_l.addWidget(card, 1)

    scr.setStyleSheet(f'background-color: {Colors.BG_HEADER}; border-radius: 4px; padding: 4px;')
    main_layout.addWidget(scr)

    win.show()
    print('企业级风格预览窗口已打开，请在屏幕上查看效果。关闭窗口即可结束。')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
