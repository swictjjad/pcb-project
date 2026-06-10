# 软著源代码材料说明

## 文件清单

| 文件名 | 行数 | 说明 |
|--------|------|------|
| 软著源代码_第一部分.py | 3000 | 主程序、模型推理、训练脚本、配置、日志、权限、追溯、报警等核心模块 |
| 软著源代码_第二部分.py | 2404 | PyQt5图形用户界面、模拟检测器、模型下载 |
| 软著源代码_第三部分.py | 2611 | DAGE模块、WIoU损失、CBAM、不确定性量化、可解释性、偏振融合、传统CV基线、单元测试 |

## 源代码统计

- **总行数**: 8015 行
- **Python文件数**: 25 个
- **覆盖模块**: 主程序、模型推理、模型训练、GUI界面、硬件控制、数据管理、深度学习模块

## 完整源代码目录结构

所有原始源代码文件位于:
`pcb_defect_system/`

```
pcb_defect_system/
├── main.py                          # 主程序入口
├── inference.py                      # 模型推理
├── train.py                         # 模型训练
├── demo_simulate.py                 # 模拟检测器
├── download_model.py                # 模型下载
├── configs/
│   └── config.yaml                  # 系统配置
├── ui/
│   └── gui.py                       # GUI界面
├── modules/
│   ├── dage.py                      # DAGE模块
│   ├── wiou.py                      # WIoU损失
│   ├── cbam.py                      # CBAM注意力
│   ├── uncertainty.py               # 不确定性量化
│   ├── explainability.py            # 可解释性
│   └── polarization_fusion.py       # 偏振融合
├── utils/
│   ├── config_loader.py             # 配置加载
│   ├── logger.py                    # 日志系统
│   ├── auth.py                      # 权限管理
│   ├── traceability.py              # 追溯数据库
│   ├── alarm.py                     # 报警管理
│   ├── equipment_monitor.py         # 设备监控
│   ├── production_manager.py        # 产线管理
│   ├── false_alarm_suppressor.py    # 误报屏蔽
│   ├── serial_control.py            # 串口控制
│   ├── production_report.py         # 报表生成
│   └── statistical_testing.py       # 统计测试
├── baselines/
│   └── traditional_cv.py             # 传统CV基线
└── tests/
    └── test_core.py                  # 单元测试
```

## 申请注意事项

1. **软件全称**: PCB缺陷检测与智能分拣系统
2. **软件简称**: PCB-YOLO
3. **版本号**: V1.0
4. **编程语言**: Python
5. **主要功能**: 基于YOLO11深度学习模型的PCB缺陷自动检测与智能分拣

生成日期: 2026年6月
