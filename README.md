# PCB缺陷检测与智能分拣系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red.svg)
![YOLO11](https://img.shields.io/badge/YOLO-11-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**基于YOLO11深度学习模型的PCB缺陷自动化检测与智能分拣系统**

</div>

---

## 📋 项目概述

本系统基于 Ultralytics YOLO11 目标检测算法，实现对 PCB（印制电路板）板上 **6类常见缺陷** 的实时自动识别，并驱动 Arduino 控制器执行自动分拣动作。

### 支持检测的缺陷类型

| 类别 | 英文名 | 中文名 |
|------|--------|--------|
| 0 | open | 开路 |
| 1 | short | 短路 |
| 2 | mousebite | 鼠咬 |
| 3 | spur | 毛刺 |
| 4 | copper | 杂铜 |
| 5 | pin-hole | 漏孔 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI 交互层（PyQt5）                        │
│  登录 / 实时预览 / 检测结果 / 专家评分 / 统计仪表盘 / 批量复核    │
├─────────────────────────────────────────────────────────────────┤
│                      业务逻辑层                                  │
│  PCBDefectSystem  主系统类                                       │
│  PCBDefectDetector 检测器类（YOLO11 + ONNX Runtime）             │
│  AlarmManager / ProductionManager / TraceabilityDB              │
├─────────────────────────────────────────────────────────────────┤
│                      硬件控制层（Arduino）                       │
│  传送带电机 / 红外传感器 / 舵机分拣                              │
├─────────────────────────────────────────────────────────────────┤
│                      深度学习层（YOLO11）                        │
│  mAP@0.5: 90.7% | Precision: 95.4% | Recall: 90.6%             │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ 功能特性

- 🎯 **YOLO11 深度学习检测**：基于 Ultralytics YOLO11，高精度 PCB 缺陷识别
- 🔧 **端到端自动化分拣**：图像采集→推理→分拣的闭环流程
- 🖥️ **PyQt5 图形界面**：实时预览、检测结果、统计仪表盘
- 🔔 **多级声光报警**：根据缺陷数量触发不同级别报警
- 📊 **全流程追溯**：SN码/操作员/设备号/检测图片的完整追溯链
- ⚙️ **一键换线**：产线配置热切换，无需重启系统
- 🛡️ **误报屏蔽**：自定义屏蔽区域，过滤固定误报
- 📈 **设备健康度监控**：光源/摄像头/推理性能实时监控
- 👥 **权限管理**：多角色用户认证系统
- 📋 **生产报表**：班次报表、缺陷趋势、操作员绩效

---

## 📁 项目结构

```
pcb_defect_system/
├── main.py                    # 主程序入口
├── inference.py              # 模型推理（YOLO11 + ONNX Runtime）
├── train.py                   # 模型训练脚本
├── demo_simulate.py           # 模拟检测器（无模型时使用）
├── requirements.txt            # Python 依赖
├── configs/
│   ├── config.yaml           # 系统配置
│   └── users.json             # 用户数据
├── ui/
│   └── gui.py                # PyQt5 图形界面
├── modules/
│   ├── dage.py               # DAGE 缺陷几何先验特征增强
│   ├── wiou.py               # WIoU v3 动态聚焦损失
│   ├── cbam.py               # CBAM 注意力机制
│   ├── uncertainty.py         # 不确定性量化
│   └── explainability.py      # 模型可解释性
├── utils/
│   ├── serial_control.py     # Arduino 串口通信
│   ├── traceability.py        # 追溯数据库（SQLite）
│   ├── alarm.py              # 声光报警管理
│   ├── equipment_monitor.py   # 设备健康度监控
│   ├── production_manager.py  # 产线配置管理
│   └── false_alarm_suppressor.py  # 误报屏蔽
├── arduino/
│   └── arduino_servo.ino     # Arduino 控制程序
└── results/
    └── 软著申请材料_20260609/   # 软著申请材料
```

---

## 🚀 快速开始

### 环境要求

- Windows 10/11
- Python 3.10+
- USB 摄像头
- Arduino UNO（可选，用于硬件控制）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行 GUI

```bash
python main.py --mode gui --mock
```

### 运行自动检测

```bash
python main.py --mode auto
```

### 运行演示模式

```bash
python main.py --mode demo
```

---

## 📊 模型训练

### 数据集

使用 [DeepPCB](https://github.com/Charmve/DeepPCB) 数据集，包含 1000 张训练图片和 500 张验证图片。

### 训练模型

```bash
python train.py --config configs/config.yaml --epochs 20
```

### 训练结果

| 指标 | 值 |
|------|------|
| mAP@0.5 | **90.7%** |
| mAP@0.5:0.95 | **61.03%** |
| Precision | **95.4%** |
| Recall | **90.6%** |

---

## 🔌 硬件连接

```
PC (USB) ─────── Arduino UNO (USB Serial)
                    │
                    ├── 电机驱动 ─── 步进电机（传送带）
                    └── 舵机 ─────── SG90 舵机（分拣）
```

### Arduino 引脚配置

| 功能 | 引脚 |
|------|------|
| 电机脉冲 | Pin 5 |
| 电机方向 | Pin 6 |
| 舵机控制 | Pin 9 |
| 红外传感器 | Pin 2 |

---

## 📝 配置说明

系统配置位于 `configs/config.yaml`：

```yaml
# 模型配置
model:
  name: "yolo11n"
  conf_threshold: 0.25
  iou_threshold: 0.45

# 训练配置
train:
  epochs: 20
  batch_size: 16
  device: "cpu"

# 硬件配置
hardware:
  camera:
    index: 0
    width: 1920
    height: 1080
  serial:
    port: "COM3"
    baudrate: 115200
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📌 注意事项

1. **模型文件**：`.pt` 模型文件较大，已从仓库中排除。如需使用，请从 [Ultralytics](https://github.com/ultralytics/ultralytics) 下载 `yolo11n.pt` 到项目根目录。

2. **数据集**：DeepPCB 数据集已从仓库中排除。请从 [GitHub Charmve/DeepPCB](https://github.com/Charmve/DeepPCB) 下载并解压到 `data/` 目录。

3. **硬件**：Arduino 控制程序需要上传到 Arduino UNO 开发板。

---

## 🙏 致谢

- [Ultralytics](https://ultralytics.com) - YOLO 算法框架
- [DeepPCB Dataset](https://github.com/Charmve/DeepPCB) - PCB 缺陷检测数据集
- [睿抗机器人开发者大赛](http://www.rcar.org.cn/) - CAIR 强体赛道

---

<div align="center">

**Made with ❤️ for PCB Defect Detection**

</div>
