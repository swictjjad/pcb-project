"""
软著源代码提取脚本
将主要源代码文件提取为软著申请所需的格式
"""
import os

PROJECT_ROOT = r"c:\Users\Uaena\Desktop\PCB_Defect_Detection\PCB_缺陷检测系统_项目\PCB_缺陷检测系统\pcb_defect_system"
OUTPUT_DIR = r"c:\Users\Uaena\Desktop\PCB_Defect_Detection\PCB_缺陷检测系统_项目\PCB_缺陷检测系统\pcb_defect_system\results\软著申请材料_20260609\02_源代码"

# 源代码文件列表（按重要性和行数排序）
SOURCE_FILES = [
    "ui/gui.py",           # GUI界面（1689行）
    "inference.py",         # 模型推理（988行）
    "demo_simulate.py",     # 模拟检测器（629行）
    "train.py",             # 模型训练（553行）
    "main.py",              # 主程序入口（523行）
    "modules/dage.py",      # DAGE模块（469行）
    "modules/uncertainty.py", # 不确定性量化（448行）
    "utils/auth.py",         # 权限管理（448行）
    "tests/test_core.py",    # 单元测试（436行）
    "modules/explainability.py", # 可解释性（416行）
    "utils/traceability.py", # 追溯数据库（364行）
    "utils/serial_control.py", # 串口控制（318行）
    "utils/false_alarm_suppressor.py", # 误报屏蔽（317行）
    "utils/production_manager.py", # 产线管理（298行）
    "modules/wiou.py",      # WIoU损失（263行）
    "utils/statistical_testing.py", # 统计测试（258行）
    "utils/production_report.py", # 生产报表（231行）
    "baselines/traditional_cv.py", # 传统CV基线（224行）
    "utils/equipment_monitor.py", # 设备监控（206行）
    "modules/polarization_fusion.py", # 偏振光融合（189行）
    "utils/alarm.py",       # 报警管理（147行）
    "modules/cbam.py",      # CBAM模块（126行）
    "utils/config_loader.py", # 配置加载（109行）
    "utils/logger.py",      # 日志系统（87行）
    "download_model.py",     # 模型下载（71行）
]


def read_file_lines(path):
    """读取文件所有行"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print(f"  读取失败: {e}")
        return []


def extract_source_for_registration():
    """提取软著申请用源代码"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_lines = 0
    file_info = []

    print("=" * 70)
    print("软著源代码提取工具")
    print("=" * 70)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 文件1: 主要模块文件合并（前部）
    print("正在生成 软著源代码_第一部分.py（前3000行核心模块）...")

    part1_lines = []
    part1_files = [
        ("main.py", "主程序入口 — 系统初始化、运行模式管理"),
        ("inference.py", "模型推理 — YOLO11检测器、NMS、CLAHE预处理"),
        ("train.py", "模型训练 — YOLO11训练、DAGE模块注册"),
        ("utils/config_loader.py", "配置加载"),
        ("utils/logger.py", "日志系统"),
        ("utils/auth.py", "权限管理 — 登录认证、角色控制"),
        ("utils/traceability.py", "追溯数据库 — SQLite记录"),
        ("utils/alarm.py", "报警管理 — 声光报警"),
        ("utils/equipment_monitor.py", "设备健康度监控"),
        ("utils/production_manager.py", "产线配置管理"),
        ("utils/false_alarm_suppressor.py", "误报屏蔽"),
        ("utils/serial_control.py", "串口通信控制 — Arduino协议"),
        ("utils/production_report.py", "生产报表生成"),
        ("utils/statistical_testing.py", "统计假设检验"),
    ]

    current_lines = 0
    for rel_path, description in part1_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  跳过（不存在）: {rel_path}")
            continue

        lines = read_file_lines(full_path)
        file_lines = len(lines)
        lines_with_comment = []

        # 添加文件分隔注释
        lines_with_comment.append(f"\n")
        lines_with_comment.append(f"# {'='*60}\n")
        lines_with_comment.append(f"# 文件: {rel_path}\n")
        lines_with_comment.append(f"# 说明: {description}\n")
        lines_with_comment.append(f"# 行数: {file_lines}\n")
        lines_with_comment.append(f"# {'='*60}\n")

        lines_with_comment.extend(lines)

        # 如果超过3000行，截断
        if current_lines + len(lines_with_comment) > 3000:
            remaining = 3000 - current_lines
            if remaining > 20:
                part1_lines.extend(lines_with_comment[:remaining])
            break

        part1_lines.extend(lines_with_comment)
        current_lines += len(lines_with_comment)
        print(f"  + {rel_path}: {file_lines}行 (累计: {current_lines}行)")

    part1_output = os.path.join(OUTPUT_DIR, "软著源代码_第一部分.py")
    with open(part1_output, 'w', encoding='utf-8') as f:
        f.writelines(part1_lines)
    print(f"  保存: {part1_output} ({len(part1_lines)}行)")
    total_lines += len(part1_lines)
    file_info.append(("软著源代码_第一部分.py", len(part1_lines), "前3000行核心模块"))

    print()
    print("正在生成 软著源代码_第二部分.py（GUI界面+高级模块）...")

    # 文件2: GUI界面 + 其他模块
    part2_lines = []
    part2_files = [
        ("ui/gui.py", "PyQt5图形用户界面 — 实时预览、检测结果、专家评分、统计仪表盘"),
        ("demo_simulate.py", "模拟检测器 — 无模型时的模拟推理"),
        ("download_model.py", "模型自动下载"),
    ]

    current_lines = 0
    for rel_path, description in part2_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  跳过（不存在）: {rel_path}")
            continue

        lines = read_file_lines(full_path)
        file_lines = len(lines)
        lines_with_comment = [
            f"\n# {'='*60}\n",
            f"# 文件: {rel_path}\n",
            f"# 说明: {description}\n",
            f"# 行数: {file_lines}\n",
            f"# {'='*60}\n",
        ]
        lines_with_comment.extend(lines)

        part2_lines.extend(lines_with_comment)
        current_lines += len(lines_with_comment)
        print(f"  + {rel_path}: {file_lines}行")

    part2_output = os.path.join(OUTPUT_DIR, "软著源代码_第二部分.py")
    with open(part2_output, 'w', encoding='utf-8') as f:
        f.writelines(part2_lines)
    print(f"  保存: {part2_output} ({len(part2_lines)}行)")
    total_lines += len(part2_lines)
    file_info.append(("软著源代码_第二部分.py", len(part2_lines), "GUI界面+模拟模块"))

    print()
    print("正在生成 软著源代码_第三部分.py（模型模块）...")

    # 文件3: 模型模块
    part3_lines = []
    part3_files = [
        ("modules/dage.py", "DAGE缺陷几何先验特征增强 — Gabor滤波器+圆环检测核+可变形卷积"),
        ("modules/wiou.py", "WIoU v3动态聚焦损失 — 边界框回归损失函数"),
        ("modules/cbam.py", "CBAM注意力机制 — 通道+空间注意力"),
        ("modules/uncertainty.py", "不确定性量化 — MC Dropout置信度波动检测"),
        ("modules/explainability.py", "模型可解释性 — Grad-CAM热力图"),
        ("modules/polarization_fusion.py", "偏振光-RGB多模态融合"),
        ("baselines/traditional_cv.py", "传统CV基线 — 阈值分割+形态学操作"),
        ("tests/test_core.py", "单元测试 — 核心功能测试用例"),
    ]

    current_lines = 0
    for rel_path, description in part3_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  跳过（不存在）: {rel_path}")
            continue

        lines = read_file_lines(full_path)
        file_lines = len(lines)
        lines_with_comment = [
            f"\n# {'='*60}\n",
            f"# 文件: {rel_path}\n",
            f"# 说明: {description}\n",
            f"# 行数: {file_lines}\n",
            f"# {'='*60}\n",
        ]
        lines_with_comment.extend(lines)

        part3_lines.extend(lines_with_comment)
        current_lines += len(lines_with_comment)
        print(f"  + {rel_path}: {file_lines}行")

    part3_output = os.path.join(OUTPUT_DIR, "软著源代码_第三部分.py")
    with open(part3_output, 'w', encoding='utf-8') as f:
        f.writelines(part3_lines)
    print(f"  保存: {part3_output} ({len(part3_lines)}行)")
    total_lines += len(part3_lines)
    file_info.append(("软著源代码_第三部分.py", len(part3_lines), "深度学习模块"))

    # 生成汇总清单
    print()
    print("=" * 70)
    print("源代码提取完成")
    print("=" * 70)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    print(f"{'文件名':<40} {'行数':>8}  说明")
    print("-" * 70)
    for fname, lines, desc in file_info:
        print(f"{fname:<40} {lines:>8}  {desc}")
    print("-" * 70)
    print(f"{'合计':<40} {total_lines:>8}  行")
    print()
    print("提示：软著申请通常要求源代码不少于2000行（功能简单的软件）或")
    print("6000行（功能复杂的软件）。本系统共提取了约 {:.0f} 行源代码，".format(total_lines))
    print("涉及24个Python源文件，涵盖了系统的所有核心功能模块。")
    print()

    # 生成README
    readme_content = f"""# 软著源代码材料说明

## 文件清单

| 文件名 | 行数 | 说明 |
|--------|------|------|
| 软著源代码_第一部分.py | {file_info[0][1]} | 主程序、模型推理、训练脚本、配置、日志、权限、追溯、报警等核心模块 |
| 软著源代码_第二部分.py | {file_info[1][1]} | PyQt5图形用户界面、模拟检测器、模型下载 |
| 软著源代码_第三部分.py | {file_info[2][1]} | DAGE模块、WIoU损失、CBAM、不确定性量化、可解释性、偏振融合、传统CV基线、单元测试 |

## 源代码统计

- **总行数**: {total_lines} 行
- **Python文件数**: {len(SOURCE_FILES)} 个
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
"""
    readme_path = os.path.join(OUTPUT_DIR, "README_源代码说明.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"README已生成: {readme_path}")


if __name__ == "__main__":
    extract_source_for_registration()
