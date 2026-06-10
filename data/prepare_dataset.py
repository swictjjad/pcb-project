# -*- coding: utf-8 -*-
"""
DeepPCB 数据集准备脚本

功能：
1. 自动下载 DeepPCB 数据集（PCB缺陷检测标准基准）
2. 转换为 YOLO 格式标注
3. 划分训练/验证/测试集
4. 生成 dataset.yaml 配置文件

DeepPCB 数据集信息：
- 来源: https://github.com/tangsanli5201/DeepPCB
- 图像数量: 1,500对（缺陷图+模板图）
- 缺陷类别: 6类 (open, short, mousebite, spur, copper, pin-hole)
- 标注格式: 原始为文本格式，需转换为YOLO格式

使用方法：
    python data/prepare_dataset.py --download
    python data/prepare_dataset.py --convert
    python data/prepare_dataset.py --all
"""

import os
import sys
import urllib.request
import zipfile
import shutil
import random
from pathlib import Path

import cv2
import numpy as np

# DeepPCB 数据集下载地址
DEEPPCB_URL = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"

# 备用下载地址（如果GitHub下载失败）
DEEPPCB_MIRROR_URLS = [
    "https://ghproxy.com/https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip",
    "https://mirror.ghproxy.com/https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip",
]

# DeepPCB 缺陷类别映射
# 原始标注中的数字 -> 类别名称
DEEPPCB_CLASSES = {
    1: "open",        # 开路
    2: "short",       # 短路
    3: "mousebite",   # 鼠咬
    4: "spur",        # 毛刺
    5: "copper",      # 杂铜
    6: "pin-hole",    # 漏孔
}

# YOLO格式类别ID（从0开始）
CLASS_TO_YOLO_ID = {name: idx for idx, name in enumerate(DEEPPCB_CLASSES.values())}


def _validate_url(url: str) -> bool:
    """
    验证URL是否安全（仅允许 HTTPS 且来自可信域名）

    Args:
        url: 待验证的URL

    Returns:
        是否安全
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        # 只允许 HTTPS
        if parsed.scheme != 'https':
            return False
        # 只允许来自 github.com 或可信镜像
        allowed_hosts = {'github.com', 'ghproxy.com', 'mirror.ghproxy.com'}
        if parsed.netloc not in allowed_hosts:
            return False
        return True
    except Exception:
        return False


def download_dataset(save_dir="./data"):
    """
    下载 DeepPCB 数据集

    Args:
        save_dir: 保存目录
    """
    import ssl
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    zip_path = save_dir / "DeepPCB.zip"
    extract_dir = save_dir / "DeepPCB_raw"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"[INFO] 数据集已存在: {extract_dir}")
        return str(extract_dir)

    # 尝试下载
    urls = [DEEPPCB_URL] + DEEPPCB_MIRROR_URLS

    for url in urls:
        try:
            # 安全检查
            if not _validate_url(url):
                print(f"\n[WARNING] URL验证失败，跳过: {url}")
                continue

            print(f"[INFO] 正在下载 DeepPCB 数据集...")
            print(f"[INFO] 下载地址: {url}")

            # 创建 SSL 上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED

            # 使用 urlopen 而非 urlretrieve 以支持 SSL
            with urllib.request.urlopen(url, context=ssl_context, timeout=120) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                block_size = 8192

                with open(str(zip_path), 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        if total_size > 0:
                            percent = min(downloaded / total_size * 100, 100)
                            sys.stdout.write(f"\r[下载进度] {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")
                            sys.stdout.flush()
            print("\n[INFO] 下载完成")

            # 解压
            print("[INFO] 正在解压...")
            with zipfile.ZipFile(str(zip_path), 'r') as zf:
                zf.extractall(str(save_dir))

            # 重命名解压目录
            extracted = list(save_dir.glob("DeepPCB-*"))
            if extracted:
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extracted[0].rename(extract_dir)

            # 清理zip文件
            zip_path.unlink(missing_ok=True)

            print(f"[INFO] 数据集已解压到: {extract_dir}")
            return str(extract_dir)

        except Exception as e:
            print(f"\n[WARNING] 下载失败: {e}")
            continue

    print("[ERROR] 所有下载地址均失败，请手动下载数据集")
    print(f"[INFO] 下载地址: {DEEPPCB_URL}")
    print(f"[INFO] 解压到: {save_dir / 'DeepPCB_raw'}")
    return None


def convert_deeppcb_to_yolo(raw_dir, output_dir="./data/DeepPCB"):
    """
    将 DeepPCB 数据集转换为 YOLO 格式

    DeepPCB 原始标注格式:
        每行: class_id x1 y1 x2 y2
        坐标为绝对像素坐标

    YOLO 标注格式:
        每行: class_id cx cy w h
        坐标为归一化坐标 (0-1)

    Args:
        raw_dir: 原始数据集目录
        output_dir: 输出目录
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # 查找图像和标注目录
    # DeepPCB目录结构: DeepPCB_raw/PCBData/
    pcb_data = raw_dir / "PCBData"
    if not pcb_data.exists():
        # 尝试其他可能的路径
        pcb_data = raw_dir
        for d in raw_dir.rglob("PCBData"):
            pcb_data = d
            break

    if not pcb_data.exists():
        print(f"[ERROR] 未找到PCBData目录: {pcb_data}")
        print("[INFO] 请确认数据集解压正确")
        return False

    print(f"[INFO] 找到PCBData目录: {pcb_data}")

    # 创建输出目录结构
    for split in ['images/train', 'images/val', 'images/test',
                   'labels/train', 'labels/val', 'labels/test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # 收集所有图像和标注对
    image_label_pairs = []

    # DeepPCB的图像和标注文件
    # 图像在 group*/ 文件夹下，格式为 */test/ 或 */train/
    # 标注在对应的 */label/ 文件夹下

    # 查找所有子目录
    subdirs = sorted(pcb_data.iterdir()) if pcb_data.is_dir() else []

    for subdir in subdirs:
        if not subdir.is_dir():
            continue

        # 查找图像文件
        for img_file in sorted(subdir.rglob("*.jpg")):
            # 对应的标注文件
            # DeepPCB标注格式: 同目录下同名.txt文件，或在label子目录
            label_file = img_file.with_suffix('.txt')

            # 尝试在label目录查找
            if not label_file.exists():
                parent = img_file.parent
                label_dir = parent / "label"
                if label_dir.exists():
                    label_file = label_dir / (img_file.stem + ".txt")

            if label_file.exists():
                image_label_pairs.append((img_file, label_file))

    if not image_label_pairs:
        # 尝试另一种目录结构
        print("[INFO] 尝试备用目录结构...")
        for img_file in sorted(pcb_data.rglob("*.jpg")):
            label_file = img_file.with_suffix('.txt')
            if not label_file.exists():
                # 在同级的label目录查找
                rel_path = img_file.relative_to(pcb_data)
                label_path = Path(str(rel_path).replace("/images/", "/labels/"))
                label_file = pcb_data / label_path.with_suffix('.txt')

            if label_file.exists():
                image_label_pairs.append((img_file, label_file))

    if not image_label_pairs:
        print("[ERROR] 未找到任何图像-标注对")
        print("[INFO] 请检查数据集目录结构")
        return False

    print(f"[INFO] 找到 {len(image_label_pairs)} 对图像-标注")

    # 随机划分数据集
    random.seed(42)
    random.shuffle(image_label_pairs)

    n_total = len(image_label_pairs)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_pairs = image_label_pairs[:n_train]
    val_pairs = image_label_pairs[n_train:n_train + n_val]
    test_pairs = image_label_pairs[n_train + n_val:]

    print(f"[INFO] 数据集划分: 训练={len(train_pairs)}, 验证={len(val_pairs)}, 测试={len(test_pairs)}")

    # 转换并保存
    stats = {'train': 0, 'val': 0, 'test': 0, 'total_labels': 0}

    for split_name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        for img_file, label_file in pairs:
            # 读取图像获取尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]

            # 生成唯一文件名（避免重名）
            unique_name = f"{img_file.parent.stem}_{img_file.stem}"

            # 复制图像
            dst_img = output_dir / 'images' / split_name / f"{unique_name}.jpg"
            shutil.copy2(str(img_file), str(dst_img))

            # 转换标注
            dst_label = output_dir / 'labels' / split_name / f"{unique_name}.txt"
            yolo_lines = convert_label_file(str(label_file), w, h)

            with open(str(dst_label), 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))

            stats[split_name] += 1
            stats['total_labels'] += len(yolo_lines)

    print(f"[INFO] 转换完成:")
    print(f"  训练集: {stats['train']} 张图像")
    print(f"  验证集: {stats['val']} 张图像")
    print(f"  测试集: {stats['test']} 张图像")
    print(f"  总标注数: {stats['total_labels']}")

    # 生成 dataset.yaml
    generate_dataset_yaml(output_dir)

    return True


def convert_label_file(label_path, img_w, img_h):
    """
    将 DeepPCB 标注转换为 YOLO 格式

    DeepPCB 格式: class_id x1 y1 x2 y2 (绝对坐标)
    YOLO 格式: class_id cx cy w h (归一化坐标)

    Args:
        label_path: 标注文件路径
        img_w: 图像宽度
        img_h: 图像高度

    Returns:
        list: YOLO格式的标注行
    """
    yolo_lines = []

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(parts[0])
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue

            # 映射类别ID (DeepPCB从1开始，YOLO从0开始)
            if class_id not in DEEPPCB_CLASSES:
                continue

            yolo_class_id = class_id - 1  # 转为0-based

            # 转换为YOLO格式 (归一化的中心坐标和宽高)
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            # 确保坐标在有效范围内
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))

            if bw > 0 and bh > 0:
                yolo_lines.append(f"{yolo_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return yolo_lines


def generate_dataset_yaml(dataset_dir):
    """
    生成 YOLO 格式的 dataset.yaml 配置文件

    Args:
        dataset_dir: 数据集目录
    """
    dataset_dir = Path(dataset_dir)

    yaml_content = f"""# DeepPCB 数据集配置 (YOLO格式)
# PCB-YOLO: 基于YOLO11的PCB缺陷检测

path: {dataset_dir.resolve()}  # 数据集根目录
train: images/train  # 训练图像目录（相对于path）
val: images/val      # 验证图像目录
test: images/test    # 测试图像目录

# 缺陷类别
nc: 6
names:
  0: open          # 开路
  1: short         # 短路
  2: mousebite     # 鼠咬
  3: spur          # 毛刺
  4: copper        # 杂铜
  5: pin-hole      # 漏孔
"""

    yaml_path = dataset_dir / "dataset.yaml"
    with open(str(yaml_path), 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"[INFO] dataset.yaml 已生成: {yaml_path}")


def verify_dataset(dataset_dir="./data/DeepPCB"):
    """
    验证数据集是否正确准备

    Args:
        dataset_dir: 数据集目录
    """
    dataset_dir = Path(dataset_dir)

    print("=" * 50)
    print("数据集验证")
    print("=" * 50)

    # 检查目录结构
    for split in ['train', 'val', 'test']:
        img_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split

        img_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

        status = "OK" if img_count > 0 and img_count == label_count else "ERROR"
        print(f"  {split}: {img_count} 图像, {label_count} 标注 [{status}]")

    # 检查dataset.yaml
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists():
        print(f"  dataset.yaml: 存在 [{yaml_path}]")
    else:
        print(f"  dataset.yaml: 不存在 [ERROR]")

    # 抽样检查标注格式
    train_labels = list((dataset_dir / 'labels' / 'train').glob("*.txt")) if (dataset_dir / 'labels' / 'train').exists() else []
    if train_labels:
        sample = train_labels[0]
        with open(str(sample), 'r') as f:
            lines = f.readlines()
        print(f"\n  标注样本 ({sample.name}):")
        for line in lines[:3]:
            print(f"    {line.strip()}")

    print("=" * 50)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='DeepPCB数据集准备脚本')
    parser.add_argument('--download', action='store_true',
                        help='下载DeepPCB数据集')
    parser.add_argument('--convert', action='store_true',
                        help='转换为YOLO格式')
    parser.add_argument('--verify', action='store_true',
                        help='验证数据集')
    parser.add_argument('--all', action='store_true',
                        help='执行全部步骤（下载+转换+验证）')
    parser.add_argument('--raw-dir', type=str, default='./data/DeepPCB_raw',
                        help='原始数据集目录')
    parser.add_argument('--output-dir', type=str, default='./data/DeepPCB',
                        help='输出目录')

    args = parser.parse_args()

    if not any([args.download, args.convert, args.verify, args.all]):
        parser.print_help()
        return

    if args.all or args.download:
        raw_dir = download_dataset("./data")
        if raw_dir is None:
            print("[ERROR] 数据集下载失败，请手动下载")
            return

    if args.all or args.convert:
        raw_dir = args.raw_dir
        if not Path(raw_dir).exists():
            # 尝试查找
            possible = list(Path("./data").glob("DeepPCB*"))
            if possible:
                raw_dir = str(possible[0])
                print(f"[INFO] 使用找到的数据集目录: {raw_dir}")
            else:
                print("[ERROR] 未找到原始数据集目录，请先下载")
                return

        success = convert_deeppcb_to_yolo(raw_dir, args.output_dir)
        if not success:
            print("[ERROR] 数据集转换失败")
            return

    if args.all or args.verify:
        verify_dataset(args.output_dir)

    print("\n[DONE] 数据集准备完成！")
    print(f"下一步: python train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
