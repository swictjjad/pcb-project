import os
import shutil
import sys
from pathlib import Path
from PIL import Image

DEEPPCB_ROOT = r'c:\Users\Uaena\Desktop\PCB_Defect_Detection\PCB_缺陷检测系统_项目\PCB_缺陷检测系统\pcb_defect_system\data\DeepPCB-master\PCBData'
OUTPUT_ROOT = r'c:\Users\Uaena\Desktop\PCB_Defect_Detection\PCB_缺陷检测系统_项目\PCB_缺陷检测系统\pcb_defect_system\data\PCBDefectDataset'

CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
IMG_W, IMG_H = 640, 640


def parse_split_file(split_path):
    """解析 trainval.txt / test.txt，返回 [(img_rel_path, annot_rel_path), ...]"""
    pairs = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def convert_annotation(annot_path, img_w, img_h):
    """将 DeepPCB bbox (x1 y1 x2 y2 class_id) 转换为 YOLO 格式 (class_id cx cy w h)"""
    yolo_lines = []
    if not os.path.exists(annot_path):
        return yolo_lines

    with open(annot_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                x1, y1, x2, y2 = [float(p) for p in parts[:4]]
                class_id = int(parts[4]) - 1  # DeepPCB 类别从 1 开始，YOLO 从 0 开始
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    continue

                # 确保坐标顺序
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)

                # 裁剪到图片范围
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)

                if xmax <= xmin or ymax <= ymin:
                    continue

                # 计算 YOLO 格式
                cx = (xmin + xmax) / 2.0 / img_w
                cy = (ymin + ymax) / 2.0 / img_h
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h

                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            except (ValueError, IndexError):
                continue

    return yolo_lines


def build_dataset(split_name, pairs, output_root):
    """构建 YOLO 数据集（images + labels）"""
    img_out_dir = os.path.join(output_root, split_name, 'images')
    lbl_out_dir = os.path.join(output_root, split_name, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    used = 0
    no_bbox = 0
    missing = 0

    for idx, (img_rel, annot_rel) in enumerate(pairs):
        img_path = os.path.join(DEEPPCB_ROOT, img_rel)
        annot_path = os.path.join(DEEPPCB_ROOT, annot_rel)

        if not os.path.exists(img_path):
            # 尝试 _test.jpg (原始 splits 引用的是 .jpg 但子目录里可能用的是 _test.jpg 或 temp.jpg)
            base = os.path.splitext(img_path)[0]
            for suffix in ['_test.jpg', '_temp.jpg', '.jpg', '.JPG']:
                candidate = base + suffix if '_' in base else base.replace('.jpg', '') + suffix
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            else:
                # 尝试通过目录名匹配
                parent_dir = os.path.dirname(img_path)
                basename = os.path.splitext(os.path.basename(img_path))[0]
                if os.path.exists(parent_dir):
                    matches = [f for f in os.listdir(parent_dir) if f.startswith(basename) and f.endswith('_test.jpg')]
                    if matches:
                        img_path = os.path.join(parent_dir, matches[0])
                    else:
                        missing += 1
                        continue
                else:
                    missing += 1
                    continue

        # 读取图片尺寸（同时验证图片有效性）
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            missing += 1
            continue

        # 转换标注
        yolo_lines = convert_annotation(annot_path, img_w, img_h)

        if not yolo_lines:
            no_bbox += 1

        # 输出文件名
        out_name = f"{idx:05d}"
        out_img = os.path.join(img_out_dir, out_name + '.jpg')
        out_lbl = os.path.join(lbl_out_dir, out_name + '.txt')

        # 复制图片（保持原图格式，提高兼容性）
        shutil.copy2(img_path, out_img)

        # 写入标签
        with open(out_lbl, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        used += 1

    print(f"  {split_name}: 共 {len(pairs)} 对，成功 {used}，缺失 {missing}，无bbox {no_bbox}")
    return used


def main():
    print("=" * 60)
    print("DeepPCB -> YOLO 格式 数据集生成")
    print("=" * 60)

    if not os.path.exists(DEEPPCB_ROOT):
        print(f"错误: DeepPCB 根目录不存在: {DEEPPCB_ROOT}")
        sys.exit(1)

    # 清空输出目录（确保干净）
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 读取 split 文件
    trainval_pairs = parse_split_file(os.path.join(DEEPPCB_ROOT, 'trainval.txt'))
    test_pairs = parse_split_file(os.path.join(DEEPPCB_ROOT, 'test.txt'))

    print(f"trainval: {len(trainval_pairs)} 对")
    print(f"test: {len(test_pairs)} 对")

    # 生成 train (从 trainval) 和 val (从 test)
    print("\n开始生成 train 集...")
    n_train = build_dataset('train', trainval_pairs, OUTPUT_ROOT)

    print("\n开始生成 val 集...")
    n_val = build_dataset('val', test_pairs, OUTPUT_ROOT)

    # 生成 dataset.yaml
    yaml_content = f"""# PCB 缺陷检测数据集（DeepPCB -> YOLO 格式）
# 由 generate_yolo_labels.py 自动生成
path: {OUTPUT_ROOT}
train: train/images
val: val/images

# 类别数量
nc: {len(CLASS_NAMES)}

# 类别名称
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = os.path.join(OUTPUT_ROOT, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\ndataset.yaml 已生成: {yaml_path}")

    print("\n" + "=" * 60)
    print(f"完成！train: {n_train}, val: {n_val}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
