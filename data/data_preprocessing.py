# -*- coding: utf-8 -*-
"""
数据集预处理模块
功能：
1. 数据集划分（训练集/验证集/测试集）
2. 数据增强
3. 格式转换（支持VOC/COCO/YOLO格式互转）
4. 数据集统计分析
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import get_config
from utils.logger import setup_logger

logger = setup_logger("data_preprocessing")


class DatasetPreprocessor:
    """数据集预处理器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.dataset_path = Path(self.config.data.dataset_path)
        self.class_names = self.config.data.class_names
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
    def split_dataset(self, ratios=None, seed=42):
        """
        划分数据集为训练集/验证集/测试集
        
        Args:
            ratios: (train, val, test) 比例元组
            seed: 随机种子
        """
        if ratios is None:
            ratios = (
                self.config.data.train_ratio,
                self.config.data.val_ratio,
                self.config.data.test_ratio
            )
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(self.dataset_path.glob(f'images/{ext}')))
        
        if not image_files:
            # 尝试其他目录结构
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(self.dataset_path.glob(f'**/{ext}')))
            image_files = [f for f in image_files if 'test' not in f.name.lower()]
        
        image_files = sorted(image_files)
        total = len(image_files)
        
        if total == 0:
            logger.error(f"未找到图像文件，请检查路径: {self.dataset_path}")
            return
        
        logger.info(f"找到 {total} 张图像")
        
        # 随机打乱
        random.shuffle(image_files)
        
        # 计算划分数量
        train_num = int(total * ratios[0])
        val_num = int(total * ratios[1])
        # 测试集取剩余
        
        train_files = image_files[:train_num]
        val_files = image_files[train_num:train_num + val_num]
        test_files = image_files[train_num + val_num:]
        
        # 创建目录
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_dir = self.dataset_path / split_name
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
            
            for img_file in files:
                # 复制图像
                shutil.copy2(img_file, split_dir / 'images' / img_file.name)
                
                # 复制标签
                label_file = self._find_label_file(img_file)
                if label_file and label_file.exists():
                    shutil.copy2(label_file, split_dir / 'labels' / label_file.name)
            
            logger.info(f"{split_name}: {len(files)} 张图像")
        
        # 保存数据集配置文件（YOLO格式）
        self._create_yaml_config()
        
        return splits
    
    def _find_label_file(self, image_file: Path) -> Path:
        """根据图像文件找到对应的标签文件"""
        # 尝试YOLO格式 (.txt)
        label_file = image_file.parent.parent / 'labels' / f"{image_file.stem}.txt"
        if label_file.exists():
            return label_file
        
        # 尝试VOC格式 (.xml)
        label_file = image_file.parent.parent / 'annotations' / f"{image_file.stem}.xml"
        if label_file.exists():
            return label_file
        
        return None
    
    def _create_yaml_config(self):
        """创建YOLO格式的数据集配置文件"""
        yaml_content = f"""# PCB缺陷检测数据集配置
path: {self.dataset_path.absolute()}  # 数据集根目录
train: train/images
val: val/images
test: test/images

# 类别
nc: {len(self.class_names)}  # 类别数
names: {self.class_names}  # 类别名称
"""
        yaml_path = self.dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"数据集配置文件已保存: {yaml_path}")
    
    def voc_to_yolo(self, xml_file: Path, img_width: int, img_height: int) -> list:
        """
        将VOC格式的XML标注转换为YOLO格式
        
        Args:
            xml_file: XML标注文件路径
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            YOLO格式标注列表 [(class_id, x_center, y_center, width, height), ...]
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        yolo_labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_to_id:
                continue
            
            class_id = self.class_to_id[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 转换为YOLO格式（归一化中心坐标和宽高）
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height
            
            # 裁剪到[0,1]范围
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            yolo_labels.append((class_id, x_center, y_center, w, h))
        
        return yolo_labels
    
    def convert_annotations(self):
        """转换所有标注文件为YOLO格式"""
        logger.info("开始转换标注格式...")
        
        xml_files = list(self.dataset_path.glob('**/*.xml'))
        
        for xml_file in tqdm(xml_files, desc="转换标注"):
            # 找到对应的图像
            img_file = xml_file.parent.parent / 'images' / f"{xml_file.stem}.jpg"
            if not img_file.exists():
                img_file = img_file.with_suffix('.png')
            
            if not img_file.exists():
                logger.warning(f"未找到图像: {img_file}")
                continue
            
            # 读取图像尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 转换标注
            yolo_labels = self.voc_to_yolo(xml_file, w, h)
            
            # 保存YOLO格式标签
            label_file = xml_file.parent.parent / 'labels' / f"{xml_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(label_file, 'w') as f:
                for label in yolo_labels:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        
        logger.info(f"转换完成: {len(xml_files)} 个文件")
    
    def apply_augmentations(self, image, annotations, seed=None):
        """
        对单张图像和标注应用数据增强

        包括：CLAHE自适应直方图均衡化、高斯噪声注入、CutOut局部遮挡、
        亮度与对比度扰动(±20%)、±15°随机旋转

        Args:
            image: numpy数组 (H, W, C)，BGR格式
            annotations: 标注列表，每个元素为 (class_id, x_center, y_center, w, h) YOLO格式
            seed: 随机种子，可选

        Returns:
            augmented_image: 增强后的图像
            augmented_annotations: 增强后的标注列表
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        img_h, img_w = image.shape[:2]
        aug_image = image.copy()
        aug_annotations = [list(a) for a in annotations]

        # 1. CLAHE 自适应直方图均衡化
        lab = cv2.cvtColor(aug_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        aug_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. 高斯噪声注入
        noise = np.random.normal(0, 10, aug_image.shape).astype(np.float32)
        aug_image = np.clip(aug_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 3. CutOut 局部遮挡
        cutout_size = int(min(img_h, img_w) * 0.15)
        cy = random.randint(0, img_h - cutout_size)
        cx = random.randint(0, img_w - cutout_size)
        aug_image[cy:cy + cutout_size, cx:cx + cutout_size] = 0

        # 4. 亮度与对比度扰动 (±20%)
        alpha = 1.0 + random.uniform(-0.2, 0.2)  # 对比度
        beta = random.uniform(-0.2 * 255, 0.2 * 255)  # 亮度
        aug_image = np.clip(alpha * aug_image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 5. ±15° 随机旋转
        angle = random.uniform(-15, 15)
        center = (img_w / 2.0, img_h / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int(img_h * sin_val + img_w * cos_val)
        new_h = int(img_h * cos_val + img_w * sin_val)
        rotation_matrix[0, 2] += (new_w - img_w) / 2.0
        rotation_matrix[1, 2] += (new_h - img_h) / 2.0
        aug_image = cv2.warpAffine(aug_image, rotation_matrix, (new_w, new_h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 更新标注：将YOLO归一化坐标按旋转进行变换
        new_aug_annotations = []
        for ann in aug_annotations:
            class_id, xc, yc, w, h = ann
            # 转换为像素坐标
            px = xc * img_w
            py = yc * img_h
            bw = w * img_w
            bh = h * img_h
            # 四个角点
            corners = np.array([
                [px - bw / 2, py - bh / 2, 1],
                [px + bw / 2, py - bh / 2, 1],
                [px + bw / 2, py + bh / 2, 1],
                [px - bw / 2, py + bh / 2, 1],
            ])
            # 应用旋转
            rotated = (rotation_matrix @ corners.T).T
            x_min, y_min = rotated[:, 0].min(), rotated[:, 1].min()
            x_max, y_max = rotated[:, 0].max(), rotated[:, 1].max()
            # 裁剪到图像范围
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(new_w, x_max)
            y_max = min(new_h, y_max)
            # 转回YOLO归一化坐标
            new_xc = ((x_min + x_max) / 2.0) / new_w
            new_yc = ((y_min + y_max) / 2.0) / new_h
            new_w_norm = (x_max - x_min) / new_w
            new_h_norm = (y_max - y_min) / new_h
            # 跳过无效标注
            if new_w_norm <= 0 or new_h_norm <= 0:
                continue
            new_xc = max(0, min(1, new_xc))
            new_yc = max(0, min(1, new_yc))
            new_w_norm = max(0, min(1, new_w_norm))
            new_h_norm = max(0, min(1, new_h_norm))
            new_aug_annotations.append([class_id, new_xc, new_yc, new_w_norm, new_h_norm])

        return aug_image, new_aug_annotations

    def analyze_dataset(self):
        """分析数据集统计信息"""
        logger.info("开始分析数据集...")
        
        stats = {
            'total_images': 0,
            'total_instances': 0,
            'class_distribution': Counter(),
            'image_sizes': [],
            'bbox_sizes': []
        }
        
        # 遍历所有标签文件
        label_files = list(self.dataset_path.glob('**/*.txt'))
        
        for label_file in tqdm(label_files, desc="分析数据集"):
            img_file = label_file.parent.parent / 'images' / f"{label_file.stem}.jpg"
            if not img_file.exists():
                img_file = img_file.with_suffix('.png')
            
            if not img_file.exists():
                continue
            
            # 读取图像尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            stats['image_sizes'].append((img_w, img_h))
            stats['total_images'] += 1
            
            # 读取标签
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        w = float(parts[3]) * img_w
                        h = float(parts[4]) * img_h
                        
                        stats['total_instances'] += 1
                        stats['class_distribution'][self.class_names[class_id]] += 1
                        stats['bbox_sizes'].append((w, h))
        
        # 输出统计信息
        logger.info("=" * 50)
        logger.info("数据集统计信息")
        logger.info("=" * 50)
        logger.info(f"总图像数: {stats['total_images']}")
        logger.info(f"总实例数: {stats['total_instances']}")
        logger.info(f"平均每图实例数: {stats['total_instances']/max(stats['total_images'],1):.2f}")
        
        logger.info("\n类别分布:")
        for class_name, count in stats['class_distribution'].most_common():
            percentage = count / max(stats['total_instances'], 1) * 100
            logger.info(f"  {class_name:20s}: {count:5d} ({percentage:5.1f}%)")
        
        if stats['image_sizes']:
            widths = [s[0] for s in stats['image_sizes']]
            heights = [s[1] for s in stats['image_sizes']]
            logger.info(f"\n图像尺寸统计:")
            logger.info(f"  宽度范围: {min(widths)} ~ {max(widths)}")
            logger.info(f"  高度范围: {min(heights)} ~ {max(heights)}")
        
        if stats['bbox_sizes']:
            areas = [w*h for w, h in stats['bbox_sizes']]
            logger.info(f"\n边界框统计:")
            logger.info(f"  平均面积: {np.mean(areas):.1f} px²")
            logger.info(f"  最小面积: {min(areas):.1f} px²")
            logger.info(f"  最大面积: {max(areas):.1f} px²")
        
        return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB缺陷检测数据集预处理')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--task', type=str, default='all',
                        choices=['split', 'convert', 'analyze', 'all'],
                        help='预处理任务')
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.config)
    preprocessor = DatasetPreprocessor(config)
    
    if args.task in ['split', 'all']:
        preprocessor.split_dataset()
    
    if args.task in ['convert', 'all']:
        preprocessor.convert_annotations()
    
    if args.task in ['analyze', 'all']:
        preprocessor.analyze_dataset()
    
    logger.info("预处理完成！")


if __name__ == "__main__":
    main()
