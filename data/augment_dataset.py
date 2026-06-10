
# -*- coding: utf-8 -*-
"""
PCB 数据集扩充脚本
使用数据增强技术将 1500 张扩充到 5000+ 张

增强方法：
1. 旋转 (90°, 180°, 270°)
2. 水平/垂直翻转
3. 亮度/对比度调整
4. 添加噪声
5. 色彩变换
"""

import os
import sys
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("augment")

# 缺陷类别名称
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']


def apply_rotation(image, angle):
    """旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def apply_flip(image, flip_code):
    """翻转图像"""
    return cv2.flip(image, flip_code)


def adjust_brightness(image, factor):
    """调整亮度"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor):
    """调整对比度"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    l_channel = l_channel.astype(np.float32)
    l_channel = ((l_channel - 128) * factor) + 128
    l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def add_noise(image, noise_type='gaussian'):
    """添加噪声"""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 15, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == 'salt':
        prob = 0.01
        output = image.copy()
        random_matrix = np.random.random(image.shape[:2])
        output[random_matrix < prob] = 255
        return output
    return image


def apply_clahe(image):
    """应用 CLAHE 增强对比度"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(lab[:, :, 0])
    lab = cv2.merge([l_channel, lab[:, :, 1], lab[:, :, 2]])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class DatasetAugmenter:
    """数据集增强器"""

    def __init__(self, source_dir, output_dir, target_count=5000):
        """
        Args:
            source_dir: 源数据集目录
            output_dir: 增强后输出目录
            target_count: 目标图片数量
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_count = target_count

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def get_augmentation_chain(self):
        """获取增强链（随机选择）"""
        chains = []

        # 链1: 旋转增强
        for angle in [90, 180, 270]:
            chains.append(('rotate', lambda img, a=angle: apply_rotation(img, a)))

        # 链2: 翻转增强
        for flip in [0, 1, -1]:
            chains.append(('flip', lambda img, f=flip: apply_flip(img, f)))

        # 链3: 亮度调整
        for factor in [0.7, 0.85, 1.15, 1.3]:
            chains.append(('brightness', lambda img, fac=factor: adjust_brightness(img, fac)))

        # 链4: 对比度调整
        for factor in [0.8, 0.9, 1.1, 1.2]:
            chains.append(('contrast', lambda img, fac=factor: adjust_contrast(img, fac)))

        # 链5: 添加噪声
        chains.append(('noise_gaussian', lambda img: add_noise(img, 'gaussian')))
        chains.append(('noise_salt', lambda img: add_noise(img, 'salt')))

        # 链6: CLAHE
        chains.append(('clahe', apply_clahe))

        # 链7: 组合增强
        chains.append(('combo1', lambda img: adjust_brightness(apply_flip(img, 1), 1.1)))
        chains.append(('combo2', lambda img: apply_rotation(adjust_contrast(img, 1.1), 90)))
        chains.append(('combo3', lambda img: adjust_brightness(apply_rotation(img, 180), 0.9)))
        chains.append(('combo4', lambda img: apply_clahe(apply_flip(apply_rotation(img, 90), 1))))

        return chains

    def process(self):
        """执行数据增强"""
        logger.info("=" * 60)
        logger.info("PCB 数据集扩充")
        logger.info("=" * 60)

        # 查找源图片
        source_images = list(self.source_dir.rglob('*_test.jpg'))
        if not source_images:
            source_images = list(self.source_dir.rglob('*.jpg'))
        if not source_images:
            source_images = list(self.source_dir.rglob('*.png'))

        logger.info(f"找到源图片: {len(source_images)} 张")

        # 计算需要的增强倍数
        current_count = len(source_images)
        augmentation_factor = max(1, (self.target_count * 2) // current_count)  # *2 因为 train/val 划分

        logger.info(f"目标数量: {self.target_count} 张")
        logger.info(f"增强倍数: ~{augmentation_factor}x")

        chains = self.get_augmentation_chain()
        random.shuffle(chains)

        # 复制原始图片到训练集
        logger.info("复制原始图片...")
        for i, img_path in enumerate(tqdm(source_images)):
            # 复制到训练集
            dst_name = f"orig_{i:04d}.jpg"
            dst_path = self.output_dir / 'train' / 'images' / dst_name
            shutil.copy(img_path, dst_path)

            # 创建对应的 label 文件（如果存在）
            label_path = img_path.parent.parent / (img_path.stem.replace('_test', '') + '_not') / (img_path.stem.replace('_test', '') + '.txt')
            if label_path.exists():
                dst_label = self.output_dir / 'train' / 'labels' / dst_name.replace('.jpg', '.txt')
                shutil.copy(label_path, dst_label)

        logger.info(f"原始图片已复制: {len(source_images)} 张")

        # 执行增强
        logger.info("执行数据增强...")
        aug_count = 0
        target_train = int(self.target_count * 0.8)  # 80% 训练集

        for i, img_path in enumerate(tqdm(source_images)):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 选择增强链
            for j, (name, func) in enumerate(chains[:augmentation_factor]):
                if aug_count >= target_train:
                    break

                try:
                    aug_img = func(img)
                    aug_name = f"aug_{i:04d}_{name}_{j}.jpg"
                    aug_path = self.output_dir / 'train' / 'images' / aug_name
                    cv2.imwrite(str(aug_path), aug_img)
                    aug_count += 1
                except Exception as e:
                    logger.warning(f"增强失败 {name}: {e}")

            if aug_count >= target_train:
                break

        logger.info(f"增强完成: +{aug_count} 张")
        logger.info(f"训练集总数: {len(source_images) + aug_count} 张")

        # 复制部分到验证集
        val_count = 0
        val_ratio = 0.1
        val_source = random.sample(source_images, int(len(source_images) * val_ratio))

        for img_path in val_source:
            if val_count >= int(self.target_count * val_ratio):
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 随机选择一个增强
            name, func = random.choice(chains)
            try:
                aug_img = func(img)
                aug_name = f"val_{val_count:04d}_{name}.jpg"
                aug_path = self.output_dir / 'val' / 'images' / aug_name
                cv2.imwrite(str(aug_path), aug_img)
                val_count += 1
            except:
                pass

        logger.info(f"验证集总数: ~{val_count} 张")
        logger.info("=" * 60)
        logger.info(f"数据集扩充完成!")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"训练集: {len(source_images) + aug_count} 张")
        logger.info(f"验证集: ~{val_count} 张")
        logger.info("=" * 60)

        # 生成 dataset.yaml
        self.create_dataset_yaml()

    def create_dataset_yaml(self):
        """生成 YOLO dataset.yaml 文件"""
        yaml_content = f"""
# PCB 缺陷检测数据集
# 自动生成
path: {self.output_dir.absolute()}
train: train/images
val: val/images

# 类别数量
nc: 6

# 类别名称
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: copper
  5: pin-hole
"""
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        logger.info(f"数据集配置文件已生成: {yaml_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PCB 数据集扩充')
    parser.add_argument('--source', type=str, default='./data/DeepPCB-master/PCBData',
                       help='源数据集路径')
    parser.add_argument('--output', type=str, default='./data/PCBDefectDataset',
                       help='输出数据集路径')
    parser.add_argument('--target', type=int, default=5000,
                       help='目标图片数量')

    args = parser.parse_args()

    augmenter = DatasetAugmenter(args.source, args.output, args.target)
    augmenter.process()


if __name__ == "__main__":
    main()
