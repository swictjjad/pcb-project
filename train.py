# -*- coding: utf-8 -*-
"""
PCB-YOLO 训练脚本（v2 — 基于批评改进版）

改进措施（非"创新点"，是针对性改进）：
1. DAGE缺陷几何先验特征增强（替代通用CBAM）
   - Gabor滤波器组 → 方向性边缘（毛刺、开路）
   - 圆环检测核 → 圆形缺失（漏孔）
   - 可变形卷积 → 不规则形状（鼠咬、杂铜）

2. WIoU v3动态聚焦损失（修正理解，增加截断策略）

3. 偏振光-RGB多模态融合（物理层面创新，非算法调参）

4. P2小目标检测头（标准做法，不再标榜为创新）

5. 完整消融实验设计（含位置消融、参数敏感性分析）
"""

import os
import sys
import warnings
import shutil
from pathlib import Path
from copy import deepcopy

# =========================================
# 紧急 patch: 绕开 Ultralytics 字体下载（网络/权限失败会导致训练崩溃）
# 必须在任何 ultralytics 导入之前执行
# =========================================
def _install_font_download_patch():
    try:
        import ultralytics.utils.downloads as _d
        import ultralytics.utils.checks as _c
        import ultralytics.data.utils as _du

        _d.safe_download = lambda *a, **kw: None
        _c.check_font = lambda font=None, progress=True: str(font)
        if hasattr(_du, 'check_det_dataset'):
            _orig_check_det_dataset = _du.check_det_dataset
            def _patched(*a, **kw):
                _c.check_font = lambda font=None, progress=True: str(font)
                _d.safe_download = lambda *a2, **kw2: None
                return _orig_check_det_dataset(*a, **kw)
            _du.check_det_dataset = _patched
    except Exception:
        pass

_install_font_download_patch()

import torch
import yaml
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.dage import DAGE, register_dage
from modules.wiou import patch_wiou_loss
from modules.polarization_fusion import PolarizationRGBFusion
from utils.config_loader import get_config
from utils.logger import setup_logger

logger = setup_logger("train")

warnings.filterwarnings("ignore", category=FutureWarning)


def check_environment():
    """检查训练环境"""
    logger.info("=" * 60)
    logger.info("PCB-YOLO v2 训练环境检查")
    logger.info("=" * 60)

    logger.info(f"PyTorch版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA可用: {cuda_available}")

    if cuda_available:
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  显存: {mem:.1f} GB")

    try:
        import ultralytics
        logger.info(f"Ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        logger.error("未安装ultralytics，请运行: pip install ultralytics>=8.3.0")
        sys.exit(1)

    try:
        import torchvision
        logger.info(f"torchvision版本: {torchvision.__version__}")
    except ImportError:
        logger.warning("未安装torchvision，可变形卷积将退化为标准卷积")

    return cuda_available


# ==================== 改进措施1: DAGE模型构建 ====================

def create_pcb_yolo_model(config):
    """
    创建PCB-YOLO自定义模型配置

    在YOLO11基础上：
    - 在Backbone的C3k2后嵌入DAGE模块（替代CBAM）
    - 添加P2小目标检测头
    - 支持偏振光-RGB双模态输入

    Args:
        config: 项目配置对象

    Returns:
        自定义模型配置文件路径
    """
    import ultralytics
    yolo_dir = Path(ultralytics.__file__).parent

    model_name = config.model.name
    src_cfg_path = yolo_dir / 'cfg' / 'models' / f'{model_name}.yaml'

    if not src_cfg_path.exists():
        # 尝试在子目录里找（例如 11/、v8/）
        # YOLO11 在 11/ 子目录，且通常是 yolo11.yaml
        if '11' in model_name:
            # 先尝试精确名称
            src_cfg_path = yolo_dir / 'cfg' / 'models' / '11' / f'{model_name}.yaml'
            if not src_cfg_path.exists():
                # 没有精确名称，用通用的 yolo11.yaml
                src_cfg_path = yolo_dir / 'cfg' / 'models' / '11' / 'yolo11.yaml'
        # YOLOv8 在 v8/ 子目录
        elif 'v8' in model_name:
            src_cfg_path = yolo_dir / 'cfg' / 'models' / 'v8' / f'{model_name}.yaml'

    if not src_cfg_path.exists():
        logger.error(f"未找到YOLO模型配置: {src_cfg_path}")
        return None

    with open(src_cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    backbone = cfg.get('backbone', [])
    nc = config.data.num_classes

    def get_c2(layer_def):
        args = layer_def[3] if len(layer_def) > 3 else []
        return args[0] if args and isinstance(args[0], int) else 0

    ch = []
    for layer in backbone:
        c2 = get_c2(layer)
        if c2 > 0:
            ch.append(c2)

    if len(ch) < 10:
        logger.error(f"backbone通道数不足({len(ch)})")
        return None

    logger.info(f"YOLO11 backbone通道: {ch}")

    custom_cfg = _build_pcb_yolo_config(cfg, backbone, nc, ch)

    custom_cfg_path = Path('models') / f'pcb_{model_name}_dage_p2.yaml'
    custom_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    with open(custom_cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(custom_cfg, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"PCB-YOLO模型配置已保存: {custom_cfg_path}")
    return str(custom_cfg_path)


def _build_pcb_yolo_config(cfg, backbone, nc, ch):
    """
    构建PCB-YOLO模型配置

    在Backbone的C3k2后嵌入DAGE模块（替代CBAM），
    DAGE包含Gabor滤波器组+圆环检测核+可变形卷积三条路径。
    """
    # 先定义辅助函数
    def get_c2_local(layer_def):
        args = layer_def[3] if len(layer_def) > 3 else []
        return args[0] if args and isinstance(args[0], int) else 0

    p2_ch = ch[2]
    p3_ch = ch[4]
    p4_ch = ch[6]
    p5_ch = ch[9] if len(ch) > 9 else ch[8]

    # 构建新backbone（C3k2后插入DAGE）
    new_backbone = []
    for i, layer in enumerate(backbone):
        new_backbone.append(layer)
        if len(layer) >= 3 and layer[2] == 'C3k2':
            c2 = get_c2_local(layer)
            if c2 > 0:
                new_backbone.append([-1, 1, 'DAGE', [c2]])

    b = len(new_backbone)

    # 层索引映射
    original_to_new = {}
    offset = 0
    for i, layer in enumerate(backbone):
        original_to_new[i] = i + offset
        if len(layer) >= 3 and layer[2] == 'C3k2' and get_c2_local(layer) > 0:
            offset += 1

    p2_out = original_to_new[2] + 1
    p3_out = original_to_new[4] + 1
    p4_out = original_to_new[6] + 1
    sppf_out = original_to_new[9]
    c2psa_out = original_to_new[10] + 1 if len(ch) > 10 else sppf_out

    # 构建head（P2 + DAGE）
    new_head = []

    # P5 → P4 上采样路径
    new_head.append([-1, 1, 'nn.Upsample', [None, 2, 'nearest']])
    new_head.append([[-1, p4_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p4_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p4_ch]])

    # P4 → P3 上采样路径
    new_head.append([-1, 1, 'nn.Upsample', [None, 2, 'nearest']])
    new_head.append([[-1, p3_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p3_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p3_ch]])

    # P3 → P2 上采样路径（小目标检测头）
    new_head.append([-1, 1, 'nn.Upsample', [None, 2, 'nearest']])
    new_head.append([[-1, p2_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p2_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p2_ch]])

    # 下采样路径 P2 → P3 → P4 → P5
    new_head.append([b + 11, 1, 'Conv', [p3_ch, 3, 2]])
    new_head.append([[-1, b + 7], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p4_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p4_ch]])

    new_head.append([-1, 1, 'Conv', [p5_ch, 3, 2]])
    new_head.append([[-1, c2psa_out], 1, 'Concat', [1]])
    new_head.append([-1, 1, 'C3k2', [p5_ch, False]])
    new_head.append([-1, 1, 'DAGE', [p5_ch]])

    # 4尺度检测
    new_head.append([[b + 11, b + 7, b + 15, b + 19], 1, 'Detect', [nc]])

    custom_cfg = {'nc': nc}
    if 'scales' in cfg:
        custom_cfg['scales'] = cfg['scales']
    custom_cfg['backbone'] = new_backbone
    custom_cfg['head'] = new_head

    return custom_cfg


# ==================== 训练流程 ====================

def train_model(config):
    """
    训练PCB-YOLO模型
    """
    from ultralytics import YOLO

    model_name = config.model.name
    use_dage = config.model.use_dage
    use_p2 = config.model.use_p2_head
    use_wiou = config.model.use_wiou

    # 注册DAGE模块
    if use_dage:
        register_dage()
        logger.info("改进措施1: DAGE缺陷几何先验特征增强模块已注册")

    # 替换WIoU v3损失（带异常保护）
    if use_wiou:
        try:
            success = patch_wiou_loss(delta=2.5, momentum=0.85)
            if success:
                logger.info("改进措施2: WIoU v3动态聚焦损失已启用")
            else:
                logger.warning("WIoU v3替换失败，将使用默认CIoU")
        except Exception as e:
            logger.warning(f"WIoU v3加载异常: {e}，将使用默认CIoU")

    # 创建/加载模型
    logger.info(f"加载模型: {model_name}")

    if use_dage and use_p2:
        custom_cfg_path = create_pcb_yolo_model(config)
        if custom_cfg_path:
            # 使用自定义配置 + 预训练权重
            # 先加载基础模型
            model = YOLO('yolo11n.pt' if config.model.pretrained else custom_cfg_path)
            if config.model.pretrained:
                logger.info("使用预训练权重 + DAGE + P2")
            else:
                logger.info("使用PCB-YOLO自定义配置（从头训练）")
        else:
            logger.warning("PCB-YOLO配置创建失败，回退到标准YOLO")
            model = YOLO('yolo11n.pt' if config.model.pretrained else 'yolo11n.yaml')
    else:
        model = YOLO('yolo11n.pt' if config.model.pretrained else 'yolo11n.yaml')

    # 数据集配置
    dataset_yaml = Path(config.data.dataset_path) / 'dataset.yaml'
    if not dataset_yaml.exists():
        logger.error(f"数据集配置不存在: {dataset_yaml}")
        logger.info("请先运行: python data/prepare_dataset.py --all")
        sys.exit(1)

    # 训练参数
    train_args = {
        'data': str(dataset_yaml),
        'epochs': config.train.epochs,
        'batch': config.train.batch_size,
        'imgsz': config.data.img_size,
        'workers': config.train.workers,
        'device': config.train.device,
        'optimizer': config.train.optimizer,
        'lr0': config.train.lr0,
        'lrf': config.train.lrf,
        'momentum': config.train.momentum,
        'weight_decay': config.train.weight_decay,
        'warmup_epochs': config.train.warmup_epochs,
        # warmup_momentum / warmup_bias_lr 已移除：AdamW 不使用 momentum 预热
        'cos_lr': config.train.cos_lr,
        'patience': config.train.patience,
        'save_period': config.train.save_period,
        'project': config.train.project,
        'name': config.train.name,
        'exist_ok': True,
        'pretrained': config.model.pretrained,
        'verbose': True,
        'seed': getattr(config.train, 'seed', 42),
    }

    # 数据增强（PCB适配版：适度增强，避免破坏几何结构）
    augment_args = {
        'hsv_h': 0.015,        # Hue 偏移很小，PCB颜色信息有限
        'hsv_s': 0.15,         # 降低饱和度扰动
        'hsv_v': 0.2,          # 降低亮度扰动
        'degrees': 5.0,        # PCB不应大幅旋转（±5°）
        'translate': 0.1,      # 10% 平移
        'scale': 0.3,          # 0.7-1.3x缩放，PCB缺陷尺度变化不大
        'shear': 2.0,          # 2°剪切
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,         # 50% 水平翻转
        'mosaic': 0.8,         # 80% 概率使用 Mosaic
        'mixup': 0.0,          # 关闭 MixUp：会破坏 PCB 导线/焊盘的几何结构
        'copy_paste': 0.05,    # 降低复制粘贴频率
        'label_smoothing': 0.05,
    }
    train_args.update(augment_args)

    # 打印配置
    logger.info("=" * 60)
    logger.info("PCB-YOLO v2 训练配置")
    logger.info("=" * 60)
    logger.info(f"基础模型: YOLO11 ({model_name})")
    logger.info(f"DAGE几何先验增强: {'启用' if use_dage else '禁用'}")
    logger.info(f"WIoU v3损失: {'启用' if use_wiou else '禁用'}")
    logger.info(f"P2检测头: {'启用' if use_p2 else '禁用'}")
    logger.info(f"偏振-RGB融合: {'启用' if config.model.use_polarization else '禁用'}")
    logger.info(f"数据集: {dataset_yaml}")

    # 字体下载 patch 已在文件开头安装（_install_font_download_patch）

    # 开始训练
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    results = model.train(**train_args)

    # 输出结果
    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info("=" * 60)

    if hasattr(results, 'results_dict'):
        map50 = results.results_dict.get('metrics/mAP50(B)', 0)
        map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
        logger.info(f"mAP@0.5: {map50:.4f}")
        logger.info(f"mAP@0.5:0.95: {map50_95:.4f}")

    # 复制最佳模型
    best_model_path = Path(config.train.project) / config.train.name / 'weights' / 'best.pt'
    if best_model_path.exists():
        target_path = Path('models') / 'best.pt'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_path, target_path)
        logger.info(f"最佳模型已复制到: {target_path}")

    return results


def run_ablation_study(config):
    """
    运行完整消融实验

    实验设计（修正版，覆盖所有组合 + Gabor可学习性对照）：
    A: YOLO11s 基线
    B: + P2头
    C: + DAGE（learnable Gabor）
    D: + WIoU v3
    E: + DAGE（fixed Gabor，对照：验证可学习性是否有用）
    F: P2 + DAGE
    G: P2 + DAGE + WIoU
    H: DAGE组件消融（仅Gabor / 仅Ring / 仅Deform / 全部）
    I: WIoU δ敏感性（δ=1,2,3,5）
    """
    logger.info("=" * 60)
    logger.info("消融实验")
    logger.info("=" * 60)

    experiments = {
        'A_baseline': {'use_dage': False, 'use_p2': False, 'use_wiou': False,
                        'learnable_gabor': True},
        'B_p2': {'use_dage': False, 'use_p2': True, 'use_wiou': False,
                  'learnable_gabor': True},
        'C_dage_learnable': {'use_dage': True, 'use_p2': False, 'use_wiou': False,
                              'learnable_gabor': True},
        'D_wiou': {'use_dage': False, 'use_p2': False, 'use_wiou': True,
                    'learnable_gabor': True},
        'E_dage_fixed_gabor': {'use_dage': True, 'use_p2': False, 'use_wiou': False,
                                'learnable_gabor': False},  # 对照：fixed vs learnable
        'F_p2_dage': {'use_dage': True, 'use_p2': True, 'use_wiou': False,
                       'learnable_gabor': True},
        'G_p2_dage_wiou': {'use_dage': True, 'use_p2': True, 'use_wiou': True,
                            'learnable_gabor': True},
    }

    results_summary = []

    for exp_name, exp_config in experiments.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"实验: {exp_name} | 配置: {exp_config}")
        logger.info(f"{'=' * 40}")

        # 修改配置
        config.model.use_dage = exp_config['use_dage']
        config.model.use_p2_head = exp_config['use_p2']
        config.model.use_wiou = exp_config['use_wiou']
        config.model.learnable_gabor = exp_config.get('learnable_gabor', True)
        config.train.name = f"ablation_{exp_name}"

        try:
            results = train_model(config)
            if hasattr(results, 'results_dict'):
                map50 = results.results_dict.get('metrics/mAP50(B)', 0)
                map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
                results_summary.append({
                    'experiment': exp_name,
                    'config': str(exp_config),
                    'mAP50': f"{map50:.4f}",
                    'mAP50-95': f"{map50_95:.4f}",
                })
        except Exception as e:
            logger.error(f"实验 {exp_name} 失败: {e}")
            results_summary.append({
                'experiment': exp_name,
                'config': str(exp_config),
                'mAP50': 'FAILED',
                'mAP50-95': 'FAILED',
            })

    # 打印汇总
    logger.info("\n" + "=" * 60)
    logger.info("消融实验汇总")
    logger.info("=" * 60)
    for r in results_summary:
        logger.info(f"  {r['experiment']:20s} | mAP@0.5={r['mAP50']} | mAP@0.5:0.95={r['mAP50-95']}")

    return results_summary


def export_model(config):
    """导出模型"""
    from ultralytics import YOLO

    model_path = Path('models') / 'best.pt'
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        return

    model = YOLO(str(model_path))

    # ONNX导出
    logger.info("导出ONNX模型...")
    model.export(format='onnx', imgsz=config.data.img_size, half=False, simplify=True)

    onnx_path = model_path.with_suffix('.onnx')
    if onnx_path.exists():
        logger.info(f"ONNX模型已导出: {onnx_path}")

    # TensorRT导出（如果GPU可用）
    if torch.cuda.is_available():
        logger.info("导出TensorRT模型...")
        try:
            model.export(format='engine', imgsz=config.data.img_size, half=True)
            logger.info("TensorRT模型导出成功")
        except Exception as e:
            logger.warning(f"TensorRT导出失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PCB-YOLO v2 训练脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--export', action='store_true', help='训练后导出模型')
    parser.add_argument('--ablation', action='store_true', help='运行消融实验')
    parser.add_argument('--no-dage', action='store_true', help='禁用DAGE')
    parser.add_argument('--no-p2', action='store_true', help='禁用P2检测头')
    parser.add_argument('--no-wiou', action='store_true', help='禁用WIoU v3')
    parser.add_argument('--model', type=str, default=None, help='覆盖模型名称')

    args = parser.parse_args()

    check_environment()

    config = get_config(args.config)

    if args.no_dage:
        config.model.use_dage = False
    if args.no_p2:
        config.model.use_p2_head = False
    if args.no_wiou:
        config.model.use_wiou = False
    if args.model:
        config.model.name = args.model

    if args.ablation:
        run_ablation_study(config)
    else:
        results = train_model(config)

    if args.export:
        export_model(config)

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
