# -*- coding: utf-8 -*-
"""
PCB缺陷检测系统 — 模拟演示脚本
无需真实硬件、摄像头或训练好的模型即可运行

功能：
1. 生成合成PCB图像（含模拟缺陷）
2. 使用yolo11n预训练权重进行推理（或模拟推理）
3. 模拟完整检测流程：采集→推理→决策→分拣
4. 展示所有6类缺陷的检测结果
5. 生成演示报告
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_loader import get_config
from utils.logger import setup_logger
from utils.serial_control import MockArduinoController

logger = setup_logger("demo")


# ==================== 合成PCB图像生成 ====================

CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short_circuit', 'spur', 'spurious_copper']
CLASS_NAMES_CN = ['漏孔', '鼠咬', '开路', '短路', '毛刺', '多余铜']
CLASS_COLORS = [
    (0, 0, 255),    # 红
    (0, 165, 255),  # 橙
    (0, 255, 255),  # 黄
    (0, 255, 0),    # 绿
    (255, 0, 255),  # 紫
    (255, 255, 0),  # 青
]


def generate_pcb_base(width=640, height=640):
    """生成PCB底板图像（绿色基板+铜走线+焊盘+过孔）"""
    # 绿色PCB基板
    img = np.ones((height, width, 3), dtype=np.uint8) * 30
    img[:, :, 1] = 120  # 绿色
    img[:, :, 0] = 40
    img[:, :, 2] = 30

    # 添加纹理噪点
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 绘制铜走线
    for _ in range(random.randint(8, 15)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        angle = random.choice([0, 90, 45, 135])
        length = random.randint(100, 400)
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = int(y1 + length * np.sin(np.radians(angle)))
        thickness = random.randint(2, 4)
        cv2.line(img, (x1, y1), (x2, y2), (180, 160, 60), thickness)

    # 绘制焊盘
    for _ in range(random.randint(10, 25)):
        cx = random.randint(30, width - 30)
        cy = random.randint(30, height - 30)
        r = random.randint(6, 12)
        cv2.circle(img, (cx, cy), r, (180, 160, 60), -1)
        cv2.circle(img, (cx, cy), r // 2, (40, 120, 30), -1)

    # 绘制过孔
    for _ in range(random.randint(5, 15)):
        cx = random.randint(30, width - 30)
        cy = random.randint(30, height - 30)
        cv2.circle(img, (cx, cy), 4, (180, 160, 60), -1)
        cv2.circle(img, (cx, cy), 2, (30, 30, 30), -1)

    # IC芯片
    for _ in range(random.randint(1, 3)):
        x = random.randint(50, width - 150)
        y = random.randint(50, height - 100)
        w = random.randint(60, 120)
        h = random.randint(40, 80)
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 160, 60), 1)
        # 引脚
        for i in range(0, w, 8):
            cv2.line(img, (x + i, y), (x + i, y - 6), (180, 160, 60), 1)
            cv2.line(img, (x + i, y + h), (x + i, y + h + 6), (180, 160, 60), 1)

    return img


def add_defect(img, defect_type, bbox_list):
    """在PCB图像上添加指定类型的缺陷"""
    h, w = img.shape[:2]
    margin = 40

    if defect_type == 0:  # missing_hole (漏孔)
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)
        r = random.randint(8, 15)
        # 画焊盘但没有孔
        cv2.circle(img, (cx, cy), r, (180, 160, 60), -1)
        # 漏孔：没有中心孔，添加一个标记点
        cv2.circle(img, (cx, cy), 2, (200, 200, 200), -1)
        bbox = [cx - r, cy - r, cx + r, cy + r]

    elif defect_type == 1:  # mouse_bite (鼠咬)
        x = random.randint(margin, w - margin - 60)
        y = random.randint(margin, h - margin)
        # 画一条走线，然后咬掉一块
        cv2.line(img, (x, y), (x + 80, y), (180, 160, 60), 3)
        bite_w = random.randint(5, 12)
        bite_h = random.randint(3, 8)
        bx = x + random.randint(20, 50)
        cv2.rectangle(img, (bx, y - bite_h), (bx + bite_w, y + 1), (40, 120, 30), -1)
        bbox = [bx - 2, y - bite_h - 2, bx + bite_w + 2, y + 3]

    elif defect_type == 2:  # open_circuit (开路)
        x = random.randint(margin, w - margin - 100)
        y = random.randint(margin, h - margin)
        cv2.line(img, (x, y), (x + 40, y), (180, 160, 60), 3)
        gap = random.randint(8, 15)
        cv2.line(img, (x + 40 + gap, y), (x + 100, y), (180, 160, 60), 3)
        # 间隙处填充背景色
        cv2.rectangle(img, (x + 40, y - 4), (x + 40 + gap, y + 4), (40, 120, 30), -1)
        bbox = [x + 38, y - 6, x + 42 + gap, y + 6]

    elif defect_type == 3:  # short_circuit (短路)
        x = random.randint(margin, w - margin - 80)
        y1 = random.randint(margin + 10, h - margin - 20)
        y2 = y1 + random.randint(15, 25)
        cv2.line(img, (x, y1), (x + 80, y1), (180, 160, 60), 3)
        cv2.line(img, (x, y2), (x + 80, y2), (180, 160, 60), 3)
        # 短路桥
        bridge_x = x + random.randint(20, 50)
        cv2.line(img, (bridge_x, y1), (bridge_x, y2), (180, 160, 60), 2)
        bbox = [bridge_x - 4, y1 - 4, bridge_x + 4, y2 + 4]

    elif defect_type == 4:  # spur (毛刺)
        x = random.randint(margin, w - margin)
        y = random.randint(margin, h - margin)
        cv2.line(img, (x - 30, y), (x + 30, y), (180, 160, 60), 3)
        # 毛刺
        spur_len = random.randint(8, 18)
        spur_angle = random.uniform(-60, 60)
        sx = int(x + spur_len * np.cos(np.radians(spur_angle)))
        sy = int(y + spur_len * np.sin(np.radians(spur_angle)))
        cv2.line(img, (x, y), (sx, sy), (180, 160, 60), 2)
        bbox = [min(x, sx) - 3, min(y, sy) - 3, max(x, sx) + 3, max(y, sy) + 3]

    elif defect_type == 5:  # spurious_copper (多余铜)
        cx = random.randint(margin + 20, w - margin - 20)
        cy = random.randint(margin + 20, h - margin - 20)
        size = random.randint(8, 18)
        # 多余铜块
        pts = np.array([
            [cx, cy - size],
            [cx + size, cy],
            [cx + size // 2, cy + size],
            [cx - size // 2, cy + size // 2],
        ], np.int32)
        cv2.fillPoly(img, [pts], (180, 160, 60))
        bbox = [cx - size - 2, cy - size - 2, cx + size + 2, cy + size + 2]

    else:
        bbox = [0, 0, 10, 10]

    bbox_list.append({
        'bbox': bbox,
        'class_id': defect_type,
        'class_name': CLASS_NAMES[defect_type],
        'class_name_cn': CLASS_NAMES_CN[defect_type],
    })
    return img


def generate_sample_image(defect_types=None, width=640, height=640):
    """生成一张含指定缺陷的PCB图像"""
    img = generate_pcb_base(width, height)
    bboxes = []

    if defect_types is None:
        defect_types = [random.randint(0, 5) for _ in range(random.randint(1, 3))]

    for dt in defect_types:
        img = add_defect(img, dt, bboxes)

    return img, bboxes


# ==================== 模拟检测器 ====================

class MockDetector:
    """模拟检测器（不需要真实模型）"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.class_names = CLASS_NAMES
        self.class_colors = CLASS_COLORS
        self.fps_history = []

    def detect(self, image, ground_truth=None):
        """
        模拟检测：如果有ground_truth则使用，否则随机生成检测结果
        """
        start_time = time.time()

        if ground_truth is not None:
            # 有ground truth时：使用ground truth + 模拟置信度
            # ground_truth为空列表时表示合格品，不生成检测
            detections = []
            for gt in ground_truth:
                conf = random.uniform(0.65, 0.98)
                # 模拟轻微的bbox偏移
                bbox = gt['bbox'].copy()
                offset = random.randint(-3, 3)
                bbox = [b + offset for b in bbox]
                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': gt['class_id'],
                    'class_name': gt['class_name'],
                })
        else:
            # 随机生成1-3个检测
            num_det = random.randint(0, 3)
            detections = []
            for _ in range(num_det):
                cls_id = random.randint(0, 5)
                x1 = random.randint(50, 500)
                y1 = random.randint(50, 500)
                x2 = x1 + random.randint(15, 40)
                y2 = y1 + random.randint(15, 40)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': random.uniform(0.55, 0.95),
                    'class_id': cls_id,
                    'class_name': CLASS_NAMES[cls_id],
                })

        # 模拟推理耗时
        inference_time = random.uniform(150, 400)
        time.sleep(inference_time / 1000.0)  # 模拟真实耗时

        fps = 1000.0 / inference_time
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history[-30:]) / len(self.fps_history[-30:])

        return {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time,
            'fps': avg_fps,
            'image_shape': image.shape[:2],
            'backend': 'mock'
        }

    def draw_results(self, image, results, show_cn=True):
        """绘制检测结果（使用PIL支持中文）"""
        from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont

        img = image.copy()
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = PILDraw.Draw(pil_img)

        # 加载字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        font = None
        font_small = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = PILFont.truetype(fp, 18)
                    font_small = PILFont.truetype(fp, 16)
                    break
                except Exception:
                    continue
        if font is None:
            font = PILFont.load_default()
            font_small = font

        for det in results['detections']:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_id = det['class_id']
            conf = det['confidence']
            label = det.get('class_name_cn', det['class_name']) if show_cn else det['class_name']
            color = self.class_colors[cls_id % len(self.class_colors)]

            # 边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 标签背景
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 22), text, font=font_small)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 22), text, fill=(255, 255, 255), font=font_small)

        # 信息栏
        info_lines = [
            f"FPS: {results['fps']:.1f}",
            f"Time: {results['inference_time']:.1f}ms",
            f"Defects: {results['num_detections']}",
            f"Backend: {results.get('backend', 'mock')}",
        ]
        for i, line in enumerate(info_lines):
            draw.text((10, 10 + i * 24), line, fill=(0, 255, 0), font=font_small)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==================== 演示流程 ====================

def run_demo():
    """运行模拟演示"""
    logger.info("=" * 60)
    logger.info("PCB缺陷检测与智能分拣系统 — 模拟演示")
    logger.info("=" * 60)

    config = get_config()
    detector = MockDetector(config)
    controller = MockArduinoController()

    # 创建输出目录
    save_dir = Path("results/demo")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    total = 0
    ok_count = 0
    ng_count = 0
    defect_counter = Counter()
    all_results = []

    logger.info("\n阶段1: 生成合成PCB样本并检测")
    logger.info("-" * 40)

    # 生成20张样本（模拟竞赛评估）
    sample_configs = [
        # 每张图指定缺陷类型
        [0], [1], [2], [3], [4], [5],           # 每种缺陷各一张
        [0, 1], [2, 3], [4, 5],                  # 组合缺陷
        [0, 0], [1, 2], [3, 4],                  # 更多组合
        [5, 0], [2, 4], [1, 3, 5],               # 三缺陷
        [], [], [], [], [], [], []                # 合格品（7张）
    ]

    for i, defect_types in enumerate(sample_configs):
        logger.info(f"\n样本 {i + 1}/20")

        # 1. 生成PCB图像
        img, ground_truth = generate_sample_image(defect_types)

        # 2. 模拟采集（保存原始图）
        raw_path = save_dir / f"raw_{i + 1:02d}.jpg"
        cv2.imwrite(str(raw_path), img)

        # 3. 模型推理
        results = detector.detect(img, ground_truth=ground_truth)

        # 4. 判定结果
        is_defective = results['num_detections'] > 0
        total += 1

        if is_defective:
            ng_count += 1
            for det in results['detections']:
                defect_counter[det['class_name']] += 1
            logger.info(f"  判定: 不合格 (检测到 {results['num_detections']} 个缺陷)")
            controller.sort_pcb(is_ok=False)
        else:
            ok_count += 1
            logger.info(f"  判定: 合格")
            controller.sort_pcb(is_ok=True)

        # 5. 绘制结果
        vis_img = detector.draw_results(img, results)
        result_path = save_dir / f"result_{i + 1:02d}.jpg"
        cv2.imwrite(str(result_path), vis_img)

        all_results.append({
            'index': i + 1,
            'defects': results['num_detections'],
            'inference_time': results['inference_time'],
            'is_defective': is_defective,
            'detection_details': [{
                'class_name': d['class_name'],
                'class_name_cn': CLASS_NAMES_CN[d['class_id']],
                'confidence': d['confidence'],
            } for d in results['detections']],
        })

    # 生成评估报告
    logger.info("\n" + "=" * 60)
    logger.info("演示评估报告")
    logger.info("=" * 60)

    pass_rate = ok_count / total * 100 if total > 0 else 0
    avg_time = sum(r['inference_time'] for r in all_results) / len(all_results)

    logger.info(f"总样本数: {total}")
    logger.info(f"合格品: {ok_count}")
    logger.info(f"不合格品: {ng_count}")
    logger.info(f"合格率: {pass_rate:.1f}%")
    logger.info(f"平均推理耗时: {avg_time:.1f}ms")
    logger.info(f"平均FPS: {1000 / avg_time:.1f}")

    logger.info("\n缺陷类别分布:")
    for name, count in defect_counter.most_common():
        cn_name = CLASS_NAMES_CN[CLASS_NAMES.index(name)]
        logger.info(f"  {name} ({cn_name}): {count}")

    # 保存JSON报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': total,
        'pass_count': ok_count,
        'fail_count': ng_count,
        'pass_rate': pass_rate,
        'avg_inference_time_ms': avg_time,
        'defect_distribution': dict(defect_counter),
        'results': all_results,
    }

    report_path = save_dir / 'demo_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n报告已保存: {report_path}")
    logger.info(f"检测图片已保存: {save_dir}/")

    # 生成汇总图
    generate_summary_image(save_dir, all_results, ok_count, ng_count, defect_counter, avg_time)

    logger.info("\n模拟演示完成！")
    return report


def generate_summary_image(save_dir, results, ok_count, ng_count, defect_counter, avg_time):
    """生成汇总展示图（使用PIL支持中文）"""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 1200, 800
    img = Image.new('RGB', (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/simsun.ttc",   # 宋体
    ]
    font_large = None
    font_medium = None
    font_small = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_large = ImageFont.truetype(fp, 28)
                font_medium = ImageFont.truetype(fp, 20)
                font_small = ImageFont.truetype(fp, 16)
                break
            except Exception:
                continue
    if font_large is None:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # 标题
    draw.text((30, 20), "PCB缺陷检测与智能分拣系统 — 演示报告", fill=(30, 30, 30), font=font_large)

    # 统计信息
    y = 80
    total = ok_count + ng_count
    pass_rate = ok_count / total * 100 if total > 0 else 0
    stats = [
        f"总样本数: {total}",
        f"合格品: {ok_count}  |  不合格品: {ng_count}",
        f"合格率: {pass_rate:.1f}%",
        f"平均推理耗时: {avg_time:.1f}ms  |  平均FPS: {1000 / avg_time:.1f}",
    ]
    for s in stats:
        draw.text((30, y), s, fill=(50, 50, 50), font=font_medium)
        y += 32

    # 缺陷分布柱状图
    y = 240
    draw.text((30, y), "缺陷类别分布:", fill=(30, 30, 30), font=font_medium)
    y += 38

    max_count = max(defect_counter.values()) if defect_counter else 1
    bar_max_w = 300
    for i, name in enumerate(CLASS_NAMES):
        cn_name = CLASS_NAMES_CN[i]
        count = defect_counter.get(name, 0)
        bar_w = int(count / max_count * bar_max_w) if max_count > 0 else 0

        # 标签
        draw.text((30, y + 2), f"{cn_name}({name})", fill=(50, 50, 50), font=font_small)

        # 柱子
        color = CLASS_COLORS[i]
        draw.rectangle([220, y, 220 + bar_w, y + 22], fill=color)

        # 数值
        draw.text((225 + bar_w, y + 2), str(count), fill=(50, 50, 50), font=font_small)
        y += 32

    # 展示4张检测结果缩略图
    thumb_size = 180
    thumb_y = 80
    result_files = sorted(save_dir.glob("result_*.jpg"))

    selected = []
    for rf in result_files:
        if len(selected) >= 4:
            break
        idx = int(rf.stem.split('_')[1]) - 1
        if idx < len(results) and results[idx]['defects'] > 0:
            selected.append(rf)

    x_offset = 550
    for i, rf in enumerate(selected[:4]):
        thumb = Image.open(str(rf)).resize((thumb_size, thumb_size))
        row = i // 2
        col = i % 2
        tx = x_offset + col * (thumb_size + 20)
        ty = thumb_y + row * (thumb_size + 20)
        img.paste(thumb, (tx, ty))

    # 保存
    summary_path = save_dir / 'summary.jpg'
    img.save(str(summary_path), quality=95)
    logger.info(f"汇总图已保存: {summary_path}")


# ==================== 交互式演示 ====================

def run_interactive_demo():
    """交互式模拟演示（OpenCV窗口）"""
    logger.info("=" * 60)
    logger.info("PCB缺陷检测 — 交互式模拟演示")
    logger.info("按 [空格] 检测新样本 | [1-6] 添加指定缺陷 | [0] 合格品 | [q] 退出")
    logger.info("=" * 60)

    config = get_config()
    detector = MockDetector(config)
    controller = MockArduinoController()

    total = 0
    ok_count = 0
    ng_count = 0

    # 初始显示
    img, gt = generate_sample_image([0])
    results = detector.detect(img, ground_truth=gt)
    vis = detector.draw_results(img, results)
    cv2.imshow("PCB Defect Detection - Interactive Demo", vis)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # 随机检测
            defects = [random.randint(0, 5) for _ in range(random.randint(0, 3))]
            img, gt = generate_sample_image(defects)
        elif key == ord('0'):
            img, gt = generate_sample_image([])
        elif ord('1') <= key <= ord('6'):
            defect_type = key - ord('1')
            img, gt = generate_sample_image([defect_type])
        else:
            continue

        # 检测
        results = detector.detect(img, ground_truth=gt)
        total += 1

        if results['num_detections'] > 0:
            ng_count += 1
            controller.sort_pcb(is_ok=False)
        else:
            ok_count += 1
            controller.sort_pcb(is_ok=True)

        # 绘制
        vis = detector.draw_results(img, results)

        # 添加统计信息
        pass_rate = ok_count / total * 100 if total > 0 else 0
        cv2.putText(vis, f"Total: {total}  Pass: {ok_count}  Fail: {ng_count}  Rate: {pass_rate:.1f}%",
                    (10, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("PCB Defect Detection - Interactive Demo", vis)

    cv2.destroyAllWindows()
    logger.info(f"交互演示结束: 检测{total}张, 合格{ok_count}, 不合格{ng_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PCB缺陷检测模拟演示')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'interactive'],
                        help='auto=自动20张样本, interactive=交互式')

    args = parser.parse_args()

    if args.mode == 'auto':
        run_demo()
    else:
        run_interactive_demo()
