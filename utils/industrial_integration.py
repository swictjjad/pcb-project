# -*- coding: utf-8 -*-
"""
工业级集成入口 - 一键运行所有模块
用法:
    python utils/industrial_integration.py [--quick] [--stability-minutes MIN]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import get_config, get_project_root
from utils.performance_benchmark import PerformanceBenchmark
from utils.confusion_matrix import ConfusionMatrixAnalyzer, FalseAlarmCostCalculator
from utils.robustness_engine import RobustnessEngine
from utils.inference_accelerator import QuantizationEngine, InferencePipeline
from utils.reliability_report import ReliabilityReportGenerator
from utils.cost_comparator import CostComparator
from utils.line_swap_enhancer import LineSwapEnhancer
from utils.logger import setup_logger

logger = setup_logger("integration")


def run_quick_test(config):
    """快速测试模式"""
    logger.info("=" * 60)
    logger.info("快速测试模式启动")
    logger.info("=" * 60)

    bench = PerformanceBenchmark(config=config)
    latency = bench.benchmark_latency(runs=50)
    logger.info(f"延迟: 均值={latency['mean_ms']:.1f}ms, P95={latency['p95_ms']:.1f}ms")

    accuracy = bench.benchmark_on_validation_set()
    logger.info(f"精度: 宏F1={accuracy.get('accuracy', {}).get('macro_f1', 0):.4f}")

    memory = bench.benchmark_memory(runs=10)
    logger.info(f"内存: 峰值={memory['peak_memory_mb']:.1f}MB")

    analyzer = ConfusionMatrixAnalyzer(config=config, detector=bench.detector)
    confusion = analyzer.compute_confusion_matrix(detector=bench.detector)
    analyzer.generate_confusion_matrix_image(confusion)
    analyzer.export_json(confusion)

    cost_calc = FalseAlarmCostCalculator(config=config)
    cost = cost_calc.calculate_from_analysis(confusion, images_per_hour=10, weeks=4)
    cost_calc.export_report(cost)

    reporter = ReliabilityReportGenerator(config=config)
    report = reporter.generate(benchmark_result=accuracy, confusion_result=confusion, memory_result=memory)
    reporter.export_markdown(report)
    reporter.export_json(report)

    logger.info("快速测试完成！")
    return {"latency": latency, "accuracy": accuracy, "memory": memory,
            "confusion": confusion, "cost": cost, "reliability_report": report}


def run_full_industrial_test(config, stability_minutes=3):
    """完整工业级测试"""
    logger.info("=" * 70)
    logger.info("完整工业级测试套件启动")
    logger.info("=" * 70)

    results = {}

    logger.info("[1/7] 完整性能基准测试...")
    bench = PerformanceBenchmark(config=config)
    full_bench = bench.run_full_benchmark(stability_minutes=stability_minutes)
    results["benchmark"] = full_bench

    logger.info("[2/7] 混淆矩阵分析...")
    analyzer = ConfusionMatrixAnalyzer(config=config, detector=bench.detector)
    confusion = analyzer.compute_confusion_matrix(detector=bench.detector)
    analyzer.generate_confusion_matrix_image(confusion)
    analyzer.export_json(confusion)
    results["confusion_matrix"] = confusion

    logger.info("[3/7] 误报成本分析...")
    cost_calc = FalseAlarmCostCalculator(config=config)
    cost = cost_calc.calculate_from_analysis(confusion, images_per_hour=10, weeks=4)
    cost_calc.export_report(cost)
    results["false_alarm_cost"] = cost

    logger.info("[4/7] 鲁棒性引擎测试...")
    robust = RobustnessEngine(config=config)
    import cv2
    val_img_dir = bench.project_root / bench.config.data.dataset_path / "val" / "images"
    val_imgs = list(val_img_dir.iterdir())
    if val_imgs:
        sample = cv2.imread(str(val_imgs[0]))
        if sample is not None:
            processed, quality, adaptation, skipped = robust.process_frame(sample)
            logger.info(f"鲁棒性: 质量={quality['quality_score']}, 适配={adaptation['method']}")
    results["robustness"] = robust.get_report()

    logger.info("[5/7] 换线功能测试...")
    swapper = LineSwapEnhancer(config=config)
    templates = swapper.get_template_names()
    logger.info(f"可用模板: {list(templates.keys())}")
    recommended = swapper.recommend_template("FR4_standard")
    logger.info(f"FR4推荐: conf={recommended['conf_threshold']}, iou={recommended['iou_threshold']}")
    swap_result = swapper.activate_profile("FR4_standard")
    logger.info(f"换线: success={swap_result['success']}, time={swap_result.get('swap_time_seconds', 0):.2f}s")
    results["line_swap"] = swap_result

    logger.info("[6/7] 生成可靠性报告...")
    reporter = ReliabilityReportGenerator(config=config)
    report = reporter.generate(
        benchmark_result=full_bench,
        stability_result=full_bench.get("stability", {}),
        confusion_result=confusion,
        cost_result=cost,
        memory_result=full_bench.get("memory", {}),
    )
    reporter.export_markdown(report)
    reporter.export_json(report)
    results["reliability_report"] = report

    logger.info("[7/7] 生成成本对比...")
    comparator = CostComparator(config=config)
    comparison = comparator.compare()
    comparator.export_json(comparison)
    comparator.export_business_case(comparison)
    results["cost_comparison"] = comparison

    logger.info("=" * 70)
    logger.info("完整工业级测试完成！")
    logger.info(f"  综合评分: {report.get('overall_score', 'N/A')}")
    logger.info(f"  等级: {report.get('grade', 'N/A')}")
    logger.info(f"  5年TCO节省: {comparison.get('summary', {}).get('savings_5yr_cny', 0):,.0f} CNY")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="PCB缺陷检测系统 - 工业级集成测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--stability-minutes", type=int, default=3, help="稳定性测试时长")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = get_config(args.config)

    if args.quick:
        results = run_quick_test(config)
    else:
        results = run_full_industrial_test(config, stability_minutes=args.stability_minutes)

    output_dir = Path(args.output_dir) if args.output_dir else get_project_root() / "results" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "integration_summary.json"
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (dict, list, str, int, float, bool, type(None))):
            serializable[k] = v
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
