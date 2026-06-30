# -*- coding: utf-8 -*-
"""
工业可靠性报告生成器
"""

import json
import os
import sys
import platform
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger

logger = setup_logger("reliability_report")


class ReliabilityReportGenerator:
    """工业可靠性报告生成器"""

    INDUSTRY_BENCHMARKS = {
        "precision": {"target": 0.95, "excellent": 0.98, "acceptable": 0.90},
        "recall": {"target": 0.93, "excellent": 0.97, "acceptable": 0.88},
        "latency_ms": {"target": 100, "excellent": 50, "acceptable": 200},
        "false_alarm_rate": {"target": 0.02, "excellent": 0.01, "acceptable": 0.05},
        "uptime": {"target": 0.99, "excellent": 0.999, "acceptable": 0.95},
    }

    def __init__(self, config=None):
        self.config = config or get_config()
        self.project_root = get_project_root()

    def _gather_system_info(self):
        return {
            "os": platform.system() + " " + platform.release(),
            "python": platform.python_version(),
            "cpu": platform.processor() or "Unknown",
            "machine": platform.machine(),
            "cuda_available": False,
            "project_root": str(self.project_root),
        }

    def _evaluate_metric(self, value, metric_name, direction="higher_is_better"):
        benchmarks = self.INDUSTRY_BENCHMARKS.get(metric_name, {})
        if not benchmarks:
            return {"status": "no_benchmark", "value": value}
        if direction == "higher_is_better":
            if value >= benchmarks.get("excellent", 0):
                status, score = "excellent", 3
            elif value >= benchmarks.get("target", 0):
                status, score = "pass", 2
            elif value >= benchmarks.get("acceptable", 0):
                status, score = "warning", 1
            else:
                status, score = "fail", 0
        else:
            if value <= benchmarks.get("excellent", 999):
                status, score = "excellent", 3
            elif value <= benchmarks.get("target", 999):
                status, score = "pass", 2
            elif value <= benchmarks.get("acceptable", 999):
                status, score = "warning", 1
            else:
                status, score = "fail", 0
        return {"status": status, "score": score, "value": value, "benchmark": benchmarks}

    def generate(self, benchmark_result=None, stability_result=None,
                 confusion_result=None, cost_result=None, memory_result=None):
        logger.info("开始生成工业可靠性报告...")
        system_info = self._gather_system_info()
        evaluations = {}

        if benchmark_result and "accuracy" in benchmark_result:
            acc = benchmark_result["accuracy"]
            evaluations["precision"] = self._evaluate_metric(acc.get("macro_precision", 0), "precision", "higher_is_better")
            evaluations["recall"] = self._evaluate_metric(acc.get("macro_recall", 0), "recall", "higher_is_better")
            evaluations["f1"] = self._evaluate_metric(acc.get("macro_f1", 0), "recall", "higher_is_better")

        if benchmark_result and "latency" in benchmark_result:
            lat = benchmark_result["latency"]
            evaluations["latency_mean"] = self._evaluate_metric(lat.get("mean_ms", 999), "latency_ms", "lower_is_better")
            evaluations["latency_p95"] = self._evaluate_metric(lat.get("p95_ms", 999), "latency_ms", "lower_is_better")

        if confusion_result and "per_class" in confusion_result:
            total_fp = sum(v.get("fp", 0) for v in confusion_result["per_class"].values())
            total_samples = confusion_result.get("total_images", 1)
            fp_rate = total_fp / total_samples
            evaluations["false_alarm_rate"] = self._evaluate_metric(fp_rate, "false_alarm_rate", "lower_is_better")

        if stability_result:
            uptime = 1.0 - (stability_result.get("error_count", 0) / max(stability_result.get("total_frames", 1), 1))
            evaluations["uptime"] = self._evaluate_metric(uptime, "uptime", "higher_is_better")

        total_score = sum(v.get("score", 0) for v in evaluations.values())
        max_score = len(evaluations) * 3
        overall_score = round(total_score / max_score * 100, 1) if max_score > 0 else 0

        if evaluations:
            avg_score = total_score / len(evaluations)
            grade = "F"
            for threshold, letter in [(3, "A+"), (2.5, "A"), (2, "B+"), (1.5, "B"), (1, "C"), (0.5, "D")]:
                if avg_score >= threshold:
                    grade = letter
                    break

        report_data = {
            "report_type": "industrial_reliability_report",
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "overall_score": overall_score,
            "grade": grade,
            "evaluations": evaluations,
            "summary": {
                "total_metrics_evaluated": len(evaluations),
                "metrics_passed": sum(1 for v in evaluations.values() if v.get("status") in ("excellent", "pass")),
                "metrics_warning": sum(1 for v in evaluations.values() if v.get("status") == "warning"),
                "metrics_failed": sum(1 for v in evaluations.values() if v.get("status") == "fail"),
            },
            "data_sources": {
                "benchmark": benchmark_result is not None,
                "stability": stability_result is not None,
                "confusion_matrix": confusion_result is not None,
                "cost_analysis": cost_result is not None,
                "memory": memory_result is not None,
            },
        }
        logger.info(f"可靠性报告生成完成: 综合评分={overall_score}, 等级={grade}")
        return report_data

    def export_markdown(self, report_data, output_path=None):
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"reliability_report_{ts}.md"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# PCB\u7f3a\u9677\u68c0\u6d4b\u7cfb\u7edf \u5de5\u4e1a\u53ef\u9760\u6027\u62a5\u544a\n")
        lines.append(f"**\u751f\u6210\u65f6\u95f4**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**\u7efc\u5408\u8bc4\u5206**: {report_data.get('overall_score', 0)} / 100\n")
        lines.append(f"**\u7b49\u7ea7**: {report_data.get('grade', 'N/A')}\n")
        lines.append("")
        lines.append("## \u7cfb\u7edf\u4fe1\u606f\n")
        si = report_data.get("system_info", {})
        lines.append(f"- **OS**: {si.get('os', 'N/A')}")
        lines.append(f"- **Python**: {si.get('python', 'N/A')}")
        lines.append(f"- **CPU**: {si.get('cpu', 'N/A')}")
        lines.append(f"- **CUDA**: {si.get('cuda_available', False)}\n")
        lines.append("## \u8bc4\u4f30\u6458\u8981\n")
        summary = report_data.get("summary", {})
        lines.append(f"- \u8bc4\u4f30\u6307\u6807: {summary.get('total_metrics_evaluated', 0)}")
        lines.append(f"- \u5408\u683c: {summary.get('metrics_passed', 0)}")
        lines.append(f"- \u8b66\u544a: {summary.get('metrics_warning', 0)}")
        lines.append(f"- \u4e0d\u5408\u683c: {summary.get('metrics_failed', 0)}\n")
        lines.append("## \u6307\u6807\u8bc4\u4f30\n")
        lines.append("| \u6307\u6807 | \u503c | \u72b6\u6001 |")
        lines.append("|------|------|---------|")
        for name, ev in report_data.get("evaluations", {}).items():
            status_cn = {"excellent": "\u2705 \u4f18\u79c0", "pass": "\u2705 \u5408\u683c",
                        "warning": "\u26a0\ufe0f \u8b66\u544a", "fail": "\u274c \u4e0d\u5408\u683c"}.get(ev.get("status"), ev.get("status"))
            lines.append(f"| {name} | {ev.get('value', 'N/A')} | {status_cn} |")
        lines.append("")
        lines.append("---\n")
        lines.append("*\u672c\u62a5\u544a\u7531 PCB-Reliability-Report \u81ea\u52a8\u751f\u6210*\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        logger.info(f"可靠性报告已保存: {output_path}")
        return str(output_path)

    def export_json(self, report_data, output_path=None):
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"reliability_report_{ts}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"可靠性报告JSON已保存: {output_path}")
        return str(output_path)
