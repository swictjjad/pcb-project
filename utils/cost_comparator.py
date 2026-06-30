# -*- coding: utf-8 -*-
"""
TCO 总拥有成本对比工具
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import get_config, get_project_root
from utils.logger import setup_logger

logger = setup_logger("cost_comparator")


class CostComparator:
    """TCO 成本对比工具"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.project_root = get_project_root()

    def compare(self, system_a_params=None, system_b_params=None):
        if system_a_params is None:
            system_a_params = {
                "name": "PCB-YOLO (本系统)", "type": "AI视觉检测",
                "hardware_cost": 15000, "software_cost": 0,
                "calibration_cost": 500, "calibration_frequency": 1,
                "maintenance_cost": 2000, "inspector_cost": 30,
                "inspectors": 0, "throughput": 600,
                "precision": 0.92, "recall": 0.89,
                "lifespan_years": 5, "power_consumption_w": 150,
                "electricity_cost": 0.8,
            }
        if system_b_params is None:
            system_b_params = {
                "name": "传统AOI系统 (竞品)", "type": "传统光学检测",
                "hardware_cost": 300000, "software_cost": 50000,
                "calibration_cost": 5000, "calibration_frequency": 4,
                "maintenance_cost": 30000, "inspector_cost": 50,
                "inspectors": 2, "throughput": 400,
                "precision": 0.97, "recall": 0.95,
                "lifespan_years": 8, "power_consumption_w": 800,
                "electricity_cost": 0.8,
            }

        years = 5

        def calc_tco(params):
            hardware = params["hardware_cost"]
            software_annual = params.get("software_cost", 0)
            calibration = params["calibration_cost"] * params["calibration_frequency"]
            maintenance = params["maintenance_cost"]
            inspector = params["inspector_cost"] * params["inspectors"] * 8 * 252
            power_kwh = params["power_consumption_w"] / 1000 * 8 * 252
            electricity = power_kwh * params["electricity_cost"]
            annual_cost = software_annual + calibration + maintenance + inspector + electricity
            tco = hardware + annual_cost * years
            return {
                "hardware_cost": hardware,
                "annual_operating_cost": round(annual_cost, 2),
                "5year_operating_cost": round(annual_cost * years, 2),
                "total_tco_5yr": round(tco, 2),
            }

        tco_a = calc_tco(system_a_params)
        tco_b = calc_tco(system_b_params)
        savings = tco_b["total_tco_5yr"] - tco_a["total_tco_5yr"]
        savings_pct = round(savings / tco_b["total_tco_5yr"] * 100, 1) if tco_b["total_tco_5yr"] > 0 else 0

        result = {
            "comparison_period_years": years,
            "timestamp": datetime.now().isoformat(),
            "system_a": {**system_a_params, "tco": tco_a},
            "system_b": {**system_b_params, "tco": tco_b},
            "summary": {
                "savings_5yr_cny": round(savings, 2),
                "savings_percentage": savings_pct,
                "cost_ratio": round(tco_a["total_tco_5yr"] / tco_b["total_tco_5yr"] * 100, 1) if tco_b["total_tco_5yr"] > 0 else 0,
                "throughput_advantage": f"{system_a_params['throughput'] / max(system_b_params['throughput'], 1):.1f}x",
                "precision_tradeoff": f"{system_a_params['precision']:.1%} vs {system_b_params['precision']:.1%}",
            },
        }
        logger.info(f"TCO对比: 本系统5年={tco_a['total_tco_5yr']:.0f} CNY, 竞品5年={tco_b['total_tco_5yr']:.0f} CNY, 节省{savings_pct}%")
        return result

    def export_business_case(self, comparison_result, output_path=None):
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"business_case_{ts}.md"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        sa = comparison_result["system_a"]
        sb = comparison_result["system_b"]
        summary = comparison_result["summary"]

        lines = []
        lines.append("# PCB\u68c0\u6d4b\u7cfb\u7edf \u5546\u4e1a\u8bba\u8bc1\u62a5\u544a\n")
        lines.append(f"**\u751f\u6210\u65f6\u95f4**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---\n\n")
        lines.append("## 1. \u6267\u884c\u6982\u8981\n\n")
        lines.append(f"\u672c\u7814\u7a76\u5bf9\u6bd4\u4e86 **{sa['name']}** \u548c **{sb['name']}** \u7684TCO\u3002\n\n")
        lines.append(f"- **5\u5e74TCO\u5dee\u5f02**: {summary['savings_5yr_cny']:,.0f} CNY ({summary['savings_percentage']}%)\n")
        lines.append(f"- **\u6210\u672c\u6bd4\u4f8b**: \u672c\u7cfb\u7edf\u4ec5\u4e3a\u7ade\u54c1\u7684 {summary['cost_ratio']}%\n\n")
        lines.append("## 2. \u6210\u672c\u5bf9\u6bd4\n\n")
        lines.append("| \u6307\u6807 | \u672c\u7cfb\u7edf | \u7ade\u54c1 |")
        lines.append("|--------|----------|---------|")
        lines.append(f"| \u786c\u4ef6\u6210\u672c | {sa['tco']['hardware_cost']:,.0f} | {sb['tco']['hardware_cost']:,.0f} |")
        lines.append(f"| \u5e74\u8fd0\u8425 | {sa['tco']['annual_operating_cost']:,.0f} | {sb['tco']['annual_operating_cost']:,.0f} |")
        lines.append(f"| 5\u5e74TCO | {sa['tco']['total_tco_5yr']:,.0f} | {sb['tco']['total_tco_5yr']:,.0f} |\n\n")
        lines.append("## 3. \u6027\u80fd\u5bf9\u6bd4\n\n")
        lines.append("| \u6307\u6807 | \u672c\u7cfb\u7edf | \u7ade\u54c1 |")
        lines.append("|--------|----------|---------|")
        lines.append(f"| \u541e\u5410 | {sa.get('throughput', 'N/A')} /h | {sb.get('throughput', 'N/A')} /h |")
        lines.append(f"| Precision | {sa.get('precision', 0):.1%} | {sb.get('precision', 0):.1%} |")
        lines.append(f"| Recall | {sa.get('recall', 0):.1%} | {sb.get('recall', 0):.1%} |\n\n")
        lines.append("## 4. \u7ed3\u8bba\n\n")
        lines.append(f"\u672c\u7cfb\u7edf\u57285\u5e74\u5185\u8282\u7701 **{summary['savings_5yr_cny']:,.0f} CNY**\uff0c\u6210\u672c\u4ec5\u4e3a\u7ade\u54c1\u7684 **{summary['cost_ratio']}%**\u3002\n\n---\n")
        lines.append("*\u672c\u62a5\u544a\u7531 PCB-Cost-Comparator \u81ea\u52a8\u751f\u6210*\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        logger.info(f"商业论证报告已保存: {output_path}")
        return str(output_path)

    def export_json(self, comparison_result, output_path=None):
        if output_path is None:
            results_dir = self.project_root / "results" / "benchmarks"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"cost_comparison_{ts}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"成本对比JSON已保存: {output_path}")
        return str(output_path)
