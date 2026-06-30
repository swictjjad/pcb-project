# -*- coding: utf-8 -*-
"""
生产报表模块
功能：
1. 班次产量统计
2. 缺陷趋势分析
3. 操作员绩效
4. 报表导出（CSV/JSON）
"""

import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from utils.logger import setup_logger

logger = setup_logger("production_report")


class ProductionReportGenerator:
    """生产报表生成器"""

    def __init__(self, traceability_db=None):
        """
        Args:
            traceability_db: 追溯数据库实例
        """
        self.db = traceability_db

    def generate_shift_report(self, shift_date: str = None,
                              operator_id: str = "",
                              equipment_id: str = "") -> dict:
        """
        生成班次报表

        Args:
            shift_date: 日期 (YYYY-MM-DD)，默认今天
            operator_id: 操作员工号
            equipment_id: 设备编号
        """
        if shift_date is None:
            shift_date = datetime.now().strftime("%Y-%m-%d")

        stats = self.db.get_statistics(
            date=shift_date,
            operator_id=operator_id,
            equipment_id=equipment_id,
        )

        report = {
            'report_type': 'shift_report',
            'generated_at': datetime.now().isoformat(),
            'shift_date': shift_date,
            'operator_id': operator_id,
            'equipment_id': equipment_id,
            'statistics': stats,
        }

        return report

    def generate_defect_trend(self, days: int = 7,
                              operator_id: str = "",
                              equipment_id: str = "") -> dict:
        """
        生成缺陷趋势报表

        Args:
            days: 统计天数
            operator_id: 操作员工号
            equipment_id: 设备编号
        """
        trend_data = []
        today = datetime.now()

        for i in range(days - 1, -1, -1):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            stats = self.db.get_statistics(
                date=date,
                operator_id=operator_id,
                equipment_id=equipment_id,
            )
            trend_data.append({
                'date': date,
                'total': stats['total'],
                'ok_count': stats['ok_count'],
                'ng_count': stats['ng_count'],
                'pass_rate': stats['pass_rate'],
                'defect_stats': stats['defect_class_stats'],
            })

        return {
            'report_type': 'defect_trend',
            'generated_at': datetime.now().isoformat(),
            'days': days,
            'operator_id': operator_id,
            'equipment_id': equipment_id,
            'trend': trend_data,
        }

    def generate_operator_report(self, date: str = None) -> dict:
        """
        生成操作员绩效报表

        Args:
            date: 日期
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        records = self.db.query_by_timerange(
            f"{date}T00:00:00", f"{date}T23:59:59"
        )

        # 按操作员分组
        operator_stats = {}
        for rec in records:
            oid = rec.get('operator_id', 'unknown')
            if oid not in operator_stats:
                operator_stats[oid] = {
                    'operator_id': oid,
                    'total': 0,
                    'ok_count': 0,
                    'ng_count': 0,
                    'inference_times': [],
                }
            s = operator_stats[oid]
            s['total'] += 1
            if rec.get('result') == 'OK':
                s['ok_count'] += 1
            else:
                s['ng_count'] += 1
            it = rec.get('inference_time_ms')
            if it is not None:
                try:
                    s['inference_times'].append(float(it))
                except (TypeError, ValueError):
                    pass

        # 计算汇总
        for s in operator_stats.values():
            s['pass_rate'] = (s['ok_count'] / s['total'] * 100) if s['total'] > 0 else 0
            s['avg_inference_ms'] = (
                sum(s['inference_times']) / len(s['inference_times'])
                if s['inference_times'] else 0
            )
            del s['inference_times']

        return {
            'report_type': 'operator_report',
            'generated_at': datetime.now().isoformat(),
            'date': date,
            'operators': list(operator_stats.values()),
        }

    def export_csv(self, report: dict, file_path: str) -> bool:
        """导出报表为CSV"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                report_type = report.get('report_type', '')

                if report_type == 'shift_report':
                    self._write_shift_csv(writer, report)
                elif report_type == 'defect_trend':
                    self._write_trend_csv(writer, report)
                elif report_type == 'operator_report':
                    self._write_operator_csv(writer, report)

            logger.info(f"报表已导出: {path}")
            return True
        except Exception as e:
            logger.error(f"导出报表失败: {e}")
            return False

    def export_json(self, report: dict, file_path: str) -> bool:
        """导出报表为JSON"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"报表已导出: {path}")
            return True
        except Exception as e:
            logger.error(f"导出报表失败: {e}")
            return False

    def _write_shift_csv(self, writer, report: dict):
        writer.writerow(["班次报表"])
        writer.writerow(["日期", report.get('shift_date', '')])
        writer.writerow(["操作员", report.get('operator_id', '')])
        writer.writerow(["设备", report.get('equipment_id', '')])
        writer.writerow([])

        stats = report.get('statistics', {})
        writer.writerow(["统计项", "数值"])
        writer.writerow(["总检测数", stats.get('total', 0)])
        writer.writerow(["合格数", stats.get('ok_count', 0)])
        writer.writerow(["缺陷数", stats.get('ng_count', 0)])
        writer.writerow(["合格率", f"{stats.get('pass_rate', 0):.1f}%"])
        writer.writerow(["平均推理耗时", f"{stats.get('avg_inference_ms', 0):.1f}ms"])
        writer.writerow([])

        defect_stats = stats.get('defect_class_stats', {})
        if defect_stats:
            writer.writerow(["缺陷类别", "数量"])
            for cls, count in defect_stats.items():
                writer.writerow([cls, count])

    def _write_trend_csv(self, writer, report: dict):
        writer.writerow(["缺陷趋势报表", f"最近{report.get('days', 7)}天"])
        writer.writerow(["日期", "总数", "合格", "缺陷", "合格率"])
        for item in report.get('trend', []):
            writer.writerow([
                item['date'], item['total'], item['ok_count'],
                item['ng_count'], f"{item['pass_rate']:.1f}%"
            ])

    def _write_operator_csv(self, writer, report: dict):
        writer.writerow(["操作员绩效报表", report.get('date', '')])
        writer.writerow(["工号", "总数", "合格", "缺陷", "合格率", "平均推理(ms)"])
        for op in report.get('operators', []):
            writer.writerow([
                op['operator_id'], op['total'], op['ok_count'],
                op['ng_count'], f"{op['pass_rate']:.1f}%",
                f"{op['avg_inference_ms']:.1f}"
            ])
