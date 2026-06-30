# -*- coding: utf-8 -*-
"""
缺陷追溯模块
功能：
1. 产品SN码关联
2. 操作员工号记录
3. 设备编号记录
4. 完整检测历史（SQLite持久化，不限1000条）
5. 质量事故追溯查询
"""

import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from utils.logger import setup_logger

logger = setup_logger("traceability")

# 默认数据库路径（使用项目根目录下的绝对路径）
_DEFAULT_DB_PATH_RESOLVED = None


def _resolve_db_path() -> Path:
    """解析追溯数据库路径的绝对路径"""
    global _DEFAULT_DB_PATH_RESOLVED
    if _DEFAULT_DB_PATH_RESOLVED is not None:
        return _DEFAULT_DB_PATH_RESOLVED
    from utils.config_loader import get_project_root
    _DEFAULT_DB_PATH_RESOLVED = get_project_root() / "results" / "traceability.db"
    return _DEFAULT_DB_PATH_RESOLVED


DEFAULT_DB_PATH = _resolve_db_path()


class TraceabilityDB:
    """缺陷追溯数据库"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = None
        self._connect()
        self._init_tables()

    def _connect(self):
        """连接数据库"""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        logger.info(f"追溯数据库已连接: {self.db_path}")

    def _init_tables(self):
        """初始化数据表"""
        with self._lock:
            cursor = self._conn.cursor()

            # 检测记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    product_sn TEXT DEFAULT '',
                    operator_id TEXT DEFAULT '',
                    equipment_id TEXT DEFAULT '',
                    line_profile TEXT DEFAULT '',
                    result TEXT NOT NULL DEFAULT 'OK',
                    num_defects INTEGER DEFAULT 0,
                    inference_time_ms REAL DEFAULT 0,
                    image_path TEXT DEFAULT '',
                    result_image_path TEXT DEFAULT '',
                    reviewed INTEGER DEFAULT 0,
                    review_result TEXT DEFAULT '',
                    reviewer_id TEXT DEFAULT '',
                    review_time TEXT DEFAULT '',
                    extra_data TEXT DEFAULT '{}'
                )
            """)

            # 缺陷明细表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    bbox TEXT DEFAULT '[]',
                    suppressed INTEGER DEFAULT 0,
                    suppressed_by TEXT DEFAULT '',
                    FOREIGN KEY (record_id) REFERENCES detection_records(id)
                )
            """)

            # 班次统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shift_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shift_date TEXT NOT NULL,
                    shift_name TEXT NOT NULL,
                    operator_id TEXT DEFAULT '',
                    equipment_id TEXT DEFAULT '',
                    line_profile TEXT DEFAULT '',
                    total_count INTEGER DEFAULT 0,
                    ok_count INTEGER DEFAULT 0,
                    ng_count INTEGER DEFAULT 0,
                    defect_class_stats TEXT DEFAULT '{}',
                    avg_inference_ms REAL DEFAULT 0,
                    start_time TEXT DEFAULT '',
                    end_time TEXT DEFAULT ''
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sn ON detection_records(product_sn)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_operator ON detection_records(operator_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equipment ON detection_records(equipment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_result ON detection_records(result)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviewed ON detection_records(reviewed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_shift_date ON shift_stats(shift_date)")

            self._conn.commit()

    def add_record(self, product_sn: str = "", operator_id: str = "",
                   equipment_id: str = "", line_profile: str = "",
                   result: str = "OK", num_defects: int = 0,
                   inference_time_ms: float = 0, image_path: str = "",
                   result_image_path: str = "", defects: List[dict] = None,
                   extra_data: dict = None) -> int:
        """
        添加检测记录

        Returns:
            记录ID
        """
        with self._lock:
            cursor = self._conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO detection_records
                (timestamp, product_sn, operator_id, equipment_id, line_profile,
                 result, num_defects, inference_time_ms, image_path, result_image_path, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, product_sn, operator_id, equipment_id, line_profile,
                result, num_defects, inference_time_ms, image_path, result_image_path,
                json.dumps(extra_data or {}, ensure_ascii=False)
            ))

            record_id = cursor.lastrowid

            # 插入缺陷明细
            if defects:
                for det in defects:
                    cursor.execute("""
                        INSERT INTO defect_details
                        (record_id, class_name, confidence, bbox, suppressed, suppressed_by)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        record_id,
                        det.get('class_name', ''),
                        det.get('confidence', 0),
                        json.dumps(det.get('bbox', [])),
                        1 if det.get('suppressed_by') else 0,
                        det.get('suppressed_by', ''),
                    ))

            self._conn.commit()
            return record_id

    def review_record(self, record_id: int, review_result: str, reviewer_id: str = ""):
        """复核记录"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                UPDATE detection_records
                SET reviewed = 1, review_result = ?, reviewer_id = ?, review_time = ?
                WHERE id = ?
            """, (review_result, reviewer_id, datetime.now().isoformat(), record_id))
            self._conn.commit()

    def batch_review(self, record_ids: List[int], review_result: str, reviewer_id: str = ""):
        """批量复核"""
        with self._lock:
            cursor = self._conn.cursor()
            now = datetime.now().isoformat()
            for rid in record_ids:
                cursor.execute("""
                    UPDATE detection_records
                    SET reviewed = 1, review_result = ?, reviewer_id = ?, review_time = ?
                    WHERE id = ?
                """, (review_result, reviewer_id, now, rid))
            self._conn.commit()
        logger.info(f"批量复核: {len(record_ids)} 条记录, 结果={review_result}")

    def get_unreviewed(self, limit: int = 50) -> List[dict]:
        """获取未复核记录"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_records
                WHERE reviewed = 0 AND result = 'NG'
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_sn(self, product_sn: str) -> List[dict]:
        """按产品SN码查询"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_records
                WHERE product_sn = ?
                ORDER BY timestamp DESC
            """, (product_sn,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_operator(self, operator_id: str, date: str = None) -> List[dict]:
        """按操作员查询"""
        with self._lock:
            cursor = self._conn.cursor()
            if date:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE operator_id = ? AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC
                """, (operator_id, date))
            else:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE operator_id = ?
                    ORDER BY timestamp DESC LIMIT 100
                """, (operator_id,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_equipment(self, equipment_id: str, date: str = None) -> List[dict]:
        """按设备编号查询"""
        with self._lock:
            cursor = self._conn.cursor()
            if date:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE equipment_id = ? AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC
                """, (equipment_id, date))
            else:
                cursor.execute("""
                    SELECT * FROM detection_records
                    WHERE equipment_id = ?
                    ORDER BY timestamp DESC LIMIT 100
                """, (equipment_id,))
            return [dict(row) for row in cursor.fetchall()]

    def query_by_timerange(self, start: str, end: str) -> List[dict]:
        """按时间范围查询"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_records
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start, end))
            return [dict(row) for row in cursor.fetchall()]

    def get_defect_details(self, record_id: int) -> List[dict]:
        """获取检测记录的缺陷明细"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM defect_details WHERE record_id = ?
            """, (record_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self, date: str = None, operator_id: str = "",
                       equipment_id: str = "") -> dict:
        """
        获取统计信息

        Args:
            date: 日期 (YYYY-MM-DD)，None=全部
            operator_id: 操作员工号
            equipment_id: 设备编号
        """
        with self._lock:
            cursor = self._conn.cursor()

            conditions = []
            params = []
            if date:
                conditions.append("DATE(timestamp) = ?")
                params.append(date)
            if operator_id:
                conditions.append("operator_id = ?")
                params.append(operator_id)
            if equipment_id:
                conditions.append("equipment_id = ?")
                params.append(equipment_id)

            where = " AND ".join(conditions) if conditions else "1=1"

            # 基本统计
            cursor.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result='OK' THEN 1 ELSE 0 END) as ok_count,
                       SUM(CASE WHEN result='NG' THEN 1 ELSE 0 END) as ng_count,
                       AVG(inference_time_ms) as avg_inference_ms
                FROM detection_records WHERE {where}
            """, params)
            row = cursor.fetchone()

            # 缺陷类别统计
            cursor.execute(f"""
                SELECT dd.class_name, COUNT(*) as count
                FROM defect_details dd
                JOIN detection_records dr ON dd.record_id = dr.id
                WHERE dd.suppressed = 0 AND {where.replace('timestamp', 'dr.timestamp')}
                GROUP BY dd.class_name ORDER BY count DESC
            """, params)

            defect_stats = {row['class_name']: row['count'] for row in cursor.fetchall()}

            total = row['total'] or 0
            ok_count = row['ok_count'] or 0
            ng_count = row['ng_count'] or 0

            return {
                'total': total,
                'ok_count': ok_count,
                'ng_count': ng_count,
                'pass_rate': (ok_count / total * 100) if total > 0 else 0,
                'avg_inference_ms': round(row['avg_inference_ms'] or 0, 1),
                'defect_class_stats': defect_stats,
            }

    def save_shift_stats(self, shift_date: str, shift_name: str,
                         operator_id: str = "", equipment_id: str = "",
                         line_profile: str = ""):
        """保存班次统计"""
        stats = self.get_statistics(date=shift_date, operator_id=operator_id,
                                    equipment_id=equipment_id)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO shift_stats
                (shift_date, shift_name, operator_id, equipment_id, line_profile,
                 total_count, ok_count, ng_count, defect_class_stats, avg_inference_ms,
                 start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                shift_date, shift_name, operator_id, equipment_id, line_profile,
                stats['total'], stats['ok_count'], stats['ng_count'],
                json.dumps(stats['defect_class_stats'], ensure_ascii=False),
                stats['avg_inference_ms'],
                datetime.now().isoformat(), datetime.now().isoformat(),
            ))
            self._conn.commit()

    def get_recent_records(self, limit: int = 100) -> List[dict]:
        """获取最近检测记录"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_records
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            logger.info("追溯数据库已关闭")
