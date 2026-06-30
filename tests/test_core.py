# -*- coding: utf-8 -*-
"""
PCB缺陷检测系统 - 单元测试
覆盖：产线管理、误报屏蔽、缺陷追溯、权限管理、设备监控
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.production_manager import ProductionManager, LineProfile
from utils.false_alarm_suppressor import FalseAlarmSuppressor, SuppressionRule
from utils.traceability import TraceabilityDB
from utils.auth import AuthManager, Role, User
from utils.equipment_monitor import EquipmentMonitor, HealthStatus
from utils.alarm import AlarmManager, AlarmLevel, AlarmConfig
from utils.config_loader import Config, load_config, merge_config


class TestLineProfile(unittest.TestCase):
    """产线配置档案测试"""

    def test_create_profile(self):
        profile = LineProfile(
            name="test_line",
            model_path="./models/test.pt",
            conf_threshold=0.3,
            class_names=["open", "short"],
        )
        self.assertEqual(profile.name, "test_line")
        self.assertEqual(profile.conf_threshold, 0.3)
        self.assertEqual(len(profile.class_names), 2)

    def test_to_dict_and_back(self):
        profile = LineProfile(name="line1", model_path="m.pt", conf_threshold=0.5)
        d = profile.to_dict()
        restored = LineProfile.from_dict(d)
        self.assertEqual(restored.name, "line1")
        self.assertEqual(restored.model_path, "m.pt")
        self.assertEqual(restored.conf_threshold, 0.5)


class TestProductionManager(unittest.TestCase):
    """产线管理器测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pm = ProductionManager(profiles_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_profile(self):
        profile = LineProfile(name="line_A", model_path="a.pt", conf_threshold=0.3)
        self.pm.save_profile(profile)

        loaded = self.pm.get_profile("line_A")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "line_A")
        self.assertEqual(loaded.conf_threshold, 0.3)

    def test_list_profiles(self):
        for name in ["line1", "line2", "line3"]:
            self.pm.save_profile(LineProfile(name=name))

        profiles = self.pm.list_profiles()
        self.assertEqual(len(profiles), 3)
        self.assertIn("line1", profiles)

    def test_switch_line(self):
        self.pm.save_profile(LineProfile(name="line1", conf_threshold=0.2))
        self.pm.save_profile(LineProfile(name="line2", conf_threshold=0.5))

        switched = []
        self.pm.on_line_switched(lambda p: switched.append(p.name))

        result = self.pm.switch_line("line2")
        self.assertTrue(result)
        self.assertEqual(self.pm.current_profile_name, "line2")
        self.assertEqual(switched, ["line2"])

    def test_switch_nonexistent_line(self):
        result = self.pm.switch_line("nonexistent")
        self.assertFalse(result)

    def test_delete_profile(self):
        self.pm.save_profile(LineProfile(name="to_delete"))
        self.assertTrue(self.pm.delete_profile("to_delete"))
        self.assertIsNone(self.pm.get_profile("to_delete"))

    def test_mode_switch(self):
        self.pm.set_mode(ProductionManager.MODE_DEBUG)
        self.assertTrue(self.pm.is_debug_mode)
        self.assertFalse(self.pm.is_production_mode)

        self.pm.set_mode(ProductionManager.MODE_PRODUCTION)
        self.assertTrue(self.pm.is_production_mode)

    def test_shift_info(self):
        self.pm.set_shift_info(operator_id="O001", equipment_id="EQ-001", shift_name="早班")
        info = self.pm.shift_info
        self.assertEqual(info['operator_id'], "O001")
        self.assertEqual(info['equipment_id'], "EQ-001")


class TestSuppressionRule(unittest.TestCase):
    """误报屏蔽规则测试"""

    def test_rect_rule_matches(self):
        rule = SuppressionRule(
            name="test_rect",
            rule_type="rect",
            region={"x1": 100, "y1": 100, "x2": 300, "y2": 300},
        )
        # 中心点在区域内
        self.assertTrue(rule.matches([150, 150, 250, 250], "open", 0.8))
        # 中心点在区域外
        self.assertFalse(rule.matches([400, 400, 500, 500], "open", 0.8))

    def test_class_filter(self):
        rule = SuppressionRule(
            name="class_filter",
            rule_type="rect",
            region={"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
            class_names=["open", "short"],
        )
        self.assertTrue(rule.matches([100, 100, 200, 200], "open", 0.8))
        self.assertFalse(rule.matches([100, 100, 200, 200], "copper", 0.8))

    def test_confidence_filter(self):
        rule = SuppressionRule(
            name="conf_filter",
            rule_type="rect",
            region={"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
            min_confidence=0.5,
        )
        self.assertTrue(rule.matches([100, 100, 200, 200], "open", 0.3))
        self.assertFalse(rule.matches([100, 100, 200, 200], "open", 0.8))

    def test_disabled_rule(self):
        rule = SuppressionRule(
            name="disabled",
            rule_type="rect",
            region={"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
            enabled=False,
        )
        self.assertFalse(rule.matches([100, 100, 200, 200], "open", 0.8))

    def test_polygon_rule(self):
        rule = SuppressionRule(
            name="polygon",
            rule_type="polygon",
            region={"points": [[0, 0], [200, 0], [200, 200], [0, 200]]},
        )
        self.assertTrue(rule.matches([50, 50, 100, 100], "open", 0.8))
        self.assertFalse(rule.matches([300, 300, 400, 400], "open", 0.8))


class TestFalseAlarmSuppressor(unittest.TestCase):
    """误报屏蔽管理器测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.suppressor = FalseAlarmSuppressor(rules_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_list_rules(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 100, 100)
        rules = self.suppressor.list_rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].name, "zone1")

    def test_filter_detections(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 200, 200)
        detections = [
            {'bbox': [50, 50, 100, 100], 'class_name': 'open', 'confidence': 0.8},
            {'bbox': [300, 300, 400, 400], 'class_name': 'short', 'confidence': 0.7},
        ]
        valid, suppressed = self.suppressor.filter_detections(detections)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(valid[0]['class_name'], 'short')

    def test_toggle_rule(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 200, 200)
        self.suppressor.toggle_rule("zone1")
        rules = self.suppressor.list_rules()
        self.assertFalse(rules[0].enabled)

    def test_delete_rule(self):
        self.suppressor.add_rect_rule("zone1", 0, 0, 100, 100)
        self.assertTrue(self.suppressor.remove_rule("zone1"))
        self.assertEqual(len(self.suppressor.list_rules()), 0)


class TestTraceabilityDB(unittest.TestCase):
    """缺陷追溯数据库测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = TraceabilityDB(db_path=os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_record(self):
        rid = self.db.add_record(
            product_sn="PCB001",
            operator_id="O001",
            equipment_id="EQ-001",
            result="OK",
            num_defects=0,
        )
        self.assertGreater(rid, 0)

    def test_query_by_sn(self):
        self.db.add_record(product_sn="PCB001", result="OK")
        self.db.add_record(product_sn="PCB002", result="NG", num_defects=2)

        records = self.db.query_by_sn("PCB001")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['result'], 'OK')

    def test_query_by_operator(self):
        self.db.add_record(operator_id="O001", result="OK")
        self.db.add_record(operator_id="O002", result="NG")

        records = self.db.query_by_operator("O001")
        self.assertEqual(len(records), 1)

    def test_review_record(self):
        rid = self.db.add_record(product_sn="PCB001", result="NG", num_defects=1)
        self.db.review_record(rid, "confirmed", "reviewer1")

        unreviewed = self.db.get_unreviewed()
        self.assertEqual(len(unreviewed), 0)

    def test_batch_review(self):
        ids = []
        for i in range(5):
            rid = self.db.add_record(product_sn=f"PCB{i:03d}", result="NG", num_defects=1)
            ids.append(rid)

        self.db.batch_review(ids[:3], "confirmed", "reviewer1")
        unreviewed = self.db.get_unreviewed()
        self.assertEqual(len(unreviewed), 2)

    def test_get_statistics(self):
        self.db.add_record(result="OK")
        self.db.add_record(result="OK")
        self.db.add_record(result="NG", num_defects=1)

        stats = self.db.get_statistics()
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['ok_count'], 2)
        self.assertEqual(stats['ng_count'], 1)
        self.assertAlmostEqual(stats['pass_rate'], 66.7, places=0)

    def test_defect_details(self):
        defects = [
            {'class_name': 'open', 'confidence': 0.9, 'bbox': [10, 20, 30, 40]},
            {'class_name': 'short', 'confidence': 0.7, 'bbox': [50, 60, 70, 80]},
        ]
        rid = self.db.add_record(result="NG", num_defects=2, defects=defects)
        details = self.db.get_defect_details(rid)
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]['class_name'], 'open')


class TestAuthManager(unittest.TestCase):
    """权限管理测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.auth = AuthManager(db_path=os.path.join(self.tmpdir, "users.json"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_users_created(self):
        users = self.auth.list_users()
        self.assertGreaterEqual(len(users), 3)

    def test_login_success(self):
        result = self.auth.login("admin", "admin123")
        self.assertTrue(result)
        self.assertTrue(self.auth.is_logged_in)
        self.assertEqual(self.auth.current_user.role, Role.ADMIN)

    def test_login_wrong_password(self):
        result = self.auth.login("admin", "wrong")
        self.assertFalse(result)

    def test_login_nonexistent_user(self):
        result = self.auth.login("nobody", "pass")
        self.assertFalse(result)

    def test_logout(self):
        self.auth.login("admin", "admin123")
        self.auth.logout()
        self.assertFalse(self.auth.is_logged_in)

    def test_permission_check(self):
        self.auth.login("admin", "admin123")
        self.assertTrue(self.auth.check_permission('manage_users'))
        self.assertTrue(self.auth.check_permission('system_config'))

    def test_operator_permissions(self):
        self.auth.login("operator", "op123")
        self.assertTrue(self.auth.check_permission('start_stop'))
        self.assertFalse(self.auth.check_permission('adjust_threshold'))
        self.assertFalse(self.auth.check_permission('manage_users'))

    def test_engineer_permissions(self):
        self.auth.login("engineer", "eng123")
        self.assertTrue(self.auth.check_permission('adjust_threshold'))
        self.assertTrue(self.auth.check_permission('view_debug_info'))
        self.assertFalse(self.auth.check_permission('manage_users'))

    def test_add_user(self):
        self.auth.login("admin", "admin123")
        result = self.auth.add_user("newuser", Role.OPERATOR, "pass1234", "New User")
        self.assertTrue(result)
        self.auth.logout()
        self.assertTrue(self.auth.login("newuser", "pass1234"))

    def test_change_password(self):
        self.auth.login("operator", "op123")
        result = self.auth.change_password("operator", "op123", "newpass1")
        self.assertTrue(result)
        self.auth.logout()
        self.assertTrue(self.auth.login("operator", "newpass1"))


class TestEquipmentMonitor(unittest.TestCase):
    """设备健康度监控测试"""

    def setUp(self):
        self.monitor = EquipmentMonitor()

    def test_initial_status(self):
        self.assertEqual(self.monitor.overall_status, HealthStatus.GOOD)

    def test_record_frame_updates_brightness(self):
        import numpy as np
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.monitor.record_frame(frame, inference_time_ms=50)
        details = self.monitor.get_details()
        self.assertGreater(details['light_source']['avg_brightness'], 0)

    def test_low_brightness_triggers_critical(self):
        import numpy as np
        # 连续记录低亮度帧
        dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 20
        for _ in range(20):
            self.monitor.record_frame(dark_frame, inference_time_ms=50)
        self.assertEqual(self.monitor.health_report['light_source'], HealthStatus.CRITICAL)

    def test_high_inference_time_triggers_warning(self):
        import numpy as np
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        for _ in range(20):
            self.monitor.record_frame(frame, inference_time_ms=250)
        self.assertEqual(self.monitor.health_report['inference'], HealthStatus.CRITICAL)

    def test_reset(self):
        import numpy as np
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.monitor.record_frame(frame, 50)
        self.monitor.reset()
        details = self.monitor.get_details()
        self.assertEqual(details['camera']['total_frames'], 0)


class TestAlarmManager(unittest.TestCase):
    """报警管理器测试"""

    def test_alarm_callback(self):
        triggered = []
        alarm = AlarmManager()
        alarm.on_alarm(lambda l, c, m: triggered.append((l, c, m)))
        alarm.trigger(AlarmLevel.WARNING, 2, "test")
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0][0], AlarmLevel.WARNING)

    def test_cooldown(self):
        triggered = []
        alarm = AlarmManager(config=AlarmConfig(cooldown_ms=5000))
        alarm.on_alarm(lambda l, c, m: triggered.append(1))
        alarm.trigger(AlarmLevel.WARNING, 1, "test1")
        alarm.trigger(AlarmLevel.WARNING, 1, "test2")  # 应被冷却
        self.assertEqual(len(triggered), 1)


class TestConfig(unittest.TestCase):
    """配置加载测试"""

    def test_config_dict_access(self):
        config = Config({'model': {'name': 'yolo11s'}, 'epochs': 200})
        self.assertEqual(config.model.name, 'yolo11s')
        self.assertEqual(config.epochs, 200)

    def test_config_get(self):
        config = Config({'key': 'value'})
        self.assertEqual(config.get('key'), 'value')
        self.assertIsNone(config.get('missing'))
        self.assertEqual(config.get('missing', 'default'), 'default')

    def test_config_to_dict(self):
        original = {'model': {'name': 'test'}, 'epochs': 100}
        config = Config(original)
        result = config.to_dict()
        self.assertEqual(result, original)

    def test_merge_config(self):
        base = Config({'model': {'name': 'a', 'size': 640}, 'epochs': 100})
        override = {'model': {'name': 'b'}, 'epochs': 200}
        merged = merge_config(base, override)
        self.assertEqual(merged.model.name, 'b')
        self.assertEqual(merged.model.size, 640)
        self.assertEqual(merged.epochs, 200)


if __name__ == '__main__':
    unittest.main()
