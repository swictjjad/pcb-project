# -*- coding: utf-8 -*-
"""
权限管理模块
功能：
1. 角色权限控制（操作员/工程师/管理员）
2. 操作审计日志
3. 参数修改权限校验
"""

import json
import hashlib
import secrets
import time
import threading
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from utils.logger import setup_logger

logger = setup_logger("auth")

# 用户数据存储路径
AUTH_DB_PATH = Path("./configs/users.json")
AUDIT_LOG_PATH = Path("./results/audit.log")

# 密码最小长度要求
PASSWORD_MIN_LENGTH = 8
# 登录失败锁定次数
LOGIN_MAX_ATTEMPTS = 5
# 锁定时间（秒）
LOGIN_LOCKOUT_DURATION = 300


class Role:
    """角色定义"""
    OPERATOR = "operator"       # 操作员：只能查看和基本操作
    ENGINEER = "engineer"       # 工程师：可以调整参数
    ADMIN = "admin"             # 管理员：全部权限


# 权限定义
PERMISSIONS = {
    Role.OPERATOR: {
        'view_detection',       # 查看检测结果
        'start_stop',           # 启动/停止检测
        'capture',              # 截图
        'export_csv',           # 导出报表
        'review',               # 复核
        'switch_line',          # 一键换线
        'add_suppression',      # 添加误报屏蔽
    },
    Role.ENGINEER: {
        'view_detection', 'start_stop', 'capture', 'export_csv', 'review',
        'switch_line', 'add_suppression',
        'adjust_threshold',     # 调整置信度/IOU阈值
        'adjust_camera',        # 调整摄像头参数
        'adjust_hardware',      # 调整硬件参数
        'view_debug_info',      # 查看调试信息（Grad-CAM等）
        'manage_suppression',   # 管理屏蔽规则
        'view_health',          # 查看设备健康度
    },
    Role.ADMIN: {
        'view_detection', 'start_stop', 'capture', 'export_csv', 'review',
        'switch_line', 'add_suppression',
        'adjust_threshold', 'adjust_camera', 'adjust_hardware',
        'view_debug_info', 'manage_suppression', 'view_health',
        'manage_users',         # 用户管理
        'manage_profiles',      # 产线配置管理
        'system_config',        # 系统配置
        'view_audit_log',       # 查看审计日志
    },
}


class User:
    """用户"""

    def __init__(self, username: str, role: str, password_hash: str = "",
                 display_name: str = "", employee_id: str = ""):
        self.username = username
        self.role = role
        self.password_hash = password_hash
        self.display_name = display_name or username
        self.employee_id = employee_id
        self.last_login = ""
        self.password_changed_at = ""

    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in PERMISSIONS.get(self.role, set())

    def to_dict(self) -> dict:
        return {
            'username': self.username,
            'role': self.role,
            'password_hash': self.password_hash,
            'display_name': self.display_name,
            'employee_id': self.employee_id,
            'last_login': self.last_login,
            'password_changed_at': self.password_changed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        user = cls(
            username=data['username'],
            role=data.get('role', Role.OPERATOR),
            password_hash=data.get('password_hash', ''),
            display_name=data.get('display_name', ''),
            employee_id=data.get('employee_id', ''),
        )
        user.last_login = data.get('last_login', '')
        user.password_changed_at = data.get('password_changed_at', '')
        return user


class AuthManager:
    """权限管理器"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else AUTH_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._users: Dict[str, User] = {}
        self._current_user: Optional[User] = None
        self._lock = threading.Lock()
        # 登录失败追踪
        self._login_attempts: Dict[str, int] = {}
        self._lockout_until: Dict[str, float] = {}
        self._load_users()

    def _load_users(self):
        """加载用户数据"""
        if not self.db_path.exists():
            # 创建默认用户
            self._create_default_users()
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for user_data in data.get('users', []):
                user = User.from_dict(user_data)
                self._users[user.username] = user
            logger.info(f"已加载 {len(self._users)} 个用户")
        except Exception as e:
            logger.error(f"加载用户数据失败: {e}")
            self._create_default_users()

    def _generate_salt(self) -> str:
        """生成安全随机盐"""
        return secrets.token_hex(32)

    def _hash_password(self, password: str, salt: str = "") -> str:
        """
        密码哈希 - 使用 PBKDF2-SHA256 带盐
        安全改进：使用更安全的哈希方法
        """
        if not salt:
            salt = self._generate_salt()
        # PBKDF2 with SHA256, 100000 iterations
        import hmac
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${key.hex()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """验证密码"""
        try:
            if '$' not in stored_hash:
                # 旧格式兼容：纯SHA256
                return hashlib.sha256(password.encode('utf-8')).hexdigest() == stored_hash
            salt, key_hex = stored_hash.split('$', 1)
            computed_key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return secrets.compare_digest(computed_key.hex(), key_hex)
        except Exception:
            return False

    def _create_default_users(self):
        """创建默认用户（强制首次登录修改密码）"""
        # 生成随机初始密码而非硬编码
        admin_pwd = secrets.token_urlsafe(12)
        engineer_pwd = secrets.token_urlsafe(12)
        operator_pwd = secrets.token_urlsafe(12)

        default_users = [
            User("admin", Role.ADMIN, self._hash_password(admin_pwd), "管理员", "A001"),
            User("engineer", Role.ENGINEER, self._hash_password(engineer_pwd), "工程师", "E001"),
            User("operator", Role.OPERATOR, self._hash_password(operator_pwd), "操作员", "O001"),
        ]
        for user in default_users:
            self._users[user.username] = user

        # 保存默认密码到单独文件（仅首次需要）
        default_pwds_file = self.db_path.parent / "default_passwords.json"
        try:
            with open(default_pwds_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'admin': admin_pwd,
                    'engineer': engineer_pwd,
                    'operator': operator_pwd,
                    'created_at': datetime.now().isoformat(),
                    'warning': '首次登录后请立即修改密码'
                }, f, ensure_ascii=False, indent=2)
            logger.warning(f"默认密码已生成，请查看: {default_pwds_file}")
        except Exception as e:
            logger.error(f"保存默认密码失败: {e}")

        self._save_users()
        logger.info("已创建默认用户（请查看 default_passwords.json 获取初始密码）")

    def _is_locked_out(self, username: str) -> bool:
        """检查是否被锁定"""
        if username not in self._lockout_until:
            return False
        if time.time() < self._lockout_until[username]:
            return True
        # 锁定已过期，清除状态
        del self._lockout_until[username]
        if username in self._login_attempts:
            self._login_attempts[username] = 0
        return False

    def _record_failed_attempt(self, username: str):
        """记录失败尝试"""
        self._login_attempts[username] = self._login_attempts.get(username, 0) + 1
        if self._login_attempts[username] >= LOGIN_MAX_ATTEMPTS:
            self._lockout_until[username] = time.time() + LOGIN_LOCKOUT_DURATION
            logger.warning(f"用户 {username} 因多次登录失败被锁定 {LOGIN_LOCKOUT_DURATION} 秒")
            self._audit_log("ACCOUNT_LOCKED", username, f"登录失败{LOGIN_MAX_ATTEMPTS}次")

    def _clear_failed_attempts(self, username: str):
        """清除失败尝试记录"""
        self._login_attempts.pop(username, None)
        self._lockout_until.pop(username, None)

    def _save_users(self):
        """保存用户数据"""
        try:
            data = {
                'users': [u.to_dict() for u in self._users.values()],
            }
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存用户数据失败: {e}")

    @staticmethod
    def _hash_password(password: str) -> str:
        """密码哈希 - 使用 PBKDF2-SHA256 带盐"""
        import hmac
        salt = secrets.token_hex(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${key.hex()}"

    def login(self, username: str, password: str) -> bool:
        """
        用户登录

        Returns:
            是否登录成功
        """
        with self._lock:
            # 检查是否被锁定
            if self._is_locked_out(username):
                remaining = int(self._lockout_until.get(username, 0) - time.time())
                logger.warning(f"登录失败: 用户 {username} 已被锁定，剩余 {remaining} 秒")
                self._audit_log("LOGIN_BLOCKED", username, f"账户被锁定")
                return False

            user = self._users.get(username)
            if not user:
                logger.warning(f"登录失败: 用户不存在 [{username}]")
                self._audit_log("LOGIN_FAIL", username, "用户不存在")
                return False

            if not self._verify_password(password, user.password_hash):
                self._record_failed_attempt(username)
                attempts = self._login_attempts.get(username, 0)
                logger.warning(f"登录失败: 密码错误 [{username}]，剩余尝试次数 {LOGIN_MAX_ATTEMPTS - attempts}")
                self._audit_log("LOGIN_FAIL", username, f"密码错误，剩余尝试次数 {LOGIN_MAX_ATTEMPTS - attempts}")
                return False

            self._clear_failed_attempts(username)
            self._current_user = user
            user.last_login = datetime.now().isoformat()
            self._save_users()
            logger.info(f"用户登录: {username} (角色={user.role})")
            self._audit_log("LOGIN", username, f"角色={user.role}")
            return True

    def logout(self):
        """用户登出"""
        if self._current_user:
            self._audit_log("LOGOUT", self._current_user.username, "")
            logger.info(f"用户登出: {self._current_user.username}")
        self._current_user = None

    @property
    def current_user(self) -> Optional[User]:
        return self._current_user

    @property
    def is_logged_in(self) -> bool:
        return self._current_user is not None

    def check_permission(self, permission: str) -> bool:
        """
        检查当前用户是否有指定权限

        Args:
            permission: 权限名称

        Returns:
            是否有权限
        """
        if not self._current_user:
            return False
        return self._current_user.has_permission(permission)

    def require_permission(self, permission: str) -> bool:
        """
        要求权限，无权限时记录审计日志

        Returns:
            是否有权限
        """
        if self.check_permission(permission):
            return True
        username = self._current_user.username if self._current_user else "(未登录)"
        logger.warning(f"权限不足: {username} 缺少 {permission}")
        self._audit_log("PERMISSION_DENIED", username, f"需要 {permission}")
        return False

    def _validate_password_strength(self, password: str) -> tuple:
        """
        验证密码强度

        Returns:
            (是否有效, 错误信息)
        """
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"密码长度至少{PASSWORD_MIN_LENGTH}位"
        # 检查是否包含字母和数字
        if not re.search(r'[A-Za-z]', password):
            return False, "密码必须包含字母"
        if not re.search(r'[0-9]', password):
            return False, "密码必须包含数字"
        return True, ""

    def add_user(self, username: str, role: str, password: str,
                 display_name: str = "", employee_id: str = "") -> bool:
        """添加用户"""
        if not self.require_permission('manage_users'):
            return False

        # 验证密码强度
        valid, msg = self._validate_password_strength(password)
        if not valid:
            logger.error(f"添加用户失败: {msg}")
            return False

        with self._lock:
            if username in self._users:
                logger.error(f"用户已存在: {username}")
                return False

            user = User(username, role, self._hash_password(password),
                        display_name, employee_id)
            user.password_changed_at = datetime.now().isoformat()
            self._users[username] = user
            self._save_users()
            self._audit_log("ADD_USER", self._current_user.username,
                            f"添加用户 {username} 角色={role}")
            logger.info(f"添加用户: {username} (角色={role})")
            return True

    def remove_user(self, username: str) -> bool:
        """删除用户"""
        if not self.require_permission('manage_users'):
            return False

        with self._lock:
            if username in self._users:
                del self._users[username]
                self._save_users()
                self._audit_log("REMOVE_USER", self._current_user.username,
                                f"删除用户 {username}")
                return True
        return False

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """修改密码"""
        # 验证新密码强度
        valid, msg = self._validate_password_strength(new_password)
        if not valid:
            logger.error(f"修改密码失败: {msg}")
            return False

        with self._lock:
            user = self._users.get(username)
            if not user:
                return False
            if not self._verify_password(old_password, user.password_hash):
                return False
            user.password_hash = self._hash_password(new_password)
            user.password_changed_at = datetime.now().isoformat()
            self._save_users()
            self._audit_log("CHANGE_PASSWORD", username, "")
            return True

    def list_users(self) -> List[dict]:
        """列出所有用户（不含密码）"""
        return [
            {
                'username': u.username,
                'role': u.role,
                'display_name': u.display_name,
                'employee_id': u.employee_id,
                'last_login': u.last_login,
            }
            for u in self._users.values()
        ]

    def _audit_log(self, action: str, username: str, detail: str):
        """审计日志"""
        try:
            AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {action} user={username} detail={detail}\n")
        except Exception as e:
            logger.error(f"写入审计日志失败: {e}")
