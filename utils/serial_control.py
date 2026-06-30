# -*- coding: utf-8 -*-
"""
串口通信控制模块
用于与Arduino进行通信，控制舵机和传送带
"""

import time
import threading
from typing import Optional, Callable

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("serial_control")


class ArduinoController:
    """Arduino控制器"""
    
    # 通信协议命令
    CMD_SERVO_OK = b'SOK\n'      # 舵机转到合格位置
    CMD_SERVO_NG = b'SNG\n'      # 舵机转到不合格位置
    CMD_SERVO_RESET = b'SRS\n'   # 舵机复位
    CMD_MOTOR_START = b'MST\n'   # 传送带启动
    CMD_MOTOR_STOP = b'MSP\n'    # 传送带停止
    CMD_SENSOR_READ = b'READ\n'  # 读取传感器状态
    
    def __init__(self, port: str = None, baudrate: int = 115200, timeout: float = 1.0):
        """
        初始化Arduino控制器

        Args:
            port: 串口名称，None则自动查找
            baudrate: 波特率
            timeout: 超时时间(秒)
        """
        # 如果未指定端口，自动查找可用端口
        if port is None:
            available_ports = self.list_ports()
            if available_ports:
                port = available_ports[0][0]  # 使用第一个可用端口
                logger.info(f"自动选择串口: {port}")
            else:
                # 没有可用串口，不硬编码 COM3，而是标记为无硬件模式
                logger.warning("未检测到可用串口，将在 connect() 时自动切换到模拟模式")
                port = "__NO_SERIAL__"  # 特殊标记，表示无硬件

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        self._lock = threading.Lock()
        self._callback: Optional[Callable] = None
        self._read_thread: Optional[threading.Thread] = None
        self._running = False
        
    @staticmethod
    def list_ports():
        """列出可用串口"""
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial未安装，无法列出串口")
            return []
        
        ports = serial.tools.list_ports.comports()
        return [(p.device, p.description) for p in ports]
    
    def connect(self) -> bool:
        """
        连接Arduino

        Returns:
            是否连接成功
        """
        # 无串口标记 → 直接返回 False，调用方会自动切换到 Mock 模式
        if self.port == "__NO_SERIAL__":
            logger.warning("无可用串口，跳过物理连接")
            self.connected = False
            return False

        if not SERIAL_AVAILABLE:
            logger.error("pyserial未安装，请运行: pip install pyserial")
            self.connected = False
            return False

        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )

            # 等待Arduino复位
            time.sleep(2)

            self.connected = True
            logger.info(f"串口已连接: {self.port} @ {self.baudrate}bps")

            # 启动读取线程
            self._running = True
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            return True

        except serial.SerialException as e:
            logger.error(f"串口连接失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2)
        
        if self.serial and self.serial.is_open:
            self.serial.close()
        
        self.connected = False
        logger.info("串口已断开")
    
    def _read_loop(self):
        """后台读取线程"""
        while self._running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        logger.debug(f"收到数据: {line}")
                        if self._callback:
                            self._callback(line)
            except Exception as e:
                logger.error(f"读取错误: {e}")
            
            time.sleep(0.01)
    
    def set_callback(self, callback: Callable):
        """
        设置数据接收回调函数
        
        Args:
            callback: 回调函数，接收字符串参数
        """
        self._callback = callback
    
    def _send_command(self, command: bytes) -> bool:
        """
        发送命令
        
        Args:
            command: 命令字节
            
        Returns:
            是否发送成功
        """
        if not self.connected or not self.serial:
            logger.warning("串口未连接")
            return False
        
        with self._lock:
            try:
                self.serial.write(command)
                self.serial.flush()
                logger.debug(f"发送命令: {command.strip()}")
                return True
            except Exception as e:
                logger.error(f"发送失败: {e}")
                return False
    
    def servo_ok(self) -> bool:
        """舵机转到合格位置"""
        logger.info("舵机 -> 合格位置")
        return self._send_command(self.CMD_SERVO_OK)
    
    def servo_ng(self) -> bool:
        """舵机转到不合格位置"""
        logger.info("舵机 -> 不合格位置")
        return self._send_command(self.CMD_SERVO_NG)
    
    def servo_reset(self) -> bool:
        """舵机复位"""
        logger.info("舵机复位")
        return self._send_command(self.CMD_SERVO_RESET)
    
    def motor_start(self) -> bool:
        """启动传送带"""
        logger.info("传送带启动")
        return self._send_command(self.CMD_MOTOR_START)
    
    def motor_stop(self) -> bool:
        """停止传送带"""
        logger.info("传送带停止")
        return self._send_command(self.CMD_MOTOR_STOP)
    
    def read_sensor(self) -> bool:
        """读取传感器状态"""
        return self._send_command(self.CMD_SENSOR_READ)
    
    def emergency_stop(self):
        """急停：立即停止传送带和舵机"""
        logger.warning("⚠ 急停！停止传送带和舵机")
        self.motor_stop()
        self.servo_reset()

    def sort_pcb(self, is_ok: bool, delay_ms: int = 500):
        """
        分拣PCB板
        
        Args:
            is_ok: 是否合格
            delay_ms: 舵机保持时间(ms)
        """
        if is_ok:
            self.servo_ok()
        else:
            self.servo_ng()
        
        time.sleep(delay_ms / 1000.0)
        self.servo_reset()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class MockArduinoController:
    """模拟Arduino控制器（用于无硬件调试）"""
    
    def __init__(self, *args, **kwargs):
        self.connected = True
        logger.info("使用模拟Arduino控制器（无硬件模式）")
    
    def connect(self):
        return True
    
    def disconnect(self):
        pass
    
    def set_callback(self, callback):
        pass
    
    def _send_command(self, command):
        cmd_str = command.decode().strip()
        logger.info(f"[模拟] 发送命令: {cmd_str}")
        return True
    
    def servo_ok(self):
        logger.info("[模拟] 舵机 -> 合格位置")
        return True
    
    def servo_ng(self):
        logger.info("[模拟] 舵机 -> 不合格位置")
        return True
    
    def servo_reset(self):
        logger.info("[模拟] 舵机复位")
        return True
    
    def motor_start(self):
        logger.info("[模拟] 传送带启动")
        return True
    
    def motor_stop(self):
        logger.info("[模拟] 传送带停止")
        return True
    
    def emergency_stop(self):
        logger.warning("[模拟] ⚠ 急停！停止传送带和舵机")
        return True
    
    def sort_pcb(self, is_ok, delay_ms=500):
        status = "合格" if is_ok else "不合格"
        logger.info(f"[模拟] 分拣PCB: {status}")
        time.sleep(delay_ms / 1000.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def create_controller(config, mock=False):
    """
    创建控制器实例
    
    Args:
        config: 配置对象
        mock: 是否使用模拟模式
        
    Returns:
        ArduinoController或MockArduinoController实例
    """
    if mock:
        return MockArduinoController()
    
    return ArduinoController(
        port=config.serial.port,
        baudrate=config.serial.baudrate,
        timeout=config.serial.timeout
    )


if __name__ == "__main__":
    # 测试串口通信
    print("可用串口:")
    for port, desc in ArduinoController.list_ports():
        print(f"  {port}: {desc}")
    
    # 测试模拟模式
    controller = MockArduinoController()
    controller.sort_pcb(True)
    controller.sort_pcb(False)
