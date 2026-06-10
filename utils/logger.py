# -*- coding: utf-8 -*-
"""
日志模块
统一日志格式和输出
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "pcb_defect",
                 log_level: str = "INFO",
                 log_file: str = None,
                 console_output: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径，None则不写入文件
        console_output: 是否输出到控制台
        
    Returns:
        logging.Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TqdmLoggingHandler(logging.Handler):
    """兼容tqdm进度条的日志处理器"""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def get_logger(name: str = "pcb_defect") -> logging.Logger:
    """获取已配置的日志器"""
    return logging.getLogger(name)


if __name__ == "__main__":
    logger = setup_logger(log_file="../results/test.log")
    logger.info("日志模块测试")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")
