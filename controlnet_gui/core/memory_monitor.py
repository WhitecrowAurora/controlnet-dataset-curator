"""
内存监控模块
用于监控应用程序的内存使用情况
"""

import psutil
import os


class MemoryMonitor:
    """内存使用监控器"""

    @staticmethod
    def get_current_memory_mb() -> float:
        """
        获取当前进程的内存使用量（MB）

        Returns:
            内存使用量（MB）
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except Exception:
            return 0.0

    @staticmethod
    def get_memory_percent() -> float:
        """
        获取当前进程的内存使用百分比

        Returns:
            内存使用百分比
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_percent()
        except Exception:
            return 0.0

    @staticmethod
    def get_system_memory_info() -> dict:
        """
        获取系统内存信息

        Returns:
            包含总内存、可用内存、使用百分比的字典
        """
        try:
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total / 1024 / 1024,
                'available_mb': mem.available / 1024 / 1024,
                'used_mb': mem.used / 1024 / 1024,
                'percent': mem.percent
            }
        except Exception:
            return {
                'total_mb': 0,
                'available_mb': 0,
                'used_mb': 0,
                'percent': 0
            }

    @staticmethod
    def format_memory_size(mb: float) -> str:
        """
        格式化内存大小显示

        Args:
            mb: 内存大小（MB）

        Returns:
            格式化后的字符串
        """
        if mb < 1024:
            return f"{mb:.1f} MB"
        else:
            return f"{mb / 1024:.2f} GB"
