"""
性能测试模块

集成Locust性能测试和监控功能
"""

try:
    from .locust_test import MilvusLoadTest
    from .performance_monitor import PerformanceMonitor
    from .test_data_generator import TestDataGenerator
except ImportError:
    try:
        # 处理相对导入问题
        from locust_test import MilvusLoadTest
        from performance_monitor import PerformanceMonitor
        from test_data_generator import TestDataGenerator
    except ImportError:
        # 如果导入失败，提供空的替代类
        class MilvusLoadTest:
            def __init__(self, config):
                self.config = config
            def start_test(self, **kwargs):
                return {"status": "error", "message": "Locust测试模块不可用"}
        
        class PerformanceMonitor:
            def __init__(self):
                self.is_monitoring = False
            def start_monitoring(self):
                pass
            def stop_monitoring(self):
                pass
            def get_current_metrics(self):
                return {}
            def get_historical_data(self, metric, duration):
                return []
        
        class TestDataGenerator:
            def __init__(self, host="localhost", port="19530"):
                pass
            def create_test_collection(self, **kwargs):
                return {"status": "error", "message": "测试数据生成器不可用"}

__all__ = ['MilvusLoadTest', 'PerformanceMonitor', 'TestDataGenerator']