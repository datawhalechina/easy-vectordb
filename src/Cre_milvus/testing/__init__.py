try:
    from .locust_test import MilvusLoadTest
    # from .performance_monitor import PerformanceMonitor
    from .test_data_generator import TestDataGenerator
except ImportError:
    try:
        from locust_test import MilvusLoadTest
        # from performance_monitor import PerformanceMonitor
        from test_data_generator import TestDataGenerator
    except ImportError:
        class MilvusLoadTest:
            def __init__(self, config):
                self.config = config
            def start_test(self, **kwargs):
                return {"status": "error", "message": "Locust测试模块不可用"}
        
        # class PerformanceMonitor:
        #     def __init__(self):
        #         self.is_monitoring = False
        #     def start_monitoring(self):
        #         pass
        #     def stop_monitoring(self):
        #         pass
        #     def get_current_metrics(self):
        #         return {}
        #     def get_historical_data(self, metric, duration):
        #         return []
        
        class TestDataGenerator:
            def __init__(self, host="localhost", port="19530"):
                pass
            def create_test_collection(self, **kwargs):
                return {"status": "error", "message": "测试数据生成器不可用"}

# __all__ = ['MilvusLoadTest', 'PerformanceMonitor', 'TestDataGenerator']
__all__ = ['MilvusLoadTest', 'TestDataGenerator']