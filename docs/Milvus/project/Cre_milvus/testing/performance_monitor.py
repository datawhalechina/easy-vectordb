# """
# 性能监控模块

# 实时监控系统性能指标
# """

# import time
# import psutil
# import threading
# from typing import Dict, List, Any, Optional, Callable
# from collections import deque
# import logging

# logger = logging.getLogger(__name__)


# class PerformanceMonitor:
#     """
#     性能监控器，收集和分析系统性能指标
#     """
    
#     def __init__(self, max_history: int = 1000):
#         """
#         初始化性能监控器
        
#         参数:
#             max_history: 最大历史记录数量
#         """
#         self.max_history = max_history
#         self.is_monitoring = False
#         self.monitor_thread = None
        
#         # 性能指标历史记录
#         self.cpu_history = deque(maxlen=max_history)
#         self.memory_history = deque(maxlen=max_history)
#         self.disk_history = deque(maxlen=max_history)
#         self.network_history = deque(maxlen=max_history)
#         self.timestamp_history = deque(maxlen=max_history)
        
#         # 自定义指标
#         self.custom_metrics = {}
#         self.custom_history = {}
        
#         # 回调函数
#         self.callbacks = []
    
#     def start_monitoring(self, interval: float = 1.0):
#         """
#         开始监控
        
#         参数:
#             interval: 监控间隔（秒）
#         """
#         if self.is_monitoring:
#             logger.warning("监控已经在运行中")
#             return
        
#         self.is_monitoring = True
#         self.monitor_thread = threading.Thread(
#             target=self._monitor_loop,
#             args=(interval,),
#             daemon=True
#         )
#         self.monitor_thread.start()
#         logger.info("性能监控已启动")
    
#     def stop_monitoring(self):
#         """停止监控"""
#         self.is_monitoring = False
#         if self.monitor_thread:
#             self.monitor_thread.join(timeout=5)
#         logger.info("性能监控已停止")
    
#     def _monitor_loop(self, interval: float):
#         """监控循环"""
#         while self.is_monitoring:
#             try:
#                 timestamp = time.time()
                
#                 # 收集系统指标
#                 cpu_percent = psutil.cpu_percent(interval=0.1)
#                 memory = psutil.virtual_memory()
#                 disk = psutil.disk_usage('/')
#                 network = psutil.net_io_counters()
                
#                 # 存储指标
#                 self.cpu_history.append(cpu_percent)
#                 self.memory_history.append({
#                     'percent': memory.percent,
#                     'used': memory.used,
#                     'available': memory.available,
#                     'total': memory.total
#                 })
#                 self.disk_history.append({
#                     'percent': disk.percent,
#                     'used': disk.used,
#                     'free': disk.free,
#                     'total': disk.total
#                 })
#                 self.network_history.append({
#                     'bytes_sent': network.bytes_sent,
#                     'bytes_recv': network.bytes_recv,
#                     'packets_sent': network.packets_sent,
#                     'packets_recv': network.packets_recv
#                 })
#                 self.timestamp_history.append(timestamp)
                
#                 # 收集自定义指标
#                 for metric_name, metric_func in self.custom_metrics.items():
#                     try:
#                         value = metric_func()
#                         if metric_name not in self.custom_history:
#                             self.custom_history[metric_name] = deque(maxlen=self.max_history)
#                         self.custom_history[metric_name].append(value)
#                     except Exception as e:
#                         logger.error(f"收集自定义指标 {metric_name} 失败: {e}")
                
#                 # 执行回调函数
#                 current_metrics = self.get_current_metrics()
#                 for callback in self.callbacks:
#                     try:
#                         callback(current_metrics)
#                     except Exception as e:
#                         logger.error(f"执行监控回调失败: {e}")
                
#                 time.sleep(interval)
                
#             except Exception as e:
#                 logger.error(f"监控循环出错: {e}")
#                 time.sleep(interval)
    
#     def get_current_metrics(self) -> Dict[str, Any]:
#         """
#         获取当前性能指标
        
#         返回:
#             当前指标字典
#         """
#         if not self.cpu_history:
#             return {}
        
#         current_metrics = {
#             'timestamp': self.timestamp_history[-1] if self.timestamp_history else time.time(),
#             'cpu': {
#                 'percent': self.cpu_history[-1],
#                 'avg_1min': self._get_average(self.cpu_history, 60),
#                 'avg_5min': self._get_average(self.cpu_history, 300)
#             },
#             'memory': self.memory_history[-1] if self.memory_history else {},
#             'disk': self.disk_history[-1] if self.disk_history else {},
#             'network': self.network_history[-1] if self.network_history else {}
#         }
        
#         # 添加自定义指标
#         for metric_name, history in self.custom_history.items():
#             if history:
#                 current_metrics[metric_name] = {
#                     'current': history[-1],
#                     'avg_1min': self._get_average(history, 60),
#                     'avg_5min': self._get_average(history, 300)
#                 }
        
#         return current_metrics
    
#     def get_historical_data(self, metric_name: str, duration: int = 300) -> List[Dict[str, Any]]:
#         """
#         获取历史数据
        
#         参数:
#             metric_name: 指标名称
#             duration: 时间范围（秒）
        
#         返回:
#             历史数据列表
#         """
#         if not self.timestamp_history:
#             return []
        
#         current_time = time.time()
#         start_time = current_time - duration
        
#         # 找到时间范围内的数据
#         historical_data = []
        
#         for i, timestamp in enumerate(self.timestamp_history):
#             if timestamp >= start_time:
#                 data_point = {'timestamp': timestamp}
                
#                 if metric_name == 'cpu' and i < len(self.cpu_history):
#                     data_point['value'] = self.cpu_history[i]
#                 elif metric_name == 'memory' and i < len(self.memory_history):
#                     data_point['value'] = self.memory_history[i]['percent']
#                 elif metric_name == 'disk' and i < len(self.disk_history):
#                     data_point['value'] = self.disk_history[i]['percent']
#                 elif metric_name in self.custom_history and i < len(self.custom_history[metric_name]):
#                     data_point['value'] = self.custom_history[metric_name][i]
                
#                 if 'value' in data_point:
#                     historical_data.append(data_point)
        
#         return historical_data
    
#     def add_custom_metric(self, name: str, metric_func: Callable[[], Any]):
#         """
#         添加自定义指标
        
#         参数:
#             name: 指标名称
#             metric_func: 指标收集函数
#         """
#         self.custom_metrics[name] = metric_func
#         logger.info(f"添加自定义指标: {name}")
    
#     def remove_custom_metric(self, name: str):
#         """
#         移除自定义指标
        
#         参数:
#             name: 指标名称
#         """
#         if name in self.custom_metrics:
#             del self.custom_metrics[name]
#         if name in self.custom_history:
#             del self.custom_history[name]
#         logger.info(f"移除自定义指标: {name}")
    
#     def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
#         """
#         添加监控回调函数
        
#         参数:
#             callback: 回调函数
#         """
#         self.callbacks.append(callback)
    
#     def get_performance_summary(self) -> Dict[str, Any]:
#         """
#         获取性能摘要
        
#         返回:
#             性能摘要字典
#         """
#         if not self.cpu_history:
#             return {}
        
#         summary = {
#             'monitoring_duration': len(self.timestamp_history),
#             'cpu': {
#                 'current': self.cpu_history[-1],
#                 'average': sum(self.cpu_history) / len(self.cpu_history),
#                 'max': max(self.cpu_history),
#                 'min': min(self.cpu_history)
#             }
#         }
        
#         if self.memory_history:
#             memory_percents = [m['percent'] for m in self.memory_history]
#             summary['memory'] = {
#                 'current': memory_percents[-1],
#                 'average': sum(memory_percents) / len(memory_percents),
#                 'max': max(memory_percents),
#                 'min': min(memory_percents)
#             }
        
#         # 添加自定义指标摘要
#         for metric_name, history in self.custom_history.items():
#             if history:
#                 try:
#                     numeric_values = [v for v in history if isinstance(v, (int, float))]
#                     if numeric_values:
#                         summary[metric_name] = {
#                             'current': numeric_values[-1],
#                             'average': sum(numeric_values) / len(numeric_values),
#                             'max': max(numeric_values),
#                             'min': min(numeric_values)
#                         }
#                 except Exception as e:
#                     logger.error(f"计算 {metric_name} 摘要失败: {e}")
        
#         return summary
    
#     def _get_average(self, data: deque, seconds: int) -> Optional[float]:
#         """
#         计算指定时间范围内的平均值
        
#         参数:
#             data: 数据队列
#             seconds: 时间范围（秒）
        
#         返回:
#             平均值
#         """
#         if not data or not self.timestamp_history:
#             return None
        
#         current_time = time.time()
#         start_time = current_time - seconds
        
#         # 找到时间范围内的数据
#         values = []
#         for i, timestamp in enumerate(reversed(self.timestamp_history)):
#             if timestamp >= start_time and i < len(data):
#                 value = data[-(i+1)]
#                 if isinstance(value, (int, float)):
#                     values.append(value)
#                 elif isinstance(value, dict) and 'percent' in value:
#                     values.append(value['percent'])
        
#         return sum(values) / len(values) if values else None