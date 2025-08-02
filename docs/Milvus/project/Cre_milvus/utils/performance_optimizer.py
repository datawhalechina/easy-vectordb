"""
性能优化和资源管理模块

提供系统性能优化、内存管理、缓存机制和资源监控功能
"""

import time
import threading
import weakref
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import logging
from functools import lru_cache, wraps
import gc
import psutil
import os

logger = logging.getLogger(__name__)


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_warnings = []
        self.cleanup_callbacks = []
    
    def register_cleanup_callback(self, callback: Callable):
        """注册内存清理回调函数"""
        self.cleanup_callbacks.append(callback)
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用情况"""
        memory = psutil.virtual_memory()
        
        status = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "free": memory.free,
            "timestamp": datetime.now().isoformat()
        }
        
        if memory.percent > self.max_memory_percent:
            warning = f"内存使用率过高: {memory.percent:.1f}%"
            self.memory_warnings.append({
                "timestamp": datetime.now().isoformat(),
                "message": warning,
                "memory_percent": memory.percent
            })
            logger.warning(warning)
            
            # 触发内存清理
            self.cleanup_memory()
        
        return status
    
    def cleanup_memory(self):
        """执行内存清理"""
        try:
            # 执行注册的清理回调
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"内存清理回调失败: {e}")
            
            # 强制垃圾回收
            collected = gc.collect()
            logger.info(f"垃圾回收完成，清理了 {collected} 个对象")
            
        except Exception as e:
            logger.error(f"内存清理失败: {e}")
    
    def get_memory_warnings(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取内存警告历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            warning for warning in self.memory_warnings
            if datetime.fromisoformat(warning["timestamp"]) > cutoff_time
        ]


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # 检查TTL
            if self._is_expired(key):
                self._remove(key)
                return None
            
            # 更新访问时间
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        with self.lock:
            # 检查缓存大小限制
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def remove(self, key: str):
        """删除缓存项"""
        with self.lock:
            self._remove(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期"""
        if key not in self.access_times:
            return True
        
        return time.time() - self.access_times[key] > self.ttl_seconds
    
    def _remove(self, key: str):
        """删除缓存项（内部方法）"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_lru(self):
        """驱逐最近最少使用的缓存项"""
        if not self.access_times:
            return
        
        # 找到最久未访问的key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            now = time.time()
            expired_count = sum(1 for key in self.cache.keys() if self._is_expired(key))
            
            return {
                "total_items": len(self.cache),
                "expired_items": expired_count,
                "valid_items": len(self.cache) - expired_count,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
            }
    
    def _estimate_memory_usage(self) -> int:
        """估算缓存内存使用量"""
        try:
            import sys
            total_size = 0
            for key, value in self.cache.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
            return total_size
        except Exception:
            return 0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = []
        self.operation_stats = {}
        self.lock = threading.RLock()
    
    def record_operation(self, operation: str, duration: float, success: bool = True, **kwargs):
        """记录操作性能"""
        with self.lock:
            timestamp = datetime.now()
            
            # 记录到历史
            metric = {
                "timestamp": timestamp.isoformat(),
                "operation": operation,
                "duration": duration,
                "success": success,
                **kwargs
            }
            
            self.metrics_history.append(metric)
            
            # 限制历史记录数量
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            # 更新操作统计
            if operation not in self.operation_stats:
                self.operation_stats[operation] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0
                }
            
            stats = self.operation_stats[operation]
            stats["count"] += 1
            stats["total_duration"] += duration
            
            if success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
            stats["min_duration"] = min(stats["min_duration"], duration)
            stats["max_duration"] = max(stats["max_duration"], duration)
    
    def get_operation_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """获取操作统计信息"""
        with self.lock:
            if operation:
                return self.operation_stats.get(operation, {})
            return self.operation_stats.copy()
    
    def get_recent_metrics(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取最近的性能指标"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            return [
                metric for metric in self.metrics_history
                if datetime.fromisoformat(metric["timestamp"]) > cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            recent_metrics = self.get_recent_metrics(60)  # 最近1小时
            
            if not recent_metrics:
                return {"message": "暂无性能数据"}
            
            total_operations = len(recent_metrics)
            successful_operations = sum(1 for m in recent_metrics if m.get("success", True))
            
            durations = [m["duration"] for m in recent_metrics]
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "operations_per_minute": total_operations,
                "timestamp": datetime.now().isoformat()
            }


class ResourceOptimizer:
    """资源优化器"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_rules = []
    
    def add_optimization_rule(self, rule: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """添加优化规则"""
        self.optimization_rules.append(rule)
    
    def optimize_clustering_performance(self, data_size: int, algorithm: str) -> Dict[str, Any]:
        """优化聚类性能"""
        recommendations = {
            "algorithm": algorithm,
            "data_size": data_size,
            "optimizations": []
        }
        
        if algorithm == "hdbscan":
            if data_size > 10000:
                recommendations["optimizations"].append({
                    "type": "parameter",
                    "suggestion": "增加min_cluster_size以提高性能",
                    "recommended_value": max(5, data_size // 2000)
                })
            
            if data_size > 50000:
                recommendations["optimizations"].append({
                    "type": "algorithm",
                    "suggestion": "考虑使用采样或分批处理",
                    "recommended_batch_size": 10000
                })
        
        elif algorithm == "kmeans":
            if data_size > 100000:
                recommendations["optimizations"].append({
                    "type": "parameter",
                    "suggestion": "使用MiniBatchKMeans以提高性能"
                })
        
        return recommendations
    
    def optimize_llm_api_calls(self, provider: str, model: str) -> Dict[str, Any]:
        """优化LLM API调用"""
        optimizations = {
            "provider": provider,
            "model": model,
            "recommendations": []
        }
        
        # 通用优化建议
        optimizations["recommendations"].extend([
            {
                "type": "caching",
                "suggestion": "启用响应缓存以减少重复调用",
                "implementation": "cache_manager"
            },
            {
                "type": "batching",
                "suggestion": "批量处理相似请求",
                "batch_size": 10
            },
            {
                "type": "retry",
                "suggestion": "实现指数退避重试机制",
                "max_retries": 3
            }
        ])
        
        # 提供商特定优化
        if provider == "openai":
            optimizations["recommendations"].append({
                "type": "parameter",
                "suggestion": "调整temperature和max_tokens以平衡质量和速度",
                "recommended_temperature": 0.1,
                "recommended_max_tokens": 10
            })
        
        return optimizations
    
    def get_system_resource_status(self) -> Dict[str, Any]:
        """获取系统资源状态"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存信息
            memory_info = self.memory_manager.check_memory_usage()
            
            # 磁盘信息
            disk_usage = psutil.disk_usage('/')
            
            # 进程信息
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": memory_info,
                "disk": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percent": (disk_usage.used / disk_usage.total) * 100
                },
                "process": {
                    "memory_rss": process_memory.rss,
                    "memory_vms": process_memory.vms,
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads()
                },
                "cache": self.cache_manager.get_stats(),
                "performance": self.performance_monitor.get_performance_summary()
            }
        
        except Exception as e:
            logger.error(f"获取系统资源状态失败: {e}")
            return {"error": str(e)}
    
    def cleanup_resources(self):
        """清理系统资源"""
        try:
            # 清理缓存
            self.cache_manager.clear()
            
            # 清理内存
            self.memory_manager.cleanup_memory()
            
            logger.info("系统资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


def performance_cache(ttl_seconds: int = 3600):
    """性能缓存装饰器"""
    def decorator(func):
        cache = CacheManager(ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                cache.set(cache_key, result)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"函数 {func.__name__} 执行时间: {duration:.3f}s")
        
        return wrapper
    return decorator


# 全局资源优化器实例
resource_optimizer = ResourceOptimizer()


def monitor_performance(operation_name: str):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                resource_optimizer.performance_monitor.record_operation(
                    operation=operation_name,
                    duration=duration,
                    success=success,
                    error=error,
                    function_name=func.__name__
                )
        
        return wrapper
    return decorator