"""
统一错误处理模块

提供系统级别的错误处理、日志记录和异常管理功能
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import json
import os

# 配置日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class SystemLogger:
    """系统日志管理器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._setup_loggers()
    
    def _setup_loggers(self):
        """设置不同类型的日志记录器"""
        
        # 主系统日志
        self.system_logger = self._create_logger(
            "system", 
            os.path.join(self.log_dir, "system.log"),
            logging.INFO
        )
        
        # 错误日志
        self.error_logger = self._create_logger(
            "error",
            os.path.join(self.log_dir, "error.log"),
            logging.ERROR
        )
        
        # 性能日志
        self.performance_logger = self._create_logger(
            "performance",
            os.path.join(self.log_dir, "performance.log"),
            logging.INFO
        )
        
        # API调用日志
        self.api_logger = self._create_logger(
            "api",
            os.path.join(self.log_dir, "api.log"),
            logging.INFO
        )
    
    def _create_logger(self, name: str, log_file: str, level: int) -> logging.Logger:
        """创建日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 避免重复添加handler
        if not logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # 设置格式
            formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_system_event(self, message: str, level: str = "info", **kwargs):
        """记录系统事件"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
            **kwargs
        }
        
        if level.lower() == "error":
            self.error_logger.error(json.dumps(log_data, ensure_ascii=False))
        elif level.lower() == "warning":
            self.system_logger.warning(json.dumps(log_data, ensure_ascii=False))
        else:
            self.system_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_api_call(self, endpoint: str, method: str, status_code: int, 
                     execution_time: float, **kwargs):
        """记录API调用"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "execution_time": execution_time,
            **kwargs
        }
        
        self.api_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_performance_metric(self, operation: str, duration: float, 
                              success: bool, **kwargs):
        """记录性能指标"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            "success": success,
            **kwargs
        }
        
        self.performance_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """记录错误信息"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            **kwargs
        }
        
        self.error_logger.error(json.dumps(error_data, ensure_ascii=False))


# 全局日志管理器实例
system_logger = SystemLogger()


class ErrorHandler:
    """错误处理器"""
    
    @staticmethod
    def handle_api_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """处理API错误"""
        system_logger.log_error(error, context)
        
        error_response = {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        # 根据错误类型返回不同的HTTP状态码
        if isinstance(error, ValueError):
            error_response["http_status"] = 400
        elif isinstance(error, FileNotFoundError):
            error_response["http_status"] = 404
        elif isinstance(error, PermissionError):
            error_response["http_status"] = 403
        elif isinstance(error, TimeoutError):
            error_response["http_status"] = 408
        else:
            error_response["http_status"] = 500
        
        return error_response
    
    @staticmethod
    def handle_service_error(error: Exception, service_name: str) -> Dict[str, Any]:
        """处理服务错误"""
        context = f"Service: {service_name}"
        system_logger.log_error(error, context)
        
        return {
            "service": service_name,
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "recoverable": ErrorHandler._is_recoverable_error(error)
        }
    
    @staticmethod
    def _is_recoverable_error(error: Exception) -> bool:
        """判断错误是否可恢复"""
        recoverable_errors = [
            ConnectionError,
            TimeoutError,
            OSError,
        ]
        
        return any(isinstance(error, err_type) for err_type in recoverable_errors)


def log_execution_time(operation_name: str):
    """装饰器：记录函数执行时间"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            success = True
            result = None
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                system_logger.log_performance_metric(
                    operation=operation_name,
                    duration=duration,
                    success=success,
                    function_name=func.__name__,
                    error=str(error) if error else None
                )
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return=None, **kwargs) -> Any:
    """安全执行函数，捕获异常并记录"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        system_logger.log_error(e, f"Safe execution of {func.__name__}")
        return default_return


def validate_config(config: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, str]:
    """验证配置参数"""
    try:
        missing_fields = []
        
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
            elif config[field] is None or config[field] == "":
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"缺少必需的配置字段: {', '.join(missing_fields)}"
            system_logger.log_system_event(error_msg, "error")
            return False, error_msg
        
        return True, "配置验证通过"
        
    except Exception as e:
        error_msg = f"配置验证失败: {str(e)}"
        system_logger.log_error(e, "配置验证")
        return False, error_msg


class HealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """注册健康检查函数"""
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = datetime.now()
                result = check_func()
                end_time = datetime.now()
                
                check_result = {
                    "status": "healthy" if result else "unhealthy",
                    "check_time": (end_time - start_time).total_seconds(),
                    "timestamp": end_time.isoformat()
                }
                
                if not result:
                    overall_healthy = False
                
                results[name] = check_result
                
            except Exception as e:
                system_logger.log_error(e, f"健康检查失败: {name}")
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                overall_healthy = False
        
        results["overall"] = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks_passed": sum(1 for r in results.values() 
                               if isinstance(r, dict) and r.get("status") == "healthy"),
            "total_checks": len(self.checks)
        }
        
        return results


# 全局健康检查器实例
health_checker = HealthChecker()


def setup_error_handling():
    """设置全局错误处理"""
    # 设置未捕获异常的处理
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许 Ctrl+C 正常退出
            return
        
        system_logger.log_error(
            exc_value, 
            "未捕获的异常",
            exc_type=exc_type.__name__,
            traceback_info=traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
    
    import sys
    sys.excepthook = handle_exception
    
    system_logger.log_system_event("错误处理系统已初始化", "info")


# 初始化错误处理
setup_error_handling()