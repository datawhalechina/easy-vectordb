import subprocess
import sys
import time
import requests
import signal
import os
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_startup.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SimpleServiceManager:
    """简化的服务管理器"""
    
    def __init__(self, backend_port: int = 12089, frontend_port: int = 12088):
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.startup_timeout = 30
        self.project_root = Path(__file__).parent
    
    def check_port_available(self, port: int) -> bool:
        """检查端口是否可用"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
        except Exception:
            return False
    
    def wait_for_service(self, url: str, timeout: int = 30, service_name: str = "服务") -> bool:
        """等待服务启动"""
        logger.info(f"⏳ 等待{service_name}启动: {url}")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name}启动成功")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error(f"❌ {service_name}启动超时 ({timeout}秒)")
        return False
    
    def start_backend(self) -> bool:
        """启动后端API服务"""
        logger.info("🚀 启动后端API服务...")
        
        # 检查端口是否可用
        if not self.check_port_available(self.backend_port):
            logger.error(f"❌ 端口 {self.backend_port} 已被占用")
            return False
        
        try:
            # 启动后端服务
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "backend_api:app",
                "--reload",
                "--port", str(self.backend_port),
                "--host", "0.0.0.0",
                "--timeout-keep-alive", "90"
            ],
            cwd=self.project_root,
            # 让子进程输出直接显示在终端
            stdout=None,
            stderr=None,
            text=True
            )
            
            logger.info(f"✅ 后端服务进程已启动 (PID: {self.backend_process.pid}, 端口: {self.backend_port})")
            
            # 等待服务启动
            if self.wait_for_service(
                f"http://localhost:{self.backend_port}/health",
                timeout=self.startup_timeout,
                service_name="后端API"
            ):
                return True
            else:
                self.stop_backend()
                return False
                
        except Exception as e:
            logger.error(f"❌ 启动后端服务失败: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """启动前端服务"""
        logger.info("🎨 启动前端界面...")
        
        # 检查端口是否可用
        if not self.check_port_available(self.frontend_port):
            logger.error(f"❌ 端口 {self.frontend_port} 已被占用")
            return False
        
        try:
            # 等待后端完全启动
            time.sleep(3)
            
            self.frontend_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run",
                "frontend.py",
                "--server.port", str(self.frontend_port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ],
            cwd=self.project_root,
            # 让子进程输出直接显示在终端
            stdout=None,
            stderr=None,
            text=True
            )
            
            logger.info(f"✅ 前端服务进程已启动 (PID: {self.frontend_process.pid}, 端口: {self.frontend_port})")
            
            # 等待前端服务启动
            if self.wait_for_service(
                f"http://localhost:{self.frontend_port}",
                timeout=45,
                service_name="前端界面"
            ):
                return True
            else:
                self.stop_frontend()
                return False
                
        except Exception as e:
            logger.error(f"❌ 启动前端服务失败: {e}")
            return False
    
    def stop_backend(self) -> None:
        """停止后端服务"""
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
                logger.info("✅ 后端服务已停止")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                logger.info("🔪 强制终止后端服务")
            except Exception as e:
                logger.error(f"❌ 停止后端服务失败: {e}")
            finally:
                self.backend_process = None
    
    def stop_frontend(self) -> None:
        """停止前端服务"""
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=10)
                logger.info("✅ 前端服务已停止")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                logger.info("🔪 强制终止前端服务")
            except Exception as e:
                logger.error(f"❌ 停止前端服务失败: {e}")
            finally:
                self.frontend_process = None
    
    def stop_all(self) -> None:
        """停止所有服务"""
        logger.info("🛑 正在停止所有服务...")
        self.stop_frontend()
        self.stop_backend()
        logger.info("👋 所有服务已停止")
    
    def health_check(self) -> Dict[str, Any]:
        """简单的健康检查"""
        backend_healthy = False
        frontend_healthy = False
        
        # 检查后端
        if self.backend_process and self.backend_process.poll() is None:
            try:
                response = requests.get(f"http://localhost:{self.backend_port}/health", timeout=3)
                backend_healthy = response.status_code == 200
            except:
                pass
        
        # 检查前端
        if self.frontend_process and self.frontend_process.poll() is None:
            try:
                response = requests.get(f"http://localhost:{self.frontend_port}", timeout=3)
                frontend_healthy = response.status_code == 200
            except:
                pass
        
        return {
            "backend": {
                "healthy": backend_healthy,
                "port": self.backend_port,
                "pid": self.backend_process.pid if self.backend_process else None
            },
            "frontend": {
                "healthy": frontend_healthy,
                "port": self.frontend_port,
                "pid": self.frontend_process.pid if self.frontend_process else None
            },
            "overall_healthy": backend_healthy and frontend_healthy,
            "timestamp": datetime.now().isoformat()
        }
    
    def monitor_processes(self) -> None:
        """简单的进程监控"""
        while True:
            try:
                # 检查后端进程
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ 后端进程意外退出")
                    break
                
                # 检查前端进程
                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.error("❌ 前端进程意外退出")
                    break
                
                time.sleep(10)  # 每10秒检查一次
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ 进程监控异常: {e}")
                break
    
    def start_services(self) -> bool:
        """启动所有服务"""
        logger.info("=" * 60)
        logger.info("🚀 Cre_milvus 系统启动")
        logger.info(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            # 启动后端服务
            if not self.start_backend():
                logger.error("❌ 后端服务启动失败")
                return False
            
            # 启动前端服务
            if not self.start_frontend():
                logger.error("❌ 前端服务启动失败")
                self.stop_backend()
                return False
            
            # 显示启动成功信息
            logger.info("=" * 60)
            logger.info("🎉 系统启动完成！")
            logger.info(f"📊 后端API: http://localhost:{self.backend_port}")
            logger.info(f"🎨 前端界面: http://localhost:{self.frontend_port}")
            logger.info(f"📚 API文档: http://localhost:{self.backend_port}/docs")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            self.stop_all()
            return False

def initialize_connections() -> bool:
    """异步初始化连接"""
    try:
        logger.info("🔄 开始后台连接初始化...")
        
        # 导入配置和连接模块
        from config_loader import load_config
        from start_simple import connect_milvus, get_milvus_status
        from milvus_lock_fix import cleanup_old_connections
        
        # 清理可能存在的旧连接
        cleanup_old_connections()
        
        # 加载配置
        config = load_config()
        logger.info("✅ 配置加载成功")
        
        # 初始化Milvus连接（优先级最高）
        logger.info("🔗 开始初始化Milvus连接（优先级最高）...")
        milvus_config = config.get("milvus", {})
        host = milvus_config.get("host", "localhost")
        port = int(milvus_config.get("port", 19530))
        milvus_success = connect_milvus(host, port)
        
        if milvus_success:
            logger.info("✅ Milvus连接初始化成功，数据插入功能已就绪")
        else:
            logger.warning("⚠️ Milvus连接初始化失败，数据插入功能可能受影响")
            status = get_milvus_status()
            if status.get("error_message"):
                logger.error(f"❌ 连接错误: {status['error_message']}")
            logger.info("💡 系统将在需要时尝试重新连接")
        
        # 可以在这里添加其他连接的初始化
        # 例如：Redis、Elasticsearch等
        
        logger.info("✅ 后台连接初始化完成")
        return milvus_success
        
    except Exception as e:
        logger.error(f"❌ 连接初始化失败: {e}")
        import traceback
        logger.debug(f"详细错误: {traceback.format_exc()}")
        return False

def get_connection_status() -> Dict[str, Any]:
    """获取连接状态"""
    try:
        from start_simple import get_milvus_status
        
        milvus_status = get_milvus_status()
        
        return {
            "milvus": milvus_status,
            "overall_healthy": milvus_status.get("connected", False),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取连接状态失败: {e}")
        return {
            "milvus": {"connected": False, "error": str(e)},
            "overall_healthy": False,
            "timestamp": datetime.now().isoformat()
        }

def start_system(backend_port: int = 12089, frontend_port: int = 12088) -> bool:
    """启动整个系统"""
    manager = SimpleServiceManager(backend_port, frontend_port)
    
    # 启动服务
    if not manager.start_services():
        return False
    
    # 异步初始化连接
    connection_thread = threading.Thread(
        target=initialize_connections,
        daemon=True,
        name="ConnectionInitializer"
    )
    connection_thread.start()
    
    # 启动进程监控
    monitor_thread = threading.Thread(
        target=manager.monitor_processes,
        daemon=True,
        name="ProcessMonitor"
    )
    monitor_thread.start()
    
    try:
        # 等待用户中断
        logger.info("按 Ctrl+C 停止系统")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 收到停止信号...")
    finally:
        manager.stop_all()
    
    return True

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"\n🛑 收到信号 {signum}，正在停止系统...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动系统
    success = start_system()
    
    if success:
        logger.info("✅ 系统正常退出")
        sys.exit(0)
    else:
        logger.error("❌ 系统异常退出")
        sys.exit(1)

if __name__ == "__main__":
    main()