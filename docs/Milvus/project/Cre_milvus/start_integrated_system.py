#!/usr/bin/env python3
"""
Cre_milvus 系统启动脚本

一键启动系统，包括后端API和前端界面
"""

import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_basic_dependencies():
    """检查基础依赖是否已安装"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package.replace('-', '_'))
            logger.info(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少基础依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install fastapi uvicorn streamlit pyyaml")
        return False
    
    return True


def initialize_connections():
    """初始化连接管理器（不强制连接）"""
    logger.info("🔗 初始化系统连接管理器...")
    
    try:
        from System.connection_initializer import startup_initialize
        success = startup_initialize()
        
        if success:
            logger.info("✅ 系统连接管理器初始化成功")
            logger.info("📝 Milvus连接将在用户配置后建立")
            return True
        else:
            logger.warning("⚠️ 系统连接管理器初始化失败，使用传统模式")
            return True  # 允许继续使用传统模式
            
    except Exception as e:
        logger.warning(f"⚠️ 新架构初始化失败，使用传统模式: {e}")
        return True  # 允许继续使用传统模式

def start_backend():
    """启动后端API服务"""
    logger.info("🚀 启动后端API服务...")
    
    # 先尝试初始化新架构连接
    connections_ready = initialize_connections()
    
    try:
        # 优先使用新架构API，如果初始化成功的话
        if connections_ready:
            try:
                # 检查新架构API是否可用
                import importlib.util
                spec = importlib.util.spec_from_file_location("new_backend_api", "new_backend_api.py")
                if spec and spec.loader:
                    logger.info("🆕 使用新架构API服务")
                    backend_process = subprocess.Popen([
                        sys.executable, "-m", "uvicorn", 
                        "new_backend_api:app", 
                        "--reload", 
                        "--port", "8509",  # 保持原端口
                        "--host", "0.0.0.0"
                    ], cwd=Path(__file__).parent)
                    
                    logger.info("✅ 新架构API服务已启动 (端口: 8509)")
                    return backend_process
            except Exception as e:
                logger.warning(f"⚠️ 新架构API启动失败，使用传统API: {e}")
        
        # 回退到传统API
        logger.info("📡 使用传统API服务")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend_api:app", 
            "--reload", 
            "--port", "8509",  # 后端使用8509端口
            "--host", "0.0.0.0"
        ], cwd=Path(__file__).parent)
        
        logger.info("✅ 传统API服务已启动 (端口: 8509)")
        return backend_process
        
    except Exception as e:
        logger.error(f"❌ 启动后端服务失败: {e}")
        return None


def start_frontend():
    """启动前端界面"""
    logger.info("🎨 启动前端界面...")
    
    try:
        # 等待后端服务启动
        time.sleep(3)
        
        # 启动Streamlit前端
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "frontend.py",
            "--server.port", "8500",
            "--server.address", "0.0.0.0"
        ], cwd=Path(__file__).parent)
        
        logger.info("✅ 前端界面已启动 (端口: 8500)")
        return frontend_process
        
    except Exception as e:
        logger.error(f"❌ 启动前端界面失败: {e}")
        return None


def open_browser():
    """打开浏览器"""
    time.sleep(5)  # 等待服务完全启动
    
    try:
        webbrowser.open("http://localhost:8500")
        logger.info("🌐 浏览器已打开")
    except Exception as e:
        logger.warning(f"⚠️ 自动打开浏览器失败: {e}")
        logger.info("请手动访问: http://localhost:8500")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🔍 Cre_milvus 系统启动器")
    logger.info("=" * 60)
    
    # 检查基础依赖
    if not check_basic_dependencies():
        sys.exit(1)
    
    # 启动后端服务
    backend_process = start_backend()
    if not backend_process:
        logger.error("❌ 后端服务启动失败，退出")
        sys.exit(1)
    
    # 启动前端界面
    frontend_process = start_frontend()
    if not frontend_process:
        logger.error("❌ 前端界面启动失败，退出")
        backend_process.terminate()
        sys.exit(1)
    
    # 在新线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    logger.info("=" * 60)
    logger.info("🎉 系统启动完成！")
    logger.info("📊 后端API: http://localhost:8509")
    logger.info("🎨 前端界面: http://localhost:8500")
    logger.info("📚 API文档: http://localhost:8509/docs")
    logger.info("=" * 60)
    logger.info("按 Ctrl+C 停止系统")
    
    try:
        # 等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 正在停止系统...")
        
        # 停止服务
        if backend_process:
            backend_process.terminate()
            logger.info("✅ 后端服务已停止")
        
        if frontend_process:
            frontend_process.terminate()
            logger.info("✅ 前端界面已停止")
        
        logger.info("👋 系统已完全停止")


if __name__ == "__main__":
    main()