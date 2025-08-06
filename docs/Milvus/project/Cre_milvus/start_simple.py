#!/usr/bin/env python3
"""
简化的系统启动脚本
只启动必要的服务，去除复杂的管理器
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

def start_backend():
    """启动简化的后端API服务"""
    logger.info("🚀 启动后端API服务...")
    
    try:
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend_api:app", 
            "--reload", 
            "--port", "8505",  
            "--host", "0.0.0.0"
        ], cwd=Path(__file__).parent)
        
        logger.info("✅ 后端API服务已启动 (端口: 8505)")
        return backend_process
        
    except Exception as e:
        logger.error(f"❌ 启动后端服务失败: {e}")
        return None

def start_frontend():
    """启动前端界面"""
    logger.info("🎨 启动前端界面...")
    
    try:
        time.sleep(3)  # 等待后端启动
        
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
    time.sleep(5)  
    
    try:
        webbrowser.open("http://localhost:8500")
        logger.info("🌐 浏览器已打开")
    except Exception as e:
        logger.warning(f"⚠️ 自动打开浏览器失败: {e}")
        logger.info("请手动访问: http://localhost:8500")

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("🔍 Cre_milvus 简化版启动器")
    logger.info("=" * 50)
    
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
    
    # 打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    logger.info("=" * 50)
    logger.info("🎉 系统启动完成！")
    logger.info("📊 后端API: http://localhost:8505")
    logger.info("🎨 前端界面: http://localhost:8500")
    logger.info("📚 API文档: http://localhost:8505/docs")
    logger.info("=" * 50)
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