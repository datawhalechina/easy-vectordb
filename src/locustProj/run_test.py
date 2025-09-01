#!/usr/bin/env python3
"""
Milvus Locust测试启动脚本
简化启动流程，自动打开浏览器
"""

import os
import sys
import time
import socket
import webbrowser
import subprocess
from pathlib import Path

def check_port_available(port):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8089):
    """找到可用端口"""
    for port in range(start_port, start_port + 50):
        if check_port_available(port):
            return port
    # 如果都不可用，让系统分配一个
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        return port

def main():
    """主函数"""
    print("🚀 启动Milvus Locust性能测试")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = Path(__file__).parent
    locustfile_path = current_dir / "locustfile.py"
    
    if not locustfile_path.exists():
        print("❌ 错误: 找不到 locustfile.py")
        sys.exit(1)
    
    # 找到可用端口
    web_port = find_available_port(8089)
    print(f"📡 使用端口: {web_port}")
    
    # 构建Locust命令
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(locustfile_path),
        "--web-port", str(web_port),
        "--host", "http://localhost:19530"
    ]
    
    print(f"🔧 启动命令: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 启动Locust
        process = subprocess.Popen(cmd, cwd=current_dir)
        
        # 等待服务启动
        print("⏳ 等待Locust服务启动...")
        time.sleep(3)
        
        # 检查进程是否还在运行
        if process.poll() is None:
            web_url = f"http://localhost:{web_port}"
            print(f"✅ Locust已启动: {web_url}")
            print("🌐 正在打开浏览器...")
            
            # 自动打开浏览器
            try:
                webbrowser.open(web_url)
            except Exception as e:
                print(f"⚠️  无法自动打开浏览器: {e}")
                print(f"请手动访问: {web_url}")
        else:
            print("❌ Locust启动失败")
            return
        
        print("\n" + "=" * 50)
        print("📋 测试说明:")
        print("1. 在浏览器中设置用户数和生成速率")
        print("2. 点击 'Start swarming' 开始测试")
        print("3. 观察实时性能指标")
        print("4. 按 Ctrl+C 停止测试")
        print("=" * 50)
        
        # 等待用户中断
        try:
            while process.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n🛑 正在停止测试...")
            try:
                process.terminate()
                # 给进程一些时间正常退出
                for _ in range(10):
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                # 如果还没退出，强制杀死
                if process.poll() is None:
                    process.kill()
                    process.wait()
            except:
                pass
            print("✅ 测试已停止")
    
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()