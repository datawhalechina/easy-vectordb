#!/usr/bin/env python3
"""
简化的Milvus Locust测试启动器
避免grpc线程阻塞问题
"""

import os
import sys
import webbrowser
import time
import socket

def find_available_port(start_port=8090):
    """找到可用端口"""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    # 如果都不可用，让系统分配一个
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]

def main():
    """主函数"""
    print("🚀 启动Milvus Locust性能测试")
    print("=" * 50)
    
    # 设置环境变量来减少grpc问题
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    
    # 找到可用端口
    port = find_available_port(8090)
    cmd = f'"{sys.executable}" -m locust -f locustfile.py --web-port {port} --host http://localhost:19530'
    
    print(f"📡 Web UI端口: {port}")
    print(f"🔧 启动命令: {cmd}")
    print("=" * 50)
    
    # 打开浏览器
    web_url = f"http://localhost:{port}"
    print(f"🌐 Web界面: {web_url}")
    
    try:
        webbrowser.open(web_url)
        print("✅ 浏览器已打开")
    except:
        print("⚠️  请手动打开浏览器访问上述地址")
    
    print("\n" + "=" * 50)
    print("📋 使用说明:")
    print("1. 等待几秒钟让服务启动")
    print("2. 在浏览器中设置用户数和生成速率")
    print("3. 点击 'Start swarming' 开始测试")
    print("4. 观察实时性能指标")
    print("5. 测试完成后直接关闭终端窗口")
    print("=" * 50)
    print()
    
    # 执行命令
    os.system(cmd)

if __name__ == "__main__":
    main()