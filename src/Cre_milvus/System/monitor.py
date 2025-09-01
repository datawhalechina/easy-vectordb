import logging

# 配置日志
logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_event(message):
    """
    记录系统事件日志。
    """
    print(message)