# from pymilvus import connections
# from elasticsearch import Elasticsearch
# import redis
# import threading
# import time

# # 全局连接对象和锁
# milvus_connected = False
# milvus_connections = None
# es_client = None
# redis_client = None
# _lock = threading.Lock()

# def init_milvus(databaseName, host, port, timeout=5):
#     """
#     已弃用的Milvus初始化函数，现在使用新的连接管理器
#     为了兼容性保留此函数，但不执行实际连接操作
#     """
#     global milvus_connected, milvus_connections
#     print(f"注意: init_milvus函数已弃用，现在使用新的连接管理器")
#     print(f"目标连接: {host}:{port}")
    
#     # 直接返回True，表示"连接成功"，实际连接由新的连接管理器处理
#     milvus_connected = True
#     return True

# def init_es(host, timeout=5):
#     global es_client
#     with _lock:
#         if es_client is None:
#             try:
#                 es_client = Elasticsearch([host], timeout=timeout)
#                 if es_client.ping():
#                     print(f"Elasticsearch连接成功: {host}")
#                 else:
#                     print("Elasticsearch连接失败")
#             except Exception as e:
#                 print(f"Elasticsearch连接异常: {str(e)}")

# def init_redis(host, port, timeout=5):
#     global redis_client
#     with _lock:
#         if redis_client is None:
#             try:
#                 redis_client = redis.StrictRedis(
#                     host=host,
#                     port=port,
#                     db=0,
#                     socket_timeout=timeout,
#                     socket_connect_timeout=timeout
#                 )
#                 # 发送测试命令验证连接
#                 if redis_client.ping():
#                     print(f"Redis连接成功: {host}:{port}")
#                 else:
#                     print("Redis连接失败")
#             except Exception as e:
#                 print(f"Redis连接异常: {str(e)}")

# def Cre_VectorDataBaseStart(
#     C_G_Choic, IP, Port, VectorName, CollectionName,
#     IndexName, ReplicaNum, Data_Location, url_split, insert_mode,
#     chunking_strategy="traditional", chunking_params=None, enable_multimodal=False
# ):
#     """
#     构建向量数据库并插入数据，参数全部由配置文件自动读取。
#     """
#     try:
#         # 初始化连接（添加超时参数）
#         print(f"初始化Milvus连接: {IP}:{Port}，超时5秒...")
#         if not init_milvus(VectorName, IP, Port, timeout=5):
#             print("⚠️ Milvus连接失败，终止操作")
#             return False
        
#         # 其他服务连接示例
#         # init_es("elastic_host", timeout=5)
#         # init_redis("redis_host", 6379, timeout=5)
        
#         print("连接成功，继续执行数据插入操作...")
#         # 这里添加数据插入逻辑
        
#         return True
        
#     except Exception as e:
#         print(f"创建向量数据库过程中出错: {str(e)}")
#         return False

# # 测试调用
# if __name__ == "__main__":
#     # 替换为您的实际配置
#     config = {
#         "IP": "localhost",
#         "Port": "19530",
#         "VectorName": "default"  # Milvus数据库名
#     }
    
#     Cre_VectorDataBaseStart(
#         C_G_Choic="",
#         IP=config["IP"],
#         Port=config["Port"],
#         VectorName=config["VectorName"],
#         CollectionName="",
#         IndexName="",
#         ReplicaNum=1,
#         Data_Location="",
#         url_split="",
#         insert_mode=""
#     )