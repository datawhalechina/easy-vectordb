from pymilvus import connections
from elasticsearch import Elasticsearch
import redis
import threading

# 全局连接对象和锁
milvus_connected = False
milvus_connections = None
es_client = None
redis_client = None
_lock = threading.Lock()

def init_milvus(databaseName, host, port):
    global milvus_connected
    with _lock:
        if not milvus_connected:
            status = connections.connect(
                alias="default",
                host=host,
                port=port,
                db_name=databaseName
            )
            milvus_connected = True
            print(f"Milvus connected: {status}")
            milvus_connections = connections

def init_es(host):
    global es_client
    with _lock:
        if es_client is None:
            es_client = Elasticsearch([host])

def init_redis(host, port):
    global redis_client
    with _lock:
        if redis_client is None:
            redis_client = redis.StrictRedis(host=host, port=port, db=0)
