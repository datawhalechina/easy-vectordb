from elasticsearch import  helpers

def create_index(es_client, index_name):
    # 定义mapping，用于创建索引
    mapping = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "file_name": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "meta": {"type": "object", "enabled": True},
                "embedding": {"type": "dense_vector", "dims": 768},
                "keywords": {"type": "keyword"}  # 新增
            }
        }
    }
    # 判断索引是否存在，如果不存在则创建索引
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mapping)

# 定义一个函数，用于将数据批量插入到Elasticsearch中
def bulk_insert_to_es(es_client, index_name, data_list):
    # 创建一个actions列表，用于存储要插入的数据
    actions = [
        {"_index": index_name, "_source": doc}
        for doc in data_list
    ]
    # 使用Elasticsearch的helpers模块的bulk函数，将数据批量插入到指定的索引中
    helpers.bulk(es_client, actions)