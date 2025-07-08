import pymilvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from zhipuai import ZhipuAI
import os
import logging
import re
import time
from urllib.parse import urlparse
# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 从环境变量获取API密钥
api_key = ""
if not api_key:
    raise ValueError("Environment variable ZHIPUAI_API_KEY is not set.")

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)
# 准备视频数据，并为每个视频生成embedding
videos = []
    
def is_valid_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_urls_with_positions(text):
    url_pattern = r'(https?://[^\s\)\]\}>]+)'
    return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]

def custom_text_splitter(text):
    # 使用正则表达式匹配句号
    sentence_end_pattern = r'(?<=[。！？])'
    
    # 初始化结果列表和当前段落起始位置
    result = []
    start = 0
    
    # 遍历文本并按照规则分割
    for match in re.finditer(sentence_end_pattern, text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:  # 忽略空字符串
            result.append(segment)
        start = end
    
    # 添加最后一个段落（如果有）
    final_segment = text[start:].strip()
    if final_segment:
        result.append(final_segment)
    
    return result
def dataMakeInsert():
        # 指定要读取的 Markdown 文件路径
    markdown_file_path = "./video_Test_Data"
    # 列出目录中的所有文件
    files_in_directory = os.listdir(markdown_file_path)

    # 过滤出所有的Markdown文件
    markdown_files = [file for file in files_in_directory if file.endswith('.md')]

    # 读取每个Markdown文件的内容
    for markdown_file in markdown_files:
        file_path = os.path.join(markdown_file_path, markdown_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            segments = custom_text_splitter(content)
            
            for i, segment in enumerate(segments, 1):
                urls_with_positions = extract_urls_with_positions(segment)
                non_url_text = segment
                
                # 移除URL并清理多余空白
                for url, (start, end) in reversed(urls_with_positions):
                    non_url_text = non_url_text[:start] + non_url_text[end:]
                non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()
                # 调用嵌入创建 API
                response = client.embeddings.create(
                    model="embedding-2",  # 替换为您想要使用的模型名称
                    input=non_url_text  # 传递单个文本
                )

                # 获取生成的 embedding 值
                embedding_value = response.data[0].embedding
                #print(embedding_value)
                # 创建新的video条目
                video_entry = {
                    "id": i,  # 可以根据实际情况调整ID生成逻辑
                    "url": urls_with_positions[0][0] if urls_with_positions else "",
                    "description": non_url_text,
                    "question_id": i,  # 后面根据业务调整
                    "embedding": embedding_value 
                }

                # 如果有多个URL，可以考虑将它们全部保存在一个列表中
                if len(urls_with_positions) > 1:
                    video_entry["urls"] = [url for url, _ in urls_with_positions]

                videos.append(video_entry)
def create():
    
    try:
        # 连接Milvus服务器
        connections.connect("default", host="localhost", port="19530")

        # 检查并创建collection
        collection_name = 'video_push'
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="question_id", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        schema = CollectionSchema(fields, "video_push")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
        collection.load()


        # 将数据插入Milvus
        entities = [
            [video['id'] for video in videos],
            [video['url'] for video in videos],
            [video['description'] for video in videos],
            [video['question_id'] for video in videos],
            [video['embedding'] for video in videos]  # 添加向量数据
        ]

        mr = collection.insert(entities)
        #logging.info(f"Insert result: {mr}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        # 关闭Milvus连接
        connections.disconnect("default")
if __name__ == "__main__":
    dataMakeInsert()
    create()