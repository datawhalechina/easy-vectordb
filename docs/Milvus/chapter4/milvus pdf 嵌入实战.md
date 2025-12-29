# milvus pdf 嵌入实战

## 一、环境准备

```
# 基础依赖
!pip install pymupdf openai sentence-transformers pymilvus
```

- 推荐Python 3.8+环境（网页8）
- 确保Milvus服务已启动（默认端口19530）
- 建议配置8GB+内存（处理大规模PDF时）

## 二、PDF文本处理

### 1. 文本提取

```python
def extract_pdf_text(path):
    """
    从PDF文件中提取文本内容
    
    参数:
        path: PDF文件路径
        
    返回:
        提取的文本字符串
    """
    try:
        doc = pymupdf.open(path)
        return " ".join([page.get_text() for page in doc])
    except Exception as e:
        raise RuntimeError(f"PDF解析失败: {str(e)}")
```

### 2. 文本重构增强

```python
def reload_text(path):
    """
    使用大模型重构PDF提取的文本，提高文本质量
    
    参数:
        path: PDF文件路径
        
    返回:
        重构后的文本字符串
    """
    text = extract_pdf_text(path)

    # 初始化OpenAI客户端，连接到阿里云DashScope
    client = OpenAI(
        api_key = api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    try:
        # 调用DeepSeek-v3模型重构文本
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "user", 
                "content": "这是pdf 解析出来的文档 请你仔细阅读，将内容调整为完整的句子。同时需要确保不能丢失原文的信息。解析文档内容如下：{}".format(text)
                }
                ]
        )
        reload_text=response.choices[0].message.content
        return reload_text
    except Exception as e:
        print(f"API调用失败，启用降级处理: {str(e)}")
        # 降级处理：清理特殊字符并合并空格
        cleaned_text = re.sub(r'[^\w\u4e00-\u9fff\s。？！；，]', '', text)
        return re.sub(r'\s+', ' ', cleaned_text)
```

**注意事项**：

- 添加异常降级处理逻辑

## 三、向量化处理

### 1. 智能分块策略

```python
def split_text(path):
    """
    对重构后的文本进行分段并生成向量表示
    
    参数:
        path: PDF文件路径
        
    返回:
        tuple: (分段文本列表, 对应的向量表示)
    """
    text=reload_text(path)
    
    # 使用正则表达式按照标点符号分割文本
    split_pattern = r'(?<=[。！？；.?!!;\n])\s*'
    chunks = [c.strip() for c in re.split(split_pattern, text) if len(c.strip()) > 20]

    # 滑动窗口合并短文本，每5个句子为一组
    final_chunks = []
    for i in range(0, len(chunks), 3):
        window = chunks[i:i+5]
        final_chunks.append(" ".join(window))

    # 打印分段结果
    for index,value in enumerate(final_chunks):
        print(f"第{index}个句子-->{value}")
    
    # 加载本地预训练的向量模型
    model = SentenceTransformer('C:/Users/Administrator/Desktop/easy-vectordb/model/bge-small-zh-v1.5')
    
    # 生成文本向量并进行L2归一化
    embeddings = model.encode(final_chunks)
    embeddings = normalize(embeddings, norm='l2')
    
    return final_chunks, embeddings
```

**最佳实践**：

- 采用滑动窗口重叠分块/单独一个句子也是可以的
- 添加文本清洗步骤：去除特殊符号/乱码
- 对短文本进行合并处理

## 四、Milvus集成

### 1. 集合配置优化

```python
# 增强型schema配置（网页3）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  # 主键字段
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),  # 512维浮点向量
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000)  # 原始文本内容
]

schema = CollectionSchema(fields, 
    description="PDF知识库",
    enable_dynamic_field=True)  # 开启动态字段
collection = Collection("pdf_test", schema)
```

**索引策略**：

```python
# 复合索引配置（网页5）
vector_index = {
    "index_type": "IVF_PQ",
    "metric_type": "IP", 
    "params": {"nlist": 2048, "m": 32}
}

collection.create_index("vector", vector_index)
collection.create_index("content", {"index_type": "TRIE"})
```

### 2. 批量插入

```python
data = [
    [i for i in range(len(texts_list))], # 主键ID列表
    embeddings.tolist(), # 向量列表（转换为Python列表）
    texts_list # 原始文本列表
]

# 执行插入并刷新内存
collection.insert(data)
collection.flush()
collection.load()  # 将索引加载到内存
```

## 五、完整工作流

```python
"""
PDF向量存储完整工作流
功能：解析PDF->文本重构->分块处理->向量生成->Milvus存储
环境要求：Python 3.10+，Milvus 2.4.x
"""
import pymupdf
from openai import OpenAI
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import re

# API配置 - 使用兼容模式连接到阿里云DashScope
api_key="sk-XXXXXXXX"
api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"

def extract_pdf_text(path):
    """
    从PDF文件中提取文本内容
    
    参数:
        path: PDF文件路径
        
    返回:
        提取的文本字符串
    """
    try:
        doc = pymupdf.open(path)
        return " ".join([page.get_text() for page in doc])
    except Exception as e:
        raise RuntimeError(f"PDF解析失败: {str(e)}")

def reload_text(path):
    """
    使用大模型重构PDF提取的文本，提高文本质量
    
    参数:
        path: PDF文件路径
        
    返回:
        重构后的文本字符串
    """
    text = extract_pdf_text(path)

    # 初始化OpenAI客户端，连接到阿里云DashScope
    client = OpenAI(
        api_key = api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    try:
        # 调用DeepSeek-v3模型重构文本
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "user", 
                "content": "这是pdf 解析出来的文档 请你仔细阅读，将内容调整为完整的句子。同时需要确保不能丢失原文的信息。解析文档内容如下：{}".format(text)
                }
                ]
        )
        reload_text=response.choices[0].message.content
        return reload_text
    except Exception as e:
        print(f"API调用失败，启用降级处理: {str(e)}")
        # 降级处理：清理特殊字符并合并空格
        cleaned_text = re.sub(r'[^\w\u4e00-\u9fff\s。？！；，]', '', text)
        return re.sub(r'\s+', ' ', cleaned_text)

def split_text(path):
    """
    对重构后的文本进行分段并生成向量表示
    
    参数:
        path: PDF文件路径
        
    返回:
        tuple: (分段文本列表, 对应的向量表示)
    """
    text=reload_text(path)
    
    # 使用正则表达式按照标点符号分割文本
    split_pattern = r'(?<=[。！？；.?!!;\n])\s*'
    chunks = [c.strip() for c in re.split(split_pattern, text) if len(c.strip()) > 20]

    # 滑动窗口合并短文本，每5个句子为一组
    # 这里有很多处理的办法 文本随便演示了一种
    final_chunks = []
    for i in range(0, len(chunks), 3):
        window = chunks[i:i+5]
        final_chunks.append(" ".join(window))

    # 打印分段结果
    for index,value in enumerate(final_chunks):
        print(f"第{index}个句子-->{value}")
    
    # 加载本地预训练的向量模型
    model = SentenceTransformer('C:/Users/Administrator/Desktop/easy-vectordb/model/bge-small-zh-v1.5')
    
    # 生成文本向量并进行L2归一化
    embeddings = model.encode(final_chunks)
    embeddings = normalize(embeddings, norm='l2')
    
    return final_chunks, embeddings

# 主程序入口
if __name__ == "__main__":
    path="C:/Users/Administrator/Desktop/easy-vectordb/code/Datawhale社区介绍.pdf"
    texts_list,embeddings=split_text(path)
    
     # 连接到Milvus向量数据库
    connections.connect(alias="default", uri="http://127.0.0.1:19530")

    # 定义向量表结构
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  # 主键字段
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),  # 512维浮点向量
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000)  # 原始文本内容
    ]

    # 创建集合（表）
    schema = CollectionSchema(
            fields,
            description="PDF知识库",
            enable_dynamic_field=True
        )
    collection = Collection("pdf_test", schema)

    # 创建混合索引
    vector_index = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024}
        }
        
    collection.create_index("vector", vector_index)
    collection.create_index("content", {"index_type": "TRIE"})

    # 批量插入（含动态字段）
    data = [
        [i for i in range(len(texts_list))], # 主键ID列表
        embeddings.tolist(), # 向量列表（转换为Python列表）
        texts_list # 原始文本列表
    ]

    # 执行插入并刷新内存
    collection.insert(data)
    collection.flush()
    collection.load()  # 将索引加载到内存

```

