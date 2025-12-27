
# 项目参考了以下开源项目
https://github.com/datawhalechina/all-in-rag/tree/main

## 运行程序的步骤
1. 配置环境
```shell
cd easy-vectordb/src/graph_rag/data
docker compose up -d
docker compose -f docker-compose.milvus.yml up -d
```
2. 安装依赖
```shell
conda create -n easy_vectordb python=3.10
conda activate easy_vectordb
pip install -r requirements.txt
```
3. 配置env.example文件
```

# Neo4j数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=all-in-rag
NEO4J_DATABASE=neo4j

# 选择向量数据库
VECTOR_DB =milvus #annoy faiss
# Milvus向量数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DIMENSION=1024
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_METRIC_TYPE =IP
MILVUS_COLLECTION_NAME=cooking_knowledge
# Annoy向量数据库配置
ANNOY_INDEX_PATH = "./annoy_index"
ANNOY_DIMENSION =1024
ANNOY_METRIC = angular # Annoy支持: "angular", "euclidean", "manhattan", "hamming"
ANNOY_N_TREES=100# 树的数量，越多越精确但索引越大


#Faiss向量数据库配置
FAISS_DIMENSION= 1024
FAISS_INDEX_PATH =./faiss_index
FAISS_INDEX_TYPE = IVF ## "Flat", "IVF", "HNSW"
FAISS_INDEX_NLIST = 100
FAISS_NLIST =100
# 模型配置
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
SILICONFLOW_EMBEDING_API_KEY=  #(可以不写)
SILICONFLOW_EMBEDDING_BASE_URL= #(可以不写)

# 月之暗面API密钥（用于LLM调用）
LLM_MODEL=
MOONSHOT_API_KEY=
MOONSHOT_BASE_URL=
# 日志级别
LOG_LEVEL=INFO
# 
```
1. 需要配置MOONSHOT_API_KEY和MOONSHOT_BASE_URL和SILICONFLOW_EMBEDING_API_KEY和SILICONFLOW_EMBEDDING_BASE_URL
2. 其中SILICONFLOW_EMBEDING_API_KEY，SILICONFLOW_EMBEDDING_BASE_URL可以不写。会EMBEDDING_MODEL自动部署BAAI/bge-small-zh-v1.5这个模型，该模型dimension为512 ，其他模型需要自行调整dimension。
3. 配置.env文件
```shell
cp env.example .env
# 需要配置月之暗面和硅基硫基的API_KEY
```
4. 运行程序
```python
python main.py
```
##  FAISS 索引构建模块 (FAISSIndexConstructionModule)
**注意** 想要学习本项目的rag的策略，需要学习 https://github.com/datawhalechina/all-in-rag/tree/main/docs/chapter9


本模块负责将文档块（Documents）转化为向量，并利用 **FAISS** 库构建高效的向量检索索引。

###  核心功能

* **多模型支持**：兼容 OpenAI API 风格的在线 Embedding 和 HuggingFace 本地模型。
* **多种索引类型**：
* `Flat`: 精确搜索（适用于小数据集）。
* `IVF`: 倒排文件索引（平衡速度与精度）。
* `HNSW`: 图索引（高性能近似搜索）。


* **持久化存储**：支持索引、元数据及配置信息的本地保存与快速加载。
* **元数据过滤**：在相似度搜索基础上提供基本的元数据过滤功能。

### 快速上手

#### 1. 初始化

```python
indexer = FAISSIndexConstructionModule(
    index_path="./my_faiss_index",
    dimension=768,
    model_name="BAAI/bge-base-zh-v1.5",
    index_type="HNSW"
)

```

#### 2. 构建索引

```python
# chunks 为 List[langchain_core.documents.Document]
indexer.build_vector_index(chunks)

```

#### 3. 相似度检索

```python
results = indexer.similarity_search("如何制作红烧肉？", k=5)

```

#### 4. 增量添加

```python
indexer.add_documents(new_chunks)

```

### 目录结构

生成的索引文件夹包含：

* `faiss.index`: 二进制向量索引文件。
* `metadata.pkl`: 存储文档文本及原始元数据。
* `config.pkl`: 存储维度、模型名等配置信息。




## 🧠Milvus 向量索引管理模块（MilvusIndexConstructionModule）

`MilvusIndexConstructionModule` 是本项目中负责“数据存储与检索”的核心大脑。它的主要任务是将文本信息转化为数学向量，并利用 **Milvus** 高性能向量数据库实现毫秒级的语义搜索。

### 1. 这个模块解决了什么问题？

在传统的 RAG 系统中，简单的文本匹配（如关键词搜索）无法理解语义。本模块通过以下技术栈解决了这个问题：

* **语义理解**：使用嵌入模型（Embedding）将“西红柿炒鸡蛋”和“番茄炒蛋”关联起来。
* **海量存储**：支持数百万级数据的快速插入与检索。
* **系统健壮性**：处理网络抖动、分词器下载失败及 API 频控等常见生产环境问题。

### 2. 关键功能拆解

#### A. 灵活的嵌入模型选择 (`_setup_embeddings`)

模块支持“双模”切换：

* **云端模式**：对接 OpenAI 兼容接口（如 DeepSeek, GPT），支持自定义 `base_url`。内置了 `request_timeout` 和 `max_retries` 机制，防止网络环境不佳导致构建中断。
* **本地模式**：支持 `BAAI/bge` 系列等 HuggingFace 本地模型，适合对数据隐私要求高或无外网连接的场景。

#### B. 生产级索引构建策略 (`build_vector_index`)

这是本模块最强大的部分，采用了 **“分批次（Batching）+ 延时保护（Sleep）”** 策略：

* **分批写入**：默认每 50 条数据为一个批次，有效防止内存溢出（OOM）。
* **延时策略**：批次间加入 `0.2s` 延迟，保护 Milvus 网关不被瞬时大流量压垮。
* **自动 Schema 管理**：自动根据模型定义的维度（Dimension）创建集合字段，支持 JSON 格式的元数据（Metadata）存储。

#### C. 高效语义搜索 (`similarity_search`)

* 支持 **IP（内积）** 或 **L2（欧氏距离）** 搜索。
* 结果返回包括：原始文本、向量相似度得分、以及存储的所有元数据（如菜谱分类、难度等）。

---

### 3. 给学员的上手指南

#### 💡 第一步：初始化

```python
# 初始化模块，指定数据库地址和使用的模型维度
milvus_manager = MilvusIndexConstructionModule(
    host="127.0.0.1",
    collection_name="my_recipe_db",
    dimension=512,  # 需与模型输出维度一致
    model_name="BAAI/bge-small-zh-v1.5"
)

```

#### 💡 第二步：一键构建索引

```python
# 传入处理好的 Document 对象列表
success = milvus_manager.build_vector_index(documents)
if success:
    print("知识库构建成功！")

```

#### 💡 第三步：测试检索

```python
# 检索最相关的 3 条结果
results = milvus_manager.similarity_search("夏天适合吃什么凉爽的菜？", k=3)
for res in results:
    print(f"得分: {res['score']}, 内容: {res['text'][:50]}...")

```

