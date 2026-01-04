# Chapter 3: Milvus 基础操作 - PyMilvus 核心API实战

## 3.1 学习目标

  - 掌握向量数据库在实际应用中的基础操作，理解其在应用系统中的核心作用  
  - 熟悉 PyMilvus 的核心 API 使用方式，包括数据库连接与基本配置  
  - 理解 Collection 的创建与管理流程，能够进行合理的 Schema 设计  
  - 掌握向量与结构化数据的写入方法，确保数据高效入库  
  - 熟练使用查询（Query）与向量搜索（Search）接口，实现向量检索功能  
  - 了解并掌握索引的创建与管理机制，提升向量检索性能  
  - 能够综合运用以上能力，基于 Milvus 构建完整的向量检索应用系统  

## 3.2 环境准备与数据库连接

Milvus 提供了两种部署模式：Milvus Standalone（单机版）和 Milvus Cluster（集群版）。对于开发测试和小规模应用，可以直接使用 Milvus Lite，这是一个轻量级的 Python 库，可以将数据存储在本地文件中。

> Milvus Lite只支持linux系统，相关的部分应用可见第6章 【Milvus Lite部署与应用】
>
> Milvus Cluster（集群版）需要使用集群服务器，并不太适合教程学习，因此本项目更加推荐Milvus Standalone模型来进行学习。

### 3.2.1 安装 PyMilvus

首先需要安装 PyMilvus 客户端库：

```bash
pip install pymilvus
```

如果你的网络环境无法访问 PyPI，可以考虑使用国内镜像源：

```bash
pip install pymilvus -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2.2 连接到 Milvus 服务

PyMilvus 提供了 `MilvusClient` 类作为与 Milvus 交互的主要接口。根据部署方式的不同，连接方式也有所区别。

> MilvusClient 是 PyMilvus 提供的**客户端接口**，用于以更简单直观的方式完成 Milvus 的连接、数据写入、查询、向量搜索和集合管理等常见操作。

**本地文件模式（Milvus Lite）**

这种方式适用于快速原型开发和小规模数据场景，所有数据存储在本地文件中：

```python
from pymilvus import MilvusClient

# 连接到本地数据库文件，如果文件不存在会自动创建
client = MilvusClient("./milvus_demo.db")
```

**服务器模式**

连接到运行在本地的 Milvus 服务器：

```python
client = MilvusClient(uri="http://localhost:19530")
```

如果 Milvus 服务器启用了认证功能，需要提供用户名和密码：

```python
client = MilvusClient(
    uri="https://localhost:19530",
    token="root:Milvus"
)
```

连接成功后，可以通过以下命令检查服务器状态：

```python
from pymilvus import utility

# 检查服务器连接状态
print(utility.get_server_version(client))
```

如果上述命令成功返回版本号，说明连接已建立。但是如果报错：`pymilvus.exceptions.ConnectionConfigException: <ConnectionConfigException: (code=1, message=Alias should be string, but <class 'pymilvus.milvus_client.milvus_client.MilvusClient'> is given.)>`，是因为新版pymilvus打印版本号的api发生了更改：

```python
# 检查服务器连接状态
print(client.get_server_version())
```

## 3.3 Collection 与 Schema 设计

Collection 是 Milvus 中存储和管理数据的基本容器,可以将其类比为关系数据库中的表。每个 Collection 都需要一个 Schema 来定义其结构,包括字段名称、数据类型、主键等。

### 3.3.1 理解 Schema

Schema 定义了 Collection 的数据结构,包括:

- **字段(Field)**: 每个字段有名称、数据类型和可选的属性
- **主键(Primary Key)**: 唯一标识每条记录的字段,支持 INT64 或 VARCHAR 类型
- **向量字段**: 用于存储向量数据的特殊字段,需要指定维度
- **标量字段**: 存储元数据的字段,如字符串、整数、浮点数等

### 3.3.2 支持的数据类型

Milvus 支持多种数据类型:

#### 数值类型

| 数据类型 | 说明         | 示例用途             |
| -------- | ------------ | -------------------- |
| INT8     | 8位整数      | 小范围计数           |
| INT16    | 16位整数     | 中等范围计数         |
| INT32    | 32位整数     | 标准整数             |
| INT64    | 64位整数     | 主键ID、大范围计数器 |
| FLOAT    | 单精度浮点数 | 价格、分数           |
| DOUBLE   | 双精度浮点数 | 高精度数值           |

#### 字符串和布尔类型

| 数据类型 | 说明       | 注意事项                   |
| -------- | ---------- | -------------------------- |
| VARCHAR  | 变长字符串 | 需要指定 `max_length` 参数 |
| BOOL     | 布尔值     | 是/否标记                  |

#### 向量类型

| 数据类型            | 说明                         | 适用场景                              |
| ------------------- | ---------------------------- | ------------------------------------- |
| FLOAT_VECTOR        | 32位浮点向量                 | 标准嵌入向量,科学计算和机器学习       |
| FLOAT16_VECTOR      | 16位半精度浮点向量           | 深度学习和GPU计算                     |
| BFLOAT16_VECTOR     | 16位Brain Floating Point向量 | 提供与Float32相同的指数范围但精度降低 |
| INT8_VECTOR         | 8位整数向量                  | 量化深度学习模型(仅支持HNSW索引)      |
| BINARY_VECTOR       | 二进制向量                   | 高效的二进制表示                      |
| SPARSE_FLOAT_VECTOR | 稀疏浮点向量                 | 稀疏数据表示                          |

#### 复合类型

| 数据类型 | 说明     | 配置要求                                  |
| -------- | -------- | ----------------------------------------- |
| JSON     | JSON对象 | 存储半结构化数据                          |
| ARRAY    | 数组     | 需要指定 `element_type` 和 `max_capacity` |

### 3.3.3 创建 Schema 示例

#### Python SDK 示例

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

# 创建 schema
schema = MilvusClient.create_schema(
    auto_id=False,  # 是否自动为主键字段生成唯一 ID。False 表示主键由你自己提供
    enable_dynamic_field=True, # 是否开启动态字段。True 表示可在插入数据时携带未在 schema 中显式定义的额外键值对，MILVUS 会以动态列存储
)

# 添加主键字段
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)

# 添加向量字段
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)

# 添加标量字段
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)
```

### 3.3.4 主键字段配置

主键字段有以下特点:

- **支持的数据类型**:INT64 或 VARCHAR
- **唯一性**:每个实体必须有唯一的主键值
- **非空性**:主键字段不能为空

#### AutoID 配置

Milvus 支持两种主键管理模式:

**1. 自动生成 ID (AutoID)**

```python
schema.add_field(
    field_name="id",
    is_primary=True,
    auto_id=True,  # Milvus 自动生成 ID
    datatype=DataType.INT64
)
```

使用 AutoID 时:

- Milvus 自动为每个实体生成唯一 ID
- 插入数据时不需要提供主键字段值
- 适合大多数不需要手动管理 ID 的场景

**2. 手动指定 ID**

```python
schema.add_field(
    field_name="product_id",
    is_primary=True,
    auto_id=False,  # 手动提供 ID
    datatype=DataType.VARCHAR,
    max_length=100
)
```

手动指定 ID 时:

- 需要在插入数据时提供主键字段值
- 适合需要与外部系统对齐的场景
- 必须确保所有 ID 的唯一性

### 3.3.5 向量字段配置

向量字段需要指定维度:

```python
schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=768  # 向量维度
)
```

**维度限制**:

- Milvus 默认支持最多 32,768 维的向量
- 可以通过配置 `Proxy.maxDimension` 支持更大维度

### 3.3.6 动态字段

启用动态字段后,可以存储未在 Schema 中定义的字段:

```python
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,  # 启用动态字段
)
```

启用动态字段时:

- Milvus 会创建一个名为 `$meta` 的保留字段
- 未定义的字段会以键值对形式存储在动态字段中
- 适合需要灵活存储额外元数据的场景

### 3.3.7 创建 Schema

创建 Schema 有两种方式：使用 `create_schema()` 方法或使用 `FieldSchema` 和 `CollectionSchema` 类。

**方式一：使用 create_schema()**

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("milvus_demo.db")

# 创建 Schema
schema = client.create_schema(
    auto_id=False,           # 手动指定ID，不自动生成
    enable_dynamic_field=True # 启用动态字段，可以插入未预定义的字段
)

# 添加字段
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True
)

schema.add_field(
    field_name="title",
    datatype=DataType.VARCHAR,
    max_length=256
)

schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=768  # 向量维度，要与使用的嵌入模型维度一致
)

schema.add_field(
    field_name="price",
    datatype=DataType.FLOAT
)
```

**方式二：使用 CollectionSchema**

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

# 定义字段
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=False
)

title_field = FieldSchema(
    name="title",
    dtype=DataType.VARCHAR,
    max_length=256
)

embedding_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=768
)

price_field = FieldSchema(
    name="price",
    dtype=DataType.FLOAT
)

# 创建 Schema
schema = CollectionSchema(
    fields=[id_field, title_field, embedding_field, price_field],
    description="Product collection",
    enable_dynamic_field=True
)
```

### 3.3.8 创建 Collection

创建 Collection 需要指定集合名称、Schema 和可选的索引参数。

```python
# 基本创建方式
client.create_collection(
    collection_name="products",
    schema=schema
)

# 创建带索引的 Collection
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",  # 需要建立索引的字段名。应为向量字段
    index_type="IVF_FLAT",  # 索引类型。IVF_FLAT表示倒排文件平铺索引，适合大规模向量的近似搜索；常见还有IVF_SQ8`、`HNSW` 等
    metric_type="COSINE",  # 距离度量方式。COSINE余弦相似度；常用还有L2`（欧氏距离）、`IP`（内积）。使用 `COSINE` 时一般需要对向量做归一化。
    params={"nlist": 128} # 索引的超参数字典。{"nlist": 128} 中nlist` 表示聚类中心数量（分桶数），影响召回与性能；值越大召回率越高但索引与搜索更慢。根据数据规模调优（例如 10K~1M 样本可从 64~4096 试配）。
)

client.create_collection(
    collection_name="products",
    schema=schema,
    index_params=index_params
)
```

**常用的度量类型**

| 度量类型 | 说明         | 距离范围 | 适用场景                     |
| -------- | ------------ | -------- | ---------------------------- |
| COSINE   | 余弦相似度   | [-1, 1]  | 文本向量，关注方向而非大小   |
| L2       | 欧几里得距离 | [0, ∞)   | 通用向量，关注绝对距离       |
| IP       | 内积         | [-∞, ∞)  | 归一化向量，与余弦相似度相关 |

###  3.3.9 Collection 的基本管理

#### 查看 Collection 信息

列出数据库中所有已创建的 Collection：

```python
# 列出所有 Collection
collections = client.list_collections()
print(collections)
```

#### 重命名 Collection

重命名现有的 Collection。注意：为 Collection 创建的别名在此操作后保持不变：

```python
client.rename_collection(
    old_name="products",
    new_name="product_catalog"
)
```

#### 删除 Collection

删除不再需要的 Collection。此操作不可逆，请谨慎使用：

```python
client.drop_collection(collection_name="product_catalog")
```

#### 加载与释放 Collection

**为什么需要加载？**

在执行相似性搜索或查询之前，必须将 Collection 加载到内存中。 加载操作会将索引文件和所有字段的原始数据加载到内存，以实现快速响应。

**加载整个 Collection**

对于小规模数据，可以加载整个 Collection：

```python
# 由于之前rename和drop操作，重新创建名为"products"的Collection
client.create_collection(
    collection_name="products",
    schema=schema
)

# 创建带索引的 Collection
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 128}
)

client.create_collection(
    collection_name="products",
    schema=schema,
    index_params=index_params
)

# 列出所有 Collection
collections = client.list_collections()
print(collections)

# 加载整个 Collection
client.load_collection(
    collection_name="products"
)
```

**选择性加载字段（优化内存使用）**

对于大规模数据，可以只加载搜索和查询所需的字段，以减少内存占用并提升性能：

```python
# 只加载指定字段
client.load_collection(
    collection_name="products",
    load_fields=["id", "embedding"], # load_fields: 可选，指定需预加载的字段列表。只加载这些字段，其它字段不加载以省内存。向量检索至少应包含向量字段（如 embedding），通常也加载主键（如 id）。若查询/输出引用未加载字段会失败或无法返回该列。默认 None 表示加载全部字段。
    skip_load_dynamic_field=True  # 可选，布尔值。集合启用动态字段时，是否跳过动态列（如内部的 \$meta）的加载。True 跳过以节省内存；False 一并加载。默认通常为 False
)
```

**注意事项：**

- `load_fields` 中必须包含主键字段和至少一个向量字段
- 只有 `load_fields` 中指定的字段可用于过滤条件和输出字段
- `skip_load_dynamic_field=True` 会跳过加载动态字段（`$meta`），使其无法用于过滤和输出

**释放 Collection**

搜索和查询是内存密集型操作。使用完毕后应释放 Collection 以节省成本：

```python
# 释放 Collection
client.release_collection(collection_name="products")

# 检查加载状态
status = client.get_load_state(collection_name="products")
print(status)  # 输出: {'state': <LoadState: Loaded>} 或 {'state': <LoadState: NotLoad>}
```

`get_load_state()` 返回的状态包括：

- `Loaded`：Collection 已加载

- `Loading`：Collection 正在加载中

- `NotLoad`：Collection 未加载

## 3.4 数据写入操作

### 3.4.1 准备数据

Milvus 接受的数据格式是字典列表，每个字典代表一条记录。在插入数据之前，需要准备好向量数据和元数据。向量数据通常由嵌入模型生成，这里我们使用随机数据进行演示。

```python
import random

# 生成模拟向量数据
def generate_embeddings(num_vectors, dim=768):
    return [[random.random() for _ in range(dim)] for _ in range(num_vectors)]

# 准备数据
data = [
    {
        "id": 1,
        "title": "Apple iPhone 15 Pro",
        "embedding": generate_embeddings(1)[0],
        "price": 999.99,
        "brand": "Apple",
        "category": "smartphone"
    },
    {
        "id": 2,
        "title": "Samsung Galaxy S24",
        "embedding": generate_embeddings(1)[0],
        "price": 899.99,
        "brand": "Samsung",
        "category": "smartphone"
    },
    {
        "id": 3,
        "title": "Sony WH-1000XM5",
        "embedding": generate_embeddings(1)[0],
        "price": 349.99,
        "brand": "Sony",
        "category": "headphones"
    }
]
```

**数据格式要求：**

- 每条记录必须包含 Collection schema 中定义的所有必填字段
- 主键字段（如 `id`）必须提供，除非设置了 `auto_id=True`
- 向量字段的维度必须与 schema 中定义的维度一致

### 3.4.2 插入数据

**单次插入**

使用 `insert()` 方法将数据插入到 Collection 中：

```python
insert_result = client.insert(
    collection_name="products",
    data=data
)

print(f"插入的记录数: {insert_result['insert_count']}")
```

`insert()` 方法返回一个包含以下信息的结果对象：

- `insert_count`：成功插入的实体数量
- `primary_keys`：插入实体的主键列表
- `timestamp`：操作完成的时间戳

注意，在新版本的 PyMilvus 中，`insert()` 方法的返回值是一个字典，而不是之前版本中的 `InsertResult` 对象。`Returns: Dict: Number of rows that were inserted and the inserted primary key list.`,如：
`{'insert_count': 3, 'ids': [1, 2, 3]}`

**批量插入**

对于大规模数据，建议分批插入以提高性能和稳定性：

```python
def batch_insert(client, collection_name, data, batch_size=1000):
    """分批插入数据"""
    total = len(data)
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        result = client.insert(
            collection_name=collection_name,
            data=batch
        )
        print(f"已插入 {min(i + batch_size, total)}/{total} 条记录")
    
    print("所有数据插入完成")

# 使用示例
large_data = [
    {
        "id": i, 
        "embedding": generate_embeddings(1)[0],
        "title": f"Product {i}",
        "price": 100.0 + i
    }
    for i in range(10000)
]
batch_insert(client, "products", large_data, batch_size=500)
```

**插入到指定分区**

如果 Collection 中有多个分区，可以指定插入到特定分区：

```python
# 创建分区
client.create_partition(
    collection_name="products",
    partition_name="electronics"
)

client.insert(
    collection_name="products",
    data=data,
    partition_name="electronics"
)
```

**注意事项：**

- 插入数据前，Collection 可以处于加载或未加载状态
- 如果启用了 `auto_id=True`，则不要在数据中包含主键字段
- 插入操作是异步的，数据可能不会立即可搜索，可以调用 `flush()` 确保数据持久化

### 3.4.3 数据更新（Upsert）

Upsert 操作结合了"更新"和"插入"：如果记录已存在（根据主键判断），则更新该记录；否则插入新记录。这是一个数据级操作，会覆盖已存在的实体。

```python
update_data = [
    {
        "id": 1,  # 已存在的记录，会被更新
        "title": "Apple iPhone 15 Pro Max",
        "price": 1099.99,
        "embedding": generate_embeddings(1)[0],
        "brand": "Apple",
        "category": "smartphone"
    },
    {
        "id": 999,  # 不存在的记录，会被插入
        "title": "Google Pixel 8",
        "price": 699.99,
        "embedding": generate_embeddings(1)[0],
        "brand": "Google",
        "category": "smartphone"
    }
]

upsert_result = client.upsert(
    collection_name="products",
    data=update_data
)

print(f"Upsert 处理的记录数: {upsert_result['upsert_count']}")
```

`upsert()` 方法返回的结果对象包含：

- `upsert_count`：成功 upsert 的实体数量
- `primary_keys`：受影响实体的主键列表
- `timestamp`：操作完成的时间戳

**Upsert 到指定分区**

同样可以将 upsert 操作限定在特定分区：

```python
client.upsert(
    collection_name="products",
    data=update_data,
    partition_name="electronics"
)
```

**使用场景：**

- 需要更新已存在记录的某些字段
- 不确定记录是否存在，希望统一处理插入和更新逻辑
- 定期同步外部数据源到 Milvus

**注意事项：**

- Upsert 会完全覆盖已存在的实体，确保提供所有必需字段
- 性能略低于纯插入操作，因为需要先检查主键是否存在
- 与 insert 一样，upsert 也是异步操作

## 3.5 向量搜索

向量搜索是 Milvus 的核心功能，用于找出与查询向量最相似的数据记录。

### 3.5.1 基本向量搜索

**单向量搜索**

```python
# 加载 Collection
client.load_collection(
    collection_name="products"
)

# 检查加载状态
status = client.get_load_state(collection_name="products")
print(status)

query_vector = generate_embeddings(1)[0]

results = client.search(
    collection_name="products",
    data=[query_vector],  # 查询向量列表
    limit=5,              # 返回前5个最相似的结果
    output_fields=["title", "price", "brand"]  # 返回的字段
)

# 处理搜索结果
for i, hits in enumerate(results):
    print(f"查询 {i+1} 的结果:")
    for hit in hits:
        print(f"  ID: {hit['id']}")
        print(f"  距离: {hit['distance']:.4f}")
        print(f"  标题: {hit['title']}")
        print(f"  价格: ${hit['price']}")
        print()
```

**批量向量搜索**

Milvus 支持在一次请求中处理多个查询向量：

```python
query_vectors = generate_embeddings(3)  # 3个查询向量

results = client.search(
    collection_name="products",
    data=query_vectors,
    limit=3,
    output_fields=["title", "price"]
)

print(results)

# results[0] 对应第一个查询向量的结果
# results[1] 对应第二个查询向量的结果
# 以此类推
```

### 3.5.2 向量搜索 + 标量过滤

在实际应用中，经常需要在向量搜索的同时添加标量过滤条件，实现精确检索和语义检索的结合。

```python
# 只搜索电子产品类别
results = client.search(
    collection_name="products",
    data=[query_vector],
    filter='category == "smartphone"',
    limit=5,
    output_fields=["title", "category", "price"]
)

print(results)

# 复杂过滤条件
results = client.search(
    collection_name="products",
    data=[query_vector],
    filter='category == "smartphone" and price < 1000',
    limit=5,
    output_fields=["title", "price"]
)

print(results)

# 使用 IN 操作符
results = client.search(
    collection_name="products",
    data=[query_vector],
    filter='brand in ["Apple", "Samsung", "Google"]',
    limit=5,
    output_fields=["title", "brand"]
)

print(results)
```

**支持的过滤表达式**

Milvus 支持丰富的布尔表达式用于标量过滤：

| 操作符       | 说明         | 示例                            |
| ------------ | ------------ | ------------------------------- |
| ==, !=       | 等于、不等于 | `price == 999`                  |
| >, >=, <, <= | 比较运算符   | `price > 500`                   |
| in, not in   | 包含、不包含 | `brand in ["Apple", "Samsung"]` |
| and, or, not | 逻辑运算符   | `price > 500 and price < 1000`  |
| like         | 模糊匹配     | `title like "%Pro%"`            |

### 3.5.3 搜索参数配置

搜索行为可以通过 `search_params` 参数进行精细控制：

```python
results = client.search(
    collection_name="products",
    data=[query_vector],
    limit=10,
    search_params={
        "metric_type": "COSINE",    # 距离度量类型
        "params": {"nprobe": 10}     # 搜索参数
    }
)

print(results)
```

**不同索引类型的搜索参数**：

- **IVF 系列索引**（IVF_FLAT, IVF_SQ8, IVF_PQ）：
  - `nprobe`：要查询的聚类单元数量，范围 [1, nlist]，默认值 8

- **HNSW 索引**：
  - `ef`：控制查询时间/准确度权衡的参数，更高的 `ef` 会带来更准确但更慢的搜索，范围 [top_k, int_max]

- **SCANN 索引**：
  - `nprobe`：要查询的聚类单元数量
  - `reorder_k`：要查询的候选单元数量，范围 [top_k, ∞]，默认值为 top_k

### 3.5.4 分页搜索

使用 `offset` 参数实现分页：

```python
# 第一页：前10条结果
page1 = client.search(
    collection_name="products",
    data=[query_vector],
    limit=10,
    offset=0
)

print(page1)

# 第二页：跳过前10条，返回第11-20条
page2 = client.search(
    collection_name="products",
    data=[query_vector],
    limit=10,
    offset=10
)

print(page2)
```

**注意事项**：

- `offset` 与 `limit` 的总和应小于 16,384
- 对于大规模分页，建议使用 `search_iterator()` 方法以获得更好的性能

## 3.6 标量查询

标量查询（Query）不涉及向量相似度计算，直接根据标量字段的值进行数据检索。

> 标量查询就体现了数据库特有优势，Miluvs相比faiss来说会更像数据库一些，保留非向量查询的一些基础功能

### 3.6.1 基本查询

```python
# 查询所有数据（限制返回数量）
results = client.query(
    collection_name="products",
    filter="",  # 空过滤条件表示查询所有
    output_fields=["id", "title", "price"],
    limit=100
)

print(results)

# 根据ID查询
results = client.query(
    collection_name="products",
    filter='id in [1, 2, 3]',
    output_fields=["id", "title", "price"]
)

print(results)

# 条件查询
results = client.query(
    collection_name="products",
    filter='category == "smartphone" and price > 500',
    output_fields=["id", "title", "brand", "price"]
)

print(results)
```

**重要说明**：

- 当 `filter` 设置为空字符串时，必须设置 `limit` 来限制返回的实体数量
- `offset` 和 `limit` 的总和应小于 16,384

### 3.6.2 字符串匹配查询

```python
# 精确匹配
results = client.query(
    collection_name="products",
    filter='brand == "Apple"',
    output_fields=["id", "title"]
)

print(results)

# 模糊匹配
results = client.query(
    collection_name="products",
    filter='title like "%Pro%"',  # 标题包含 "Pro"
    output_fields=["id", "title"]
)

print(results)
```

### 3.6.3 数值范围查询

```python
# 价格区间查询
results = client.query(
    collection_name="products",
    filter='price >= 100 and price <= 1000',
    output_fields=["id", "title", "price"]
)

print(results)

# 多条件组合
results = client.query(
    collection_name="products",
    filter='(category == "smartphone" and price > 800) or (category == "headphones" and price < 200)',
    output_fields=["id", "category", "title", "price"]
)

print(results)
```

### 3.6.4 统计查询

Milvus 支持使用 `count(*)` 来统计满足条件的实体数量：

```python
# 统计总数
count_result = client.query(
    collection_name="products",
    filter="",
    output_fields=["count(*)"]
)
total_count = count_result[0]["count(*)"]
print(f"总记录数: {total_count}")

# 按类别统计
categories = ["smartphone", "headphones", "tablet"]
for category in categories:
    result = client.query(
        collection_name="products",
        filter=f'category == "{category}"',
        output_fields=["count(*)"]
    )
    count = result[0]["count(*)"]
    print(f"{category}: {count} 条记录")
```

**查询优化建议**：

- 对于大规模数据集的查询，建议使用 `query_iterator()` 方法进行迭代查询。query()：一次性返回满足过滤条件的所有结果（可配合limit、output_fields）。适合结果集较小的场景。
  query_iterator()：返回一个迭代器，按批次分页拉取结果，内存友好，适合结果集较大的场景或需要流式处理。
- 查询前确保 Collection 已加载到内存中
- 可以通过 `consistency_level` 参数调整一致性级别以平衡性能和准确性

举个query()和query_iterator()的对比例子：

```python
# 一次性查询：直接拿到所有匹配记录
all_results = client.query(
    collection_name="products",
    filter='category == "smartphone" and price < 1000',
    output_fields=["id", "title", "price", "category"],
    limit=1000  # 可选，控制最大返回条数
)
print(f"总结果数: {len(all_results)}")
for row in all_results:
    print(row["id"], row["title"], row["price"])

# 迭代式查询：分批获取，适合海量数据
it = client.query_iterator(
    collection_name="products",
    filter='category == "smartphone" and price < 1000',
    output_fields=["id", "title", "price", "category"],
    batch_size=200
)

total = 0
while True:
    batch = it.next()
    if not batch:
        break
    for row in batch:
        total += 1
        # 处理每条记录
        print(row["id"], row["title"], row["price"])

print(f"累计结果数: {total}")
```

## 3.7 索引管理

索引是提升向量搜索性能的关键。Milvus 支持多种索引类型,不同的索引适用于不同的场景。

> 如果忘记了索引相关的内容，可以跳转到向量基础【向量ANN搜索算法】重新复习一下~

### 3.7.1 索引类型选择

| 索引类型   | 特点                                             | 适用场景                                  |
| ---------- | ------------------------------------------------ | ----------------------------------------- |
| FLAT       | 精确搜索,无压缩                                  | 相对较小的数据集,需要100%召回率           |
| IVF_FLAT   | 高速查询,尽可能高的召回率                        | 中等规模数据                              |
| IVF_SQ8    | 非常高速查询,内存资源有限,可接受轻微召回率损失   | 内存受限场景                              |
| IVF_PQ     | 高速查询,内存资源有限,可接受轻微召回率损失       | 大规模数据                                |
| IVF_RABITQ | 高精度1-bit量化                                  | 大规模数据集,需要高召回率同时优化资源利用 |
| HNSW       | 非常高速查询,尽可能高的召回率,大内存资源         | 高性能需求                                |
| HNSW_SQ    | 非常高速查询,内存资源有限,可接受轻微召回率损失   | 高性能需求且内存受限                      |
| HNSW_PQ    | 中速查询,非常有限的内存资源,可接受轻微召回率损失 | 内存极度受限                              |
| HNSW_PRQ   | 中速查询,非常有限的内存资源,可接受轻微召回率损失 | 内存极度受限                              |
| SCANN      | 非常高速查询,尽可能高的召回率,大内存资源         | 高性能需求                                |
| DISKANN    | 磁盘索引                                         | 超大规模数据                              |
| AUTOINDEX  | 自动优化                                         | 快速启动,所有字段类型                     |

### 3.7.2 创建索引

**创建向量索引**

```python
# 准备索引参数
index_params = client.prepare_index_params()

# 添加向量字段索引
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,
        "efConstruction": 256
    }
)

# 在创建 Collection 时指定索引
client.create_collection(
    collection_name="products1",
    schema=schema,
    index_params=index_params
)
```

**为已存在的 Collection 创建索引**

```python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 128}
)

client.create_index(
    collection_name="products",
    index_params=index_params
)
```

### 3.7.3 标量字段索引

对于经常用于过滤的标量字段,可以创建标量索引以提升查询性能[(3)](https://milvus.io/docs/scalar_index.md):

```python
# 为数值字段创建索引
index_params.add_index(
    field_name="price",
    index_type="INVERTED"
)

client.create_index(
    collection_name="products",
    index_params=index_params
)
```

**标量字段支持的索引类型**:

- **INVERTED**: 倒排索引,适用于除JSON外的所有标量字段
- **STL_SORT**: 排序索引,适用于数值类型字段
- **TRIE**: 字典树索引,仅适用于VarChar字段
- **BITMAP**: 位图索引,适用于Bool、Int8、Int16、Int32、Int64、VarChar和Array字段(元素类型为上述之一)
- **AUTOINDEX**: 自动索引,根据数据类型自动选择最佳索引

### 3.7.4 索引管理操作

```python
# 查看索引信息
indexes = client.list_indexes(collection_name="products")
print(indexes)

# 描述索引详情
index_info = client.describe_index(
    collection_name="products",
    index_name="embedding_vector_idx"
)
print(index_info)

# 删除索引
client.drop_index(
    collection_name="products",
    index_name="embedding_vector_idx"
)
```

### 3.7.5 使用 AUTOINDEX

AUTOINDEX 是 Milvus 提供的智能索引选项,会根据数据类型自动选择最合适的索引类型:

```python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

client.create_collection(
    collection_name="products2",
    schema=schema,
    index_params=index_params
)
```

**AUTOINDEX 自动选择规则**:

| 数据类型               | 自动选择的索引算法 |
| ---------------------- | ------------------ |
| VARCHAR                | 倒排索引           |
| INT8/INT16/INT32/INT64 | 倒排索引           |
| FLOAT/DOUBLE           | 倒排索引           |

## 3.8 数据删除操作

### 3.8.1 根据主键删除

```python
# 删除指定ID的记录
client.delete(
    collection_name="products",
    ids=[1, 2, 3]  # 要删除的主键ID列表
)
```

### 3.8.2 根据条件删除

```python
# 加载 Collection
client.load_collection(
    collection_name="products"
)

# 根据单个条件删除
client.delete(
    collection_name="products",
    filter='price < 100'
)

# 根据复杂条件删除
client.delete(
    collection_name="products",
    filter='brand == "OldBrand" or (category == "discontinued" and price < 50)'
)

# 删除特定时间之前的数据
client.delete(
    collection_name="products",
    filter='created_at < "2024-01-01"'
)
```

**注意**：Milvus 的删除操作是异步执行的，数据不会立即从磁盘中删除，而是在后续的数据压缩过程中清理。
 根据可用的信息源,我对您的文档进行了修改和补充。以下是修改后的完整内容:

### 3.8.3 批量删除优化

Milvus 支持通过布尔表达式删除实体。对于大规模数据删除,建议使用过滤表达式:

```python
# 使用过滤表达式删除
res = client.delete(
    collection_name="products",
    filter='price < 100 and category == "old_items"'
)
```

**在分区中删除数据**

```python
# 在指定分区中删除
client.delete(
    collection_name="products",
    partition_name="electronics",
    filter="id in [0, 1]"
)
```

删除操作的注意事项:

- 删除请求必须包含主键或过滤表达式
- 目标集合必须已加载并可用于查询
- 删除是异步操作,在删除完成前搜索和查询仍然可能返回已删除的实体

## 3.9 分区管理

分区是集合的子集,每个分区与其父集合共享相同的数据结构,但只包含集合中数据的一个子集。合理使用分区可以提升查询性能并简化数据管理。

### 3.9.1 创建和管理分区

**默认分区**

创建集合时,Milvus 会自动创建一个名为 **_default** 的分区。如果不添加其他分区,所有插入的实体都会进入默认分区。

**创建分区**

一个集合最多可以有 1,024 个分区。

```python
# 创建分区
client.create_partition(
    collection_name="products",
    partition_name="partitionA"
)
```

**查看分区**

```python
# 列出所有分区
res = client.list_partitions(
    collection_name="products"
)
print(res)
```

**检查分区是否存在**

```python
res = client.has_partition(
    collection_name="products",
    partition_name="partitionA"
)
print(res)
# 输出: True
```

**删除分区**

删除分区前需要确保分区已释放。

```python
# 释放分区
client.release_partitions(
    collection_name="products",
    partition_names=["partitionA"]
)

# 删除分区
client.drop_partition(
    collection_name="products",
    partition_name="partitionA"
)

# 验证删除
res = client.list_partitions(
    collection_name="products"
)
print(res)
```

### 3.9.2 加载和释放分区

**加载分区**

您可以单独加载集合中的特定分区。如果集合中有未加载的分区,集合的加载状态将保持为未加载。

```python
# 创建分区
client.create_partition(
    collection_name="products",
    partition_name="partitionA"
)

client.load_partitions(
    collection_name="products",
    partition_names=["partitionA"]
)

res = client.get_load_state(
    collection_name="products",
    partition_name="partitionA"
)
print(res)
# 输出: {"state": "<LoadState: Loaded>"}
```

**释放分区**

```python
client.release_partitions(
    collection_name="products",
    partition_names=["partitionA"]
)

res = client.get_load_state(
    collection_name="products",
    partition_name="partitionA"
)
print(res)
# 输出: {"state": "<LoadState: NotLoaded>"}
```

### 3.9.3 使用 Partition Key

Partition Key 是一种基于分区的搜索优化解决方案。通过指定特定的标量字段作为 Partition Key,并在搜索时基于 Partition Key 指定过滤条件,可以将搜索范围缩小到若干分区,从而提升搜索效率。

**设置 Partition Key**

在添加标量字段时,将其 `is_partition_key` 属性设置为 `True`:

```python
schema.add_field(
    field_name="my_varchar", 
    datatype=DataType.VARCHAR, 
    max_length=512,
    is_partition_key=True,
)
```

**设置分区数量**

当指定标量字段作为 Partition Key 时,Milvus 会自动在集合中创建 16 个分区。您也可以自定义分区数量:

```python
client.create_collection(
    collection_name="my_collection",
    schema=schema,
    num_partitions=128
)
```

**基于 Partition Key 的过滤**

在启用 Partition Key 功能的集合中进行 ANN 搜索时,需要在搜索请求中包含涉及 Partition Key 的过滤表达式:

```python
# 基于单个 partition key 值过滤
filter='partition_key == "x" && <other conditions>'

# 基于多个 partition key 值过滤
filter='partition_key in ["x", "y", "z"] && <other conditions>'
```

**启用 Partition Key Isolation**

Partition Key Isolation 功能可以进一步提升搜索性能。启用后,Milvus 会基于 Partition Key 值对实体进行分组,并为每个组创建单独的索引。

```python
client.create_collection(
    collection_name="my_collection1",
    schema=schema,
    properties={"partitionkey.isolation": True}
)
```

启用 Partition Key Isolation 后,基于 Partition Key 的过滤器应仅包含一个特定的 Partition Key 值。目前,Partition Key Isolation 功能仅适用于索引类型设置为 HNSW 的搜索。


## 3.10 数据过期管理（TTL）

对于某些应用场景,如临时数据存储或会话数据,可以设置 Collection 的 TTL(Time To Live),让数据在指定时间后自动过期。

### 3.10.1 TTL 概述

TTL(Time-to-Live)是数据库中常用的功能,用于在插入或修改数据后的特定时间段内保持数据有效或可访问。例如,如果您每天导入数据但只需要保留 14 天的记录,可以通过将集合的 TTL 设置为 **14 × 24 × 3600 = 1209600** 秒来配置 Milvus 自动删除超过该时间的数据。

**TTL 工作原理**:

- 过期的实体不会出现在任何搜索或查询结果中
- 过期数据可能会保留在存储中,直到下一次数据压缩(通常在 24 小时内进行)
- 可以通过设置 Milvus 配置文件中的 `dataCoord.compaction.expiry.tolerance` 配置项来控制何时触发数据压缩
- 该配置项默认为 `-1`,表示使用现有的数据压缩间隔。当更改为正整数(如 `12`)时,数据压缩将在任何实体过期后的指定小时数后触发

### 3.10.2 创建带 TTL 的 Collection

**Python**

```python
from pymilvus import MilvusClient

# 创建 Collection 时设置 TTL
client.create_collection(
    collection_name="my_collection2",
    schema=schema,
    properties={
        "collection.ttl.seconds": 1209600  # 14天
    }
)
```

### 3.10.3 修改已有 Collection 的 TTL

**Python**

```python
client.alter_collection_properties(
    collection_name="my_collection2",
    properties={"collection.ttl.seconds": 1209600}
)
```

### 3.10.4 删除 TTL 设置

如果您决定无限期保留集合中的数据,可以从该集合中删除 TTL 设置。

**Python**

```python
client.drop_collection_properties(
    collection_name="my_collection2",
    property_keys=["collection.ttl.seconds"]
)
```


## 3.11 小结

本章介绍了 PyMilvus 的核心 API 和基础操作，包括：

1. **环境准备**：如何安装和连接 Milvus
2. **Schema 设计**：定义 Collection 的数据结构
3. **Collection 管理**：创建、加载、释放、删除 Collection
4. **数据操作**：插入、更新、删除数据
5. **向量搜索**：基本搜索、过滤搜索、批量搜索
6. **标量查询**：基于条件的精确查询
7. **索引管理**：创建和管理索引以提升性能
8. **分区管理**：使用分区和分区密钥优化查询
9. **数据管理**：使用TTL管理数据

看到这里，恭喜你掌握Milvus相关基础知识，那么现在你就可以开始构建基于 Milvus 的向量检索应用了。

还等什么，出发吧~