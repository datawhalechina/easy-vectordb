## Milvus基础操作

Milvus 是一款开源向量数据库，适配各种规模的 AI 应用，在本指南中，将引领你在数分钟内完成 Milvus 的本地设置，并借助 Python 客户端库实现向量的生成、存储与搜索。这里运用的 Milvus Lite，是pymilvus中包含的 Python 库，可嵌入客户端应用程序。

1. 设置向量数据库

要创建本地的 Milvus 向量数据库，仅需实例化一个`MilvusClient`，并指定用于存储所有数据的文件名，如`"milvus_demo.db"`。

在 Milvus 里，需要借助 `Collections` 来存储向量及其相关元数据，可将其类比为传统 SQL 数据库中的表格。创建 `Collections` 时，能定义 `Schema` 和索引参数，以此配置向量规格，包括维度、索引类型和远距离度量等。此外，还有一些复杂概念用于优化索引，提升向量搜索性能。但就目前而言，重点关注基础知识，并尽量采用默认设置。至少，需设定 Collections 的名称和向量场的维度。例如：
```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("milvus_demo.db")  # 数据库文件路径

# 创建schema
schema = client.create_schema(
    auto_id=False,  # 是否自动生成主键ID
    enable_dynamic_field=True,  # 是否启用动态字段
)

# 添加字段
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)  # 字段名、数据类型、是否为主键
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)  # 向量字段，维度为128

# 创建集合
client.create_collection(
    collection_name="my_collection",  # 集合名称
    schema=schema  # 集合的模式定义
)
```
上述代码中，主键和向量字段采用默认名称`（"id"和"vector"）`，度量类型（向量距离定义）设为默认值（COSINE） 。

2. 插入向量

Milvus 期望数据以字典列表的形式插入，每个字典代表一条数据记录，称作一个实体。假设已有向量化后的数据vectors（为浮点数数组列表），以及对应的实体 ID 列表ids，可按如下方式插入数据：
```python
entities = [
    {"id": id, "vector": vector}  # 实体字典，包含ID和向量数据
    for id, vector in zip(ids, vectors)  # 将ID列表和向量列表组合
]
client.insert("my_collection", entities)  # 集合名称，要插入的实体列表
```

3. 向量搜索

Milvus 可同时处理一个或多个向量搜索请求。`query_vectors`变量是一个向量列表，其中每个向量都是一个浮点数数组。
```python
query_vectors = embedding_fn.encode_queries(("Who is Alan Turing?",))  # 将查询文本编码为向量
```
执行搜索的示例代码如下：
```python
results = client.search(
    collection_name="my_collection",  # 要搜索的集合名称
    data=query_vectors,  # 查询向量数据
    limit=5,  # 返回结果的最大数量
    output_fields=["id"]  # 需要返回的字段列表
)
```
输出结果是一个结果列表，每个结果对应一个向量搜索查询。每个查询包含一个结果列表，其中每个结果涵盖实体主键、到查询向量的距离，以及指定output_fields的实体详细信息。

还能在过滤指定的标量（标量指非向量数据）的同时进行向量搜索，可通过指定特定条件的过滤表达式达成。例如，假设集合中存在一个名为"category"的标量字段，要搜索"category"为"tech"的相关向量，可这样操作：

```python
results = client.search(
    collection_name="my_collection",
    data=query_vectors,
    limit=5,
    output_fields=["id"],
    filter='category == "tech"'  # 过滤条件表达式
)
```

4. 加载现有数据

由于 Milvus Lite 的所有数据都存储于本地文件，即便程序终止，也能通过创建带有现有文件的MilvusClient，将所有数据加载至内存。例如，恢复"milvus_demo.db"文件中的集合，并继续写入数据：
```python
client = MilvusClient("milvus_demo.db")  # 连接到现有的数据库文件
```

## Collection

每个数据库中可包含多个Collection，类似于关系数据库中的表和记录。
Collection是一个二维表格，拥有固定的列和行，每一列表示一个字段，每一行表示一个实体，

### 构建Collection

构建一个Collection需要如下三个步骤：

1. 需要创建Schema，Schema定义了Collection的列和字段的类型。
2. 设置索引参数（可选）。
3. 创建Collection

首先，对于Schema，参考如下代码：
```python
from pymilvus import MilvusClient, DataType

# 创建MilvusClient实例
client = MilvusClient(uri="http://localhost:19530")  # Milvus服务器地址

# 定义collection schema
schema = client.create_schema(
    auto_id=False,  # 不自动生成ID
    enable_dynamic_fields=True,  # 启用动态字段
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=1024)  # 文本字段，最大长度1024
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)  # 1024维向量字段
```
其次创建索引，加载集合
```python
# 设置索引参数
index_params = {
    "index_type": "IVF_FLAT",  # 索引类型：倒排文件平面索引
    "metric_type": "IP",  # 距离度量类型：内积
    "params": {
        "nlist": 1024  # 聚类中心数量
    }
}

# 在vector字段上创建索引
collection.create_index(
    field_name="embedding",  # 要创建索引的字段名
    index_params=index_params,  # 索引参数配置
    timeout=None  # 超时时间，None表示无限等待
)

# 加载整个collection
collection.load()  # 将集合加载到内存中

# 或加载指定字段
collection.load(
    load_fields=["id", "embedding"],  # 指定要加载的字段列表
    skip_load_dynamic_field=True  # 跳过动态字段的加载
)
```
你可以使用如下代码检查你数据库中存在的Collection:
```python
res = client.describe_collection(
    collection_name="quick_setup"  # 要查看的集合名称
)

print(res)  # 打印集合详细信息

```
您可以按以下方式重命名一个 Collection:
```python
client.rename_collection(
    old_name="video_push",  # 原集合名称
    new_name="Dw_easy_vectorDB"  # 新集合名称
)
```
Milvus在查询时速度很快，是因为每次加载的Collection的数据会缓存在内存中。为了减少内存的消耗，您可以使用动态字段的方式加载你需要的数据进入Milvus。
```python
client.load_collection(
    collection_name="Dw_easy_vectorDB",  # 要加载的集合名称
    load_fields=["id", "embedding"],  # 指定加载的字段
    skip_load_dynamic_field=True  # 跳过动态字段加载
)
```
这样做除了能减少内存的消耗外，还有一个好处，即后续使用GPU HMB和SSD对大规模数据进行优化存储和加速检索时，id字段所占用的空间非常小，往往只有几bit，在CPU和SSD、HMB之间传输时，性能差异不大，因此可以忽略消耗。
关于这一部分，请学习完成所有知识后，浏览[FusionANNS]()。

当你使用结束此Collection后，请及时释放Collection，释放内存。
```python
client.release_collection(
    collection_name="Dw_easy_vectorDB"  # 要释放的集合名称
)

res = client.get_load_state(
    collection_name="Dw_easy_vectorDB"  # 要查询加载状态的集合名称
)

print(res)  # 打印加载状态
```
释放collection后，get_load_state()会返回NotLoad状态，表明collection已成功从内存中释放。
这样做可以有效减少内存消耗，特别是在处理大规模数据时非常重要

如果你的使用场景是大模型问答系统，对于用户提供的信息数据，你需要快速的向量化存储，以便提供更加高质量的回答，并且你使用的模型上下文比较短的情况下，可以通过设置Collection的TTL来实现。用户投喂的大规模文档数据，存储到Milvus中，设置TTL后，Milvus会自动删除超过指定时间的数据：

* TTL以秒为单位指定
* 删除过程是异步的，可能会有延迟
* 过期的实体不会出现在搜索或查询结果中
* 实际删除会在后续的数据压缩过程中进行，通常在24小时内
  

你可以使用如下代码来实现：
```python
#  创建新的集合
from pymilvus import MilvusClient

client.create_collection(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    schema=schema,  # 集合模式
    properties={
        "collection.ttl.seconds": 604800  # TTL设置，7天后自动删除数据
    }
)

```
```python
#  修改已存在的集合
client.alter_collection_properties(
    collection_name="Dw_easy_vectorDB",  # 要修改的集合名称
    properties={"collection.ttl.seconds": 604800}  # 新的TTL属性设置
)
```

### Collection中设置数据分区
当你第一次创建一个Collection时，所有的数据都会被存储在一个默认分区中。然而，为了更好地管理数据，你可以创建多个分区，并将数据分布到这些分区中。分区可以让你更有效地管理和查询数据，例如，你可以根据时间戳将数据分布到不同的分区中，以便按时间范围查询数据。

创建分区：
```python
client.create_partition(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    partition_name="partition_1"  # 分区名称
)
```
释放分区：
```python
client.release_partitions(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    partition_names=["partition_1"]  # 要释放的分区名称列表
)
```
删除分区之前必须要先释放分区。

删除分区：
```python
client.drop_partition(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    partition_name="partition_1"  # 要删除的分区名称
)
```
查询分区：
```python
client.list_partitions(
    collection_name="Dw_easy_vectorDB"  # 要查询分区的集合名称
)
```
向分区中插入数据：
```python
client.insert(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    partition_name="partition_1",  # 目标分区名称
    data=entities  # 要插入的实体数据
)
```
查询分区中的数据：
```python
client.query(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    partition_names=["partition_1"],  # 要查询的分区名称列表
    filter="your_filter_expression"  # 过滤条件表达式
)
```
不过，对于某些问答系统，分区的设计会影响查询性能。我们很难确定对于某一个问题的答案，应该从哪个分区中查询，除此之外，我们不能保证另一个不相干的分区中是否包含了某条可能对最终回答产生重要影响的数据。所以，不建议使用分区。

### 索引
参考[Milvus 索引介绍](./milvus%20索引介绍.md)
## 基本向量搜索
核心概念：ANN 搜索和 kNN 搜索
这俩是向量搜索的两种方式，先搞懂它们的区别。
1. 什么是 “kNN 搜索”？
全称是 “k 近邻搜索”。简单说就是：拿你的查询向量，和数据库里所有向量一个个比，找出最像的前 k 个（k 是你指定的数量，比如前 5 个）。

举个例子：
你有 1000 张动物图片，每张都转成了向量。现在你拿一张 “猫” 的图片向量去搜，kNN 会把这张向量和 1000 张的向量全比一遍，最后挑出最像的 5 张（大概率也是猫）。

缺点：如果数据库里有 10 亿个向量，一个个比就太慢了，像在 10 亿本书里找一本相似的，从头翻到尾，耗时又耗力。

2. 什么是 “ANN 搜索”？
全称是 “近似近邻搜索”。它不跟所有向量比，而是用 “偷懒” 的办法：提前给向量做 “分类整理”（也就是建 “索引”），然后只在 “可能相似的小范围” 里找，最后返回差不多像的前 k 个。

还是刚才的例子：
提前把 1000 张动物图片分好类（比如 “猫科”“犬科”“鸟类”），建一个 “索引”（类似分类标签）。搜 “猫” 的向量时，ANN 会先通过索引定位到 “猫科” 分类，只在这个小分类里比，不用看犬科和鸟类，速度就快多了。

特点：虽然可能不是 100% 最像的（但差距很小），但速度快太多，适合大数据量的场景（比如手机上的图片搜索、聊天机器人回答问题）。

1. ANN 搜索基础
与 kNN 的区别：kNN 需比较所有向量，耗时耗资源；ANN 依赖预建索引文件，快速定位相似向量子组，平衡性能与正确性。
AUTOINDEX：自动分析集合数据分布，优化索引参数，降低使用门槛，适配多种度量类型。
度量类型：不同度量方式对应不同相似度判断标准，如L2（值越小越相似）、IP（值越大越相似）等，距离范围各有不同。
1. 主要搜索操作
* 单向量搜索：针对单个查询向量，根据索引和度量类型返回前 K 个最相似向量。
* 批量向量搜索：同时处理多个查询向量，Milvus 并行执行搜索并返回对应结果集，代码结构与单向量类似，仅需传入向量列表。
* 分区中的 ANN 搜索：通过指定partition_names参数将搜索范围限制在特定分区，减少数据量以提升性能，适用于集合内有多个分区的场景。
* 使用输出字段：默认返回主键和距离，可通过output_fields指定额外字段（如color），使结果包含更多实体信息。
* 限制与偏移：limit控制单次返回结果数（top-K），offset用于分页查询（跳过指定数量结果），两者总和需小于 16384。


了解完这些基本的概念后，我们可以开始编写代码来使用 Milvus 进行搜索。


### 标量查询（Query）

与向量搜索不同，标量查询主要用于根据标量字段的条件来检索数据，不涉及向量相似度计算。

#### 基本查询操作

```python
from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")  # 连接到Milvus服务器

results = client.query(
    collection_name="product_recommendation",  # 要查询的集合名称
    filter="",  # 空表达式查询所有数据
    output_fields=["id", "category", "brand", "price"],  # 要返回的字段列表
    limit=100  # 限制返回100条记录
)

print("所有商品数据：")
for result in results:
    print(f"ID: {result['id']}, 类别: {result['category']}, "
          f"品牌: {result['brand']}, 价格: {result['price']}")
```

#### 条件查询

```python
# 2. 基于单个条件查询
results = client.query(
    collection_name="product_recommendation",
    filter='category == "electronics"',  # 单个条件过滤
    output_fields=["id", "category", "brand", "price"]
)

print("电子产品：")
for result in results:
    print(f"ID: {result['id']}, 品牌: {result['brand']}, 价格: {result['price']}")

# 3. 基于数值范围查询
results = client.query(
    collection_name="product_recommendation",
    filter='price >= 100 and price <= 1000',  # 数值范围过滤条件
    output_fields=["id", "category", "brand", "price"]
)

print("价格在100-1000之间的商品：")
for result in results:
    print(f"ID: {result['id']}, 类别: {result['category']}, "
          f"品牌: {result['brand']}, 价格: {result['price']}")
```

#### 复杂查询表达式

```python
# 4. 使用 IN 操作符
results = client.query(
    collection_name="product_recommendation",
    filter='category in ["electronics", "clothing"] and price < 500',  # IN操作符和逻辑AND组合
    output_fields=["id", "category", "brand", "price"]
)

# 5. 使用 LIKE 操作符（字符串模糊匹配）
results = client.query(
    collection_name="product_recommendation",
    filter='brand like "App%"',  # LIKE操作符，%为通配符
    output_fields=["id", "category", "brand", "price"]
)

# 6. 使用逻辑运算符组合条件
results = client.query(
    collection_name="product_recommendation",
    filter='(category == "electronics" and price > 500) or (category == "clothing" and price < 100)',  # 复杂逻辑组合
    output_fields=["id", "category", "brand", "price"]
)
```

#### 查询结果限制和排序

```python
# 7. 限制返回结果数量
results = client.query(
    collection_name="product_recommendation",
    filter='category == "electronics"',
    output_fields=["id", "category", "brand", "price"],
    limit=10  # 只返回前10条结果
)

# 8. 使用偏移量实现分页
results = client.query(
    collection_name="product_recommendation",
    filter='category == "electronics"',
    output_fields=["id", "category", "brand", "price"],
    limit=10,  # 每页返回10条
    offset=20  # 跳过前20条，实现分页
)
```

### 数据删除操作

Milvus 支持根据条件删除数据，删除操作是异步执行的,过期的实体不会立即从搜索或查询结果中消失，而是会在后续的数据压缩过程中被移除，通常在24小时内完成。

```python
# 1. 根据主键删除
client.delete(
    collection_name="product_recommendation",  # 目标集合名称
    filter="id in [1, 2, 3]"  # 删除条件：ID在指定列表中的记录
)

# 2. 根据条件删除
client.delete(
    collection_name="product_recommendation",
    filter='category == "discontinued" and price < 10'  # 复合删除条件
)

# 3. 删除特定品牌的所有商品
client.delete(
    collection_name="product_recommendation",
    filter='brand == "OldBrand"'  # 按品牌删除
)

print("删除操作已提交，正在异步执行...")
```

### 数据更新操作（Upsert）

Milvus 支持 Upsert 操作，即如果数据存在则更新，不存在则插入。
当您执行upsert操作时，Milvus会执行以下流程：

* 检查集合的主字段是否启用了AutoId
* 如果启用了AutoId，Milvus会用自动生成的主键替换实体中的主键并插入数据
* 如果没有启用，Milvus会使用实体携带的主键来插入数据
* 基于upsert请求中包含的实体的主键值执行删除操作
```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",  # Milvus服务器地址
    token="root:Milvus"  # 认证令牌，格式为用户名:密码
)


res = client.upsert(
    collection_name="test_collection",  # 目标集合名称
    data=[  # 要插入或更新的数据列表
        {
            'id': 1,  # 实体ID
             'vector': [  # 向量数据
                 0.3457690490452393,
                 -0.9401784221711342,
                 0.9123948134344333,
                 0.49519396415367245,
                 -0.558567588166478
             ]
       },
       {
           'id': 2,
           'vector': [
               0.42349086179692356,
               -0.533609076732849,
               -0.8344432775467099,
               0.675761846081416,
               0.57094256393761057
           ]
       }
   ]
)

# {'upsert_count': 2}  # 返回结果：更新插入的记录数量
```
### 数据统计和聚合

```python
# 1. 统计总记录数
count_result = client.query(
    collection_name="product_recommendation",
    filter="",  # 空过滤条件
    limit=10,  # 使用空filter时必须指定limit
    output_fields=["count(*)"]  # 返回计数聚合函数结果
)
print(f"总记录数: {count_result[0]['count(*)']}")

# 2. 按条件统计
electronics_count = client.query(
    collection_name="product_recommendation",
    filter='category == "electronics"',  # 按类别过滤
    output_fields=["count(*)"]  # 统计函数
)
print(f"电子产品数量: {electronics_count[0]['count(*)']}")

# 3. 统计不同类别的商品数量
categories = ["electronics", "clothing", "books"]  # 类别列表
for category in categories:
    count = client.query(
        collection_name="product_recommendation",
        filter=f'category == "{category}"',  # 动态构建过滤条件
        output_fields=["count(*)"]
    )
    print(f"{category} 商品数量: {count[0]['count(*)']}")
```

## 混合搜索（Hybrid Search）

在Milvus中，"混合搜索"（Hybrid Search）特指对多个向量字段进行搜索并重新排序的功能

### 向量搜索 + 标量过滤
```python
# 1. 基本混合搜索
from pymilvus import AnnSearchRequest

# 创建多个搜索请求
search_param_1 = {
    "data": [query_dense_vector],  # 密集向量查询数据
    "anns_field": "text_dense",  # 要搜索的密集向量字段名
    "param": {"nprobe": 10},  # 搜索参数：探测的聚类数量
    "limit": 2  # 返回结果数量限制
}
request_1 = AnnSearchRequest(**search_param_1)  # 创建ANN搜索请求对象

search_param_2 = {
    "data": [query_text],  # 稀疏向量查询数据
    "anns_field": "text_sparse",  # 要搜索的稀疏向量字段名
    "param": {"drop_ratio_search": 0.2},  # 搜索参数：丢弃比例
    "limit": 2
}
request_2 = AnnSearchRequest(**search_param_2)

reqs = [request_1, request_2]  # 搜索请求列表
```

### 多条件复合过滤

```python
# 2. 复杂条件组合
results = client.search(
    collection_name="product_recommendation",
    data=[query_vector],  # 查询向量
    limit=5,
    # 复杂的过滤条件
    filter='(category == "electronics" and brand in ["Apple", "Samsung"]) or (category == "clothing" and price < 200)',  # 多条件逻辑组合
    output_fields=["id", "category", "brand", "price"]
)

# 3. 时间范围过滤（假设有时间字段）
# 注意：需要在Schema中定义时间字段
results = client.search(
    collection_name="product_recommendation",
    data=[query_vector],
    limit=10,
    filter='category == "electronics" and created_time >= "2024-01-01" and created_time <= "2024-12-31"',  # 时间范围过滤
    output_fields=["id", "category", "brand", "price", "created_time"]  # 包含时间字段的输出
)
```

### 地理位置搜索示例

```python
# 假设有地理位置相关的Collection
# 4. 地理位置范围搜索
results = client.search(
    collection_name="location_based_products",  # 基于位置的产品集合
    data=[query_vector],
    limit=10,
    # 搜索特定地理范围内的商品
    filter='latitude >= 39.9 and latitude <= 40.1 and longitude >= 116.3 and longitude <= 116.5',  # 地理坐标范围过滤
    output_fields=["id", "name", "latitude", "longitude", "category"]  # 包含地理位置的输出字段
)
```

## 批量操作和事务

### 批量插入优化

```python
import random
from pymilvus import MilvusClient

def batch_insert_large_data(client, collection_name, data, batch_size=1000):
    """
    分批插入大量数据，避免单次插入过多导致的性能问题
    """
    total_count = len(data)  # 总数据量

    for i in range(0, total_count, batch_size):  # 按批次大小分割数据
        batch_data = data[i:i + batch_size]  # 当前批次数据

        try:
            client.insert(
                collection_name=collection_name,  # 目标集合
                data=batch_data  # 当前批次的数据
            )
            print(f"已插入 {min(i + batch_size, total_count)}/{total_count} 条记录")

        except Exception as e:
            print(f"批次 {i//batch_size + 1} 插入失败: {e}")
            continue  # 跳过失败的批次，继续处理下一批

# 创建客户端
client = MilvusClient(uri="http://localhost:19530")

# 生成测试数据
large_dataset = []
for i in range(10000):  # 生成10000条测试数据
    large_dataset.append({
        "id": i,  # 唯一ID
        "category": f"category_{i % 10}",  # 循环生成10个类别
        "brand": f"brand_{i % 100}",  # 循环生成100个品牌
        "price": 10.0 + (i % 1000),  # 价格范围10-1010
        "embedding": [random.random() for _ in range(768)]  # 768维随机向量
    })

# 调用函数
batch_insert_large_data(client, "product_recommendation", large_dataset)  # 执行批量插入
```

### 批量删除

```python
import random
from pymilvus import MilvusClient

def batch_delete_by_ids(client, collection_name, ids, batch_size=100):
    """
    分批删除大量数据 - 使用 ids 参数
    """
    total_count = len(ids)  # 要删除的ID总数

    for i in range(0, total_count, batch_size):  # 按批次处理
        batch_ids = ids[i:i + batch_size]  # 当前批次的ID列表

        try:
            result = client.delete(
                collection_name=collection_name,  # 目标集合
                ids=batch_ids  # 要删除的ID列表
            )
            print(f"已删除 {min(i + batch_size, total_count)}/{total_count} 条记录")
            print(f"删除数量: {result.get('delete_count', 0)}")  # 实际删除数量

        except Exception as e:
            print(f"批次删除失败: {e}")
            continue  # 跳过失败批次

def batch_delete_by_filter(client, collection_name, filter_expr, batch_size=100):
    """
    使用过滤条件批量删除数据
    """
    try:
        result = client.delete(
            collection_name=collection_name,  # 目标集合
            filter=filter_expr  # 删除条件表达式
        )
        print(f"根据条件删除完成，删除数量: {result.get('delete_count', 0)}")

    except Exception as e:
        print(f"条件删除失败: {e}")

# 创建客户端
client = MilvusClient(uri="http://localhost:19530")

# 方式1: 通过 ID 列表删除
ids_to_delete = list(range(1000, 2001))  # 生成ID范围1000-2000的列表
batch_delete_by_ids(client, "product_recommendation", ids_to_delete)

# 方式2: 通过过滤条件删除
batch_delete_by_filter(
    client,
    "product_recommendation",
    "id >= 1000 and id <= 2000"  # 删除条件：ID在指定范围内
)
```

### 分区中的 ANN 搜索
假设您在 Collections 中创建了多个分区，您可以将搜索范围缩小到特定数量的分区。在这种情况下，您可以在搜索请求中包含目标分区名称，将搜索范围限制在指定的分区内。减少搜索所涉及的分区数量可以提高搜索性能。

下面的代码片段假定在你的 Collections 中有一个名为PartitionA的分区。
```python
query_vector = [.........]  # 查询向量数据
res = client.search(
    collection_name="Dw_easy_vectorDB",  # 目标集合名称
    partition_names=["partitionA"],  # 指定要搜索的分区列表
    data=[query_vector],  # 查询向量列表
    limit=3,  # 返回前3个最相似结果
)

for hits in res:  # 遍历搜索结果
    print("TopK results:")
    for hit in hits:  # 遍历每个命中结果
        print(hit)  # 打印结果详情
```
### 使用输出字段

在搜索结果中，Milvus 默认包含包含 top-K 向量嵌入的实体的主字段值和相似性距离/分数。您可以在搜索请求中包含目标字段（包括向量和标量字段）的名称作为输出字段，以使搜索结果携带这些实体中其他字段的值。
```python
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592],  # 查询向量

res = client.search(
    collection_name="Dw_easy_vectorDB",  # 集合名称
    data=[query_vector],  # 查询向量数据
    limit=3,  # 返回结果数量限制
    search_params={"metric_type": "IP"},  # 搜索参数：使用内积距离度量
    output_fields=["color"]  # 指定返回的额外字段
)

print(res)  # 打印搜索结果
```

### 使用限制和偏移
您可能会注意到，搜索请求中携带的参数limit 决定了搜索结果中包含的实体数量。该参数指定了单次搜索中返回实体的最大数量，通常称为top-K。
比如搜出来 100 个相似结果，一页显示 20 个，就可以用 “limit=20”（每页 20 个）和 “offset=20”（跳过前 20 个，看第 2 页）。但注意：一次最多看 16384 个结果，太多了会变慢。
```python
query_vector = [.............],  # 查询向量

res = client.search(
    collection_name="Dw_easy_vectorDB",
    data=[query_vector],
    limit=3,  # 每页返回3条结果
    search_params={
        "metric_type": "IP",  # 距离度量类型
        "offset": 10  # 偏移量：跳过前10条结果
    }
)
```

### 使用分区密钥

分区密钥（Partition Key）是一种基于分区的搜索优化解决方案。通过指定特定标量字段作为 Partition Key，并在搜索过程中根据 Partition Key 指定过滤条件，可以将搜索范围缩小到多个分区，从而提高搜索效率。

#### 什么是分区密钥？

分区密钥是一种特殊的标量字段，用于自动将数据分布到不同的分区中。与手动创建分区不同，使用分区密钥可以让 Milvus 根据字段值自动管理数据分布，实现更高效的查询性能。

举个例子：
假设你有一个电商推荐系统，存储了不同类别商品的向量数据。如果将 "category"（商品类别）设置为分区密钥，Milvus 会自动将 "electronics"、"clothing"、"books" 等不同类别的商品数据分布到不同的分区中。

#### 创建带分区密钥的 Collection

首先，我们需要在创建 Collection 时指定分区密钥：

```python
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

client = MilvusClient("http://localhost:19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  # 主键字段
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),  # 分区密钥字段
    FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=64),  # 品牌字段，最大长度64
    FieldSchema(name="price", dtype=DataType.FLOAT),  # 价格字段，浮点数类型
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # 768维向量字段
]

schema = CollectionSchema(
    fields=fields,  # 字段定义列表
    description="Product recommendation collection with partition key"  # 集合描述
)

# 创建 Collection
client.create_collection(
    collection_name="product_recommendation",  # 集合名称
    schema=schema,  # 集合模式
    num_partitions=64  # 分区数量
)
```

#### 插入数据到分区密钥 Collection

插入数据时，Milvus 会根据分区密钥字段的值自动将数据分配到相应的分区：

```python
# 准备插入数据
entities = [
    {
        "id": 1,  # 实体ID
        "category": "electronics",  # 分区密钥：电子产品
        "brand": "Apple",  # 品牌
        "price": 999.99,  # 价格
        "embedding": [0.1, 0.2, 0.3, ...]  # 768维向量数据
    },
    {
        "id": 2,
        "category": "clothing",  # 分区密钥：服装
        "brand": "Nike",
        "price": 89.99,
        "embedding": [0.4, 0.5, 0.6, ...]  # 768维向量
    },
    {
        "id": 3,
        "category": "electronics",  # 分区密钥：电子产品
        "brand": "Samsung",
        "price": 799.99,
        "embedding": [0.7, 0.8, 0.9, ...]  # 768维向量
    },
    {
        "id": 4,
        "category": "books",  # 分区密钥：图书
        "brand": "Penguin",
        "price": 19.99,
        "embedding": [0.2, 0.4, 0.6, ...]  # 768维向量
    }
]

# 插入数据，Milvus 会根据 category 字段自动分区
client.insert(
    collection_name="product_recommendation",  # 目标集合
    data=entities  # 要插入的实体数据列表
)

print("数据插入完成，已根据 category 字段自动分区")
```

#### 使用分区密钥进行高效搜索

使用分区密钥进行搜索时，可以显著提升查询性能：

```python
# 1. 基于分区密钥的精确搜索
# 只在 "electronics" 分区中搜索
query_vector = [0.1, 0.2, 0.3, ...]  # 查询向量

res = client.search(
    collection_name="product_recommendation",
    data=[query_vector],  # 查询向量列表
    limit=5,  # 返回结果数量
    # 使用分区密钥过滤，只搜索电子产品分区
    filter='category == "electronics"',  # 分区密钥过滤条件
    output_fields=["id", "category", "brand", "price"]  # 返回字段
)

print("电子产品搜索结果：")
for hits in res:  # 遍历搜索结果
    for hit in hits:  # 遍历每个命中结果
        print(f"ID: {hit['id']}, 品牌: {hit['entity']['brand']}, "
              f"价格: {hit['entity']['price']}, 距离: {hit['distance']}")  # 打印结果详情
```

```python
# 2. 多分区搜索
# 在多个分区中搜索
res = client.search(
    collection_name="product_recommendation",
    data=[query_vector],
    limit=5,
    # 搜索多个类别
    filter='category in ["electronics", "clothing"]',  # 多分区密钥过滤
    output_fields=["id", "category", "brand", "price"]
)

print("电子产品和服装搜索结果：")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 类别: {hit['entity']['category']}, "
              f"品牌: {hit['entity']['brand']}, 距离: {hit['distance']}")
```

```python
# 3. 结合其他过滤条件
# 在特定分区中进行复合条件搜索
res = client.search(
    collection_name="product_recommendation",
    data=[query_vector],
    limit=5,
    # 组合分区密钥和其他条件
    filter='category == "electronics" && price < 900',  # 分区密钥+价格条件组合
    output_fields=["id", "category", "brand", "price"]
)

print("价格低于900的电子产品：")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 品牌: {hit['entity']['brand']}, "
              f"价格: {hit['entity']['price']}, 距离: {hit['distance']}")
```

#### 查看分区信息

可以查看 Collection 的分区分布情况：

```python
# 查看所有分区
partitions = client.list_partitions(collection_name="product_recommendation")  # 获取分区列表
print("分区列表：", partitions)

# 查看 Collection 详细信息
collection_info = client.describe_collection(collection_name="product_recommendation")  # 获取集合详细信息
print("Collection 信息：", collection_info)

# 统计不同类别的数据量
categories = ["electronics", "clothing", "books"]  # 分区密钥值列表

for category in categories:  # 遍历每个类别
    count = client.query(
        collection_name="product_recommendation",
        filter=f'category == "{category}"',  # 按类别过滤
        output_fields=["count(*)"]  # 返回计数结果
    )
    print(f"类别 {category} 数据量: {count}")
```

**注意事项：**
1. **分区密钥选择**：
   - 选择具有良好分布特性的字段（避免数据倾斜）
   - 常用于查询过滤的字段
   - 基数适中的字段（不要太少也不要太多）

```txt
# 好的分区密钥示例
# - 用户地区：["北京", "上海", "广州", "深圳", ...]
# - 商品类别：["electronics", "clothing", "books", ...]
# - 时间分片：["2024-01", "2024-02", "2024-03", ...]
```
```txt
# 不好的分区密钥示例
# - 用户ID：基数太大，分区过多
# - 性别：基数太小，分区太少
# - 连续数值：如价格，分布不均匀
```

2. **查询模式**：
   - 尽量在查询中包含分区密钥过滤条件
   - 避免跨所有分区的全局搜索

```python
# 推荐的查询方式
filter='category == "electronics"'  # 利用分区密钥

# 不推荐的查询方式  
filter='price > 100'  # 没有使用分区密钥，需要扫描所有分区

# 基于单个分区密钥值的过滤
filter='partition_key == "x" && <other conditions>'

# 基于多个分区密钥值的过滤
filter='partition_key in ["x", "y", "z"] && <other conditions>'
```

3. **分区数量限制**：
   - 默认最大分区数为 1024
   - 分区过多会影响性能
   - 建议根据实际数据分布调整

#### 实际应用场景

**1. 多租户系统**
```python
# 以租户ID作为分区密钥
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  # 主键字段
    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=32, is_partition_key=True),  # 租户ID作为分区密钥
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=1000),  # 文档内容字段
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # 文档向量字段
]

# 查询时只搜索特定租户的数据
res = client.search(
    collection_name="multi_tenant_docs",  # 多租户文档集合
    data=[query_vector],  # 查询向量
    filter='tenant_id == "company_a"',  # 租户过滤条件
    limit=10  # 返回结果数量
)
```

**2. 时间序列数据**
```python
# 以时间分片作为分区密钥
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  # 主键字段
    FieldSchema(name="time_bucket", dtype=DataType.VARCHAR, max_length=16, is_partition_key=True),  # 时间分片作为分区密钥，如"2024-01"
    FieldSchema(name="sensor_data", dtype=DataType.FLOAT_VECTOR, dim=128)  # 传感器数据向量字段
]

# 查询特定时间段的数据
res = client.search(
    collection_name="sensor_data",  # 传感器数据集合
    data=[query_vector],  # 查询向量
    filter='time_bucket in ["2024-01", "2024-02"]',  # 时间范围过滤条件
    limit=10  # 返回结果数量
)
```

通过合理使用分区密钥，可以在大规模向量数据场景下获得显著的性能提升，同时简化数据管理的复杂度。


