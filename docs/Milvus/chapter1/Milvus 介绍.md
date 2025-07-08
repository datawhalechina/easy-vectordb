# 《Milvus 文档》

Milvus 是一款开源向量数据库，适配各种规模的 AI 应用，在本指南中，将引领你在数分钟内完成 Milvus 的本地设置，并借助 Python 客户端库实现向量的生成、存储与搜索。这里运用的 Milvus Lite，是pymilvus中包含的 Python 库，可嵌入客户端应用程序。
你需要先理解：

1. 什么是 “向量”？
你可以把 “向量” 理解成一串数字组成的 “特征密码”。生活中任何东西（比如一张图片、一句话、一个水果）都有自己的特征，向量就是把这些特征转换成数字的形式，方便后续进行相似度检索。

举个例子：

比如描述一个苹果，它的特征可能是：颜色（红色 = 1，绿色 = 0）、形状（圆形 = 1，方形 = 0）、甜度（0-10 分，假设 8 分）、大小（直径 5cm=5）。
那这个苹果的向量可能就是 [1, 1, 8, 5]—— 这串数字就代表了苹果的 “特征密码”。

在 AI 里，图片、文字都会被转换成这样的向量。比如一句话 “我爱吃苹果”，AI 会提取它的语义特征，变成一串更长的数字（比如 128 个数字），这就是 “文本向量”。

2. 什么是 “向量数据库”？
普通数据库（比如 Excel 表格）存的是文字、数字（比如 “苹果，红色，5 元”），而向量数据库专门存上面说的 “向量”（也就是 “特征密码”）。

为啥要专门存向量？因为 AI 处理数据时，不是直接比文字，而是比向量。比如想找和 “我爱吃苹果” 意思相似的句子，AI 会先把这句话转成向量，再去向量数据库里找 “数字长得最像” 的向量，对应的句子就是相似的。

Milvus 就是这样一个专门存向量的数据库，就像 “向量的仓库”。
## 快速入门
1. 设置向量数据库
要创建本地的 Milvus 向量数据库，仅需实例化一个`MilvusClient`，并指定用于存储所有数据的文件名，如`"milvus_demo.db"`。
```python
client = MilvusClient("milvus_demo.db")
```
在 Milvus 里，需要借助 `Collections` 来存储向量及其相关元数据，可将其类比为传统 SQL 数据库中的表格。创建 `Collections` 时，能定义 `Schema` 和索引参数，以此配置向量规格，包括维度、索引类型和远距离度量等。此外，还有一些复杂概念用于优化索引，提升向量搜索性能。但就目前而言，重点关注基础知识，并尽量采用默认设置。至少，需设定 Collections 的名称和向量场的维度。例如：
```python
from pymilvus import CollectionSchema, FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields)
client.create_collection("my_collection", schema)
```
上述代码中，主键和向量字段采用默认名称`（"id"和"vector"）`，度量类型（向量距离定义）设为默认值（COSINE） 。
2. 插入向量
Milvus 期望数据以字典列表的形式插入，每个字典代表一条数据记录，称作一个实体。假设已有向量化后的数据vectors（为浮点数数组列表），以及对应的实体 ID 列表ids，可按如下方式插入数据：
```python
entities = [
    {"id": id, "vector": vector}
    for id, vector in zip(ids, vectors)
]
client.insert("my_collection", entities)
```
3. 向量搜索
Milvus 可同时处理一个或多个向量搜索请求。`query_vectors`变量是一个向量列表，其中每个向量都是一个浮点数数组。
```python
query_vectors = embedding_fn.encode_queries(("Who is Alan Turing?",))
```
执行搜索的示例代码如下：
```python
results = client.search(
    collection_name="my_collection",
    query_vectors=query_vectors,
    top_k=5,
    output_fields=["id"]
)
```
输出结果是一个结果列表，每个结果对应一个向量搜索查询。每个查询包含一个结果列表，其中每个结果涵盖实体主键、到查询向量的距离，以及指定output_fields的实体详细信息。
还能在过滤指定的标量（标量指非向量数据）的同时进行向量搜索，可通过指定特定条件的过滤表达式达成。例如，假设集合中存在一个名为"category"的标量字段，要搜索"category"为"tech"的相关向量，可这样操作：
```python
expr = 'category == "tech"'
results = client.search(
    collection_name="my_collection",
    query_vectors=query_vectors,
    top_k=5,
    output_fields=["id"],
    expr=expr
)
```
4. 加载现有数据
由于 Milvus Lite 的所有数据都存储于本地文件，即便程序终止，也能通过创建带有现有文件的MilvusClient，将所有数据加载至内存。例如，恢复"milvus_demo.db"文件中的集合，并继续写入数据：
```python
client = MilvusClient("milvus_demo.db")
collection = client.get_collection("my_collection")
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
        # 连接Milvus服务器
        connections.connect("default", host="localhost", port="19530")

        # 检查并创建collection
        collection_name = 'video_push'
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            # 你可以选择构建一些其他的字段
        ]
        schema = CollectionSchema(fields, collection_name)
        collection = Collection(name=collection_name, schema=schema)
```
其次创建索引，加载集合
```python
        collection.create_index(
            # 索引字段名
            field_name="embedding",
            # 索引参数设置
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
        collection.load()
```
你可以使用如下代码检查你数据库中存在的Collection:
```python
res = client.describe_collection(
    collection_name="quick_setup"
)

print(res)

```
您可以按以下方式重命名一个 Collection:
```python
client.rename_collection(
    old_name="video_push",
    new_name="Dw_easy_vectorDB"
)
```
Milvus在查询时速度很快，是因为每次加载的Collection的数据会缓存在内存中。为了减少内存的消耗，您可以使用动态字段的方式加载你需要的数据进入Milvus。
```python
client.load_collection(
    collection_name="Dw_easy_vectorDB",
    load_fields=["id", "embedding"]
    skip_load_dynamic_field=True 
)
```
这样做除了能减少内存的消耗外，还有一个好处，即后续使用GPU HMB和SSD对大规模数据进行优化存储和加速检索时，id字段的N Byte因为非常小，在CPU和SSD、HMB之间传输时，性能差异不大，因此可以忽略消耗。
关于这一部分，请学习完成所有知识后，浏览[FusionANNS]()。

当你使用结束此Collection后，请及时释放Collection，释放内存。
```python
client.release_collection(
    collection_name="Dw_easy_vectorDB"
)

res = client.get_load_state(
    collection_name="Dw_easy_vectorDB"
)

print(res)
```
如果你的使用场景是大模型问答系统，对于用户提供的信息数据，你需要快速的向量化存储，以便提供更加高质量的回答，并且你使用的模型上下文比较短的情况下，可以通过设置Collection的TTL来实现。用户投喂的大规模文档数据，存储到Milvus中，并设置TTL，Milvus会自动删除过期的数据。
你可以使用如下代码来实现：
```python
#  创建新的集合
from pymilvus import MilvusClient

client.create_collection(
    collection_name="Dw_easy_vectorDB",
    schema=schema,
    properties={
        "collection.ttl.seconds": 604800 # 7天
    }
)

```
```python
#  修改已存在的集合
client.alter_collection_properties(
    collection_name="Dw_easy_vectorDB",
    properties={"collection.ttl.seconds": 604800}
)
```

### Collection中设置数据分区
当你第一次创建一个Collection时，所有的数据都会被存储在一个默认分区中。然而，为了更好地管理数据，你可以创建多个分区，并将数据分布到这些分区中。分区可以让你更有效地管理和查询数据，例如，你可以根据时间戳将数据分布到不同的分区中，以便按时间范围查询数据。

创建分区：
```python
client.create_partition(
    collection_name="Dw_easy_vectorDB",
    partition_name="partition_1"
)
```
删除分区：
```python
client.drop_partition(
    collection_name="Dw_easy_vectorDB",
    partition_name="partition_1"
)
```
查询分区：
```python
client.list_partitions(
    collection_name="Dw_easy_vectorDB"
)
```
向分区中插入数据：
```python
client.insert(
    collection_name="Dw_easy_vectorDB",
    partition_name="partition_1",
    entities=entities
)
```
查询分区中的数据：
```python
client.query(
    collection_name="Dw_easy_vectorDB",
    expr="partition_name == 'partition_1'",
)
```
不过，对于某些问答系统，分区的设计会影响查询性能。我们很难确定对于某一个问题的答案，应该从哪个分区中查询，除此之外，我们不能保证另一个不相干的分区中是否包含了某条可能对最终回答产生重要影响的数据。所以，不建议使用分区。

### Shema
待完善
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
### 单向量搜索
单向量搜索指的是只涉及一个查询向量的搜索。根据预建索引和搜索请求中携带的度量类型
>  “度量类型” 是什么？
    简单说，是 “判断两个向量有多像的尺子”。不同的场景用不同的尺子，结果也不一样。

    举几个常见的：
    * L2（欧氏距离）：像量两个点之间的直线距离。比如比较两个水果的向量（颜色、大小、甜度），L2 值越小，说明两个水果越像。
    * COSINE（余弦相似度）：像看两个向量的 “方向” 有多一致。比如两句话 “我爱吃苹果” 和 “苹果是我最爱”，它们的向量方向很像，余弦值接近 1，说明意思相似（忽略句子长度，只看语义方向）。
    * IP（内积）：可以理解成 “特征重叠度”。比如两个用户的兴趣向量（喜欢的电影、音乐），IP 值越大，说明兴趣越重合，适合推荐系统。

本节将介绍如何进行单向量搜索。搜索请求携带单个查询向量，要求 Milvus 使用内积（IP）计算查询向量与 Collections 中向量的相似度，并返回三个最相似的向量。
```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530")

query_vector = [...................]
res = client.search(
    collection_name="Dw_easy_vectorDB",
    anns_field="embedding",
    data=[query_vector],
    limit=3,
    search_params={"metric_type": "IP"}# 注意，这里的metric_type 必须与创建索引时设置的一致
)

for hits in res:
    for hit in hits:
        print(hit)

# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ]
# ]
```
Milvus 根据搜索结果与查询向量的相似度得分从高到低排列搜索结果。相似度得分也称为与查询向量的距离，其值范围随使用的度量类型而变化。

### 批量向量搜索

批量向量搜索
同样，您也可以在一个搜索请求中包含多个查询向量。Milvus 将并行对查询向量进行 ANN 搜索，并返回两组结果。
```python
query_vectors = [
    [.....],
    [.....],
    [.....],
    [.....]
]

res = client.search(
    collection_name="Dw_easy_vectorDB",
    data=query_vectors,
    limit=3,
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)

```

### 分区中的 ANN 搜索
假设您在 Collections 中创建了多个分区，您可以将搜索范围缩小到特定数量的分区。在这种情况下，您可以在搜索请求中包含目标分区名称，将搜索范围限制在指定的分区内。减少搜索所涉及的分区数量可以提高搜索性能。

下面的代码片段假定在你的 Collections 中有一个名为PartitionA的分区。
```python
query_vector = [.........]
res = client.search(
    collection_name="Dw_easy_vectorDB",
    partition_names=["partitionA"],
    data=[query_vector],
    limit=3,
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
```
### 使用输出字段

在搜索结果中，Milvus 默认包含包含 top-K 向量嵌入的实体的主字段值和相似性距离/分数。您可以在搜索请求中包含目标字段（包括向量和标量字段）的名称作为输出字段，以使搜索结果携带这些实体中其他字段的值。
```python
# 4. Single vector search
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592],

res = client.search(
    collection_name="Dw_easy_vectorDB",
    data=[query_vector],
    limit=3, # The number of results to return
    search_params={"metric_type": "IP"}，
    output_fields=["color"]
)

print(res)
```

### 使用限制和偏移
您可能会注意到，搜索请求中携带的参数limit 决定了搜索结果中包含的实体数量。该参数指定了单次搜索中返回实体的最大数量，通常称为top-K。
比如搜出来 100 个相似结果，一页显示 20 个，就可以用 “limit=20”（每页 20 个）和 “offset=20”（跳过前 20 个，看第 2 页）。但注意：一次最多看 16384 个结果，太多了会变慢。
```python
query_vector = [.............],

res = client.search(
    collection_name="Dw_easy_vectorDB",
    data=[query_vector],
    limit=3, 
    search_params={
        "metric_type": "IP", 
        "offset": 10 
    }
)
```

