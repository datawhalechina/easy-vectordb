# Milvus Lite部署与应用教程

Milvus Lite 是[Milvus](https://github.com/milvus-io/milvus) 的轻量级版本，Milvus Lite 仅适用于小规模向量搜索使用案例，

系统要求：

- 操作系统：Ubuntu 20.04+/CentOS 7+/macOS 11.0+ 【**不支持windows系统**】
- Python版本：3.8及以上（推荐3.10以上）   
- 存储空间：建议预留至少500MB空间用于本地数据库文件

## 安装 Milvus

Milvus Lite，它是`pymilvus` 中包含的一个 python 库，直接安装即可：

```
pip install -U pymilvus  # 包含Milvus Lite
```

验证安装

```python
import pymilvus
print(pymilvus.__version__)  # 应输出类似2.5.8的版本号
```

## 创建数据库

```python
from pymilvus import MilvusClient

# 创建本地数据库文件（自动生成）
client = MilvusClient("./milvus_demo.db")  # 文件路径可自定义
```

> 会在本地创建milvus_demo.db数据库文件，持久化存储所有数据

![fig1](/images/fig1.png)

先来简单了解一下向量数据库内一些东西的基本概念

|      **概念**      |                           **描述**                           |
| :----------------: | :----------------------------------------------------------: |
| 数据库（Database） |            类似与MySQL的database，首先需要一个库             |
| 集合 (Collection)  | 集合类似于MySQL中的表，它是用来存储向量的容器。集合可以有多个字段，每个字段对应一个向量属性。 |
|   向量 (Vector)    | 向量是多维空间中的点，通常用于表示数据的特征，是集合中的基本存储单元。 |
|    索引 (Index)    | 索引是用来加速向量搜索的数据结构，有多种索引类型，如 FLAT、IVF、HNSW 等，各自都有特定的适用场景。 |
|       Field        |                字段，可以是结构化数据、向量；                |
|       Entity       |                一组Filed，类似表的一条记录。                 |

> 在milvus中 数据库（Database）类似与MySQL的database，首先需要一个库，在Milvus Lite模式中，milvus_demo.db就是数据库
>
> 如果是独立模型 可执行如下命令查看数据库
>
> ```
> print(client.list_databases())
> ```

集合 (Collection) 集合类似于MySQL中的表，它是用来存储向量的容器。集合可以有多个字段，每个字段对应一个向量属性。

查看数据库中的集合，显示是空的列表，因为数据库中没有数据

```python
print(client.list_collections())
```

## 创建集合

```python
# 创建集合，重复运行，不会反复创建集合。
client.create_collection(
  collection_name="test_collection",  # 集合名称
  dimension=2048  # 向量的维度，这里的维度，关系到后面添加的向量数据的维度
)
print(client.list_collections())
#['test_collection']
```

这里可以看到集合创建成果了

```python
client.has_collection('test_collection')  # 判断集合是否存在
# 返回True False
```

也可以查看集合

```python
res = client.describe_collection('test_collection')
print(res)
"""
{'collection_name': 'test_collection', 'auto_id': False, 'num_shards': 0, 'description': '', 'fields': [{'field_id': 100, 'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'is_primary': True}, {'field_id': 101, 'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 2048}}], 'functions': [], 'aliases': [], 'collection_id': 0, 'consistency_level': 0, 'properties': {}, 'num_partitions': 0, 'enable_dynamic_field': True}
"""
```

- **`collection_name`**：集合名称，用于唯一标识存储数据的容器，类似于关系型数据库中的表名。命名需简洁且具业务意义，如`test_collection`

- **`auto_id`**：主键自动生成开关。此处为`False`，表示需手动指定主键值（如插入数据时提供`id`字段）；若为`True`，则Milvus自动生成全局唯一的64位整数主键

- **`num_shards`**：分片数量。此处为`0`，表示使用系统默认分片策略（通常自动按集群规模分配）

- **`description`**：集合描述，此处为空字符串，可补充业务用途说明（如“存储用户行为向量”）

- 字段定义（`fields`)

  - 主键字段 `id`：

    - `type: DataType.INT64`：64位整数类型，唯一标识每条数据。

    - `is_primary`: True：指定为主键，用于高效检索和去重

    - `params: {}`：无额外参数，因主键类型为标量，无需配置向量维度等属性。

  - 向量字段 `vector`：

    - `type: DataType.FLOAT_VECTOR`：浮点向量类型，用于存储高维向量数据（如文本/图像嵌入）。
    - `params: {'dim': 2048}`：向量维度为2048，需与插入数据的实际维度一致
  

也可以删除集合

```python
client.drop_collection('demo_collection')
```

## 插入数据

创建一个小的向量集合（Collection）

```python
client.create_collection(
    collection_name="test", # 集合的名称
    dimension=5, # 向量的维度
    primary_field_name="id", # 主键字段名称
    id_type="int", # 主键的类型
    vector_field_name="vector", # 向量字段的名称
    metric_type="L2", # 指标类型，用于测量向量嵌入之间的相似性的算法。
    auto_id=True # 主键ID自动递增
)
```

 **insert 写入数据：**

```python
# 写入一条
res1 = client.insert(
    collection_name="test",  # 前面创建的 collection 名称
    data={
        "id": 0,  # 主键ID
        "vector": [  # 向量
            0.1,
            0.2,
            0.3,
            0.4,
            0.5
        ],
        "name": "测试1"  # 其他字段
    }
)
# 查看插入的数据
result = client.query(collection_name="test", filter="", output_fields=["*"],limit=100)
print(len(result))  #查看条数
print(result) #查看插入的数据
```

当然也支持批量插入数据

```python
import numpy as np

# 生成随机向量数据（包含9条记录）
data = [
    {
        "id": i,
        "vector": np.random.rand(5).tolist(),  # 生成5维随机向量
        "name": f"测试{i+2}"  
    } 
    for i in range(1, 10) 
]

# 插入数据到集合
res2 = client.insert(
    collection_name="test",
    data=data
)

client.flush(collection_name="test")

result = client.query(collection_name="test", filter="", output_fields=["*"],limit=100)
print(len(result))  #查看条数
print(result) #查看插入的数据
```

## 数据查询

### 向量相似查询

```python
res = client.search(
    collection_name="test",
    data=[[0.05, 0.23, 0.07, 0.45, 0.13]],
    limit=3,
    search_params={
        "metric_type": "L2",
        "params": {}
    }
)

for row in res[0]:
    print(row)
```

- `collection_name`：指定搜索的集合名称test

- `data`：查询向量，需为二维列表格式。此处输入一个 5 维向量，与集合中向量字段的维度定义必须一致，否则会报错

- `limit`：限制返回结果数量为 3，即返回相似度最高的前 3 条记录

- `search_params`：搜索参数配置：
- `metric_type: "L2"`：使用欧氏距离（L2）计算向量相似度，适用于关注绝对距离的场景（如图像特征匹配）
  
- **`params: {}`**：索引相关参数，当前为空，需根据集合索引类型补充参数（如 IVF_FLAT 需设置 `nprobe` 指定搜索簇数量）

### 向量相似查询 + 过滤

```python
res = client.search(
    collection_name="test",
    data=[[0.05, 0.23, 0.07, 0.45, 0.13]],
    limit=3,
    filter='name LIKE "%测试%"'
)

for row in res[0]:
    print(row)
```

###  query 普通查询数据

```python
res = client.query(
    collection_name="test1",
    filter="id > 1",
    limit=3,
    output_fields=["*"]
)
for row in res:
    print(row)
```

## upsert 插入或更新数据

```python
## 查询 name = '测试1' 的数据
res = client.query(
    collection_name="test",
    filter="name == '测试1'",
    output_fields=["*"]
)
row = res[0]
print(row)

# 修改name为datawhale
row['name'] = "datawhale"

# 保存修改
client.upsert(
    collection_name="test",
    data=[row]
)
```

查询数据

```python
res = client.query(
    collection_name="test",
    filter="name == 'datawhale'",
    output_fields=["*"],
    limit=10
)
for row in res:
    print(row)
```

