# Milvus 中 mmap（内存映射）详解
**首先，什么是 “mmap”？**

“mmap” 全称 “内存映射”，可以简单理解为：给硬盘上的大文件在内存中开一个 “直接窗口”。
举个例子：比如我们有一段非常大的素材文件，有20GB，我们要打开这个文件就需要20GB的内存，但我们电脑并没有这么大的内存，怎么办呢，此时mmp应运而生，mmp为这个大文件打开了一个内存窗口，可以直接访问到SSD中的文件，就像该文件已经在内存中一样，这样我们就可以直接访问该文件了，而不需要把整个文件加载到内存中。

而对于Milvus这种非常吃内存的数据库，加载一个集合时，会把所有的标量字段、向量字段和索引等全部加载到内存中，如果数据量太大，还会出现加载失败的问题，使用mmp优化存储后，我们可以加载一个非常大的Collection到内存中，并且在不占用太大的内存的情况下，就可以处理这类大规模的向量数据。


**mmp是怎么实现的呢？**

在使用mmp时，每次加载Collection时，Milvus会调用mmap将用于保障搜索速度的关键的索引加载到内存中，而其他的标量或者向量数据将会继续存放在SSD中，查询时，将通过**内存映射的方式**访问数据。


**注意点：**
性能可能波动：如果访问的数据不在内存缓存里（比如第一次访问某个冷数据），需要从硬盘读，速度会比纯内存慢一点（这叫 “缓存未命中”）。
索引仍需内存：为了保证搜索速度，索引还是要加载到内存，不能映射到硬盘。

**mmap 的配置级别（怎么用？）**
Milvus 的 mmap 可以在 4 个级别配置，优先级从高到低是：字段 / 索引级别 > 集合级别 > 全局级别（优先级高的会覆盖低的）。
1. 全局级别（整个集群默认设置）
是整个 Milvus 集群的基础设置，保存在milvus.yaml文件里，影响所有集合。
我们在配置Milvus的时候，可以修改`milvus.yaml`中的，storage参数，将mmapEnabled设置为true
```yaml
# milvus.yaml
storage:
  mmapEnabled: true  # 全局启用 mmap
  mmapDirPath: /opt/milvus/data/mmap_files  # 映射文件存储路径
```

2. 集合级别（针对单个集合）
可以给某个集合单独设置 mmap，覆盖全局设置。
创建集合时启用：在创建集合（Collection）时，通过properties={"mmap.enabled": "true"}参数开启，这样集合里的所有字段默认用 mmap。
修改已有集合：先释放集合（release_collection），再用alter_collection_properties修改 mmap 设置，最后重新加载集合（load_collection）生效。

举例：给 大大大数据集 这个集合启用 mmap，就能单独让它的数占内存少一点。

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 创建集合时启用 mmap
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000)
]

schema = CollectionSchema(
    fields,
    description="大大大数据集",
    properties={"mmap.enabled": "true"}  # 关键配置：集合级启用 mmap
)

collection = Collection("Dw_easy_db", schema)

# 修改已有集合的 mmap 配置
coll = Collection("Dw_easy_db")
coll.release()  # 必须先释放集合

# 修改 mmap 属性并重新加载
coll.alter_properties({"mmap.enabled": "false"})  # 关闭 mmap
coll.load()
```
3. 字段级别（针对单个字段）
更灵活 —— 可以只给某个字段（比如超大的标量字段）启用 mmap，其他字段正常加载到内存。
创建字段时启用：在add_field时加mmap_enabled=True参数（比如给存储长文本的 “doc_chunk” 字段启用）。
修改已有字段：用alter_collection_field修改字段的mmap.enabled属性，同样需要先释放再加载集合。
适合场景：比如一个集合里，“向量” 字段常用（放内存），“详细描述” 字段很大且不常用（用 mmap）。
```python
# 创建字段时启用 mmap（需 Milvus v2.3.0+）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(
        name="large_text", 
        dtype=DataType.VARCHAR, 
        max_length=65535,
        properties={"mmap.enabled": "true"}  # 字段级启用 mmap
    )
]

# 修改已有字段的 mmap 属性
coll = Collection("text_collection")
coll.release()

# 修改字段属性
coll.alter_field("large_text", {"mmap.enabled": "false"})  # 关闭该字段 mmap
coll.load()
```
4. 索引级别（针对单个索引）
给某个字段的索引单独设置 mmap，比如给 “标题” 字段的索引启用 mmap。
创建索引时启用：在add_index的参数里加{"mmap.enabled": "true"}。
修改已有索引：用alter_index_properties调整，同样需要释放再加载集合生效。
```python
# 创建索引时启用 mmap
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024},
    "properties": {"mmap.enabled": "true"}  # 索引级启用 mmap
}

coll = Collection("vector_collection")
coll.create_index("vector", index_params)

# 修改索引 mmap 属性
coll.release()
coll.alter_index("vector_index", {"mmap.enabled": "false"})  # 关闭索引 mmap
coll.load()
```

Milvus 官方建议：常用的数据和索引一定要放内存，不常用的再用 mmap。
比如：
高频访问的 “用户画像向量” 和其索引 —— 放内存，保证搜索快。
低频访问的 “历史日志向量”—— 用 mmap，存硬盘省内存。
