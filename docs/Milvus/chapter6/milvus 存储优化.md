# Milvus 存储优化
## Milvus 中 mmap（内存映射）详解
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

## 数据切分处理
文本切分是整个流程里最基础也最关键的环节，尤其当面对PDF、技术文档、多模态资料这些结构复杂的材料时，切分效果直接影响后续的向量表达和检索质量。

首先：我们结合场景来看，你可以先看看下面的场景中，是否存在你正在面对的。

1. 中文/技术文档：技术手册、法律合同等强逻辑文档。
2. 高逻辑连贯性要求：学术论文、产品说明书等结构化内容。
3. 多模态/扫描文档：医疗报告、技术白皮书、扫描版图文文档。

---

一、语义切分：以逻辑单元为最小单位，确保信息完整

这是处理复杂文档最根本的原则，核心是避免在句子或段落中间生硬切断语义：

1. 递归分割法：按层级切分（段落→句子→单词），配合重叠机制（20%-30%重叠）保留上下文。例如，用 RecursiveCharacterTextSplitter 工具，设置 chunk_size=512、chunk_overlap=150，兼顾效率与语义连续。
2. 语义边界检测：用NLP工具识别逻辑转折点：
   - BERT-NSP模型：判断相邻段落是否语义衔接，低于阈值则切分；
   - OpenNLP句子解析：适合英文文档，避免因缩写/小数点误切；
   - 中文专用工具（如HanLP/jieba）：解决中文无空格分词问题。

---

二、动态分块技术：根据内容结构自适应调整粒度

固定分块在面对标题层级复杂或图文混排文档时容易失效：

1. 标题引导分块：识别PDF中的多级标题，将同一子标题下的段落合并为一个语义块；
2. 相似性聚类分块：计算句子间余弦相似度，低于阈值时断开（如 SemanticSplitterNodeParser 工具）；
3. 混合分块策略：
   - 正文按语义段切分；
   - 表格/代码块整块保留，避免碎片化。

---

三、专业文档处理：针对领域特性设计切分规则

领域术语、特殊符号容易导致误分割：

- 医疗/法律文档：建立领域缩略词库（如 “q.d.” 不视为句尾），保护条款编号完整性；
- 含代码/公式的文档：用正则隔离非自然语言片段，独立嵌入；
- 多模态文档（VisRAG方案）：
  - 文本与图像协同切分：将关联的图文段落绑定为同一块；
  - 三种召回策略：页面拼接、加权选择、多图像VLM，保留视觉信息。

---

四、智能切分方法：基于大模型的新兴方案（适合高阶优化）

适合对效果有极致要求的场景，依赖LLM推理能力：

1. Meta-Chunking（论文《Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception》）  
   通过两种策略识别深层逻辑关系：
   - Margin Sampling Chunking：LLM判断连续句子是否应分割，按概率差动态阈值切分；
   - Perplexity Chunking：低困惑度句子视为分块边界（因模型更“确定”此处语义完整）。
2. LumberChunker：迭代调用LLM定位语义转折点，资源消耗较大但更精准。

---

五、向量检索协同优化：从分块到检索的端到端设计

切分最终服务于检索，需全局优化：

1. 关键信息索引：构建二级索引，仅对摘要性“关键信息”做向量化（如标题/实体词），原始文本作为附加内容，减少噪声；
2. 多粒度向量存储：同步存储句子级、段落级向量，应对粗细粒度查询需求；
3. 检索后重排序：先召回Top-K块，再用Cross-Encoder重排，提升精度。

 效果：在ChatPDF类应用中广泛验证，回答准确率提升30%+。

若希望深入前沿方法，可重点阅读：

1. 《Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception》（2024）  
2. 《VisRAG: Vision-based Retrieval-augmented Generation》（多模态文档处理）


