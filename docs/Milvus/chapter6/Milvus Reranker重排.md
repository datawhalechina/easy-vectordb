# Milvus Reranker重排

## 前言

### 什么是 Milvus Reranker？

在向量数据库的检索场景中，我们经常需要从多个维度对搜索结果进行综合评估。例如，在推荐系统中，我们可能需要同时考虑内容相似度、用户偏好、地理位置、时间新鲜度等多个因素。单一的向量相似度搜索往往无法满足这种复杂的业务需求。

**Reranker（重排器）** 就是为了解决这个问题而设计的。它的工作流程是：
1. 首先通过向量检索获得初步的候选结果集
2. 然后根据业务规则和多维度特征对这些结果重新打分
3. 最后按照新的分数重新排序，得到最终的推荐列表

这种"先召回、后精排"的两阶段策略，在搜索引擎、推荐系统等领域被广泛应用。

### Milvus Reranker 的功能特性

从 Milvus 2.6 版本开始，系统内置了强大的 Reranker 功能，主要包括以下几种排名策略：

1. **加权排名（Weighted Ranker）**：为不同的搜索维度分配不同的权重，适合明确知道各因素重要性的场景
2. **RRF 排名（Reciprocal Rank Fusion）**：基于排名位置的融合算法，无需手动设置权重，适合多路召回融合
3. **衰变排名（Decay Ranker）**：根据数值字段（如距离、时间）对结果进行衰减打分，包含三种衰减函数：
   - 高斯衰减（Gaussian Decay）：平滑的钟形曲线衰减
   - 指数衰减（Exponential Decay）：快速的指数级衰减
   - 线性衰减（Linear Decay）：均匀的线性衰减
4. **Booster Reranker**：通过权重提升特定特征的影响力

### 学习路径

本教程将通过一个完整的"美食推荐系统"案例，带你逐步掌握每种 Reranker 的使用方法。每个部分都包含：
- **原理解释**：用通俗易懂的语言说明工作机制
- **应用场景**：什么情况下使用这种策略
- **代码示例**：可直接运行的完整代码
- **结果分析**：帮助理解排名效果

### 环境准备

在开始之前，请确保你的电脑**已安装 Milvus 2.6**或更高版本的 Python SDK：

```bash
! pip install "pymilvus>=2.6.0"
```

---

## 第0步：准备工作（必须先运行）

### 案例背景

为了让大家更好地理解各种 Reranker 的工作原理，我们将构建一个**美食推荐系统**作为演示案例。

想象你正在开发一个美食 App，用户可以通过多种方式搜索餐厅：
- 通过文字描述搜索（如"辣的火锅"）
- 通过图片外观搜索（如"红通通的菜品"）
- 考虑地理位置（离用户的距离）
- 参考用户评分

这个场景完美地展示了多维度检索和重排的需求。在实际应用中，单纯依靠向量相似度往往不够，我们需要综合多个因素来给出最佳推荐结果。

### 数据结构设计

我们的数据集包含以下字段：

| 字段名 | 类型 | 说明 | 用途 |
|--------|------|------|------|
| id | INT64 | 餐厅ID | 主键标识 |
| desc_vector | FLOAT_VECTOR(2) | 口味描述向量 | 用于文本相似度搜索 |
| pic_vector | FLOAT_VECTOR(2) | 图片外观向量 | 用于图像相似度搜索 |
| distance | FLOAT | 距离（公里） | 用于衰变排名演示 |
| rating | FLOAT | 用户评分 | 用于加权排名演示 |

> **注意**：为了简化演示，这里使用 2 维向量。在实际应用中，向量维度通常是768、1024 等更高维度。

### 测试数据说明

我们将插入 15 条测试数据，按相似度分为四个组：

**高度相似组**（内积 > 0.7）：
- ID=1: 完美匹配 [1.0, 0.0]，但距离 100km，评分 5.0
- ID=2: 非常匹配 [0.9, 0.1]，距离 5km，评分 4.8
- ID=3: 很匹配 [0.85, 0.15]，距离 25km，评分 4.2
- ID=4: 匹配 [0.8, 0.2]，距离 0km，评分 3.5

**中度相似组**（内积 0.4-0.7）：ID=5,6,7,8
**低度相似组**（内积 < 0.4）：ID=9,10,11
**极低相似组**（内积接近 0）：ID=12,13,14,15

通过这些数据，我们可以清晰地观察：
1. 向量相似度与距离的权衡（ID=1 很匹配但很远，ID=4 中等匹配但最近）
2. 不同 Reranker 策略对中等相似度结果的排序影响
3. 衰减函数如何根据距离重新调整排名

### 初始化代码

**请先运行以下代码，创建集合并插入测试数据：**

```python
from pymilvus import MilvusClient, DataType

# 1. 连接到本地的一个临时数据库文件（使用 Milvus Lite）
client = MilvusClient("milvus_demo.db")

# 2. 如果集合已经存在，先删除，保证每次运行都是新的
col_name = "food_collection"
if client.has_collection(col_name):
    client.drop_collection(col_name)

# 3. 创建表结构（Schema）
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="desc_vector", datatype=DataType.FLOAT_VECTOR, dim=2)  # 描述向量
schema.add_field(field_name="pic_vector", datatype=DataType.FLOAT_VECTOR, dim=2)   # 图片向量
schema.add_field(field_name="distance", datatype=DataType.FLOAT)                   # 距离(km)
schema.add_field(field_name="rating", datatype=DataType.FLOAT)                     # 评分

# 4. 创建索引（向量搜索必须要有索引）
# 注意：使用 IP (内积) 作为度量类型，值越大表示越相似
# 这样与 WeightedRanker 的"分数越大越好"逻辑一致
index_params = client.prepare_index_params()
index_params.add_index(field_name="desc_vector", index_type="FLAT", metric_type="IP")
index_params.add_index(field_name="pic_vector", index_type="FLAT", metric_type="IP")
client.create_collection(col_name, schema=schema, index_params=index_params)

# 5. 插入模拟数据（扩展到 15 条，充分展示 Reranker 效果）
data = [
    # 高度相似组（与查询向量 [1.0, 0.0] 的内积 > 0.7）
    {"id": 1, "desc_vector": [1.0, 0.0], "pic_vector": [1.0, 0.0], "distance": 100.0, "rating": 5.0},    # 完美匹配，但很远
    {"id": 2, "desc_vector": [0.9, 0.1], "pic_vector": [0.8, 0.2], "distance": 5.0, "rating": 4.8},      # 很匹配，较近
    {"id": 3, "desc_vector": [0.85, 0.15], "pic_vector": [0.85, 0.15], "distance": 25.0, "rating": 4.2}, # 很匹配，中等距离
    {"id": 4, "desc_vector": [0.8, 0.2], "pic_vector": [0.7, 0.3], "distance": 0.0, "rating": 3.5},      # 匹配，最近

    # 中度相似组（内积在 0.4-0.7 之间）
    {"id": 5, "desc_vector": [0.6, 0.4], "pic_vector": [0.5, 0.5], "distance": 2.0, "rating": 4.5},
    {"id": 6, "desc_vector": [0.55, 0.45], "pic_vector": [0.6, 0.4], "distance": 15.0, "rating": 3.8},
    {"id": 7, "desc_vector": [0.5, 0.5], "pic_vector": [0.5, 0.5], "distance": 50.0, "rating": 4.0},
    {"id": 8, "desc_vector": [0.45, 0.55], "pic_vector": [0.4, 0.6], "distance": 8.0, "rating": 3.2},

    # 低度相似组（内积 < 0.4）
    {"id": 9, "desc_vector": [0.3, 0.7], "pic_vector": [0.3, 0.7], "distance": 1.0, "rating": 2.8},      # 不匹配，但很近
    {"id": 10, "desc_vector": [0.2, 0.8], "pic_vector": [0.2, 0.8], "distance": 30.0, "rating": 2.5},
    {"id": 11, "desc_vector": [0.15, 0.85], "pic_vector": [0.1, 0.9], "distance": 60.0, "rating": 2.0},

    # 极低相似组（内积接近 0）
    {"id": 12, "desc_vector": [0.0, 1.0], "pic_vector": [0.0, 1.0], "distance": 10.0, "rating": 1.5},    # 完全不匹配
    {"id": 13, "desc_vector": [0.1, 0.9], "pic_vector": [0.1, 0.9], "distance": 0.5, "rating": 1.8},
    {"id": 14, "desc_vector": [-0.1, 0.9], "pic_vector": [0.0, 1.0], "distance": 40.0, "rating": 1.2},
    {"id": 15, "desc_vector": [0.05, 0.95], "pic_vector": [0.05, 0.95], "distance": 20.0, "rating": 1.0},
]
client.insert(col_name, data)
print(f"✅ 准备工作完成！已插入 {len(data)} 条测试数据。")
```

**代码说明：**
- 使用 `MilvusClient` 连接到本地 Milvus Lite 数据库（无需单独安装 Milvus 服务）
- 创建包含两个向量字段和两个标量字段的 Schema
- 为两个向量字段创建 FLAT 索引，使用 **IP（内积）** 度量类型（值越大越相似）
- 插入 15 条测试数据，覆盖高、中、低、极低四个相似度等级
- 数据包含不同距离（0-100km）和评分（1.0-5.0），充分展示 Reranker 效果

---

## 第1部分：加权排名 (Weighted Ranker)

### 原理解释

**加权排名（Weighted Ranker）** 是最直观的多路召回融合方法。它的核心思想是：为每个检索维度分配一个权重系数，然后将各维度的得分按权重加权求和，得到最终分数。

数学公式：
```
最终分数 = w1 × score1 + w2 × score2 + ... + wn × scoren
```

其中：
- `w1, w2, ..., wn` 是各维度的权重（通常要求权重之和为 1）
- `score1, score2, ..., scoren` 是各维度的原始得分

### 应用场景

加权排名适合以下场景：
- **明确知道各因素的重要性**：例如，在电商搜索中，文本相关性占 70%，图片相似度占 30%
- **需要精细控制排名策略**：可以通过调整权重来优化业务指标
- **A/B 测试不同权重组合**：快速验证不同权重配置的效果

### 实际案例

在我们的美食推荐系统中，假设用户更看重文字描述的匹配度（比如菜品口味），而图片外观相对次要。我们可以设置：
- 文字描述权重：0.8（80%）
- 图片外观权重：0.2（20%）

### 代码示例

```python
from pymilvus import AnnSearchRequest, WeightedRanker

# 模拟用户的搜索向量（用户想找 [1.0, 0.0] 这种特征的餐厅）
query_vec = [[1.0, 0.0]]

# ===== 先运行基线对比：单一向量搜索（无 Reranker）=====
print("=== 基线：仅使用描述向量搜索（无 Reranker）===")
baseline_res = client.search(
    collection_name="food_collection",
    data=query_vec,
    anns_field="desc_vector",
    limit=5
)
for hits in baseline_res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, 内积分数={hit.score:.4f}, "
              f"距离={hit.entity.get('distance')}km, 评分={hit.entity.get('rating')}")

# ===== 1. 准备两个搜索请求 =====
# 请求A：在文字描述向量字段中搜索（使用 IP 内积，分数越大越相似）
req_desc = AnnSearchRequest(
    data=query_vec,
    anns_field="desc_vector",
    param={"metric_type": "IP"},  # IP: 内积，值越大越相似
    limit=10
)

# 请求B：在图片外观向量字段中搜索
req_pic = AnnSearchRequest(
    data=query_vec,
    anns_field="pic_vector",
    param={"metric_type": "IP"},
    limit=10
)

# ===== 2. 执行混合搜索，使用 WeightedRanker =====
res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc, req_pic],           # 列表里有两个请求
    ranker=WeightedRanker(0.8, 0.2),    # 权重分别对应 req_desc 和 req_pic
    limit=5
)

print("\n=== 加权排名结果（描述80% + 图片20%）===")
for hits in res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, 加权分数={hit.score:.4f}, "
              f"距离={hit.entity.get('distance')}km, 评分={hit.entity.get('rating')}")

```

### 结果分析

运行上述代码，你会观察到：

**基线结果（仅描述向量）**：
- ID=1 排第一（完美匹配，内积=1.0）
- ID=2 排第二（很匹配，内积=0.9）
- ID=3 排第三（内积=0.85）
- 排名完全由向量相似度决定，不考虑距离

**加权排名结果**：
- 前 3 名仍然主要是 ID=1,2,3（因为它们在两个向量字段都很匹配）
- 如果某条记录在描述向量上中等匹配，但在图片向量上很匹配，排名会上升
- 权重 0.8 和 0.2 表示文字描述的重要性是图片的 4 倍

**对比观察**：
- 注意 ID=4（描述向量 0.8，图片向量 0.7），它在混合搜索中的排名可能比纯描述搜索略有下降（因为图片匹配度较低）

### 参数调优建议

- **权重之和建议为 1**：虽然不强制要求，但这样更容易理解和调试
- **从均等权重开始**：如果不确定权重，可以先设置为相等（如 0.5, 0.5），然后根据业务效果调整
- **使用业务指标验证**：通过 A/B 测试观察点击率、转化率等指标来优化权重

---

## 第2部分：RRF 排名 (RRFRanker)

### 原理解释

**RRF（Reciprocal Rank Fusion，倒数排名融合）** 是一种不依赖具体分数值的排名融合算法。它的核心思想是：只关注每个结果在各个检索路径中的**排名位置**，而不关心具体的分数值。

数学公式：
```
RRF_score = Σ (1 / (k + rank_i))
```

其中：
- `rank_i` 是该结果在第 i 个检索路径中的排名（从 1 开始）
- `k` 是平滑系数（通常设为 60），用于避免排名靠前的结果得分差距过大

### 为什么需要 RRF？

在多路召回场景中，不同检索路径的分数可能不在同一量级：
- 文本检索的分数可能在 0-100 之间
- 图像检索的分数可能在 0-1 之间
- 直接相加会导致分数大的路径主导结果

RRF 通过只看排名位置，巧妙地解决了这个问题，实现了真正的"公平投票"。

### 应用场景

RRF 适合以下场景：
- **不知道如何设置权重**：当你无法确定各因素的重要性时
- **多路召回分数量级不同**：避免分数归一化的麻烦
- **快速实现多路融合**：无需调参，开箱即用
- **学术研究和基线对比**：RRF 是信息检索领域的经典方法

### 实际案例

在我们的美食推荐系统中，如果我们不确定文字和图片哪个更重要，或者想要一个公平的融合策略，就可以使用 RRF。

### 代码示例

```python
from pymilvus import RRFRanker

# 使用前面定义的 req_desc 和 req_pic
# 这里不需要设置权重，只需要给一个 k 值（平滑系数，官方推荐 60）

res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc, req_pic],
    ranker=RRFRanker(k=60),  # k 值越大，排名靠前和靠后的结果得分差距越小
    limit=3
)

print("=== RRF 排名结果 ===")
for hits in res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, RRF得分={hit.score:.6f}, 距离={hit.entity.get('distance')}km")
```

### 结果分析

运行上述代码，你会发现：
- RRF 的分数通常很小（0 到 1 之间），这是正常的
- 如果某个餐厅在两个检索路径中都排名靠前，它的 RRF 得分会更高
- 即使某个路径的原始分数很高，如果排名不靠前，RRF 得分也不会太高

### 参数说明

- **k 值的作用**：
  - k 越小（如 10），排名第 1 和第 2 的得分差距越大（更激进）
  - k 越大（如 100），排名第 1 和第 2 的得分差距越小（更平滑）
  - 官方推荐值：60（在大多数场景下表现良好）

---

## 第3部分：衰变排名 (Decay Ranker)

### 原理解释

**衰变排名（Decay Ranker）** 是一种基于数值字段的打分调整策略。它的核心思想是：某些数值字段（如距离、时间）会影响结果的价值，我们需要根据这些字段对原始分数进行"衰减"。
这里的衰减是在向量相似度基础上乘以一个动态权重(衰减因子),而不同的衰减函数类型决定了这个权重如何随数值字段(如时间、距离)变化而变化。


或者我们用更加大白话的逻辑去描述什么是衰变排名：就是在加权重排的基础上，增加一个控制权重变化的值k，而高斯衰减和线性或者指数衰减就是控制这个k的变化趋势的快慢平滑


衰变函数的通用形式：
```
最终分数 = 原始分数 × decay_factor
```

其中 `decay_factor` 是根据数值字段计算出的衰减系数（0 到 1 之间）。

### 三个关键参数

在使用衰变排名时，需要理解三个核心参数：

1. **Origin（原点）**：最理想的数值，衰减系数为 1（不衰减）
   - 例如：origin = 0 (公里)
   - 意思就是：如果房子就在公司楼下（距离为0），那就是最完美的，给你打满分（1.0分）。

2. **Scale（范围）**：衰减的尺度单位
   - 例如：每 50 公里作为一个衰减单位
   - 例如：每 7 天作为一个衰减单位

3. **Decay（衰减值）**：在 Origin + Scale 位置处的衰减系数
   - 例如：decay=0.5 表示在 50 公里处，分数衰减为原来的 50%
   - 例如：decay=0.1 表示在 7 天后，分数衰减为原来的 10%

> 2.Scale范围和Decay衰减值必须要成对理解，这一对共同组成了一个路标，表示当距离到达这么远的时候，分数应该掉到了多少。
> *   **例子**：
    `scale = 10` (公里), `decay = 0.5`
    意思就是：**“嘿，兄弟，当你走到 10公里 (Scale) 的地方时，分数要正好打 5折 (Decay)，变成 0.5分。”**

4. **Offset（偏移量）**：无所谓的缓冲区，无衰减区域
   - 例如：offset = 0.5 (公里) 在这个范围内，虽然距离增加了，但我不扣分，依然算满分。
   - 意思就是：虽然我要离公司近，但 500米 以内我都当成是“零距离”。

**综合演示：把参数串起来看**

我们假设参数如下：
*   `origin = 0`
*   `offset = 1` (1公里内不扣分)
*   `scale = 10` (路标插在10公里处)
*   `decay = 0.5` (路标处剩0.5分)

**你的打分心理活动如下：**

1.  **0 km (Origin)**: 完美！满分！**(1.0分)**
2.  **0.8 km (Within Offset)**: 哎呀，这就几百米，不算远，当你是满分！**(1.0分)**
3.  **1.1 km (Over Offset)**: 超过缓冲区了，开始扣分了！**(0.99...分)**
4.  **...中间怎么掉分，由高斯/指数/线性函数决定...**
5.  **10 km (Scale)**: 终于到了路标这儿了，按照规定，这里必须是 **0.5分 (Decay)**。
6.  **20 km**: 比路标还远，分数继续掉，可能只剩 **0.1分** 了。

```python
gauss_ranker = Function(
    name="rent_house_score",
    input_field_names=["distance"],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "decay",
        "function": "gauss",
        
        "origin": 0.0,   # 完美位置：公司楼下
        "offset": 1.0,   # 宽容度：1公里内我都觉得跟住公司楼下没区别（满分）
        
        # 下面两行定了一个基准点：
        "scale": 10.0,   # 当距离达到 10公里 时...
        "decay": 0.5     # ...分数要降到 0.5 (一半)
    }
)
```

### 应用场景

衰变排名适合以下场景：
- **地理位置搜索**：离得越远，推荐优先级越低（外卖、打车、找房）
- **时间敏感内容**：越久远的内容，价值越低（新闻、热搜、社交动态）
- **价格敏感推荐**：价格偏离预算越多，推荐优先级越低
- **库存管理**：库存越少，推荐优先级越低

### 三种衰减函数对比

Milvus 提供了三种衰减函数，它们的衰减速度和形状各不相同：

| 衰减类型 | 衰减速度 | 曲线形状 | 适用场景 |
|---------|---------|---------|---------|
| 高斯衰减 | 中等，平滑 | 钟形曲线 | 地理位置、用户偏好 |
| 指数衰减 | 快速，激进 | 指数曲线 | 时间敏感、热度排名 |
| 线性衰减 | 均匀，可预测 | 直线 | 硬性过滤、倒计时 |

---

### 3.1 高斯衰减 (Gaussian Decay)

#### 特点

**高斯衰减**采用正态分布（钟形曲线）的衰减方式，具有以下特点：
- **平滑过渡**：在原点附近，分数几乎不衰减
- **逐渐加速**：远离原点后，衰减速度逐渐加快
- **永不归零**：即使非常远，分数也不会完全变成 0

*   **理解**：变化是**平滑缓冲**的。
*   **比喻**：瑕疵如果在不起眼的地方（近距离），**几乎不扣钱**（权重保持在高位）。只有当瑕疵大到一定程度，价格才开始快跌。就算瑕疵很大，也还会保留一点价值，不会直接变0。
*   **K的变化**：`1.0 -> 0.99 -> 0.95 -> (突然加速) -> 0.5 -> 0.1`
*   **感觉**：**最自然**。就像找对象，离我500米和离我800米，我觉得没区别（不衰减）；但离我50公里和80公里，那区别就大了。


#### 数学公式

```
decay_factor = exp(-((distance - origin) / scale)² / ln(1/decay))
```

验证：当 distance = origin + scale 时：
```
decay_factor = exp(-(scale/scale)² / ln(1/decay))
             = exp(-1 / ln(1/decay))
             = decay
```

例如：origin=0, scale=50, decay=0.5
- 在 0km 处：decay_factor = exp(0) = 1.0（不衰减）
- 在 50km 处：decay_factor = exp(-1/ln(2)) ≈ 0.5（衰减一半）
- 在 100km 处：decay_factor = exp(-4/ln(2)) ≈ 0.0625（严重衰减）

#### 适用场景

- **餐厅推荐**：家门口的餐厅最好，稍微远一点也可以接受，太远就不考虑了
- **房源搜索**：理想位置附近的房子都可以看看，但不会完全排除远处的房子
- **用户偏好**：偏好值接近理想值的商品优先推荐

#### 代码示例

```python
from pymilvus import Function, FunctionType

# 定义高斯衰减函数
gauss_ranker = Function(
    name="gauss_distance_decay",           # 函数名称（自定义）
    function_type=FunctionType.Rerank,     # 指定为 Rerank 类型
    params={
        "origin": 0.0,          # 完美距离：0km（衰减系数=1）
        "scale": 50.0,          # 衰减范围：50km
        "decay": 0.5,           # 在 50km 处，衰减系数=0.5
        "field_name": "distance"  # 对哪个字段进行衰减
    }
)

# 执行混合搜索（需要先定义 req_desc）
# 先定义搜索请求
query_vec = [[1.0, 0.0]]
req_desc = AnnSearchRequest(
    data=query_vec,
    anns_field="desc_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 使用高斯衰减进行搜索
res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc],
    ranker=gauss_ranker,
    limit=5
)

print("=== 高斯衰减结果（考虑距离）===")
for hits in res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, 衰减后分数={hit.score:.4f}, "
              f"距离={hit.entity.get('distance')}km, 评分={hit.entity.get('rating')}")
```

#### 结果分析

运行上述代码，你会观察到：

**对比基线（纯向量搜索）**：
- ID=1 会排在第一（完美匹配，内积=1.0）
- ID=2,3,4 紧随其后

**应用高斯衰减后**：
- **ID=4**（距离 0km，匹配度 0.8）排名显著上升，可能超过 ID=1
- **ID=2**（距离 5km，匹配度 0.9）排名也会上升
- **ID=1**（距离 100km，完美匹配）排名大幅下降，因为距离衰减严重
- **ID=3**（距离 25km，匹配度 0.85）处于中间位置

这展示了距离衰减如何平衡相似度和地理位置：近但不太完美的餐厅可以超过远但完美的餐厅。

---

### 3.2 指数衰减 (Exponential Decay)

#### 特点

**指数衰减**采用指数函数的衰减方式，具有以下特点：
- **快速衰减**：一旦偏离原点，分数迅速下降
- **激进策略**：对距离/时间非常敏感
- **适合时效性**：强调"新鲜度"或"近距离"

*   **理解**：变化是**极快**的。
*   **比喻**：衣服只要有一点点瑕疵，价格立马腰斩！瑕疵再大一点，直接当废品卖。
*   **K的变化**：`1.0 -> 0.5 -> 0.1 -> 0.01`
*   **感觉**：**很狠**。只在乎完美的（最新的/最近的），稍微差一点分就不值钱了。

#### 数学公式

```
decay_factor = decay^(|distance - origin| / scale)
```

验证：当 distance = origin + scale 时：
```
decay_factor = decay^(scale/scale) = decay^1 = decay
```

例如：origin=0, scale=50, decay=0.5
- 在 0km 处：decay_factor = 0.5^0 = 1.0（不衰减）
- 在 50km 处：decay_factor = 0.5^1 = 0.5（衰减一半）
- 在 100km 处：decay_factor = 0.5^2 = 0.25（衰减到1/4）

#### 适用场景

- **新闻热搜**：刚发布的新闻最重要，几小时后热度骤降
- **社交动态**：最新的动态优先展示，旧动态快速沉底
- **限时促销**：越接近截止时间，推荐优先级越低
- **实时库存**：库存快速变化的商品

#### 代码示例

```python
# 定义指数衰减函数
exp_ranker = Function(
    name="exp_distance_decay",
    function_type=FunctionType.Rerank,
    params={
        "origin": 0.0,
        "scale": 50.0,
        "decay": 0.5,
        "field_name": "distance"
    }
)

# 执行混合搜索（使用指数衰减）
res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc],
    ranker=exp_ranker,
    limit=5
)

print("=== 指数衰减结果（激进的距离衰减）===")
for hits in res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, 衰减后分数={hit.score:.4f}, "
              f"距离={hit.entity.get('distance')}km, 评分={hit.entity.get('rating')}")
```

#### 结果分析

运行上述代码，你会发现：

**与高斯衰减对比**：
- 近距离餐厅（0-25km）的排名变化不大
- **ID=1**（100km）在指数衰减下下降更明显
- 指数衰减对中等距离（30-60km）更"宽容"，但对极远距离（>80km）更严厉

**适用场景判断**：
- 如果业务是"只推荐附近餐厅"：使用指数衰减
- 如果业务是"推荐好餐厅，距离远点也行"：使用高斯衰减

#### 与高斯衰减的对比

假设 origin=0, scale=50, decay=0.5：
- 在 25km 处：高斯衰减约 0.73，指数衰减约 0.71（相近）
- 在 50km 处：两者都是 0.5（定义点）
- 在 100km 处：高斯衰减约 0.06，指数衰减约 0.25（指数更温和）

**关键区别**：
- 指数衰减在初期（0-50km）更激进
- 高斯衰减在远距离（>50km）衰减更猛烈，呈现"断崖式"下跌

---

### 3.3 线性衰减 (Linear Decay)

#### 特点

**线性衰减**采用线性函数的衰减方式，具有以下特点：
- **匀速衰减**：分数按固定速率下降
- **可预测性强**：衰减规律简单明了
- **硬性截断**：超过一定距离后，分数直接变为 0


*   **理解**：变化是**匀速**的。
*   **比喻**：瑕疵每大 1mm，价格就扣 1块钱。扣完为止。
*   **K的变化**：`1.0 -> 0.9 -> 0.8 ... -> 0`
*   **感觉**：**很硬**。一旦过了临界点，直接淘汰。

#### 数学公式

```
decay_factor = max(0, 1 - (1 - decay) × |distance - origin| / scale)
```

#### 适用场景

- **配送范围限制**：外卖只配送 5 公里内，超过就不显示
- **保质期管理**：商品还有 7 天过期，每天价值减少 1/7
- **预算硬约束**：价格超过预算 20% 就完全不考虑
- **服务半径**：服务只覆盖特定范围

#### 代码示例

```python
# 定义线性衰减函数
linear_ranker = Function(
    name="linear_distance_decay",
    function_type=FunctionType.Rerank,
    params={
        "origin": 0.0,
        "scale": 50.0,
        "decay": 0.5,
        "field_name": "distance"
    }
)

# 执行混合搜索（使用线性衰减）
res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc],
    ranker=linear_ranker,
    limit=5
)

print("=== 线性衰减结果（硬性范围限制）===")
for hits in res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, 衰减后分数={hit.score:.4f}, "
              f"距离={hit.entity.get('distance')}km, 评分={hit.entity.get('rating')}")
```

#### 结果分析

运行上述代码，你会发现：

**线性衰减的特点**：
- **ID=1**（100km）的分数会直接变为 0，被完全过滤掉
- **ID=7**（50km）的分数衰减为一半
- 距离 > 100km 的餐厅完全不会出现在结果中

**硬性截断效果**：
- 线性衰减会产生明确的"服务半径"边界
- 超过 100km 的餐厅完全不推荐
- 适合"只服务特定区域"的业务场景

#### 截断距离计算

根据参数 origin=0, scale=50, decay=0.5：
- 在 0km 处：decay_factor = 1.0（不衰减）
- 在 25km 处：decay_factor = max(0, 1 - 0.5 × 25/50) = 0.75
- 在 50km 处：decay_factor = 0.5（衰减一半）
- 在 100km 处：decay_factor = max(0, 1 - 0.5 × 100/50) = 0（完全截断）

**截断距离** = origin + scale × 2 = 0 + 50 × 2 = 100km

---

### 衰变排名总结

三种衰减函数的选择建议：

| 场景类型 | 推荐算法 | 理由 |
|---------|---------|------|
| 地理位置推荐（餐厅、房源） | 高斯衰减 | 平滑过渡，用户体验好 |
| 时间敏感内容（新闻、动态） | 指数衰减 | 强调新鲜度，快速淘汰旧内容 |
| 硬性范围限制（配送、服务） | 线性衰减 | 明确边界，超出范围直接过滤 |
| 价格偏好 | 高斯衰减 | 允许一定偏差，但不会完全排除 |
| 库存预警 | 指数衰减 | 库存越少，优先级下降越快 |

---

## 第4部分：Booster Reranker（提升排名）

### 原理解释

**Booster Reranker** 并不是 Milvus 中的一个独立类，而是一种设计思想：通过某种机制"提升"（Boost）特定结果的排名。

在 Milvus 2.6 中，实现 Booster 效果的主要方式有：

1. **使用加权排名**：给某个维度分配更高的权重
2. **使用衰变排名的反向逻辑**：让某个数值字段越大，分数越高
3. **组合多个 Reranker**：先用一个 Reranker 初排，再用另一个精排

### 应用场景

Booster 适合以下场景：
- **商业推广**：付费商家的结果排名提升
- **用户偏好**：用户收藏/点赞过的类型优先展示
- **热度加权**：浏览量、销量高的商品优先推荐
- **会员特权**：VIP 用户看到的结果排序不同

### 实现方式一：加权排名实现 Boost

最简单的 Booster 实现就是利用加权排名，给需要提升的维度分配更高的权重。

#### 代码示例

```python
from pymilvus import AnnSearchRequest, WeightedRanker

# 假设我们想要"提升"评分高的餐厅
# 我们可以创建一个基于 rating 字段的虚拟搜索请求

# 方式1：通过调整权重来 Boost 某个维度
# 给文字描述权重 0.6，图片权重 0.4（相比之前的 0.8, 0.2，图片权重提升了）
res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc, req_pic],
    ranker=WeightedRanker(0.6, 0.4),  # 提升了图片的权重
    limit=3
)

print("=== Booster 效果（提升图片权重）===")
for hits in res:
    for i, hit in enumerate(hits):
        print(f"排名 {i+1}: ID={hit.id}, 分数={hit.score:.4f}, 评分={hit.entity.get('rating')}")
```

### 实现方式二：组合使用多个 Reranker

虽然 Milvus 的 `hybrid_search` 一次只能使用一个 Reranker，但我们可以通过多次搜索来实现组合效果：

1. 第一次搜索：使用向量相似度召回候选集
2. 第二次处理：在应用层根据其他字段（如 rating）重新排序

#### 代码示例

```python
# 第一步：使用向量搜索召回
res = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc, req_pic],
    ranker=WeightedRanker(0.8, 0.2),
    limit=10  # 先召回更多结果
)

# 第二步：在应用层根据 rating 进行 Boost
for hits in res:
    # 将结果转换为列表，方便排序
    results = []
    for hit in hits:
        # Boost 公式：最终分数 = 原始分数 × (1 + rating / 10)
        rating = hit.entity.get('rating', 0)
        boosted_score = hit.score * (1 + rating / 10)
        results.append({
            'id': hit.id,
            'original_score': hit.score,
            'rating': rating,
            'boosted_score': boosted_score
        })
    
    # 按 boosted_score 重新排序
    results.sort(key=lambda x: x['boosted_score'], reverse=True)
    
    print("=== Booster 效果（评分加权）===")
    for i, r in enumerate(results[:3]):
        print(f"排名 {i+1}: ID={r['id']}, 原始分数={r['original_score']:.4f}, "
              f"评分={r['rating']}, Boost后分数={r['boosted_score']:.4f}")
```

### Booster 策略建议

1. **透明度**：如果是商业推广，建议明确标注"广告"或"推荐"
2. **适度提升**：Boost 系数不宜过大，避免破坏用户体验
3. **A/B 测试**：通过实验验证 Boost 策略对业务指标的影响
4. **多样性平衡**：避免所有结果都是高评分/高销量的商品，保持结果多样性

---

## 完整示例：组合使用多种 Reranker

在实际应用中，我们经常需要组合使用多种策略。下面是一个完整的示例，展示如何在一个推荐流程中综合运用多种 Reranker。

### 场景描述

用户在美食 App 中搜索"川菜"，我们需要：
1. 同时匹配文字描述和图片外观（加权排名）
2. 优先推荐距离近的餐厅（高斯衰减）
3. 适当提升高评分餐厅（Booster）

### 实现方案

```python
from pymilvus import AnnSearchRequest, WeightedRanker, Function, FunctionType

# 用户搜索向量
query_vec = [[1.0, 0.0]]

# 步骤1：使用加权排名进行多路召回
req_desc = AnnSearchRequest(data=query_vec, anns_field="desc_vector",
                            param={"metric_type": "IP"}, limit=10)
req_pic = AnnSearchRequest(data=query_vec, anns_field="pic_vector",
                           param={"metric_type": "IP"}, limit=10)

res_step1 = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc, req_pic],
    ranker=WeightedRanker(0.7, 0.3),  # 文字权重 70%，图片权重 30%
    limit=10
)

print("=== 步骤1：加权排名结果（不考虑距离）===")
for hits in res_step1:
    for i, hit in enumerate(hits[:5]):
        print(f"排名 {i+1}: ID={hit.id}, 加权分数={hit.score:.4f}, 距离={hit.entity.get('distance')}km")

# 步骤2：对结果应用距离衰减（高斯衰减）
# 注意：这里我们需要重新搜索，因为 hybrid_search 一次只能用一个 ranker
gauss_ranker = Function(
    name="gauss_distance_decay",
    function_type=FunctionType.Rerank,
    params={
        "origin": 0.0,
        "scale": 30.0,  # 30km 作为衰减单位
        "decay": 0.5,
        "field_name": "distance"
    }
)

res_step2 = client.hybrid_search(
    collection_name="food_collection",
    reqs=[req_desc, req_pic],
    ranker=gauss_ranker,
    limit=10
)

print("\n=== 步骤2：距离衰减结果（高斯衰减）===")
for hits in res_step2:
    for i, hit in enumerate(hits[:5]):
        print(f"排名 {i+1}: ID={hit.id}, 衰减后分数={hit.score:.4f}, "
              f"距离={hit.entity.get('distance')}km, 评分={hit.entity.get('rating')}")

# 步骤3：在应用层应用评分 Boost
results = []
for hits in res_step2:
    for hit in hits:
        rating = hit.entity.get('rating', 0)
        distance = hit.entity.get('distance', 0)
        # 综合分数 = 衰减后分数 × (1 + rating / 10)
        final_score = hit.score * (1 + rating / 10)
        results.append({
            'id': hit.id,
            'score': hit.score,
            'rating': rating,
            'distance': distance,
            'final_score': final_score
        })

# 按最终分数排序
results.sort(key=lambda x: x['final_score'], reverse=True)

print("\n=== 步骤3：最终排名（应用评分 Boost）===")
for i, r in enumerate(results[:5]):
    print(f"排名 {i+1}: ID={r['id']}, 最终分数={r['final_score']:.4f}, "
          f"评分={r['rating']}, 距离={r['distance']}km")
```

### 结果分析

通过这个组合策略，我们实现了：
- **多维度匹配**：同时考虑文字和图片相似度
- **地理位置优化**：距离近的餐厅优先推荐
- **质量保证**：高评分餐厅获得额外加分

这种多阶段的排名策略在实际推荐系统中非常常见，可以根据业务需求灵活调整。

---

## 最佳实践与注意事项

### 1. 选择合适的 Reranker

| 需求 | 推荐方案 | 原因 |
|------|---------|------|
| 明确知道各因素权重 | WeightedRanker | 精确控制，效果可预测 |
| 不确定权重配置 | RRFRanker | 无需调参，公平融合 |
| 需要考虑距离/时间 | Decay Ranker | 自动处理数值衰减 |
| 需要商业化排名 | Booster + 其他 | 灵活组合，满足业务需求 |

### 2. 性能优化建议

- **控制召回数量**：`limit` 参数不宜过大，建议 10-100 之间
- **索引选择**：
  - 小数据集（< 10万）：使用 FLAT 索引（精确搜索）
  - 大数据集（> 10万）：使用 IVF_FLAT 或 HNSW 索引（近似搜索）
- **批量查询**：如果有多个查询，使用批量接口可以提升性能

### 3. 常见问题与解决方案

#### 问题1：Reranker 结果不符合预期

**可能原因**：
- 权重设置不合理
- 衰减参数（origin, scale, decay）配置错误
- 数据分布不均匀

**解决方案**：
- 打印中间结果，观察每个阶段的分数变化
- 使用小数据集进行参数调优
- 绘制衰减曲线，直观理解参数效果

#### 问题2：多个向量字段的分数量级不同

**可能原因**：
- 不同向量字段使用了不同的 metric_type（L2 vs IP vs COSINE）
- 向量维度不同导致分数范围不同

**解决方案**：
- 统一使用相同的 metric_type
- 使用 RRFRanker 代替 WeightedRanker（RRF 不受分数量级影响）
- 在应用层对分数进行归一化

#### 问题3：衰减效果不明显

**可能原因**：
- scale 参数设置过大，导致衰减太慢
- decay 参数接近 1，衰减幅度太小

**解决方案**：
- 减小 scale 参数（如从 100 改为 50）
- 减小 decay 参数（如从 0.8 改为 0.5）
- 尝试更激进的指数衰减

## 总结

### 核心要点回顾

1. **WeightedRanker（加权排名）**
   - 适合：明确知道各因素重要性的场景
   - 优点：精确控制，效果可预测
   - 缺点：需要手动调参，分数量级敏感

2. **RRFRanker（倒数排名融合）**
   - 适合：不确定权重配置，或分数量级不同的场景
   - 优点：无需调参，公平融合
   - 缺点：无法精细控制各因素权重

3. **Decay Ranker（衰变排名）**
   - **高斯衰减**：平滑过渡，适合地理位置
   - **指数衰减**：快速衰减，适合时间敏感内容
   - **线性衰减**：匀速衰减，适合硬性范围限制

4. **Booster Reranker（提升排名）**
   - 通过加权或应用层逻辑实现
   - 适合商业化推荐和个性化排序

### 下一步

- 尝试在自己的数据集上应用这些 Reranker
- 通过 A/B 测试验证不同策略的效果
- 探索 Milvus 的其他高级功能（如分区、动态 Schema 等）

---

## 参考资源

- [Milvus 官方文档 - Rerankers](https://milvus.io/docs/zh/decay-ranker-overview.md)


---

**恭喜你完成了 Milvus Reranker 的学习！** 

现在你已经掌握了多种重排策略，可以根据实际业务需求灵活选择和组合使用。记住，没有最好的 Reranker，只有最适合的 Reranker。多实验、多对比，找到最适合你的排名策略！
