# Chapter 2：FAISS 数据结构与索引类型

本章将围绕FAISS中核心的数据结构与各类索引类型展开，从基础的精确检索索引到复杂的复合索引，逐步深入讲解其原理、API使用、参数调优及实战应用。

通过本章学习，你将能够根据不同的业务场景和数据规模，选择合适的FAISS索引类型并完成检索任务。

## 1 精确检索索引：IndexFlat 系列

### 1.1 核心原理：暴力检索的本质

IndexFlat 系列是 FAISS 中最基础的索引类型，其核心特征是“精确检索”，即通过遍历数据库中所有向量，计算查询向量与每个数据库向量之间的距离，最终返回距离最近的 Top-K 结果，因此也被称为“暴力检索”索引。

该系列索引不进行任何向量压缩或结构优化，确保检索结果的绝对准确性，是衡量其他近似检索索引性能的“黄金标准”。

根据距离度量方式的不同，IndexFlat 系列主要包含以下三种常用类型：

**IndexFlatL2：基于 L2（欧氏距离）度量**
欧氏距离计算方式为两个向量对应维度差值的平方和的平方根，公式为：
$$
  d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

适用于需要衡量向量空间中物理距离的场景，如计算机视觉中的图像特征匹配。

**IndexFlatIP：基于内积（Inner Product）度量**

内积为两个向量对应维度的乘积和，公式为：

$$
  x \cdot y = \sum_{i=1}^{n} x_i y_i
$$

在 FAISS 中，**内积被用作相似度而非距离**。由于检索排序默认按“距离升序”，FAISS 实际使用 **负内积作为距离**：

$$
  d(x,y) = - (x \cdot y)
$$

因此，IndexFlatIP 返回的是 **内积最大的向量**。  
适用于向量已归一化或本身表示相似度空间的场景，如推荐系统中的用户-物品向量匹配。

**IndexFlatCOSINE：基于余弦相似度（Cosine Similarity）度量**

余弦相似度定义为：

$$
  sim(x,y)=\frac{\sum_{i=1}^{n}x_i y_i}{\sqrt{\sum_{i=1}^{n}x_i^2}\sqrt{\sum_{i=1}^{n}y_i^2}}
$$

用于衡量向量方向的一致性，与向量模长无关，因此适合文本嵌入、聚类、推荐等特征空间。

在 FAISS 中，余弦相似度的实现方式是：

  1. **对所有向量进行 L2 归一化**；
  2. **使用内积计算相似度**（归一化后内积值即为余弦相似度）；
  3. 使用负相似度作为距离，以满足按升序排序的检索逻辑。

因此，IndexFlatCOSINE 本质上是通过 **“归一化 + 内积”** 等价实现余弦相似度检索，而不是直接使用公式进行计算。

### 1.2 API 使用与参数说明

IndexFlat 系列是 FAISS 中最简单、最常用的精确向量检索索引，API 十分简洁。核心步骤包括：索引初始化、添加向量、执行检索。

以下为 **可直接运行的 Python 示例**：

```python
import faiss
import numpy as np

# 1. 数据准备（模拟数据，维度为128，数据库向量数10000，查询向量数10）
dim = 128  # 向量维度
db_size = 10000  # 数据库向量数量
query_size = 10  # 查询向量数量
np.random.seed(42)  # 固定随机种子，确保结果可复现

db_vectors = np.random.random((db_size, dim)).astype('float32')  # 数据库向量（必须为float32类型）
query_vectors = np.random.random((query_size, dim)).astype('float32')  # 查询向量

# 2. 索引初始化（三种方式，按需选择）

# 方式1：IndexFlatL2（L2 距离，最常用）
index_l2 = faiss.IndexFlatL2(dim)

# 方式2：IndexFlatIP（内积，用于点积相似度或余弦相似度）
index_ip = faiss.IndexFlatIP(dim)

# 方式3：IndexFlat + 归一化实现余弦相似度（推荐）
# 某些 FAISS 版本没有 IndexFlatCOSINE，因此余弦相似度需手动归一化并使用 IndexFlatIP
db_vectors_norm = db_vectors.copy()
faiss.normalize_L2(db_vectors_norm)  # 手动归一化为单位向量
index_cosine = faiss.IndexFlatIP(dim)
index_cosine.add(db_vectors_norm)

# 3. 添加向量到索引（IndexFlat 系列都是直接 add）
index_l2.add(db_vectors)
print(f"索引中已添加的向量数量：{index_l2.ntotal}")

# 4. 执行检索（核心参数：query 向量、Top-K）
k = 5  # 返回最近邻数量
distances, indices = index_l2.search(query_vectors, k)

# 5. 结果解读
print("查询向量 0 的检索结果：")
print(f"距离：{distances[0]}")  # 与数据库向量的 L2 距离
print(f"索引：{indices[0]}")    # 对应数据库向量的编号
```

**关键 API 与参数说明**

 **索引初始化参数**

- `IndexFlatL2(dim)`：使用 L2 距离
- `IndexFlatIP(dim)`：使用内积
- `IndexFlat + normalize_L2`：实现余弦相似度（最兼容所有版本）

IndexFlat 系列非常轻量，仅需传入向量维度 `dim`，无其他复杂参数。

**其他常用方法**：        

`index.reset()`：清空索引中的所有向量；

`index.remove_ids(ids)`：根据向量ID删除指定向量（需配合IDMap使用）；

`index.save(filename)` / `faiss.read_index(filename)`：索引的保存与加载。

### 1.3 精确检索的性能瓶颈

IndexFlat系列虽能保证检索精度，但在数据规模扩大时，会面临严重的时间和内存性能瓶颈，这也是其仅适用于小规模数据场景的核心原因。

**1. 时间瓶颈：O(N) 的线性检索复杂度**

精确检索的时间复杂度与数据库向量数量 N 成正比，即每一次查询都需要与 N 个向量进行距离计算。假设每个向量维度为 d，单次距离计算的时间复杂度为 O(d)，则单次查询的总时间复杂度为 O(N*d)。

当 N 达到 100 万、d = 128 时，单次查询需执行约 1.28 亿次浮点运算；若查询量较大（如每秒 100 次查询），系统将难以承受，检索延迟会明显上升。

**2. 内存瓶颈：无压缩的向量存储**

IndexFlat 系列存储原始 float32 向量，不做压缩。  
float32 每元素 4 字节，128 维向量占 128×4=512 字节。

- N = 100 万 → 约 512MB  
- N = 1 亿 → 约 51.2GB

普通服务器难以完全加载如此大的索引，从而进一步增加检索延迟。

### 1.4 实战：SIFT10k 数据集精确检索对比

本实战将使用SIFT10k数据集（包含10000个128维的SIFT图像特征向量），对比IndexFlatL2、IndexFlatIP、IndexFlatCOSINE三种索引的检索结果差异，并分析其性能表现。

#### 1. 数据集准备

SIFT10k数据集可通过FAISS官方提供的工具下载，或使用模拟的SIFT特征向量（此处采用模拟数据方便学习使用）

```python
import faiss
import numpy as np
import time

print("FAISS version:", faiss.__version__)

# 1. 数据准备
dim = 128
db_size = 10000
query_size = 50

np.random.seed(42)
db_vectors = np.random.random((db_size, dim)).astype('float32')
query_vectors = np.random.random((query_size, dim)).astype('float32')

# 2. 创建索引：L2、IP
index_l2 = faiss.IndexFlatL2(dim)
index_ip = faiss.IndexFlatIP(dim)

# 3. 余弦相似度：手动归一化 + IP
db_vectors_norm = db_vectors.copy()
query_vectors_norm = query_vectors.copy()
faiss.normalize_L2(db_vectors_norm)
faiss.normalize_L2(query_vectors_norm)

index_cos = faiss.IndexFlatIP(dim)

# 添加向量
index_l2.add(db_vectors)
index_ip.add(db_vectors)
index_cos.add(db_vectors_norm)

print("三个索引均已添加向量：", index_l2.ntotal)

# 4. 检索
k = 10

# ---- L2 ----
t0 = time.time()
dist_l2, idx_l2 = index_l2.search(query_vectors, k)
t1 = time.time()
print(f"IndexFlatL2 检索时间：{t1 - t0:.4f} 秒")

# ---- IP ----
t0 = time.time()
dist_ip, idx_ip = index_ip.search(query_vectors, k)
t1 = time.time()
print(f"IndexFlatIP 检索时间：{t1 - t0:.4f} 秒")

# ---- COSINE ----
t0 = time.time()
dist_cos, idx_cos = index_cos.search(query_vectors_norm, k)
t1 = time.time()
print(f"Cosine（归一化+IP）检索时间：{t1 - t0:.4f} 秒")

# 5. Top-K 重合率对比（第 0 个查询向量）
print("\n=== 第 0 个查询向量的 Top-K 对比 ===")
print("L2 Top-10：", idx_l2[0])
print("IP  Top-10：", idx_ip[0])
print("COS Top-10：", idx_cos[0])

def overlap(a, b):
    return len(set(a) & set(b))

print(f"\nL2 vs IP   重合度：{overlap(idx_l2[0], idx_ip[0])}/10")
print(f"L2 vs COS  重合度：{overlap(idx_l2[0], idx_cos[0])}/10")
print(f"IP vs COS  重合度：{overlap(idx_ip[0], idx_cos[0])}/10")
```

运行结果

```
(base) PS C:\Users\xiong\Desktop\easy-vectordb\easy-vectordb> & E:/anaconda/python.exe c:/Users/xiong/Desktop/easy-vectordb/easy-vectordb/test.py
FAISS version: 1.11.0
三个索引均已添加向量： 10000
IndexFlatL2 检索时间：0.0145 秒
IndexFlatIP 检索时间：0.0033 秒
Cosine（归一化+IP）检索时间：0.0037 秒

=== 第 0 个查询向量的 Top-K 对比 ===
L2 Top-10： [8769 9385   82 5125 9571 3491 6267 3948 4436 1056]
IP  Top-10： [6267 8290 1274 5831 2135 1323 9231 2825 2099 1717]
COS Top-10： [6267 9385 9571   82 8769 9352 4195 3491 3848 2825]

L2 vs IP   重合度：1/10
L2 vs COS  重合度：6/10
IP vs COS  重合度：2/10
```

#### 2. 结果分析

根据本次实验的真实运行结果，我们可以观察到以下现象：

**（1）检索时间表现**

| 索引类型                     | 检索时间（秒） |
| ---------------------------- | -------------- |
| IndexFlatL2                  | ~0.014 秒      |
| IndexFlatIP                  | ~0.0034 秒     |
| 归一化 + IndexFlatIP（余弦） | ~0.0037 秒     |

**结论：三种索引的速度都很快，属于线性暴力检索，性能差异不大。**

- IP（内积）和 COS（归一化 + 内积）因为内部实现简单，略快。
- L2 由于涉及平方和加法操作，相比内积稍慢，但差距非常小。

整体上，三者的计算复杂度相同（O(N × d)），只有常数因子有所不同。

**（2）不同距离度量下的 Top-K 差异非常大**

实验给出的重合度如下：

| 对比      | 重合度          |
| --------- | --------------- |
| L2 vs IP  | **1/10（10%）** |
| L2 vs COS | **6/10（60%）** |
| IP vs COS | **2/10（20%）** |

这组数据非常“鲜活”，体现了检索度量之间巨大的差异性：

**① L2 vs IP: 仅重合 10%，几乎完全不同**

这没问题、非常正常，因为：

- **L2 距离关注的是“绝对位置”**  → 谁在欧氏空间里离你更近。

- **IP（内积）关注的是“向量长度 + 方向”**  → 长度大的向量会得到更高内积，即便方向差。

**所以它们天然衡量的是不同的东西，结果几乎不可能一致。**

**② L2 vs COS: 重合度 60%，较为相似**

你得到的 **60% 非常合理**。

- 余弦相似度关注向量方向  
- L2 距离同时考虑位置和向量模长

在随机向量（本实验模拟数据）中，模长差异比较小，因此：

> **方向比较一致的向量，空间位置也常常相对接近**  → 因此 L2 和 COS 的结果有相对较高的重叠。

**③ IP vs COS: 重合度 20%，差异依旧很大**

这是最“有教育意义”的地方。

我们会以为“内积 ≈ 方向一致”，但：

- **IP 受模长影响很大**  
- **COS 完全不受模长影响（已归一化）**

因此：

> **如果某些向量本身特别长，它们在 IP 中得分会异常高，但不一定方向相似**  → 这会造成 IP 与 COS 的结果差异巨大。
>
> 这次实验结果（20% 重合度）正好印证了这个特性。

#### 3. 学习总结

结合本次实验，我们可以得出如下结论：

**（1）不同索引检索速度差异不大**

三种 Flat 索引都使用暴力扫描，**因此“选择索引类型”不是为了速度，而是为了“度量方式”。**

（2）不同距离度量会导致截然不同的检索结果**

从本次实验的重合度可以看到：

- L2 vs IP → **10%，几乎完全不同**  
- IP vs COS → **20%，差别依旧巨大**  
- L2 vs COS → **60%，部分一致，但差异仍明显**

这是因为不同距离度量衡量的是完全不同的“相似性”：

- **L2**：空间位置更近  
- **IP**：模长大 + 方向一致  
- **COS**：方向一致（不考虑模长）

**所以选错距离度量，会得到完全错误的检索结果。**