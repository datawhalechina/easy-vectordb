# Chapter 2：FAISS数据结构与索引类型

本章将围绕FAISS中核心的数据结构与各类索引类型展开，从基础的精确检索索引到复杂的复合索引，逐步深入讲解其原理、API使用、参数调优及实战应用。

通过本章学习，你将能够根据不同的业务场景和数据规模，选择合适的FAISS索引类型并完成检索任务。

## 1 IndexFlat 系列

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

## 2 IVF 系列索引
### 2.1 核心原理：倒排文件与聚类分桶

IVF（Inverted File，倒排文件）系列索引是FAISS中用于解决大规模数据检索的核心索引类型，其核心思想是“先聚类分桶，再局部检索”，通过牺牲极小的精度来换取检索效率的大幅提升。

IVF的工作流程可分为“索引构建”和“检索”两个阶段：

#### 1. 索引构建阶段

1. **聚类分桶**：使用K-Means算法将数据库中的所有向量聚类成nlist个聚类（也称为“桶”或“ Voronoi 单元”），每个聚类对应一个中心向量（聚类中心）。
2. **倒排索引构建**：为每个聚类建立一个“倒排链表”，链表中存储属于该聚类的所有数据库向量的索引及向量本身（或其量化形式）。同时，单独存储所有聚类中心，形成一个“聚类中心索引”。

#### 2. 检索阶段

1. **确定候选聚类**：计算查询向量与所有聚类中心的距离，选择距离最近的nprobe个聚类作为候选聚类（nprobe为核心参数，控制候选聚类数量）。
2. **局部精确检索**：仅遍历候选聚类对应的倒排链表，在这些局部向量中执行精确检索，最终返回距离最近的Top-K结果。

通过“聚类分桶”，IVF将原本O(N)的线性检索复杂度降低为O(nprobe*N/nlist)，当nlist较大且nprobe较小时，检索效率将得到数量级的提升。
例如，当N=100万、nlist=1000时，每个聚类平均包含1000个向量，若nprobe=10，则仅需检索10*1000=1万个向量，检索规模缩小为原来的1%。

### 2.2 核心索引类型与API使用

IVF系列索引根据“局部检索时的向量存储方式”可分为两类：IndexIVF_FLAT（局部存储原始向量，精度较高）和IndexIVF_PQ（局部存储PQ量化后的向量，内存占用更低）。
两者的API使用逻辑一致，核心区别在于索引构建时是否引入PQ量化。

#### 1. IndexIVF_FLAT：局部精确的IVF索引

IndexIVF_FLAT的“倒排链表”中存储的是原始向量，因此在局部检索时能保证较高的精度，是IVF系列中精度最高的索引类型，同时兼顾效率提升。

```python
import faiss
import numpy as np

# 1. 数据准备（模拟SIFT100k数据集：10万条128维向量）
dim = 128
db_size = 100000
query_size = 50
np.random.seed(42)
db_vectors = np.random.uniform(low=0, high=1, size=(db_size, dim)).astype('float32')
query_vectors = np.random.uniform(low=0, high=1, size=(query_size, dim)).astype('float32')

# 2. 索引核心参数配置
nlist = 100  # 聚类数量（核心参数1），通常建议为数据库规模的平方根附近（如10万数据取100-1000）
metric = faiss.METRIC_L2  # 距离度量方式（L2），可选METRIC_INNER_PRODUCT（内积）

# 3. 索引构建（IVF索引需先初始化量化器，通常用IndexFlat作为量化器）
quantizer = faiss.IndexFlatL2(dim)  # 量化器：用于聚类和计算查询与聚类中心的距离
index_ivf_flat = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)

# 4. 训练索引（IVF索引必须先训练，训练数据需与数据库向量分布一致，此处直接用数据库向量训练）
assert not index_ivf_flat.is_trained  # 初始状态为未训练
index_ivf_flat.train(db_vectors)
assert index_ivf_flat.is_trained  # 训练后状态变为已训练

# 5. 添加向量到索引
index_ivf_flat.add(db_vectors)
print(f"IndexIVF_FLAT 索引规模：{index_ivf_flat.ntotal} 条向量，聚类数：{index_ivf_flat.nlist}")

# 6. 执行检索（核心参数nprobe：检索的候选聚类数）
k = 10  # Top-K结果
nprobe = 10  # 核心参数2，检索10个候选聚类
index_ivf_flat.nprobe = nprobe  # 设置nprobe参数
distances, indices = index_ivf_flat.search(query_vectors, k)

# 7. 结果解读
print(f"第0个查询向量的Top-{k}近邻索引：{indices[0]}")
print(f"第0个查询向量的Top-{k}近邻距离：{distances[0]}")
```
运行结果
```
IndexIVF_FLAT 索引规模：100000 条向量，聚类数：100
第0个查询向量的Top-10近邻索引：[ 3890 20772 48424 36273 31498 96926 73345 64609 37120 21128]
第0个查询向量的Top-10近邻距离：[13.286436  13.52108   13.8645115 13.870218  14.027049  14.071011 14.091234  14.111423  14.166992  14.230012 ]
```

#### 2. IndexIVF_PQ：结合PQ量化的IVF索引

IndexIVF_PQ在IVF的基础上引入了PQ（乘积量化）技术，将“倒排链表”中的原始向量替换为PQ量化后的码本，进一步降低内存占用，适用于超大规模数据场景（如N≥1000万）。其API使用与IndexIVF_FLAT的主要区别在于PQ相关参数的配置。

```python
import faiss
import numpy as np

# 1. 数据准备（同IndexIVF_FLAT，SIFT100k数据集）
dim = 128
db_size = 100000
query_size = 50
db_vectors = np.random.uniform(low=0, high=1, size=(db_size, dim)).astype('float32')
query_vectors = np.random.uniform(low=0, high=1, size=(query_size, dim)).astype('float32')

# 2. 核心参数配置
nlist = 100  # IVF聚类数
nprobe = 10  # 检索候选聚类数
metric = faiss.METRIC_L2
# PQ核心参数
m = 16  # 向量分割的段数（需整除向量维度，128/16=8，每段8维）
bits = 8  # 每段量化后的比特数（8比特对应256个码本中心）

# 3. 索引构建（量化器仍用IndexFlatL2）
quantizer = faiss.IndexFlatL2(dim)
# IndexIVF_PQ构造函数：quantizer, dim, nlist, m, bits, metric
index_ivf_pq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits, metric)

# 4. 训练索引（PQ量化需要训练码本，必须执行训练步骤）
index_ivf_pq.train(db_vectors)

# 5. 添加向量与检索
index_ivf_pq.add(db_vectors)
index_ivf_pq.nprobe = nprobe
distances, indices = index_ivf_pq.search(query_vectors, k=10)

# 内存占用对比（粗略计算）
# IndexIVF_FLAT内存：10万 * 128 * 4字节 = 51.2MB
# IndexIVF_PQ内存：10万 * (16 * 1字节) = 1.6MB（每段8比特即1字节，16段共16字节）
print(f"IndexIVF_PQ 索引规模：{index_ivf_pq.ntotal} 条向量")
print(f"IndexIVF_PQ 近似内存占用：{db_size * m * bits / 8 / 1024:.2f} KB")
```
运行结果
```
IndexIVF_PQ 索引规模：100000 条向量
IndexIVF_PQ 近似内存占用：1562.50 KB
```
### 2.3 核心参数调优：nlist 与 nprobe 的权衡

IVF系列索引的性能（检索效率、精度）主要由nlist（聚类数）和nprobe（检索候选聚类数）两个核心参数决定，两者需根据数据规模和业务需求进行权衡调优。

#### 1. nlist（聚类数）的调优逻辑

nlist决定了索引构建时的聚类数量，直接影响每个聚类的平均向量数（N/nlist），其调优核心是“聚类粒度与检索效率的平衡”：

- **nlist过大**：每个聚类的向量数过少，聚类中心数量增多。优点是聚类粒度更细，查询向量与候选聚类的匹配更精准；缺点是聚类过程（训练阶段）耗时增加，且检索时需要遍历更多聚类才能保证精度（需增大nprobe），反而降低检索效率。
- **nlist过小**：每个聚类的向量数过多，聚类粒度粗糙。优点是训练速度快，检索时只需遍历少量聚类；缺点是局部检索的向量规模增大，检索效率提升有限，且聚类中心代表性不足，可能导致精度下降。

经验值建议：nlist通常设置为数据库向量数量N的平方根附近，例如：    N=10万 → nlist=100-1000；N=100万 → nlist=1000-5000；N=1亿 → nlist=5000-20000。

#### 2. nprobe（检索候选聚类数）的调优逻辑

nprobe决定了检索时需要遍历的候选聚类数量，是“检索精度与效率的直接权衡点”：

- **nprobe增大**：遍历的候选聚类更多，包含目标近邻向量的概率更高，检索精度（Recall）提升，但需要计算的距离数量增加，检索延迟增大。当nprobe=nlist时，IVF索引退化为暴力检索，精度与IndexFlat一致，但效率更低。
- **nprobe减小**：遍历的候选聚类更少，检索速度更快，但可能遗漏包含目标近邻的聚类，导致精度下降。当nprobe=1时，仅检索距离查询向量最近的一个聚类，效率最高但精度可能最低。

调优方法：在保证业务所需Recall（如Recall@10≥95%）的前提下，尽可能减小nprobe。通常先固定nlist，通过实验测试不同nprobe对应的Recall和检索时间，选择最优值。

### 2.4 实战：IVF 索引检索 SIFT100k，对比 nprobe 对 Recall 的影响

本实战使用SIFT100k数据集，以IndexIVF_FLAT为例，测试不同nprobe值对检索Recall（召回率）和检索时间的影响，掌握IVF索引的调参方法。

#### 1. 实验准备

Recall@k的定义：检索结果中包含的“真实近邻”数量占总真实近邻数量的比例。此处以IndexFlatL2的检索结果作为“真实近邻”基准。

```python
import faiss
import numpy as np
import time

# 1. 数据准备（SIFT100k模拟数据）
dim = 128
db_size = 100000  # 数据库向量数：10万
query_size = 50   # 查询向量数：50个
k = 10            # 检索Top-K，计算Recall@10
# 生成随机向量（FAISS要求float32类型）
db_vectors = np.random.uniform(low=0, high=1, size=(db_size, dim)).astype('float32')
query_vectors = np.random.uniform(low=0, high=1, size=(query_size, dim)).astype('float32')

# 2. 生成真实近邻（用IndexFlatL2作为基准，暴力检索的结果是"绝对准确"的）
index_flat = faiss.IndexFlatL2(dim)  # L2距离（欧氏距离）
index_flat.add(db_vectors)
# 关键修正：search返回 (距离矩阵, 近邻索引矩阵) → 只解包2个值
distances_true, true_indices = index_flat.search(query_vectors, k)

# 3. 定义Recall计算函数（逻辑不变，保持原功能）
def calculate_recall(pred_indices, true_indices, k):
    """
    计算Recall@k：预测结果中命中真实近邻的比例
    pred_indices: 模型预测的近邻索引（shape: [query_size, k]）
    true_indices: 真实近邻索引（shape: [query_size, k]）
    """
    recall_sum = 0.0
    for pred, true in zip(pred_indices, true_indices):
        pred_set = set(pred)    # 预测的Top-K索引集合
        true_set = set(true)    # 真实的Top-K索引集合
        overlap = len(pred_set & true_set)  # 交集数量（命中数）
        recall_sum += overlap / k  # 单个查询的Recall@k
    return recall_sum / len(pred_indices)  # 所有查询的平均Recall@k

# 4. 初始化IVF索引（固定nlist=100，测试不同nprobe的影响）
nlist = 100  # 聚类数（按10万数据的经验值设置）
quantizer = faiss.IndexFlatL2(dim)  # 量化器（IVF的聚类中心用L2距离计算）
# IndexIVFFlat：IVF+Flat（只聚类，不量化，精度较高，适合对比）
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
index_ivf.train(db_vectors)  # IVF必须先训练聚类中心
index_ivf.add(db_vectors)    # 向索引中添加数据

# 5. 测试不同nprobe值的性能（核心实验逻辑）
nprobe_list = [1, 5, 10, 20, 50, 100]  # 待测试的nprobe（从1到nlist=100）
results = []

for nprobe in nprobe_list:
    index_ivf.nprobe = nprobe  # 设置当前测试的nprobe
    # 计时开始
    start_time = time.time()
    # 关键修正：search返回 (距离矩阵, 近邻索引矩阵) → 只解包2个值
    distances_pred, pred_indices = index_ivf.search(query_vectors, k)
    # 计时结束
    end_time = time.time()
    
    # 计算Recall@10和平均查询时间
    recall = calculate_recall(pred_indices, true_indices, k)
    total_time = end_time - start_time  # 总耗时（秒）
    avg_time = total_time / query_size * 1000  # 单个查询平均时间（毫秒）
    
    results.append((nprobe, recall, avg_time))

# 格式化输出结果（清晰易读，适合课堂展示）
print("=" * 40)
print(f"nlist固定为 {nlist}，测试不同nprobe的性能")
print("=" * 40)
print(f"{'nprobe':<6} {'Recall@10':<10} {'平均查询时间(ms)':<15}")
print("-" * 40)
for nprobe, recall, avg_time in results:
    print(f"{nprobe:<6} {recall:.4f}        {avg_time:.2f}")
print("=" * 40)
```

#### 2. 实验结果分析

本次实验基于10万条128维随机向量（模拟无真实语义关联的向量数据），固定nlist=100，测试了不同nprobe值对IVF索引检索性能（Recall@10和查询时间）的影响，实际运行结果如下：
```
========================================
nlist固定为 100，测试不同nprobe的性能
========================================
nprobe Recall@10  平均查询时间(ms)
----------------------------------------
1      0.0620        0.01
5      0.2140        0.02
10     0.3220        0.03
20     0.5020        0.05
50     0.8260        0.14
100    1.0000        0.27
========================================
```
结合实验结果和课堂理论，可得出以下核心结论：

1. **nprobe与精度（Recall@10）：从“快速爬升”到“完全饱和”**
   - 当nprobe从1增加到50时，Recall@10从6.2%快速提升至82.6%——这是因为随着候选聚类数增加，包含目标近邻的概率大幅上升，精度提升效率很高；
   - 当nprobe从50增加到100时，Recall@10从82.6%提升至100%——此时精度完全饱和（与暴力检索一致），但精度提升幅度仅17.4%，体现了“边际收益递减”的规律（多花了近1倍的时间，只多拿了17.4%的精度）；
   - 特别说明：本次实验用随机均匀分布的向量，聚类代表性较弱，因此相同nprobe下的Recall比真实语义向量（如图片、文本嵌入）更低——真实场景中，向量有语义关联性，聚类更集中，nprobe=10时可能达到80%以上的Recall。

2. **nprobe与效率（查询时间）：近似线性增长**
   - 查询时间随nprobe的增加近似线性上升：nprobe从1（0.01ms）增加到100（0.27ms），时间增长了27倍；
   - 当nprobe=100时，IVF索引退化为暴力检索（需遍历所有聚类），因此查询时间与IndexFlatL2（暴力检索）基本一致，验证了课堂上“nprobe=nlist→暴力检索”的理论。

3. **“性价比最优”nprobe的选择逻辑（核心实操技能）**
   实验结果中，不同需求对应不同最优nprobe：
   - 极速场景（允许低精度）：nprobe=20，Recall@10=50.2%，平均查询时间仅0.05ms，适合对精度要求不高、追求极致速度的场景；
   - 均衡场景（精度+速度兼顾）：nprobe=50，Recall@10=82.6%，平均查询时间0.14ms——相比nprobe=100，时间减少48%（0.27→0.14ms），精度仅损失17.4%，是大多数业务的“性价比之选”；
   - 高精度场景（允许慢查询）：nprobe=100，Recall@10=100%，适合对精度要求极高（如科研、医疗）、对延迟不敏感的场景。

## 3 PQ量化索引：压缩检索
### 3.1 核心原理：高维向量的“空间压缩术”

当处理百万级以上高维向量时，直接存储完整向量会面临严重的内存瓶颈。例如 100 万个 768 维 float32 向量，原始存储需占用 3GB 内存，而 PQ 技术可将其压缩至 8MB，实现 99.7% 的内存节省，核心在于“分而治之”的量化思想。

#### 核心原理：三维拆解

PQ 把复杂的高维向量压缩过程拆解为“拆分-量化-编码”三个步骤，类似将一本厚书拆分成章节，再给每章制作精简索引：

1. **维度拆分**：将 d 维原始向量均匀分割为 m 个互不重叠的子向量，每个子向量维度为 d/m（需满足 d 能被 m 整除）。例如 768 维向量可拆分为 8 个 96 维子向量。
2. **子空间量化**：对每个子向量所在的子空间，通过 K-Means 聚类算法训练出 k 个聚类中心，形成该子空间的“码本”。若每个码本用 nbits 比特表示，则 k=2^nbits（常用 nbits=8，即每个子空间有 256 个聚类中心）。
3. **向量编码**：将每个子向量与对应码本中的聚类中心对比，用距离最近的聚类中心索引（即“码字”）替代原始子向量。最终整个高维向量被转化为 m 个码字组成的编码序列，实现大幅压缩。

### 3.2 **核心索引类型与API使用**

FAISS 提供了两种核心 PQ 索引实现：IndexPQ 适用于中小规模数据，IndexIVF_PQ 结合倒排文件（IVF）技术，专为大规模数据设计，是工业界最常用的方案。

以下通过完整代码案例演示其使用流程。

#### 2.1 基础工具：IndexPQ 使用步骤

IndexPQ 直接对所有向量进行 PQ 量化，核心参数包括向量维度（d）、子向量数量（m）和每个子量化器的位数（nbits）。

```python
import faiss
import numpy as np
import time

# =========================
# 1. 数据准备
# =========================
d = 64          # 向量维度（需被 m 整除）
nb = 100_000     # 数据库向量数量
nq = 100        # 查询向量数量
k = 10          # Top-K 检索结果数

# 随机生成向量数据（实际使用可替换为真实向量，如 SIFT1M）
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# =========================
# 2. IndexPQ 初始化与训练
# =========================
m = 8           # 子向量数量
nbits = 8       # 每个子量化器位数（2^nbits=256个聚类中心）

index_pq = faiss.IndexPQ(d, m, nbits)

print(f"训练前索引状态：{'已训练' if index_pq.is_trained else '未训练'}")
index_pq.train(xb)  # PQ 依赖码本，必须先训练
print(f"训练后索引状态：{'已训练' if index_pq.is_trained else '未训练'}")

index_pq.add(xb)  # 添加向量到索引

# =========================
# 3. 检索
# =========================
start = time.time()
distances_pq, indices_pq = index_pq.search(xq, k)
end = time.time()

print(f"IndexPQ 检索用时：{end - start:.4f} 秒")
print("Top-10 检索结果（前5个查询向量的前3个结果）：")
print(indices_pq[:5, :3])
```

运行结果

```
训练前索引状态：未训练
训练后索引状态：已训练
IndexPQ 检索用时：0.0025 秒
Top-10 检索结果（前5个查询向量的前3个结果）：
[[99049 27684 55525]
 [11788 54242 22682]
 [74453 19620 22005]
 [82073 42234 85680]
 [68340 92079 13408]]
```

#### 2.2 大规模优化：IndexIVF_PQ 核心用法

IndexIVF_PQ 采用“粗筛选+精检索”的两级架构：先通过 IVF 层将向量分到多个聚类分区，再在目标分区内用 PQ 进行精确匹配，大幅提升检索速度。

```python
# =========================
# 1. 核心参数配置
# =========================
nlist = 100                    # IVF 聚类分区数
param_str = f"IVF{nlist},PQ{m}"  # 索引参数字符串

# =========================
# 2. IndexIVF_PQ 初始化
# =========================
index_ivf_pq = faiss.index_factory(d, param_str, faiss.METRIC_L2)  # L2距离度量

# =========================
# 3. 训练与添加数据
# =========================
index_ivf_pq.train(xb)
index_ivf_pq.add(xb)

# 设置 nprobe（搜索分区数），平衡速度与精度
index_ivf_pq.nprobe = 10  # 搜索10个分区（推荐 nlist 的 5%-20%）

# =========================
# 4. 检索
# =========================
start = time.time()
distances_ivf, indices_ivf = index_ivf_pq.search(xq, k)
end = time.time()
print(f"IndexIVF_PQ 检索用时：{end - start:.4f} 秒")
print("IndexIVF_PQ 检索结果（前5个查询向量的前3个结果）：")
print(indices_ivf[:5, :3])
```

运行结果

```
IndexIVF_PQ 检索用时：0.0008 秒
IndexIVF_PQ 检索结果（前5个查询向量的前3个结果）：
[[85667 10807 78521]
 [42467 19404 40628]
 [65152 61047 47008]
 [98877 57937 81528]
 [88937 16874 97116]]
```

**实验小结：**

- **IndexPQ**：适合中小规模数据，速度较慢但精度最高
- **IndexIVF_PQ**：适合大规模数据，通过 IVF 分区加 PQ 精确匹配，大幅提升速度
- **nlist 与 nprobe**：平衡精度与速度的关键参数，nlist ~ 数据集大小平方根，nprobe 5%-20%

### 3.3 核心问题解析：码本训练与精度权衡

PQ 检索的核心矛盾是“压缩率”与“精度”的平衡，而码本训练质量直接决定了这种平衡的上限。

#### 3.3.1 码本训练：量化效果的“基石”

码本是子空间聚类中心的集合，其质量取决于训练数据和过程控制：

- **训练数据代表性**：必须使用与数据库分布一致的数据（优先用全量数据库向量），否则码本无法覆盖真实数据分布，导致量化误差剧增。  
- **迭代次数控制**：FAISS 中 K-Means 聚类默认迭代 20 次，若数据分布复杂，可通过 `faiss.Kmeans(d, k, niter=50)` 手动增加迭代次数。  
- **空码本问题**：若子空间内部分聚类中心无向量匹配（常见于 m 过大或 nbits 过小），需减少子向量数量或增大码本规模。  

#### 3.3.2 压缩率计算：量化程度的“度量衡”

PQ 压缩率由子向量数量（m）和码本位数（nbits）共同决定：

$$
压缩率 = \frac{原始向量大小}{PQ编码大小} = \frac{d \times 4}{m \times (nbits/8)}
$$

示例：d=64、m=8、nbits=8 时，压缩率 = (64×4)/(8×1) = 32 倍  
即原始 256 字节的向量被压缩为 8 字节（存储量），检索时仍需解码计算距离。  

#### 3.3.3 精度权衡：参数调优的“核心逻辑”

精度损失源于量化误差和搜索范围限制，可通过以下参数调优实现平衡：

| 参数              | 调优方向                 | 对精度/速度的影响                                            |
| :---------------- | :----------------------- | :----------------------------------------------------------- |
| 子向量数量 m      | 增大 m（如 8→16）        | 精度提升（子空间更细），速度略降，压缩率降低                 |
| 码本位数 nbits    | 增大 nbits（如 8→12）    | 精度显著提升（聚类中心更多），训练时间增加                   |
| IVF 分区数 nlist  | 增大 nlist（如 100→500） | 精度提升（分区更细），单次搜索分区速度略快，索引构建时间增加 |
| 搜索分区数 nprobe | 增大 nprobe（如 10→50）  | 精度大幅提升（覆盖更多相关数据），速度降低                   |

实践表明：当 nprobe 从1增加到64时，IndexIVF_PQ 的召回率可从 34% 提升至 52%（示例数值，实际依赖数据集和参数配置），接近暴力搜索水平。

> **提示**：m/nbits 控制 PQ 压缩率与量化精度；nlist/nprobe 控制 IVF 搜索范围与召回率。理解这两类参数的作用，是调优 PQ+IVF 索引的核心。

### 3.4实战：不同索引性能比较

```python
import faiss
import numpy as np
import time
import sys
from prettytable import PrettyTable

# -------------------------- 配置参数 --------------------------
d = 64  # 向量维度（模拟真实高维特征向量）
nb = 100000  # 数据库向量数量
nq = 100  # 查询向量数量
k = 10  # 每个查询返回top-k结果

# 算法参数
nlist = 100  # IVF聚类中心数量
m = 8  # PQ子向量数量（整除d）
bits = 8  # PQ每个子向量编码位数
ivf_nprobe = 10  # IVF搜索时检查的簇数量

# 真实场景数据配置
n_clusters = 50  # 模拟真实数据的聚类数量
cluster_std = 0.3  # 每个聚类的标准差（模拟真实数据的聚集特性）

# ------------------------------------------------------------------------------

def generate_realistic_data(d, nb, nq, n_clusters, cluster_std, seed=42):
    """生成模拟真实场景的数据（带聚类结构的特征向量）
    模拟真实场景：数据具有自然聚类特性（如图像特征、文本嵌入的分布）
    """
    np.random.seed(seed)
    
    # 1. 生成聚类中心（模拟不同类别的核心特征）
    cluster_centers = np.random.randn(n_clusters, d).astype('float32')
    # 对聚类中心归一化，模拟真实特征的单位长度特性
    faiss.normalize_L2(cluster_centers)
    
    # 2. 生成数据库向量（按聚类分配数据）
    db_vectors = []
    # 每个聚类的样本数量（随机分配，模拟真实数据的类别不平衡）
    cluster_sizes = np.random.multinomial(nb, [1/n_clusters]*n_clusters)
    
    for i in range(n_clusters):
        if cluster_sizes[i] == 0:
            continue
        # 围绕聚类中心生成带噪声的向量
        cluster_vectors = cluster_centers[i:i+1] + cluster_std * np.random.randn(cluster_sizes[i], d).astype('float32')
        faiss.normalize_L2(cluster_vectors)
        db_vectors.append(cluster_vectors)
    
    db_vectors = np.vstack(db_vectors).astype('float32')
    
    # 3. 生成查询向量（从聚类中心附近采样，模拟真实查询场景）
    query_vectors = []
    for _ in range(nq):
        # 随机选择一个聚类中心
        center_idx = np.random.randint(n_clusters)
        # 生成带噪声的查询向量
        query_vec = cluster_centers[center_idx:center_idx+1] + 0.2 * np.random.randn(1, d).astype('float32')
        faiss.normalize_L2(query_vec)
        query_vectors.append(query_vec)
    
    query_vectors = np.vstack(query_vectors).astype('float32')
    
    return db_vectors, query_vectors

def calculate_memory_usage(index):
    """计算索引内存占用（MB）"""
    if isinstance(index, faiss.IndexFlatL2):
        memory = nb * d * 4  # float32占4字节
    elif isinstance(index, faiss.IndexPQ):
        code_size = m * bits // 8
        centroid_memory = (1 << bits) * m * 4
        memory = nb * code_size + centroid_memory
    elif isinstance(index, faiss.IndexIVFFlat):
        memory = nb * d * 4 + nlist * d * 4
    elif isinstance(index, faiss.IndexIVFPQ):
        code_size = m * bits // 8
        centroid_memory = nlist * d * 4
        subquantizer_memory = nlist * m * (1 << bits) * 4
        memory = nb * code_size + centroid_memory + subquantizer_memory
    else:
        memory = sys.getsizeof(index)
    
    return memory / (1024 * 1024)  # 转换为MB

def evaluate_index(index, query_vectors, brute_force_indices):
    """评估索引性能：速度、召回率、内存占用"""
    # 内存占用
    memory_mb = calculate_memory_usage(index)
    
    # 搜索速度（5次平均）
    n_runs = 5
    total_time = 0
    for _ in range(n_runs):
        start_time = time.time()
        _, indices = index.search(query_vectors, k)
        total_time += (time.time() - start_time)
    avg_time_ms = (total_time / n_runs) * 1000
    
    # 召回率计算
    recall = 0.0
    for i in range(nq):
        true_set = set(brute_force_indices[i])
        pred_set = set(indices[i])
        recall += len(true_set & pred_set) / k
    recall /= nq
    
    return memory_mb, avg_time_ms, recall

def build_indices(db_vectors, query_vectors):
    """构建所有索引并评估"""
    # 1. 暴力搜索（基准）
    brute_force = faiss.IndexFlatL2(d)
    brute_force.add(db_vectors)
    _, brute_force_indices = brute_force.search(query_vectors, k)
    bf_mem, bf_time, bf_recall = evaluate_index(brute_force, query_vectors, brute_force_indices)
    
    # 2. IndexPQ
    pq = faiss.IndexPQ(d, m, bits)
    pq.train(db_vectors)
    pq.add(db_vectors)
    pq_mem, pq_time, pq_recall = evaluate_index(pq, query_vectors, brute_force_indices)
    
    # 3. IndexIVF
    ivf_quantizer = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(ivf_quantizer, d, nlist, faiss.METRIC_L2)
    ivf.train(db_vectors)
    ivf.add(db_vectors)
    ivf.nprobe = ivf_nprobe
    ivf_mem, ivf_time, ivf_recall = evaluate_index(ivf, query_vectors, brute_force_indices)
    
    # 4. IndexIVF_PQ
    ivf_pq_quantizer = faiss.IndexFlatL2(d)
    ivf_pq = faiss.IndexIVFPQ(ivf_pq_quantizer, d, nlist, m, bits)
    ivf_pq.train(db_vectors)
    ivf_pq.add(db_vectors)
    ivf_pq.nprobe = ivf_nprobe
    ivfpq_mem, ivfpq_time, ivfpq_recall = evaluate_index(ivf_pq, query_vectors, brute_force_indices)
    
    # 整理结果
    results = [
        ["暴力搜索 (IndexFlatL2)", f"{bf_mem:.2f}", f"{bf_time:.2f}", f"{bf_recall:.4f}"],
        ["乘积量化 (IndexPQ)", f"{pq_mem:.2f}", f"{pq_time:.2f}", f"{pq_recall:.4f}"],
        ["倒排文件 (IndexIVF)", f"{ivf_mem:.2f}", f"{ivf_time:.2f}", f"{ivf_recall:.4f}"],
        ["IVF+PQ (IndexIVF_PQ)", f"{ivfpq_mem:.2f}", f"{ivfpq_time:.2f}", f"{ivfpq_recall:.4f}"]
    ]
    
    return results

def print_results_table(results):
    """打印结果表格"""
    table = PrettyTable()
    table.field_names = ["算法名称", "内存占用(MB)", "平均搜索时间(ms)", "召回率"]
    table.align["算法名称"] = "l"
    table.align["内存占用(MB)"] = "r"
    table.align["平均搜索时间(ms)"] = "r"
    table.align["召回率"] = "r"
    
    for row in results:
        table.add_row(row)
    
    print("\n" + "="*80)
    print("检索算法性能对比结果")
    print("="*80)
    print(table)

if __name__ == "__main__":
    # 生成模拟真实场景的数据
    print("生成模拟真实场景的数据...")
    db_vectors, query_vectors = generate_realistic_data(d, nb, nq, n_clusters, cluster_std)
    
    # 构建索引并评估
    print("构建索引并评估性能...")
    results = build_indices(db_vectors, query_vectors)
    
    # 打印结果表格
    print_results_table(results)
```

运行结果

```
+------------------------+--------------+------------------+--------+
| 算法名称               | 内存占用(MB) | 平均搜索时间(ms) | 召回率 |
+------------------------+--------------+------------------+--------+
| 暴力搜索 (IndexFlatL2) |        24.41 |            49.56 | 1.0000 |
| 乘积量化 (IndexPQ)     |         0.77 |             2.23 | 0.2430 |
| 倒排文件 (IndexIVF)    |        24.44 |             0.90 | 0.8930 |
| IVF+PQ (IndexIVF_PQ)   |         1.57 |             0.53 | 0.2950 |
+------------------------+--------------+------------------+--------+
```

**结果分析**

暴力搜索精准但最慢最耗内存，PQ 省内存但精度损失大，IVF 兼顾精度与速度却不省内存，IVF_PQ 最快最省内存但精度中等。

三者核心是速度、召回率、内存不可兼得的三角权衡。

工程选择无最优解，仅需根据场景适配 —— 精度优先选暴力搜索 / IVF，速度 + 内存优先选 IVF_PQ，内存极度受限选 PQ。

### 3.5 总结与学习路径

#### 3.5.1 核心知识点梳理

1. **PQ（Product Quantization）**：通过“维度拆分 + 子空间量化”压缩高维向量，可通过参数 `m`（子向量数量）和 `nbits`（码本位数）调节压缩率与精度。
2. **IndexIVF_PQ**：适用于大规模向量数据检索，`nlist`（聚类中心数）和 `nprobe`（搜索分区数）是平衡速度与精度的关键参数。

#### 3.5.2 进阶学习建议

- **复现实验**：使用公开数据集如 SIFT1M 或 GIST1M进行实验，对比不同参数组合下的性能表现。
- **源码阅读**：重点查看 FAISS 中 `IndexPQ.cpp` 核心代码，理解 PQ 的量化实现原理。
- **拓展应用**：尝试将 IVF_PQ 应用于推荐系统（用户偏好向量检索）或图像检索场景，优化实际业务性能。

#### 3.5.3 常见问题解答

- **Q1：训练索引时提示“维度不匹配”？**
   A1：确保子向量数量 `m` 能整除向量维度 `d`，例如 `d=128` 时，`m` 可设为 8、16、32 等。
- **Q2：检索精度远低于预期？**
   A2：检查以下三点：训练数据是否具有代表性、`nprobe` 是否足够大（建议 ≥ `nlist` 的 5%）、码本位数 `nbits` 是否过小（建议 ≥ 8）。
