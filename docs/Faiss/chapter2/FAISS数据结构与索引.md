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

## 4 HNSW索引：图结构近邻检索

### 4.1HNSW 原理回顾：分层图的高效导航

在高维向量检索场景中，传统的精确最近邻（k-NN）搜索因计算复杂度极高而难以实用，近似最近邻（ANN）算法应运而生。HNSW（Hierarchical Navigable Small World，分层可导航小世界）作为当前业界主流的 ANN 算法之一，其核心创新在于通过分层图结构实现“快速导航+精准定位”的平衡。

#### 4.1.1 核心思想：小世界网络与分层结构

HNSW 借鉴了“小世界网络”特性——网络中任意两个节点之间都存在短路径，同时结合分层策略构建索引，结构如下：

- 多层图结构：将向量数据集构建为多层图，上层图为“粗粒度导航层”，节点连接稀疏，用于快速跨区域跳转；下层图为“细粒度精确层”，节点连接密集，用于精准定位近邻。最底层（第 0 层）包含全部数据节点，是检索的最终区域。

- 搜索流程：检索从顶层图的随机入口点开始，采用“贪婪搜索”策略向查询向量的近似近邻移动，直到找到当前层的局部最优节点；随后下降到下一层，以该最优节点为新入口点重复搜索，直至到达最底层，最终在底层的局部最优节点附近筛选出目标近邻。

这种分层导航策略既避免了暴力搜索的高复杂度，又解决了传统单一层图检索易陷入局部最优的问题，实现了检索速度与精度的高效平衡。

#### 4.1.2 图结构检索的核心优势

与 IVF（倒排文件）等基于聚类的索引相比，HNSW 的图结构检索具有显著优势：

- 高召回率：丰富的节点连接关系提供了多条导航路径，减少了因聚类划分导致的近邻遗漏问题，尤其在高维数据中表现更优。

- 稳定性能：检索性能受数据分布影响较小，对于非均匀分布的数据集，仍能保持稳定的搜索速度和精度。

- 灵活可调：通过核心参数可精准控制性能权衡，既能满足低延迟的实时检索需求，也能通过参数调优达到接近精确检索的精度。

### 4.2IndexHNSWFlat 核心参数解析

IndexHNSWFlat 是 Faiss 中 HNSW 索引的基础实现，采用“扁平存储”方式（不压缩向量，保证检索精度），其性能完全依赖于三个核心参数的配置。理解这些参数的作用是实现高效调参的关键。

**核心参数定义与作用**

| 参数名         | 核心作用                                                     | 取值范围                            | 性能权衡                                                     |
| :------------- | :----------------------------------------------------------- | :---------------------------------- | :----------------------------------------------------------- |
| M              | 定义图中每层节点的最大出度（第 0 层通常为 2*M），决定节点连接的密集程度 | 5-48（常用 8-32）                   | M 增大 → 导航路径更丰富、召回率更高，但索引构建时间更长、内存占用更大、查询延迟可能增加 |
| efConstruction | 索引构建时，动态筛选邻居的候选列表大小，决定邻居选择的充分性 | 几十到上千（通常远大于 M）          | efConstruction 增大 → 构建的图结构更优、检索精度更高，但索引构建时间显著增加 |
| efSearch       | 查询时，每层探索的候选邻居数量，直接控制查询精度与速度       | 不小于查询的近邻数 k（常用 10-200） | efSearch 增大 → 探索范围更广、召回率更高，但查询延迟增加；高质量索引（高 efConstruction）可降低对 efSearch 的依赖 |

### 4.3HNSW vs IVF-PQ 性能对比

本实战将通过完整代码构建 HNSW 与 IVF-PQ 索引，从索引构建时间、查询时间、召回率、内存占用四个维度进行对比，验证 HNSW 的性能优势。

#### 4.3.1实验设计

模拟高维向量场景（如文本或图像嵌入），参数设置如下：

- 向量维度 d=128（常见的嵌入维度）

- 基础数据集大小 nb=10000（1万条向量，模拟中等规模数据）

- 查询数据集大小 nq=100（100 条查询向量）

- 目标近邻数 k=5（检索 Top-10 近邻）

#### 4.3.2完整代码实现

```python
import time
import numpy as np
import faiss

# -------------------------- 数据配置（平衡速度与参考价值） --------------------------
np.random.seed(1234)
d = 128  # 向量维度（适中，避免计算量过大）
nb = 100000  # 基础数据量（1万条，比5千更有参考性）
nq = 100  # 查询数据量（30条，平衡测试速度与召回率稳定性）
k = 5  # 近邻数（5个，减少查询计算量）

# 生成数据（float32，Faiss默认格式）
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')
print(f"✅ 数据生成完成：{nb}条基础向量，{nq}条查询向量（维度{d}）")

# -------------------------- 精确检索基准（召回率对比标准） --------------------------
print("\n📊 计算精确检索基准结果...")
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D_flat, I_flat = index_flat.search(xq, k)
print("✅ 精确检索基准计算完成！")

# -------------------------- 召回率计算函数 --------------------------
def calculate_recall(I_pred, I_true, k):
    """预测结果命中真实近邻的平均比例"""
    recall_sum = 0.0
    for pred, true in zip(I_pred, I_true):
        hit_count = len(set(pred) & set(true))
        recall_sum += hit_count / k
    return recall_sum / len(I_pred)

# -------------------------- 1. 构建IVF-PQ索引（稳定优先，先运行） --------------------------
print("\n" + "="*60)
print("🚀 构建 IVF-PQ 索引")
# IVF-PQ参数（平衡速度、召回率、内存）
nlist = 30  # 聚类桶数量（30个，减少训练时间）
m = 8  # 子向量维度（64/8=8，必须整除）
bits = 6  # 量化位数（6位，平衡精度与内存）

# 初始化索引
quantizer = faiss.IndexFlatL2(d)  # 粗量化器（Flat索引）
index_ivf_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

# 训练+构建索引（单线程稳定运行）
print("  - 正在训练IVF-PQ聚类中心...")
start_time = time.time()
index_ivf_pq.train(xb)  # 训练聚类中心（IVF类索引必须先训练）
index_ivf_pq.add(xb)    # 添加向量构建索引
ivf_pq_build_time = time.time() - start_time

# 查询配置（nprobe越小越快，越大召回率越高）
index_ivf_pq.nprobe = 4  # 探索4个桶（平衡速度与召回率）
print("  - 正在执行查询...")
start_time = time.time()
D_ivf_pq, I_ivf_pq = index_ivf_pq.search(xq, k)
ivf_pq_search_time = time.time() - start_time

# 计算核心指标
ivf_pq_recall = calculate_recall(I_ivf_pq, I_flat, k)
ivf_pq_memory = (nlist * d * 4 + nb * m * bits / 8) / 1024 / 1024  # 换算为MB

# 输出IVF-PQ结果
print(f"  - 构建时间：{ivf_pq_build_time:.4f} 秒")
print(f"  - 查询时间：{ivf_pq_search_time:.4f} 秒")
print(f"  - 召回率：{ivf_pq_recall:.4f}")
print(f"  - 内存占用：{ivf_pq_memory:.2f} MB")
print("✅ IVF-PQ索引测试完成！")

# -------------------------- 2. 构建HNSW索引 --------------------------
print("\n" + "="*60)
print("🚀 构建 HNSW 索引")
# HNSW参数（极简稳定配置）
M = 6  # 节点出度（6个，平衡速度与召回率）
efConstruction = 50  # 构建选列表（50，保证基础质量）
efSearch = 20  # 查询选列表（20，平衡速度与召回率）

# 初始化索引
index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
index_hnsw.hnsw.efConstruction = efConstruction
index_hnsw.hnsw.efSearch = efSearch

# 分批添加（避免阻塞，每批10000条）
batch_size = 10000
print(f"  - 分批添加向量（每批{batch_size}条）...")
start_time = time.time()
for i in range(0, nb, batch_size):
    end = min(i + batch_size, nb)
    index_hnsw.add(xb[i:end])
    print(f"    > 已添加 {end:5d}/{nb} 条")
hnsw_build_time = time.time() - start_time

# 执行查询
print("  - 正在执行查询...")
start_time = time.time()
D_hnsw, I_hnsw = index_hnsw.search(xq, k)
hnsw_search_time = time.time() - start_time

# 计算核心指标
hnsw_recall = calculate_recall(I_hnsw, I_flat, k)
hnsw_memory = index_hnsw.ntotal * d * 4 / 1024 / 1024  # float32占4字节

# 输出HNSW结果
print(f"  - 构建时间：{hnsw_build_time:.4f} 秒")
print(f"  - 查询时间：{hnsw_search_time:.4f} 秒")
print(f"  - 召回率：{hnsw_recall:.4f}")
print(f"  - 内存占用：{hnsw_memory:.2f} MB")
print("✅ HNSW索引测试完成！")

# -------------------------- 性能对比汇总 --------------------------
print("\n" + "="*80)
print("📋 性能对比汇总表（HNSW vs IVF-PQ）")
print("="*80)
# 表头（左对齐指标名，右对齐数值，宽度固定）
header = f"{'指标':<15} {'HNSW':<18} {'IVF-PQ':<18}"
print(header)
print("-"*80)  # 分隔线
# 每行数据（统一格式：时间4位小数，召回率4位小数，内存2位小数）
rows = [
    (f"构建时间", f"{hnsw_build_time:.4f} 秒", f"{ivf_pq_build_time:.4f} 秒"),
    (f"查询时间", f"{hnsw_search_time:.4f} 秒", f"{ivf_pq_search_time:.4f} 秒"),
    (f"召回率", f"{hnsw_recall:.4f}", f"{ivf_pq_recall:.4f}"),
    (f"内存占用", f"{hnsw_memory:.2f} MB", f"{ivf_pq_memory:.2f} MB")
]
# 格式化输出（确保列对齐）
for metric, hnsw_val, ivf_pq_val in rows:
    print(f"{metric:<15} {hnsw_val:<18} {ivf_pq_val:<18}")
print("="*80)
```

运行结果

```
✅ 数据生成完成：100000条基础向量，100条查询向量（维度128）

📊 计算精确检索基准结果...
✅ 精确检索基准计算完成！

============================================================
🚀 构建 IVF-PQ 索引
  - 正在训练IVF-PQ聚类中心...
  - 正在执行查询...
  - 构建时间：0.5369 秒
  - 查询时间：0.0056 秒
  - 召回率：0.0160
  - 内存占用：0.59 MB
✅ IVF-PQ索引测试完成！

============================================================
🚀 构建 HNSW 索引
  - 分批添加向量（每批10000条）...
    > 已添加 10000/100000 条
    > 已添加 20000/100000 条
    > 已添加 30000/100000 条
    > 已添加 40000/100000 条
    > 已添加 50000/100000 条
    > 已添加 60000/100000 条
    > 已添加 70000/100000 条
    > 已添加 80000/100000 条
    > 已添加 90000/100000 条
    > 已添加 100000/100000 条
  - 正在执行查询...
  - 构建时间：0.7805 秒
  - 查询时间：0.0003 秒
  - 召回率：0.0520
  - 内存占用：48.83 MB
✅ HNSW索引测试完成！

================================================================================
📋 性能对比汇总表（HNSW vs IVF-PQ）
================================================================================
指标              HNSW               IVF-PQ
--------------------------------------------------------------------------------
构建时间            0.7805 秒           0.5369 秒
查询时间            0.0003 秒           0.0056 秒
召回率             0.0520             0.0160
内存占用            48.83 MB           0.59 MB
================================================================================

```

#### 4.3.3 结果分析与调参实践

运行代码后，通常会观察到以下规律：

- 召回率recall@5：HNSW 召回率（5.2%）高于 IVF-PQ（1.6%），体现图结构检索的精度优势。

- 时间：HNSW 构建时间略长于 IVF-PQ，但查询时间更稳定；IVF-PQ 查询时间受 nprobe 影响大，增大 nprobe 会接近 HNSW 精度但耗时增加。

- 内存：IVF-PQ 因量化压缩，内存占用显著低于 HNSW；HNSW 为扁平存储，内存占用与原始向量接近。

#### 4.3.4 自主学习实践

修改 HNSW 核心参数，观察性能变化，完成以下实验并记录结论：

1. 固定 efConstruction=200、efSearch=64，调整 M=8、16、32，观察召回率、内存与查询时间的变化。
2. 固定 M=16、efSearch=64，调整 efConstruction=100、200、400，观察索引构建时间与召回率的关系。
3. 固定 M=16、efConstruction=200，调整 efSearch=32、64、128，绘制“efSearch-召回率”“efSearch-查询时间”曲线。

### 4.5学习延伸：高维向量检索优化技巧

#### 4.5.1 核心知识点回顾

- HNSW 通过分层图结构实现“快速导航+精准检索”，是高维向量检索的优选方案。
- M、efConstruction、efSearch 是 HNSW 调参核心，需根据“召回率-速度-内存”需求动态平衡。
- 与 IVF-PQ 相比，HNSW 优势在高召回率和稳定性能，劣势是内存占用较高。

#### 4.5.2 核心参数调优策略

遵循“先定结构，再优质量，最后调速度”的迭代流程：

1. **确定 M 值**：根据数据维度选择初始值（高维数据 M 取 16-32，低维数据 M 取 8-16），以“内存占用可接受”为前提，优先保证图结构的鲁棒性。
2. **优化 efConstruction**：在构建时间允许的情况下，尽可能增大 efConstruction（如 200-500），构建高质量索引，为后续查询优化预留空间。
3. **调整 efSearch**：以“满足目标召回率”为目标，选择最小的 efSearch（如召回率要求 0.95 时，逐步增大 efSearch 直至达标）。

​      调参口诀：M 定连接密度，efConstruction 筑索引质量，efSearch 控查询快慢，三者平衡是关键。    

#### 4.5.3 硬件与工程优化

1. **GPU 加速**：对于超大规模数据（千万级以上），使用 GPU 版本 Faiss 可将查询速度提升 10-100 倍，核心代码只需修改索引初始化：
 ```python
# 初始化GPU索引（需安装faiss-gpu）
index_hnsw_gpu = faiss.index_cpu_to_gpu(res, 0, index_hnsw)  # 迁移CPU索引到GPU
 ```

2. **批量查询**：将单条查询合并为批量查询（如一次查询 100 条），利用向量计算的并行性，降低单位查询时间。

3. **索引持久化**：将构建好的索引保存到磁盘，避免重复构建，节省时间：
```python
# 保存索引 
faiss.write_index(index_hnsw, "hnsw_index.index") 
# 加载索引 
index_hnsw_load = faiss.read_index("hnsw_index.index")
```

## 5LSH 索引：哈希检索
### 5.1IndexLSH原理：哈希检索的核心逻辑
#### 5.1.1传统哈希与局部敏感哈希的区别

传统哈希函数的设计目标是*最小化哈希冲突*，确保不同输入映射到不同哈希值；而LSH（局部敏感哈希）则反其道而行之，通过*最大化相似向量的哈希冲突*，将相似的高维向量映射到同一个哈希桶中，非相似向量映射到同一桶的概率极低。这种特性使LSH能在不遍历全量数据的情况下，快速筛选出潜在的相似向量，大幅提升近似检索效率。

#### 5.1.2 FAISS IndexLSH的实现原理

FAISS中的IndexLSH采用*随机超平面哈希*（Random Hyperplanes）实现，核心流程如下：

1. 构建随机超平面：生成n_bits个随机超平面（每个超平面由一个d维法向量表示，d为向量维度），这些超平面将高维空间分割为多个区域；
2. 向量哈希编码：对于每个输入向量，计算其与所有超平面法向量的点积，点积为正则编码为1，负则编码为0，最终形成一个n_bits位的二进制哈希码；
3. 桶存储与检索：将具有相同哈希码的向量归入同一桶，查询时仅需计算查询向量的哈希码，在对应桶内筛选相似向量，避免全量比对。

关键结论：n_bits（哈希码长度）是核心参数——n_bits越大，哈希码区分度越高，检索精度越高，但哈希桶数量呈2^n_bits增长，可能导致桶内向量数量过少，反而降低效率；n_bits越小则相反。

#### 5.2IndexLSH核心API全解析
IndexLSH核心API

| API接口                   | 功能描述                    | 关键参数                                                     |
| :------------------------ | :-------------------------- | :----------------------------------------------------------- |
| faiss.IndexLSH(d, n_bits) | 初始化LSH索引               | d：向量维度；n_bits：哈希码长度（推荐16-64）                 |
| index.add(xb)             | 将基础向量集加入索引        | xb：形状为(nb, d)的float32向量集（nb为向量数量）             |
| index.search(xq, k)       | 检索查询向量的k个近似最近邻 | xq：(nq, d)的查询向量集；k：需返回的近邻数；返回值(D, I)：D为距离矩阵，I为索引矩阵 |
| index.reset()             | 清空索引中的向量            | 无关键参数，用于重新构建索引                                 |

### 5.3IndexLSH适用场景：低维数据的高效近似检索
#### 5.3.1 核心适用场景

IndexLSH的核心优势是*低计算成本与简单实现*，最适合以下场景：

- 低维向量检索（维度d<=512）：高维空间中随机超平面的区分度会下降，导致检索精度大幅降低；
- 近似检索优先的场景：如推荐系统的候选召回、图像/文本的快速相似性筛选，允许少量精度损失以换取速度提升；
- 中小规模数据集（向量数量10万-1000万）：无需复杂的聚类预处理，索引构建速度快于IVF等聚类类索引。

#### 5.3.2 不适用场景

1. 高维向量检索（d>1024）：哈希码的区分能力不足，召回率会显著低于HNSW、IVF-PQ等索引；
2. 精确检索需求：LSH是近似检索算法，无法保证返回绝对最优的近邻结果；
3. 超大规模数据集（>1亿向量）：内存占用会随向量数量线性增长，不如IVF-PQ通过量化压缩内存的优势明显。

### 5.4实战：LSH vs IVF-PQ 性能对比
#### 5.4.1实验设计

- 数据集：随机生成10万条128维float32向量（基础集xb），100条查询向量（查询集xq）；
- 对比对象：IndexLSH、IndexIVFPQ；
- 评估指标：索引构建时间、单条查询平均时间、召回率（以IndexFlatL2的精确结果为基准）。

#### 5.4.2完整代码实现
```python
import time
import numpy as np
import faiss

# -------------------------- 1. 实验配置与数据准备 --------------------------
# 基础配置
d = 128  # 向量维度
nb = 100000  # 基础向量数量
nq = 1000  # 查询向量数量
k = 10  # 检索的近邻数
np.random.seed(1234)  # 固定随机种子，保证结果可复现

# 生成测试数据（float32类型是FAISS的要求）
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 构建精确索引作为召回率基准
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D_flat, I_flat = index_flat.search(xq, k)  # 精确结果，用于计算召回率

# -------------------------- 2. 定义性能评估函数 --------------------------
def calculate_recall(I_pred, I_true, k):
    """
    计算召回率：预测结果中命中真实近邻的比例
    参数：I_pred-模型预测的索引矩阵，I_true-精确结果的索引矩阵，k-近邻数
    返回：平均召回率
    """
    recall_list = []
    for i in range(len(I_pred)):
        pred_set = set(I_pred[i])
        true_set = set(I_true[i])
        hit = len(pred_set & true_set)
        recall = hit / k
        recall_list.append(recall)
    return np.mean(recall_list)

# -------------------------- 3. LSH索引构建与性能测试 --------------------------
print("=== 测试IndexLSH性能 ===")
# 初始化LSH索引（n_bits设为32，平衡精度与速度）
index_lsh = faiss.IndexLSH(d, 32)

# 索引构建时间
start_time = time.time()
index_lsh.add(xb)
lsh_index_time = time.time() - start_time
print(f"LSH索引构建时间：{lsh_index_time:.4f} 秒")

# 检索性能测试
start_time = time.time()
D_lsh, I_lsh = index_lsh.search(xq, k)
lsh_search_time = (time.time() - start_time) 
print(f"LSH查询时间：{lsh_search_time:.6f} 秒")

# 计算召回率
lsh_recall = calculate_recall(I_lsh, I_flat, k)
print(f"LSH召回率：{lsh_recall:.4f}")

# -------------------------- 4. IVF-PQ索引构建与性能测试 --------------------------
print("\n=== 测试IndexIVFPQ性能 ===")
# IVF-PQ需要先定义量化器（通常用Flat索引）
quantizer = faiss.IndexFlatL2(d)
nlist = 100  # 聚类桶数量
m = 16  # 乘积量化的分段数（需整除向量维度d）
nbits_per_idx = 8  # 每个分段的编码位数

# 初始化IVF-PQ索引
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits_per_idx)

# IVF-PQ需先训练（聚类过程）
start_time = time.time()
index_ivfpq.train(xb)  # 训练聚类中心
index_ivfpq.add(xb)    # 加入向量构建索引
ivfpq_index_time = time.time() - start_time
print(f"IVF-PQ索引构建（含训练）时间：{ivfpq_index_time:.4f} 秒")

# 设置查询时的探测桶数量（nprobe越大，召回率越高但速度越慢）
index_ivfpq.nprobe = 10

# 检索性能测试
start_time = time.time()
D_ivfpq, I_ivfpq = index_ivfpq.search(xq, k)
ivfpq_search_time = (time.time() - start_time)
print(f"IVF-PQ查询时间：{ivfpq_search_time:.6f} 秒")

# 计算召回率
ivfpq_recall = calculate_recall(I_ivfpq, I_flat, k)
print(f"IVF-PQ召回率：{ivfpq_recall:.4f}")

print("\n" + "="*80)
print("📋 性能对比汇总表（LSH vs IVF-PQ）")
print("="*80)
# 表头（左对齐指标名，右对齐数值，宽度固定）
header = f"{'指标':<15} {'LSH':<18} {'IVF-PQ':<18}"
print(header)
print("-"*80)  # 分隔线
# 每行数据（统一格式：时间4位小数，召回率4位小数，内存2位小数）
rows = [
    (f"构建时间", f"{lsh_index_time:.4f} 秒", f"{ivfpq_index_time:.4f} 秒"),
    (f"查询时间", f"{lsh_search_time:.4f} 秒", f"{ivfpq_search_time:.4f} 秒"),
    (f"召回率", f"{lsh_recall:.4f}", f"{ivfpq_recall:.4f}")
]
# 格式化输出（确保列对齐）
for metric, lsh_val, ivfpq_val in rows:
    print(f"{metric:<15} {lsh_val:<18} {ivfpq_val:<18}")
print("="*80)

```
运行结果：
```
=== 测试IndexLSH性能 ===
LSH索引构建时间：0.0133 秒
LSH查询时间：0.007148 秒
LSH召回率：0.0012

=== 测试IndexIVFPQ性能 ===
IVF-PQ索引构建（含训练）时间：0.6926 秒
IVF-PQ查询时间：0.007568 秒
IVF-PQ召回率：0.1419

================================================================================
📋 性能对比汇总表（LSH vs IVF-PQ）
================================================================================
指标              LSH                IVF-PQ
--------------------------------------------------------------------------------
构建时间            0.0133 秒           0.6926 秒
查询时间            0.0071 秒           0.0076 秒
召回率             0.0012             0.1419
================================================================================
```

结果分析：
1. 索引时间：LSH无需训练，索引构建速度远快于IVF-PQ（需聚类训练）；
2. 检索时间：低维场景下LSH略快，高维或大规模数据中IVF-PQ的优势会更明显；
3. 召回率：IVF-PQ通过量化与聚类优化，召回率通常高于LSH，但需更多计算资源。

### 5.5总结与拓展
#### 5.5.1核心知识点梳理
- LSH原理：通过随机超平面生成哈希码，将相似向量聚集到同一桶，实现快速近似检索；
- API核心：IndexLSH(d, n_bits)初始化，add()加向量，search()做检索，n_bits是精度与速度的调节关键；
- 适用场景：低维、中小规模、近似检索优先的场景，如快速候选召回。

#### 5.5.2拓展练习
- 尝试不同的n_bits值（如16、32、64），观察索引时间、查询时间与召回率的变化趋势；
- 对比IVF-PQ的nlist（聚类桶数）与m（乘积量化分段数）对性能的影响；
- 分析高维数据（如512维）下LSH与IVF-PQ的性能差异，思考原因。
