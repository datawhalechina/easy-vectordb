# Chapter 3：FAISS 核心功能进阶学习教程

通过“理论解析+核心 API+实战案例”的结构，帮助大家掌握复合索引设计、向量归一化、索引持久化及 GPU 加速等关键技能，解决大规模向量检索中的“精度-效率”平衡问题。

**前置准备**：

1. 安装 FAISS（CPU 版本：`pip install faiss-cpu`；GPU 版本：`pip install faiss-gpu`）；
2. 下载 SIFT1M 数据集（含 100 万张图片的 128 维特征向量，获取链接：https://huggingface.co/datasets/fzliu/sift1m/tree/main）；

## 1向量归一化与相似度适配：避免检索偏差

FAISS 中相似度计算依赖距离度量，而 COSINE 相似度（衡量向量方向一致性）需通过“L2 归一化”预处理才能正确计算，否则会导致结果偏差。

### 1.1 核心原理：COSINE 相似度与归一化的关系

COSINE 相似度用于衡量两个向量在方向上的一致性，公式如下：
$$
\text{cosine}(a,b) = \frac{a \cdot b}{\|a\| \|b\|}
$$
其中，`a⋅b`是向量 `a` 和 `b` 的内积，`|a|`和 `|b|` 是向量 `a` 和 `b` 的 L2 范数（即向量的长度）。


#### 1.1.1 L2 归一化对 COSINE 相似度的影响

如果我们对向量 `a` 和 `b` 进行 **L2 归一化**（即将它们的范数归一化为 1），那么：
$$
\|a\| = 1, \quad \|b\| = 1
$$
此时，COSINE 相似度就简化为：
$$
\text{cosine}(a,b) = a \cdot b
$$
也就是说，归一化后的向量之间的 COSINE 相似度直接等于它们的内积。

#### 1.1.2 L2 距离与 COSINE 相似度的关系

L2 距离（即欧几里得距离）可以通过 COSINE 相似度计算得到，具体关系如下：
$$
\|a - b\|^2 = 2(1 - \text{cosine}(a,b))
$$
由此可见，L2 距离与 COSINE 相似度是 **负相关** 的：当 COSINE 相似度越高，L2 距离越小。

#### 1.1.3 结论

1. 在计算 COSINE 相似度时，必须先对向量进行 L2 归一化处理（即将向量的长度规范化为 1）。
2. 使用 COSINE 相似度进行检索时，可以选择 **内积** 或 **L2 距离** 作为度量标准，前提是已对向量进行归一化。

### 1.2 核心 API：faiss.normalize_L2

```python
### 函数作用：对矩阵的每行（向量）做 L2 归一化
faiss.normalize_L2(x)  # x 为 numpy 数组（shape: N×d），原地修改

# 示例
x = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
faiss.normalize_L2(x)
print("归一化后向量：", x)
print("归一化后向量的 L2 范数：", np.linalg.norm(x, axis=1))  # 输出 [1. 1.]
```

> 如果运行后输出不是 `[1. 1.]`，大概率是代码执行时的精度显示问题（如 `[1.0000001 0.9999999]`），属于浮点数计算误差，本质仍满足归一化要求，不影响后续检索逻辑。

### 1.3 实战：归一化前后检索结果对比

**实战目标**：以 COSINE 相似度为目标，对比“未归一化”与“归一化”的检索精度差异。

```python
import faiss
import numpy as np

# 1. 构造测试数据（3 个数据库向量，2 个查询向量，模拟语义相似性）
xb = np.array([[1,2,3], [2,4,6], [3,3,3]], dtype=np.float32)  # 向量0与向量1方向完全一致（COSINE=1），向量2方向独特
xq = np.array([[0.5,1,1.5], [1,1,1]], dtype=np.float32)  # 查询0与向量0/1方向一致，查询1与向量2方向一致
d = xb.shape[1]  # 向量维度：3
k = 1  # 检索Top-1结果（只取最相似的1个）

# 方案 1：未归一化，用内积度量（错误方式）
index_no_norm = faiss.IndexFlatIP(d)  # 内积索引
index_no_norm.add(xb)  # 加入未归一化的数据库向量
D_no_norm, I_no_norm = index_no_norm.search(xq, k)  # 检索

# 方案 2：归一化后，用内积度量（正确方式）
xb_norm = xb.copy()
xq_norm = xq.copy()
faiss.normalize_L2(xb_norm)  # 数据库向量归一化（按行）
faiss.normalize_L2(xq_norm)  # 查询向量必须同步归一化（关键！）

index_norm = faiss.IndexFlatIP(d)
index_norm.add(xb_norm)
D_norm, I_norm = index_norm.search(xq_norm, k)

# 输出结果对比
print("=== 检索结果对比（目标：COSINE 相似度 Top-1）===")
print("数据库向量：\n", xb)
print("查询向量 0：", xq[0], "（与数据库向量0/1方向一致，COSINE=1）")
print("查询向量 1：", xq[1], "（与数据库向量2方向一致）")
print("\n【未归一化+内积】检索结果（错误，受向量长度干扰）：")
print(f"查询0 最相似向量索引：{I_no_norm[0][0]}，内积：{D_no_norm[0][0]:.2f}")  
print(f"查询1 最相似向量索引：{I_no_norm[1][0]}，内积：{D_no_norm[1][0]:.2f}")  

print("\n【归一化+内积】检索结果（正确，内积=COSINE相似度）：")
print(f"查询0 最相似向量索引：{I_norm[0][0]}，内积（COSINE）：{D_norm[0][0]:.2f}")  
print(f"查询1 最相似向量索引：{I_norm[1][0]}，内积（COSINE）：{D_norm[1][0]:.2f}")  
```

**结果分析**

未归一化时，内积受向量模长影响（向量 2 模长更大，虽与查询 1 方向一致但内积更高），导致检索错误；归一化后，内积直接等价于 COSINE 相似度，结果符合预期。