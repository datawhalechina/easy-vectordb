# Annoy入门与环境搭建

## 1. Annoy 简介

**Annoy**（Approximate Nearest Neighbors Oh Yeah）是由 **Spotify** 开发并开源的轻量级近似最近邻搜索库。它被广泛应用于 Spotify 的音乐推荐系统中，用于快速查找相似的歌曲和用户。

### 核心优势

| 特性 | 说明 |
|------|------|
| **内存映射（mmap）** | 索引文件可以通过内存映射加载，不需要将整个索引加载到内存中 |
| **多进程共享** | 多个进程可以共享同一份索引文件，极大降低内存开销 |
| **API 简洁** | 只有几个核心方法，5 分钟即可上手 |
| **跨平台** | 支持 Python、C++、Java、Go 等多种语言绑定 |

### 适用场景

Annoy 特别适合以下场景：

- **单机部署**：不需要分布式架构
- **只读索引**：数据不频繁更新，可以定期重建
- **数据量中等**：百万到千万级向量
- **多进程服务**：Web 服务需要多个 worker 共享索引

### 不适用场景

- 需要频繁增删改的场景（Annoy 不支持增量更新）
- 需要分布式部署的超大规模数据（考虑 Milvus）
- 需要丰富索引类型和 GPU 加速（考虑 Faiss）

---

## 2. 环境安装

### 2.1 pip 安装（推荐）

```bash
pip install annoy
```

### 2.2 conda 安装

```bash
conda install -c conda-forge python-annoy
```

### 2.3 验证安装

```python
import annoy

# 创建一个简单的索引测试
t = annoy.AnnoyIndex(3, 'angular')
t.add_item(0, [1, 0, 0])
t.add_item(1, [0, 1, 0])
t.add_item(2, [0, 0, 1])
t.build(10)

# 搜索测试
result = t.get_nns_by_vector([1, 0.5, 0], 2)
print(f"搜索结果: {result}")
print("Annoy 安装成功！")
```

运行上述代码，如果输出类似以下内容，说明安装成功：

```
搜索结果: [0, 1]
Annoy 安装成功！
```

### 2.4 常见安装问题

**问题 1：Windows 上编译失败**

Annoy 需要 C++ 编译器。Windows 用户如果遇到编译错误，有以下解决方案：

**方案 A：使用预编译的 wheel 文件（推荐）**

从 GitHub 下载预编译的 wheel 文件：[https://github.com/Sprocketer/annoy-wheels](https://github.com/Sprocketer/annoy-wheels)

根据你的 Python 版本下载对应的 `.whl` 文件，然后本地安装：

```bash
# 示例：Python 3.11 + Windows 64位
pip install annoy-1.17.3-cp311-cp311-win_amd64.whl
```

**方案 B：使用 conda 安装**

```bash
conda install -c conda-forge python-annoy
```

**方案 C：安装 C++ 编译环境**

安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，然后重新 `pip install annoy`。

**问题 2：Apple Silicon (M1/M2) 兼容性**

较新版本的 Annoy 已支持 Apple Silicon，直接 pip 安装即可。如遇问题，尝试：
```bash
pip install --upgrade annoy
```

---

## 3. 快速上手：5 分钟入门

让我们通过一个完整的例子，快速了解 Annoy 的核心用法。

### 3.1 完整示例

```python
from annoy import AnnoyIndex
import numpy as np

# ============ 第一步：创建索引 ============
# 参数说明：
# - 128: 向量维度
# - 'angular': 距离度量（余弦距离）
index = AnnoyIndex(128, 'angular')

# ============ 第二步：添加向量 ============
# 生成一些随机向量模拟数据
np.random.seed(42)
n_vectors = 10000
vectors = np.random.randn(n_vectors, 128).astype('float32')

# 添加向量到索引
# add_item(i, vector): i 是向量的唯一标识（整数）
for i, vec in enumerate(vectors):
    index.add_item(i, vec)

print(f"已添加 {index.get_n_items()} 个向量")

# ============ 第三步：构建索引 ============
# build(n_trees): n_trees 是树的数量，越多精度越高但占用更多内存
n_trees = 10
index.build(n_trees)
print(f"索引构建完成，共 {index.get_n_trees()} 棵树")

# ============ 第四步：保存索引 ============
index.save('demo_index.ann')
print("索引已保存到 demo_index.ann")

# ============ 第五步：加载索引 ============
# 创建新的索引对象并加载
index2 = AnnoyIndex(128, 'angular')
index2.load('demo_index.ann')
print("索引加载成功")

# ============ 第六步：搜索 ============
# 创建一个查询向量
query = np.random.randn(128).astype('float32')

# get_nns_by_vector(vector, n, search_k=-1, include_distances=False)
# - vector: 查询向量
# - n: 返回最近的 n 个结果
# - search_k: 搜索时检查的节点数，-1 表示使用默认值 n_trees * n
# - include_distances: 是否返回距离
results = index2.get_nns_by_vector(query, 5, include_distances=True)

print(f"\n查询结果（前5个最近邻）:")
print(f"索引: {results[0]}")
print(f"距离: {results[1]}")

# ============ 第七步：根据已有向量搜索 ============
# get_nns_by_item(i, n): 找到与第 i 个向量最相似的 n 个向量
similar_to_0 = index2.get_nns_by_item(0, 5, include_distances=True)
print(f"\n与向量 0 最相似的 5 个向量:")
print(f"索引: {similar_to_0[0]}")
print(f"距离: {similar_to_0[1]}")
```

运行代码后，你将看到类似输出：

```
已添加 10000 个向量
索引构建完成，共 10 棵树
索引已保存到 demo_index.ann
索引加载成功

查询结果（前5个最近邻）:
索引: [3122, 5514, 1890, 1286, 2316]
距离: [1.254392, 1.2594, 1.261284, 1.277733, 1.285302]

与向量 0 最相似的 5 个向量:
索引: [0, 1144, 24, 6459, 4111]
距离: [0.000135, 1.266683, 1.277138, 1.280767, 1.28747]
```

::: tip 注意
由于随机数生成，你的具体数值可能不同，但格式是一致的。第一个结果的距离为 0 是因为查询的是向量本身。
:::

### 3.2 代码要点总结

| 步骤 | 方法 | 说明 |
|------|------|------|
| 创建索引 | `AnnoyIndex(dim, metric)` | 指定维度和距离度量 |
| 添加向量 | `add_item(i, vector)` | i 必须是非负整数 |
| 构建索引 | `build(n_trees)` | 构建后不能再添加向量 |
| 保存索引 | `save(filename)` | 保存到文件 |
| 加载索引 | `load(filename)` | 从文件加载 |
| 搜索（向量） | `get_nns_by_vector(v, n)` | 根据向量搜索 |
| 搜索（索引） | `get_nns_by_item(i, n)` | 根据已有向量的索引搜索 |

---

## 4. 与 base 章节的衔接

如果你已经学习了 base 章节的内容，你可能注意到：

- **base/chapter5/Annoy算法.md** 中我们**手动实现**了一个 `SimpleAnnoy` 类，深入理解了随机投影树的原理
- 本章使用的是 **Spotify 官方的 annoy 库**，它是用 C++ 实现的，性能更高

两者的关系：

| 对比项 | base 中的 SimpleAnnoy | 本章的 annoy 库 |
|--------|----------------------|----------------|
| 目的 | 理解算法原理 | 生产环境使用 |
| 实现语言 | Python | C++（Python 绑定） |
| 性能 | 较慢（教学用） | 高性能 |
| 功能 | 基础功能 | 完整功能（mmap、多线程等） |

::: tip 推荐学习路径
1. 先阅读 [base/chapter5/Annoy算法](/base/chapter5/Annoy算法) 理解随机投影树原理
2. 再学习本章，掌握官方库的实际使用
:::
