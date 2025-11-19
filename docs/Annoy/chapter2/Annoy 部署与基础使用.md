# Annoy 部署与基础使用指南

本章节聚焦于 Annoy 的 **环境准备、安装方式与基础 API 使用**，帮助你在几分钟内搭建一个可用的向量检索 Demo。

## 1. 环境要求与依赖

Annoy 的核心是用 C++ 实现的，Python 只是包装层，因此：

- 操作系统：Linux / macOS / Windows 均支持
- Python 版本：建议 Python 3.8+
- 编译工具链：
  - Linux/macOS：通常系统自带 `gcc/clang` 即可
  - Windows：如果用 `pip` 安装二进制轮子，一般无需额外配置；如需从源码构建，则需要 VS Build Tools

> 一般学习/实验场景下，直接 `pip install annoy` 即可，无需关心底层编译细节。

## 2. 安装 Annoy

### 2.1 使用 pip 安装（推荐）

```bash
pip install annoy
```

安装完成后，可以简单验证：

```python
import annoy
print(annoy.__version__)
```

如能正常打印版本号，说明安装成功。

### 2.2 使用 conda 安装

在部分发行版中，也可以通过 conda 获取 Annoy：

```bash
conda install -c conda-forge python-annoy
```

> 实际使用时，优先推荐 pip；如果你整个项目依赖都通过 conda 管理，可以保持一致。

### 2.3 从源码安装（了解即可）

在需要自定义编译选项或做二次开发时，可以从源码安装：

```bash
git clone https://github.com/spotify/annoy.git
cd annoy
python setup.py install
```

## 3. 快速开始示例

下面通过一个完整的“构建索引 → 查询近邻”的最小示例，串联起 Annoy 的基础使用流程。

```python
from annoy import AnnoyIndex
import numpy as np

# 1. 定义向量维度与距离度量
f = 64
index = AnnoyIndex(f, "angular")  # 也可以是 "euclidean" 等

# 2. 构造示例向量并添加到索引
np.random.seed(42)
num_items = 1000
for i in range(num_items):
    vector = np.random.normal(size=f)
    vector = vector / np.linalg.norm(vector)
    index.add_item(i, vector)

# 3. 构建索引（n_trees 越大，精度越高，构建时间越久）
index.build(n_trees=20)

# 4. 查询：给定一个向量 ID，找出最相似的 5 个向量
neighbors, distances = index.get_nns_by_item(
    0,        # 查询 ID
    5,        # 返回 Top-5
    include_distances=True
)
print("neighbors:", neighbors)
print("distances:", distances)
```

这个示例展示了 Annoy 的完整最小闭环：

1. 指定 **维度 + 距离度量** 创建索引
2. 通过 `add_item()` 添加向量
3. 使用 `build()` 构建随机投影树索引
4. 使用 `get_nns_by_item()` 做相似向量检索

## 4. 基础 API 使用说明

Annoy 的核心类是 `AnnoyIndex`，围绕它掌握几个常用方法即可完成绝大多数入门/实战需求。

### 4.1 AnnoyIndex 初始化

```python
from annoy import AnnoyIndex

# f: 向量维度, metric: 距离度量方式
index = AnnoyIndex(f=128, metric="angular")
```

常用距离度量：

- `"angular"`：角度距离，最常用（等价于余弦相似度）
- `"euclidean"`：欧氏距离
- `"manhattan"`：曼哈顿距离
- `"hamming"`：汉明距离（二进制向量）
- `"dot"`：点积

### 4.2 添加向量与构建索引

```python
import numpy as np
from annoy import AnnoyIndex

f = 32
index = AnnoyIndex(f, "angular")

# 添加若干条向量，ID 需要是 0 ~ n-1 的整数
for item_id in range(1000):
    vec = np.random.randn(f)
    index.add_item(item_id, vec)

# 构建索引
index.build(n_trees=10)
```

常见经验参数：

- `n_trees`：
  - 10～50：一般推荐范围
  - 越大精度越高，但构建时间和内存消耗也会增加
- 构建完成后索引即为只读结构，无法继续 `add_item()`

### 4.3 基础查询接口

Annoy 提供两种常用查询方式：

1. **根据已有向量 ID 查询**：`get_nns_by_item()`
2. **根据新的向量查询**：`get_nns_by_vector()`

```python
# 方式一：根据 ID 查询
neighbors, distances = index.get_nns_by_item(
    item_id=0,
    n=10,
    include_distances=True
)

# 方式二：根据向量查询
query_vec = np.random.randn(f)
neighbors2, distances2 = index.get_nns_by_vector(
    query_vec,
    n=10,
    include_distances=True
)
```

调参提示：

- `n`：返回 Top-K 的大小
- `search_k`（可选）：控制搜索候选数，值越大精度越高、耗时越长；不指定时 Annoy 会给出一个经验值

### 4.4 索引持久化与加载

在生产环境中，通常会将构建好的索引保存为文件，并在服务启动时加载：

```python
from annoy import AnnoyIndex

# 保存索引
index.save("movie_embedding.ann")

# 线上服务中加载索引
new_index = AnnoyIndex(f, "angular")
new_index.load("movie_embedding.ann")  # 内部使用 mmap 映射

print("loaded items:", new_index.get_n_items())
```

> 由于使用 mmap，多个进程加载同一索引文件时，并不会复制多份数据到物理内存，对多进程 Web 服务非常友好。

## 5. 一个简单的文本相似检索示例（思路版）

真实项目中，Annoy 常常与 **文本/图片嵌入模型** 联合使用。下面是一个简化的“文本相似检索”流程示意：

```python
from annoy import AnnoyIndex
# 示例：伪代码，假设已经有一个 encode(text) → 向量 的函数

sentences = [
    "向量数据库入门教程",
    "使用 Annoy 做相似度检索",
    "Milvus 部署与实践",
]

# 1. 生成文本向量
evectors = [encode(s) for s in sentences]

# 2. 构建 Annoy 索引
f = len(evectors[0])
index = AnnoyIndex(f, "angular")
for i, v in enumerate(evectors):
    index.add_item(i, v)
index.build(20)

# 3. 查询与输入文本最相似的句子
query = "如何用 Annoy 做向量检索？"
q_vec = encode(query)
ids, dists = index.get_nns_by_vector(q_vec, 3, include_distances=True)

print("最相似的句子:")
for idx in ids:
    print("-", sentences[idx])
```

在后续章节中，你可以结合本仓库中 `docs/Annoy` 目录下的其他文档（如算法对比、参数调优等），逐步将 Annoy 嵌入到自己的向量检索项目中。
