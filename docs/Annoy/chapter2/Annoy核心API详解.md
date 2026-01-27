# Annoy核心API详解

本章详细介绍 Annoy 库的所有 API，帮助你全面掌握这个库的使用方法。

## 1. AnnoyIndex 类

`AnnoyIndex` 是 Annoy 的核心类，所有操作都围绕它进行。

### 1.1 构造函数

```python
from annoy import AnnoyIndex

# AnnoyIndex(f, metric)
# - f: 向量维度（正整数）
# - metric: 距离度量类型（字符串）
index = AnnoyIndex(128, 'angular')
```

### 1.2 支持的距离度量

Annoy 支持以下 5 种距离度量：

| metric | 名称 | 公式 | 适用场景 |
|--------|------|------|----------|
| `'angular'` | 角距离（余弦距离） | $\sqrt{2(1 - \cos\theta)}$ | **推荐**，文本、图像嵌入 |
| `'euclidean'` | 欧氏距离 | $\sqrt{\sum(x_i - y_i)^2}$ | 物理空间距离 |
| `'manhattan'` | 曼哈顿距离 | $\sum\|x_i - y_i\|$ | 网格路径距离 |
| `'hamming'` | 汉明距离 | 不同位的数量 | 二进制特征 |
| `'dot'` | 点积（内积） | $\sum x_i \cdot y_i$ | 最大化内积（推荐系统） |

::: warning 关于 angular 和余弦相似度
`angular` 返回的是**角距离**，不是余弦相似度。转换公式：
- 余弦相似度 = 1 - (angular_distance² / 2)
- 值越小表示越相似
:::

**示例：不同度量的使用**

```python
from annoy import AnnoyIndex
import numpy as np

# 创建测试向量
v1 = [1.0, 0.0, 0.0]
v2 = [0.0, 1.0, 0.0]
v3 = [1.0, 1.0, 0.0]

metrics = ['angular', 'euclidean', 'manhattan', 'dot']

for metric in metrics:
    index = AnnoyIndex(3, metric)
    index.add_item(0, v1)
    index.add_item(1, v2)
    index.add_item(2, v3)
    index.build(10)
    
    # 查询与 v1 最相似的向量
    results = index.get_nns_by_item(0, 3, include_distances=True)
    print(f"{metric:12} -> 索引: {results[0]}, 距离: {[f'{d:.4f}' for d in results[1]]}")
```

输出：
```
angular      -> 索引: [0, 2, 1], 距离: ['0.0000', '0.7654', '1.4142']
euclidean    -> 索引: [0, 2, 1], 距离: ['0.0000', '1.0000', '1.4142']
manhattan    -> 索引: [0, 2, 1], 距离: ['0.0000', '1.0000', '2.0000']
dot          -> 索引: [0, 2, 1], 距离: ['1.0000', '1.0000', '0.0000']
```

::: tip dot 度量的特殊性
使用 `dot` 时，Annoy 会返回转换后的距离值。对于点积度量，值越小表示相似度越高（内积越大）。
:::

---

## 2. 索引构建 API

### 2.1 add_item(i, vector)

向索引中添加一个向量。

```python
index.add_item(i, vector)
```

**参数说明：**
- `i`（int）：向量的唯一标识符，必须是**非负整数**
- `vector`（list/array）：向量数据，长度必须等于创建索引时指定的维度

**注意事项：**
- 索引 `i` 不需要连续，但建议使用 0, 1, 2, ... 的连续序列
- 如果 `i` 已存在，新向量会**覆盖**旧向量
- 添加操作必须在 `build()` 之前完成

```python
index = AnnoyIndex(3, 'angular')

# 正确用法
index.add_item(0, [1.0, 2.0, 3.0])
index.add_item(1, [4.0, 5.0, 6.0])

# 也可以使用 numpy 数组
import numpy as np
index.add_item(2, np.array([7.0, 8.0, 9.0]))

# 索引不需要连续（但不推荐）
index.add_item(100, [10.0, 11.0, 12.0])

print(f"向量数量: {index.get_n_items()}")  # 输出: 向量数量: 4
```

### 2.2 build(n_trees, n_jobs=-1)

构建索引。调用此方法后，索引进入**只读状态**，不能再添加新向量。

```python
index.build(n_trees, n_jobs=-1)
```

**参数说明：**
- `n_trees`（int）：构建的树的数量
  - 越多 → 精度越高，但构建时间和内存占用增加
  - 推荐值：10-100，根据数据量和精度需求调整
- `n_jobs`（int）：并行构建的线程数
  - `-1`：使用所有 CPU 核心
  - `1`：单线程
  - `n`：使用 n 个线程

```python
# 单线程构建
index.build(10, n_jobs=1)

# 多线程构建（推荐）
index.build(50, n_jobs=-1)
```

### 2.3 unbuild()

解除索引的构建状态，允许继续添加向量。

```python
index.unbuild()
```

**使用场景：** 当你需要在已构建的索引上添加更多向量时。

```python
index = AnnoyIndex(3, 'angular')
index.add_item(0, [1, 2, 3])
index.build(10)

# 此时不能添加向量
# index.add_item(1, [4, 5, 6])  # 会报错

# 解除构建状态
index.unbuild()

# 现在可以添加了
index.add_item(1, [4, 5, 6])
index.build(10)  # 需要重新构建
```

::: warning 注意
`unbuild()` 后需要重新调用 `build()`，这意味着索引会完全重建。对于大型索引，这可能非常耗时。
:::

---

## 3. 索引持久化 API

### 3.1 save(filename, prefault=False)

将索引保存到文件。

```python
index.save(filename, prefault=False)
```

**参数说明：**
- `filename`（str）：保存的文件路径
- `prefault`（bool）：是否预加载所有页面到内存
  - `False`（默认）：按需加载（推荐）
  - `True`：保存时预加载所有数据到内存

```python
# 保存索引
index.save('my_index.ann')

# 带 prefault 的保存（一般不需要）
index.save('my_index.ann', prefault=True)
```

### 3.2 load(filename, prefault=False)

从文件加载索引。

```python
index.load(filename, prefault=False)
```

**参数说明：**
- `filename`（str）：索引文件路径
- `prefault`（bool）：是否预加载所有页面到内存
  - `False`（默认）：使用内存映射，按需加载（**推荐**）
  - `True`：将整个文件加载到内存

**两种加载模式的对比：**

| 模式 | prefault=False（默认） | prefault=True |
|------|------------------------|---------------|
| 内存占用 | 低（按需加载） | 高（全部加载） |
| 首次查询 | 稍慢（需要磁盘IO） | 快（已在内存中） |
| 多进程共享 | **支持** | 不支持 |
| 适用场景 | 生产环境 | 对延迟极度敏感且内存充足 |

```python
# 创建新的索引对象
index2 = AnnoyIndex(128, 'angular')

# 使用内存映射加载（推荐）
index2.load('my_index.ann')

# 完全加载到内存
index2.load('my_index.ann', prefault=True)
```

### 3.3 unload()

卸载索引，释放内存。

```python
index.unload()
```

**使用场景：** 当你需要释放内存但保留索引对象时。

```python
index.load('my_index.ann')
# ... 使用索引 ...

# 释放内存
index.unload()

# 可以重新加载
index.load('another_index.ann')
```

### 3.4 on_disk_build(filename)

直接在磁盘上构建索引，适用于内存不足以容纳整个索引的情况。

```python
index.on_disk_build(filename)
```

**使用方法：**

```python
index = AnnoyIndex(128, 'angular')

# 指定在磁盘上构建
index.on_disk_build('large_index.ann')

# 添加向量（数据直接写入磁盘）
for i in range(1000000):
    vec = np.random.randn(128).tolist()
    index.add_item(i, vec)

# 构建索引
index.build(10)

# 索引已自动保存到 large_index.ann
```

::: tip 适用场景
当数据量太大，无法全部放入内存时，使用 `on_disk_build` 可以边添加边写入磁盘。
:::

---

## 4. 搜索 API

### 4.1 get_nns_by_vector(vector, n, search_k=-1, include_distances=False)

根据向量搜索最近邻。

```python
results = index.get_nns_by_vector(vector, n, search_k=-1, include_distances=False)
```

**参数说明：**
- `vector`（list/array）：查询向量
- `n`（int）：返回最近的 n 个结果
- `search_k`（int）：搜索时检查的节点数量
  - `-1`（默认）：使用 `n_trees * n`
  - 值越大 → 精度越高，但速度越慢
- `include_distances`（bool）：是否返回距离

**返回值：**
- `include_distances=False`：返回索引列表 `[i1, i2, ...]`
- `include_distances=True`：返回元组 `([i1, i2, ...], [d1, d2, ...])`

```python
# 只返回索引
neighbors = index.get_nns_by_vector([1.0, 2.0, 3.0], 10)
print(neighbors)  # [5, 23, 17, 8, 42, ...]

# 返回索引和距离
neighbors, distances = index.get_nns_by_vector([1.0, 2.0, 3.0], 10, include_distances=True)
print(neighbors)   # [5, 23, 17, 8, 42, ...]
print(distances)   # [0.123, 0.234, 0.345, ...]

# 提高精度（增加 search_k）
neighbors = index.get_nns_by_vector([1.0, 2.0, 3.0], 10, search_k=10000)
```

### 4.2 get_nns_by_item(i, n, search_k=-1, include_distances=False)

根据已有向量的索引搜索最近邻。

```python
results = index.get_nns_by_item(i, n, search_k=-1, include_distances=False)
```

**参数说明：**
- `i`（int）：查询向量的索引
- 其他参数同 `get_nns_by_vector`

```python
# 找到与向量 0 最相似的 10 个向量
neighbors = index.get_nns_by_item(0, 10)

# 注意：结果中会包含向量自身（距离为 0）
neighbors, distances = index.get_nns_by_item(0, 10, include_distances=True)
print(neighbors[0])   # 0（向量自身）
print(distances[0])   # 0.0
```

### 4.3 search_k 参数调优

`search_k` 是影响搜索精度和速度的关键参数。

**经验公式：**
```
search_k = n_trees * n * factor
```

其中 `factor` 根据需求调整：
- `factor = 1`：默认值，速度快但精度一般
- `factor = 10-100`：精度高但速度较慢
- `factor = 1000+`：接近暴力搜索的精度

**示例：不同 search_k 的效果对比**

```python
from annoy import AnnoyIndex
import numpy as np
import time

# 创建索引
dim = 128
n_items = 100000
n_trees = 10

index = AnnoyIndex(dim, 'angular')
np.random.seed(42)

for i in range(n_items):
    index.add_item(i, np.random.randn(dim))
    
index.build(n_trees)

# 查询向量
query = np.random.randn(dim)

# 测试不同 search_k
search_k_values = [-1, 100, 1000, 10000, 100000]

print("search_k 参数对比测试")
print("-" * 50)

for search_k in search_k_values:
    start = time.time()
    results = index.get_nns_by_vector(query, 10, search_k=search_k)
    elapsed = time.time() - start
    
    sk_display = search_k if search_k > 0 else f"默认({n_trees * 10})"
    print(f"search_k={sk_display:>8}  耗时: {elapsed*1000:.3f}ms  结果: {results[:3]}...")
```

---

## 5. 辅助 API

### 5.1 get_item_vector(i)

获取指定索引的向量。

```python
vector = index.get_item_vector(i)
```

```python
index.add_item(0, [1.0, 2.0, 3.0])
index.build(10)

vec = index.get_item_vector(0)
print(vec)  # [1.0, 2.0, 3.0]
```

### 5.2 get_n_items()

获取索引中的向量数量。

```python
count = index.get_n_items()
```

### 5.3 get_n_trees()

获取索引中的树的数量。

```python
n_trees = index.get_n_trees()
```

### 5.4 get_distance(i, j)

获取两个向量之间的距离。

```python
distance = index.get_distance(i, j)
```

```python
index.add_item(0, [1, 0, 0])
index.add_item(1, [0, 1, 0])
index.build(10)

dist = index.get_distance(0, 1)
print(f"向量 0 和 1 的距离: {dist}")  # 取决于使用的 metric
```

### 5.5 set_seed(seed)

设置随机种子，使结果可重现。

```python
index.set_seed(42)
```

::: warning 注意
必须在 `build()` 之前调用 `set_seed()`。
:::

---

## 6. API 速查表

| 类别 | 方法 | 说明 |
|------|------|------|
| **构造** | `AnnoyIndex(f, metric)` | 创建索引，f=维度 |
| **添加** | `add_item(i, v)` | 添加向量 |
| **构建** | `build(n_trees, n_jobs)` | 构建索引 |
| **解构** | `unbuild()` | 解除构建状态 |
| **保存** | `save(fn, prefault)` | 保存到文件 |
| **加载** | `load(fn, prefault)` | 从文件加载 |
| **卸载** | `unload()` | 释放内存 |
| **磁盘构建** | `on_disk_build(fn)` | 直接在磁盘上构建 |
| **搜索** | `get_nns_by_vector(v, n, ...)` | 根据向量搜索 |
| **搜索** | `get_nns_by_item(i, n, ...)` | 根据索引搜索 |
| **获取向量** | `get_item_vector(i)` | 获取指定向量 |
| **向量数量** | `get_n_items()` | 获取向量总数 |
| **树数量** | `get_n_trees()` | 获取树的数量 |
| **计算距离** | `get_distance(i, j)` | 计算两向量距离 |
| **设置种子** | `set_seed(seed)` | 设置随机种子 |

