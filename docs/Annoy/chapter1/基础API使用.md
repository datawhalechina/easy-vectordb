# 基础API使用：掌握Annoy核心功能

## AnnoyIndex类详解

AnnoyIndex是Annoy库的核心类，提供了构建和查询向量索引的所有功能。理解这个类的设计和使用方法是掌握Annoy的关键。

### 类初始化

```python
from annoy import AnnoyIndex

# 基本初始化
index = AnnoyIndex(f, metric)
```

**参数说明**：
- `f`：向量维度，必须是正整数，一旦设定不可更改
- `metric`：距离度量方式，支持以下选项：
  - `'angular'`：角度距离（余弦相似度），适合文本和推荐场景
  - `'euclidean'`：欧氏距离，适合图像和几何数据
  - `'manhattan'`：曼哈顿距离，适合稀疏数据
  - `'hamming'`：汉明距离，适合二进制数据
  - `'dot'`：点积距离，适合特定的机器学习场景

**初始化示例**：
```python
# 创建128维向量的角度距离索引
text_index = AnnoyIndex(128, 'angular')

# 创建512维向量的欧氏距离索引
image_index = AnnoyIndex(512, 'euclidean')

# 创建64位二进制向量的汉明距离索引
binary_index = AnnoyIndex(64, 'hamming')
```

## 核心API方法详解

### 添加向量数据

**add_item()方法**：
```python
index.add_item(i, vector)
```

这是构建索引的基础方法，用于向索引中添加向量数据。

**参数详解**：
- `i`：向量的唯一标识符，必须是非负整数
- `vector`：向量数据，可以是list、numpy数组或其他可迭代对象

**使用示例**：
```python
import numpy as np
from annoy import AnnoyIndex

# 创建索引
f = 10  # 向量维度
index = AnnoyIndex(f, 'angular')

# 方法1：使用Python列表
vector1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
index.add_item(0, vector1)

# 方法2：使用NumPy数组
vector2 = np.random.normal(size=f)
index.add_item(1, vector2)

# 方法3：批量添加
for i in range(1000):
    vector = np.random.normal(size=f)
    index.add_item(i, vector)
```

**注意事项**：
- 向量ID必须从0开始连续编号
- 向量维度必须与初始化时指定的维度一致
- 添加完所有向量后必须调用build()方法构建索引

### 构建索引

**build()方法**：
```python
index.build(n_trees, n_jobs=-1)
```

构建索引是Annoy使用流程中的关键步骤，将添加的向量数据组织成可搜索的树结构。

**参数详解**：
- `n_trees`：构建的树的数量，影响查询精度和构建时间
- `n_jobs`：并行构建的线程数，-1表示使用所有可用CPU核心

**参数选择指南**：
```python
# 根据数据规模选择树的数量
data_size = 100000
if data_size < 10000:
    n_trees = 10
elif data_size < 100000:
    n_trees = 20
else:
    n_trees = 50

# 构建索引
index.build(n_trees)
```

**构建过程监控**：
```python
import time

print("Building index...")
start_time = time.time()
index.build(20)
build_time = time.time() - start_time
print(f"Index built in {build_time:.2f} seconds")
```

### 查询相似向量

Annoy提供了多种查询方法来满足不同的使用需求。

**get_nns_by_item()方法**：
```python
neighbors = index.get_nns_by_item(item_id, n, search_k=-1, include_distances=False)
```

根据已存在的向量ID查找最近邻。

**参数说明**：
- `item_id`：查询向量的ID
- `n`：返回的最近邻数量
- `search_k`：搜索的候选数量，-1表示使用默认值
- `include_distances`：是否返回距离信息

**使用示例**：
```python
# 基本查询
neighbors = index.get_nns_by_item(0, 10)
print(f"Item 0's 10 nearest neighbors: {neighbors}")

# 包含距离信息的查询
neighbors, distances = index.get_nns_by_item(0, 10, include_distances=True)
print(f"Neighbors: {neighbors}")
print(f"Distances: {distances}")
```

**get_nns_by_vector()方法**：
```python
neighbors = index.get_nns_by_vector(vector, n, search_k=-1, include_distances=False)
```

根据给定的向量查找最近邻，适用于查询不在索引中的新向量。

**使用示例**：
```python
# 查询新向量
query_vector = np.random.normal(size=f)
neighbors = index.get_nns_by_vector(query_vector, 5)
print(f"Query vector's 5 nearest neighbors: {neighbors}")

# 实时查询示例
def real_time_search(query_vector, top_k=10):
    """实时搜索函数"""
    start_time = time.time()
    neighbors, distances = index.get_nns_by_vector(
        query_vector, top_k, include_distances=True
    )
    query_time = time.time() - start_time
    
    return {
        'neighbors': neighbors,
        'distances': distances,
        'query_time': query_time
    }
```

### 获取向量数据

**get_item_vector()方法**：
```python
vector = index.get_item_vector(item_id)
```

根据ID获取存储在索引中的向量数据。

**使用示例**：
```python
# 获取特定向量
original_vector = index.get_item_vector(0)
print(f"Vector 0: {original_vector}")

# 验证向量完整性
def verify_vectors(index, sample_ids):
    """验证向量数据完整性"""
    for item_id in sample_ids:
        try:
            vector = index.get_item_vector(item_id)
            print(f"✓ Vector {item_id}: {len(vector)} dimensions")
        except Exception as e:
            print(f"✗ Vector {item_id}: Error - {e}")
```

### 距离计算

**get_distance()方法**：
```python
distance = index.get_distance(i, j)
```

计算索引中两个向量之间的距离。

**使用示例**：
```python
# 计算两个向量间的距离
dist = index.get_distance(0, 1)
print(f"Distance between item 0 and 1: {dist}")

# 构建距离矩阵
def build_distance_matrix(index, item_ids):
    """构建距离矩阵"""
    n = len(item_ids)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = index.get_distance(item_ids[i], item_ids[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    
    return distance_matrix
```

## 索引持久化操作

### 保存索引

**save()方法**：
```python
index.save(filename)
```

将构建好的索引保存到磁盘文件。

**使用示例**：
```python
# 保存索引
index_file = "my_index.ann"
index.save(index_file)
print(f"Index saved to {index_file}")

# 检查文件大小
import os
file_size = os.path.getsize(index_file)
print(f"Index file size: {file_size / (1024*1024):.2f} MB")
```

### 加载索引

**load()方法**：
```python
index.load(filename, prefault=False)
```

从磁盘文件加载已保存的索引。

**参数说明**：
- `filename`：索引文件路径
- `prefault`：是否预加载所有页面到内存

**使用示例**：
```python
# 加载索引
new_index = AnnoyIndex(f, 'angular')
new_index.load(index_file)

# 验证加载成功
print(f"Loaded index with {new_index.get_n_items()} items")

# 预加载优化
large_index = AnnoyIndex(f, 'angular')
large_index.load(index_file, prefault=True)  # 适合频繁查询的场景
```

## 索引信息查询

### 获取索引统计信息

```python
# 获取向量数量
n_items = index.get_n_items()
print(f"Number of items: {n_items}")

# 获取向量维度
n_dims = index.f
print(f"Vector dimension: {n_dims}")

# 获取树的数量（仅在构建后可用）
n_trees = index.get_n_trees()
print(f"Number of trees: {n_trees}")
```

### 索引状态检查

```python
def check_index_status(index):
    """检查索引状态"""
    try:
        n_items = index.get_n_items()
        if n_items == 0:
            return "Empty index"
        
        # 尝试查询以检查是否已构建
        try:
            index.get_nns_by_item(0, 1)
            return f"Ready - {n_items} items indexed"
        except:
            return f"Not built - {n_items} items added but not indexed"
    except:
        return "Invalid index"

print(f"Index status: {check_index_status(index)}")
```

## 完整使用示例

以下是一个完整的Annoy使用示例，展示了从数据准备到查询的完整流程：

```python
#!/usr/bin/env python3
"""Annoy基础API完整使用示例"""

import numpy as np
import time
from annoy import AnnoyIndex

def create_sample_data(n_items=10000, n_dims=128):
    """创建示例数据"""
    print(f"Creating {n_items} random {n_dims}-dimensional vectors...")
    
    # 生成随机向量数据
    vectors = []
    for i in range(n_items):
        # 生成正态分布的随机向量
        vector = np.random.normal(0, 1, n_dims)
        # 归一化向量（对angular距离很重要）
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    
    return vectors

def build_index(vectors, metric='angular', n_trees=10):
    """构建Annoy索引"""
    n_dims = len(vectors[0])
    print(f"Building index with {len(vectors)} vectors, {n_dims} dimensions...")
    
    # 创建索引
    index = AnnoyIndex(n_dims, metric)
    
    # 添加向量
    for i, vector in enumerate(vectors):
        index.add_item(i, vector)
    
    # 构建索引
    start_time = time.time()
    index.build(n_trees)
    build_time = time.time() - start_time
    
    print(f"Index built in {build_time:.2f} seconds with {n_trees} trees")
    return index

def query_examples(index, vectors):
    """查询示例"""
    print("\nRunning query examples...")
    
    # 示例1：根据ID查询
    item_id = 0
    neighbors = index.get_nns_by_item(item_id, 5, include_distances=True)
    print(f"Top 5 neighbors of item {item_id}:")
    for i, (neighbor, distance) in enumerate(zip(neighbors[0], neighbors[1])):
        print(f"  {i+1}. Item {neighbor} (distance: {distance:.4f})")
    
    # 示例2：根据向量查询
    query_vector = vectors[100]  # 使用已知向量作为查询
    neighbors = index.get_nns_by_vector(query_vector, 3, include_distances=True)
    print(f"\nTop 3 neighbors of query vector:")
    for i, (neighbor, distance) in enumerate(zip(neighbors[0], neighbors[1])):
        print(f"  {i+1}. Item {neighbor} (distance: {distance:.4f})")
    
    # 示例3：性能测试
    print(f"\nPerformance test:")
    query_times = []
    for _ in range(100):
        start = time.time()
        index.get_nns_by_item(np.random.randint(0, index.get_n_items()), 10)
        query_times.append(time.time() - start)
    
    avg_time = np.mean(query_times)
    print(f"Average query time: {avg_time*1000:.2f}ms")
    print(f"Queries per second: {1/avg_time:.0f}")

def main():
    """主函数"""
    # 参数设置
    n_items = 10000
    n_dims = 128
    n_trees = 20
    
    # 创建数据
    vectors = create_sample_data(n_items, n_dims)
    
    # 构建索引
    index = build_index(vectors, 'angular', n_trees)
    
    # 保存索引
    index_file = "example_index.ann"
    index.save(index_file)
    print(f"Index saved to {index_file}")
    
    # 加载索引（演示）
    new_index = AnnoyIndex(n_dims, 'angular')
    new_index.load(index_file)
    print(f"Index loaded successfully")
    
    # 查询示例
    query_examples(new_index, vectors)
    
    # 索引信息
    print(f"\nIndex information:")
    print(f"  Items: {new_index.get_n_items()}")
    print(f"  Dimensions: {n_dims}")
    print(f"  Trees: {n_trees}")
    print(f"  Metric: angular")

if __name__ == "__main__":
    main()
```

这个完整示例展示了Annoy的核心API使用方法，包括数据准备、索引构建、持久化和查询等所有关键步骤。通过运行这个示例，您可以快速了解Annoy的基本使用流程和性能特征。

在下一章节中，我们将深入学习索引管理的高级技巧，包括索引优化、版本控制和生产环境的最佳实践。
