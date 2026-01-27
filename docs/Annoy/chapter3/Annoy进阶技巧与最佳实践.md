# Annoy进阶技巧与最佳实践

本章介绍 Annoy 的高级用法和生产环境中的最佳实践。

## 1. 内存映射的威力

内存映射（Memory Mapping, mmap）是 Annoy 最重要的特性之一，也是它区别于其他 ANN 库的核心优势。

### 1.1 什么是内存映射？

内存映射是一种将文件内容直接映射到进程虚拟地址空间的技术：

- **传统方式**：将整个文件读入内存 → 占用大量 RAM
- **内存映射**：文件保留在磁盘上，按需加载页面 → 节省内存

```
传统加载：
[磁盘文件] --全部复制--> [进程内存]

内存映射：
[磁盘文件] <--按需映射--> [进程虚拟地址空间]
                              ↓
                         操作系统自动管理页面换入换出
```

### 1.2 Annoy 的内存映射模式

默认情况下，`load()` 使用内存映射：

```python
from annoy import AnnoyIndex

# 使用内存映射加载（默认）
index = AnnoyIndex(128, 'angular')
index.load('my_index.ann')  # prefault=False 是默认值

# 完全加载到内存
index.load('my_index.ann', prefault=True)
```

### 1.3 多进程共享索引

这是 Annoy 最强大的功能：**多个进程可以共享同一份索引文件，而不需要复制内存**。

**示例：多进程搜索服务**

```python
# multiprocess_demo.py
from annoy import AnnoyIndex
import multiprocessing as mp
import numpy as np
import time
import os

def worker(worker_id, index_path, dim, queries):
    """工作进程：加载索引并执行搜索"""
    # 每个进程独立加载索引（但共享同一份物理内存）
    index = AnnoyIndex(dim, 'angular')
    index.load(index_path)
    
    results = []
    for q in queries:
        neighbors = index.get_nns_by_vector(q, 5)
        results.append(neighbors)
    
    print(f"Worker {worker_id} (PID: {os.getpid()}) 完成 {len(queries)} 次查询")
    return results

def main():
    # ============ 第一步：创建索引 ============
    dim = 128
    n_items = 100000
    index_path = 'shared_index.ann'
    
    print("创建索引...")
    index = AnnoyIndex(dim, 'angular')
    np.random.seed(42)
    
    for i in range(n_items):
        index.add_item(i, np.random.randn(dim))
    
    index.build(10)
    index.save(index_path)
    print(f"索引已保存: {index_path}")
    
    # ============ 第二步：多进程查询 ============
    n_workers = 4
    queries_per_worker = 100
    
    # 生成查询向量
    all_queries = [np.random.randn(dim).tolist() for _ in range(n_workers * queries_per_worker)]
    
    # 分配查询到各个 worker
    query_chunks = [all_queries[i::n_workers] for i in range(n_workers)]
    
    print(f"\n启动 {n_workers} 个工作进程...")
    start_time = time.time()
    
    # 使用进程池
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(
            worker,
            [(i, index_path, dim, query_chunks[i]) for i in range(n_workers)]
        )
    
    elapsed = time.time() - start_time
    total_queries = n_workers * queries_per_worker
    print(f"\n总计 {total_queries} 次查询，耗时 {elapsed:.2f} 秒")
    print(f"平均 QPS: {total_queries / elapsed:.2f}")

if __name__ == '__main__':
    main()
```

运行此脚本后，你会看到多个进程并行处理查询，而索引文件只需要一份物理内存。

### 1.4 内存占用对比

```python
# memory_comparison.py
from annoy import AnnoyIndex
import numpy as np
import os

def get_file_size(path):
    """获取文件大小（MB）"""
    return os.path.getsize(path) / (1024 * 1024)

# 创建索引
dim = 128
n_items = 100000

index = AnnoyIndex(dim, 'angular')
np.random.seed(42)

for i in range(n_items):
    index.add_item(i, np.random.randn(dim))

index.build(10)
index.save('test_index.ann')

file_size = get_file_size('test_index.ann')
print(f"索引文件大小: {file_size:.2f} MB")
print(f"向量数量: {n_items}")
print(f"向量维度: {dim}")
print(f"每个向量的存储开销: {file_size * 1024 / n_items:.2f} KB")
```

::: tip 内存共享的关键点
1. 所有进程必须使用 `prefault=False`（默认值）
2. 索引文件必须是只读的
3. 操作系统会自动管理页面缓存
:::

---

## 2. Annoy 的局限与应对策略

### 2.1 不支持增量更新

**问题**：Annoy 不支持在已构建的索引上添加新向量。

**解决方案 1：定期重建**

```python
import schedule
import time
from annoy import AnnoyIndex

def rebuild_index():
    """定期重建索引"""
    # 从数据库加载所有向量
    vectors = load_all_vectors_from_db()
    
    # 创建新索引
    new_index = AnnoyIndex(128, 'angular')
    for i, vec in enumerate(vectors):
        new_index.add_item(i, vec)
    new_index.build(10)
    
    # 保存为临时文件
    new_index.save('index_new.ann')
    
    # 原子替换（避免服务中断）
    import os
    os.rename('index_new.ann', 'index.ann')
    
    print(f"索引重建完成，共 {len(vectors)} 个向量")

# 每小时重建一次
schedule.every(1).hours.do(rebuild_index)
```

**解决方案 2：分片策略**

```python
class ShardedAnnoyIndex:
    """分片索引：新数据写入新分片"""
    
    def __init__(self, dim, metric='angular', shard_size=100000):
        self.dim = dim
        self.metric = metric
        self.shard_size = shard_size
        self.shards = []  # [(index, id_offset), ...]
        self.current_shard = None
        self.current_count = 0
        self.total_items = 0
        
    def add_item(self, vector):
        """添加向量，自动管理分片"""
        if self.current_shard is None or self.current_count >= self.shard_size:
            # 构建当前分片并创建新分片
            if self.current_shard is not None:
                self.current_shard.build(10)
                self.shards.append((self.current_shard, self.total_items - self.current_count))
            
            self.current_shard = AnnoyIndex(self.dim, self.metric)
            self.current_count = 0
        
        self.current_shard.add_item(self.current_count, vector)
        self.current_count += 1
        self.total_items += 1
        
        return self.total_items - 1  # 返回全局 ID
    
    def search(self, query, n=10):
        """在所有分片中搜索"""
        all_results = []
        
        # 搜索已构建的分片
        for shard, offset in self.shards:
            results = shard.get_nns_by_vector(query, n, include_distances=True)
            for idx, dist in zip(results[0], results[1]):
                all_results.append((idx + offset, dist))
        
        # 搜索当前分片（如果已有数据）
        if self.current_count > 0:
            self.current_shard.build(10)
            results = self.current_shard.get_nns_by_vector(query, n, include_distances=True)
            offset = self.total_items - self.current_count
            for idx, dist in zip(results[0], results[1]):
                all_results.append((idx + offset, dist))
            self.current_shard.unbuild()
        
        # 合并结果并排序
        all_results.sort(key=lambda x: x[1])
        return all_results[:n]
```

### 2.2 不支持删除

**问题**：Annoy 不支持删除向量。

**解决方案：标记删除 + 定期压缩**

```python
class AnnoyWithDeletion:
    """支持逻辑删除的 Annoy 封装"""
    
    def __init__(self, dim, metric='angular'):
        self.index = AnnoyIndex(dim, metric)
        self.dim = dim
        self.metric = metric
        self.deleted_ids = set()
        self.is_built = False
        
    def add_item(self, i, vector):
        self.index.add_item(i, vector)
        
    def build(self, n_trees):
        self.index.build(n_trees)
        self.is_built = True
        
    def delete(self, i):
        """标记为删除（不是真正删除）"""
        self.deleted_ids.add(i)
        
    def search(self, query, n=10):
        """搜索时过滤已删除的向量"""
        # 多请求一些结果，以弥补被过滤的
        search_n = n + len(self.deleted_ids)
        results = self.index.get_nns_by_vector(query, search_n, include_distances=True)
        
        # 过滤已删除的
        filtered = [(idx, dist) for idx, dist in zip(results[0], results[1]) 
                    if idx not in self.deleted_ids]
        
        return filtered[:n]
    
    def compact(self):
        """压缩：重建索引，真正删除被标记的向量"""
        new_index = AnnoyIndex(self.dim, self.metric)
        id_mapping = {}  # old_id -> new_id
        new_id = 0
        
        for old_id in range(self.index.get_n_items()):
            if old_id not in self.deleted_ids:
                vec = self.index.get_item_vector(old_id)
                new_index.add_item(new_id, vec)
                id_mapping[old_id] = new_id
                new_id += 1
        
        new_index.build(self.index.get_n_trees())
        
        self.index = new_index
        self.deleted_ids.clear()
        
        return id_mapping
```

### 2.3 何时需要切换到其他方案

| 场景 | 推荐方案 |
|------|----------|
| 需要频繁增删改 | **Milvus** 或 **Elasticsearch** |
| 需要 GPU 加速 | **Faiss** |
| 数据量 > 1 亿 | **Milvus**（分布式） |
| 需要多种索引类型 | **Faiss** |
| 单机、只读、内存受限 | **Annoy**（本库） |

---

## 3. 性能优化技巧

### 3.1 n_trees 和 search_k 的调优经验

**n_trees 调优：**

```python
from annoy import AnnoyIndex
import numpy as np
import time

def benchmark_n_trees(dim=128, n_items=100000, n_queries=100):
    """测试不同 n_trees 的构建时间和搜索精度"""
    
    # 准备数据
    np.random.seed(42)
    vectors = np.random.randn(n_items, dim).astype('float32')
    queries = np.random.randn(n_queries, dim).astype('float32')
    
    # 计算真实最近邻（暴力搜索）
    from sklearn.metrics.pairwise import cosine_similarity
    true_neighbors = []
    for q in queries:
        sims = cosine_similarity([q], vectors)[0]
        true_neighbors.append(np.argsort(-sims)[:10].tolist())
    
    results = []
    
    for n_trees in [5, 10, 20, 50, 100]:
        # 构建索引
        index = AnnoyIndex(dim, 'angular')
        for i, vec in enumerate(vectors):
            index.add_item(i, vec)
        
        start = time.time()
        index.build(n_trees)
        build_time = time.time() - start
        
        # 计算召回率
        recall_sum = 0
        for i, q in enumerate(queries):
            pred = index.get_nns_by_vector(q, 10)
            recall = len(set(pred) & set(true_neighbors[i])) / 10
            recall_sum += recall
        
        avg_recall = recall_sum / n_queries
        
        results.append({
            'n_trees': n_trees,
            'build_time': build_time,
            'recall': avg_recall
        })
        
        print(f"n_trees={n_trees:3d}  构建时间: {build_time:.2f}s  召回率: {avg_recall:.4f}")
    
    return results

# 运行基准测试
# benchmark_n_trees()
```

**推荐值：**
- 小数据集（< 10 万）：`n_trees = 10-20`
- 中等数据集（10-100 万）：`n_trees = 20-50`
- 大数据集（> 100 万）：`n_trees = 50-100`

### 3.2 批量添加优化

虽然 Annoy 没有原生的批量添加 API，但可以通过减少 Python 循环开销来优化：

```python
from annoy import AnnoyIndex
import numpy as np

def batch_add_items(index, vectors, start_id=0):
    """批量添加向量"""
    for i, vec in enumerate(vectors):
        index.add_item(start_id + i, vec)
    return start_id + len(vectors)

# 使用示例
index = AnnoyIndex(128, 'angular')
vectors = np.random.randn(100000, 128)

# 分批添加
batch_size = 10000
next_id = 0
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    next_id = batch_add_items(index, batch, next_id)
    print(f"已添加 {next_id} 个向量")

index.build(10)
```

---

## 4. 与 Embedding 模型集成

### 4.1 与 sentence-transformers 集成

```python
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np

# ============ 第一步：加载模型 ============
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384  # all-MiniLM-L6-v2 的输出维度

# ============ 第二步：准备文档 ============
documents = [
    "Python 是一种流行的编程语言",
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络处理数据",
    "自然语言处理用于理解人类语言",
    "向量数据库用于存储和检索向量",
    "Annoy 是一个高效的近似最近邻搜索库",
    "Faiss 是 Facebook 开发的向量搜索库",
    "Milvus 是一个分布式向量数据库",
]

# ============ 第三步：生成向量并建立索引 ============
print("生成文档向量...")
embeddings = model.encode(documents)

index = AnnoyIndex(dim, 'angular')
for i, emb in enumerate(embeddings):
    index.add_item(i, emb)

index.build(10)
index.save('documents.ann')
print(f"索引已保存，共 {len(documents)} 个文档")

# ============ 第四步：语义搜索 ============
def semantic_search(query, top_k=3):
    """语义搜索"""
    query_embedding = model.encode([query])[0]
    indices, distances = index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    
    results = []
    for idx, dist in zip(indices, distances):
        results.append({
            'document': documents[idx],
            'distance': dist,
            'similarity': 1 - (dist ** 2 / 2)  # 转换为余弦相似度
        })
    return results

# 测试搜索
query = "如何使用向量搜索？"
print(f"\n查询: {query}")
print("-" * 50)

results = semantic_search(query)
for i, r in enumerate(results):
    print(f"{i+1}. [{r['similarity']:.4f}] {r['document']}")
```

### 4.2 与 OpenAI Embedding 集成

```python
from openai import OpenAI
from annoy import AnnoyIndex
import json

# ============ 初始化 ============
client = OpenAI()  # 需要设置 OPENAI_API_KEY 环境变量
dim = 1536  # text-embedding-3-small 的维度

def get_embedding(text, model="text-embedding-3-small"):
    """获取 OpenAI Embedding"""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# ============ 构建索引 ============
documents = [
    "向量数据库是存储和检索向量的数据库系统",
    "Annoy 适合单机只读场景的向量检索",
    "语义搜索可以理解查询的含义而不仅仅是关键词匹配",
]

print("获取文档向量...")
index = AnnoyIndex(dim, 'angular')
id_to_doc = {}

for i, doc in enumerate(documents):
    emb = get_embedding(doc)
    index.add_item(i, emb)
    id_to_doc[i] = doc
    print(f"  已处理: {doc[:30]}...")

index.build(10)
index.save('openai_index.ann')

# 保存文档映射
with open('id_to_doc.json', 'w', encoding='utf-8') as f:
    json.dump(id_to_doc, f, ensure_ascii=False)

print("索引构建完成！")

# ============ 搜索 ============
def search(query, top_k=3):
    query_emb = get_embedding(query)
    indices, distances = index.get_nns_by_vector(query_emb, top_k, include_distances=True)
    
    return [(id_to_doc[i], d) for i, d in zip(indices, distances)]

# 测试
query = "什么是向量搜索？"
results = search(query)
print(f"\n查询: {query}")
for doc, dist in results:
    print(f"  [{dist:.4f}] {doc}")
```

---

## 5. 生产环境部署建议

### 5.1 FastAPI 封装 REST API

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from annoy import AnnoyIndex
import numpy as np
from typing import List
import os

app = FastAPI(title="Annoy Vector Search API")

# 全局索引（在启动时加载）
index = None
dim = 128

class SearchRequest(BaseModel):
    vector: List[float]
    top_k: int = 10

class SearchResponse(BaseModel):
    indices: List[int]
    distances: List[float]

@app.on_event("startup")
async def load_index():
    """启动时加载索引"""
    global index
    index_path = os.getenv("INDEX_PATH", "index.ann")
    
    if not os.path.exists(index_path):
        raise RuntimeError(f"索引文件不存在: {index_path}")
    
    index = AnnoyIndex(dim, 'angular')
    index.load(index_path)  # 使用内存映射
    print(f"索引加载成功: {index.get_n_items()} 个向量")

@app.get("/health")
async def health_check():
    """健康检查"""
    if index is None:
        raise HTTPException(status_code=503, detail="索引未加载")
    return {"status": "healthy", "n_items": index.get_n_items()}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """向量搜索"""
    if index is None:
        raise HTTPException(status_code=503, detail="索引未加载")
    
    if len(request.vector) != dim:
        raise HTTPException(
            status_code=400, 
            detail=f"向量维度错误: 期望 {dim}, 实际 {len(request.vector)}"
        )
    
    indices, distances = index.get_nns_by_vector(
        request.vector, 
        request.top_k, 
        include_distances=True
    )
    
    return SearchResponse(indices=indices, distances=distances)

# 启动命令：uvicorn api_server:app --workers 4
```

### 5.2 多进程部署架构

```
                    ┌─────────────────┐
                    │   Nginx/LB      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │ Worker 1│        │ Worker 2│        │ Worker 3│
    │ (uvicorn)│       │ (uvicorn)│       │ (uvicorn)│
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │   index.ann     │ ← 共享文件（mmap）
                    │   (磁盘文件)     │
                    └─────────────────┘
```

**部署命令：**

```bash
# 使用 uvicorn 多 worker 模式
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 5.3 索引更新策略（蓝绿部署）

```python
# index_updater.py
import os
import shutil
import time
from annoy import AnnoyIndex

class BlueGreenIndexUpdater:
    """蓝绿部署索引更新器"""
    
    def __init__(self, dim, metric='angular', base_path='./indexes'):
        self.dim = dim
        self.metric = metric
        self.base_path = base_path
        self.blue_path = os.path.join(base_path, 'blue.ann')
        self.green_path = os.path.join(base_path, 'green.ann')
        self.active_path = os.path.join(base_path, 'active.ann')
        
        os.makedirs(base_path, exist_ok=True)
        
    def build_new_index(self, vectors, n_trees=10):
        """构建新索引到非活跃路径"""
        # 确定当前活跃的是哪个
        if os.path.exists(self.active_path):
            current = os.path.realpath(self.active_path)
            if 'blue' in current:
                new_path = self.green_path
            else:
                new_path = self.blue_path
        else:
            new_path = self.blue_path
        
        # 构建新索引
        index = AnnoyIndex(self.dim, self.metric)
        for i, vec in enumerate(vectors):
            index.add_item(i, vec)
        index.build(n_trees)
        index.save(new_path)
        
        return new_path
    
    def switch_active(self, new_path):
        """切换活跃索引（原子操作）"""
        # 创建临时符号链接
        temp_link = self.active_path + '.tmp'
        
        # 创建新的符号链接
        if os.path.exists(temp_link):
            os.remove(temp_link)
        os.symlink(os.path.basename(new_path), temp_link)
        
        # 原子替换
        os.rename(temp_link, self.active_path)
        
        print(f"已切换活跃索引到: {new_path}")
    
    def update(self, vectors, n_trees=10):
        """完整的更新流程"""
        print("开始构建新索引...")
        new_path = self.build_new_index(vectors, n_trees)
        
        print("切换活跃索引...")
        self.switch_active(new_path)
        
        print("索引更新完成！")
```

---

## 6. 总结

::: tip Annoy 最佳实践清单
- 使用默认的 `prefault=False` 加载索引（启用内存映射）
- 多进程服务共享同一索引文件
- 根据数据量选择合适的 `n_trees`（10-100）
- 使用蓝绿部署实现无缝索引更新
- 对于需要删除的场景，使用标记删除 + 定期压缩
- 大数据集使用 `on_disk_build()` 直接在磁盘上构建
:::

### 何时使用 Annoy

| 适合场景 | 不适合场景 |
|---------|----------|
| 单机部署 | 分布式需求 |
| 只读或低频更新 | 频繁增删改 |
| 内存受限环境 | 需要 GPU 加速 |
| 多进程共享 | 需要复杂索引类型 |
| 百万级数据 | 十亿级数据 |

::: info 学习收获
完成 Annoy 教程后，你已经掌握了：

1. Annoy 的安装和快速上手
2. 完整的 API 使用方法
3. 内存映射和多进程共享
4. 生产环境部署的最佳实践
:::

::: tip 推荐阅读
- 在实战项目中应用 Annoy 构建推荐系统
- 对比学习 [Faiss 教程](/Faiss/chapter1/FAISS入门与环境搭建) 和 [Milvus 教程](/Milvus/chapter1/Milvus向量数据库入门)
- 回顾 [base/chapter5/Annoy算法](/base/chapter5/Annoy算法) 深入理解算法原理
:::
