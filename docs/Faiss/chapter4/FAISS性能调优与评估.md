# Chapter 4：FAISS 性能调优与评估

## 1 检索性能评估指标：量化你的检索系统

### 1.1 为什么需要性能评估？

FAISS的检索性能是“准确率”与“效率”的平衡艺术——没有评估的调优就是盲目试错。

通过量化指标，我们能明确：

①当前系统的短板（是召回率不足还是延迟过高？）；

②调优策略的实际效果；

③是否满足业务场景需求（如推荐系统需Recall@10≥95%，实时检索需延迟≤50ms）。

### 1.2 四大核心评估指标详解

| 指标名称                       | 核心定义                                                     | 业务意义                                                     |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Recall@k（召回率）**         | 在 Top-k 检索结果中，被正确找回的真实最近邻占所有真实最近邻的比例（真实最近邻由精确搜索获得） | 衡量近似检索的准确性，反映“漏检”程度，是 ANN 最核心的效果指标 |
| **QPS（Queries Per Second）**  | 系统在单位时间内可处理的查询请求数量                         | 衡量系统吞吐能力，决定是否能支撑高并发在线服务               |
| **延迟（Latency）**            | 单次查询从请求到返回结果的耗时，通常统计 p50 / p95 / p99     | 衡量系统响应速度，直接影响实时体验（如搜索、推荐、对话系统） |
| **索引构建时间（Build Time）** | 从原始向量数据构建完整索引结构所需的时间                     | 衡量索引初始化与更新成本，影响离线构建效率与在线更新可行性   |

> ⚠在 ANN Top-k 评测中，由于每个查询的真实相关结果数量固定为 k，Precision@k 与 Recall@k 在数值上等价，因此通常仅报告 Recall@k。

### 1.3 评估数据集准备：SIFT1M

SIFT1M是向量检索领域的“标准测试集”，包含100万条128维SIFT特征向量（数据库向量）和1万条查询向量，且提供查询向量对应的真实Top-100结果，是评估FAISS性能的理想数据。

**数据集获取与预处理代码**：

数据集下载地址:https://huggingface.co/datasets/fzliu/sift1m/tree/main

数据预处理代码

```python
import numpy as np
# 1. 加载 SIFT1M 数据库向量（仅需 xb）
def load_sift1m(path):
    # 加载文件（fvecs/ivecs均为二进制格式，按对应 dtype 读取）
    xb = np.fromfile(f"{path}/sift_base.fvecs", dtype=np.float32)
    xq = np.fromfile(f"{path}/sift_query.fvecs", dtype=np.float32)
    gt = np.fromfile(f"{path}/sift_groundtruth.ivecs", dtype=np.int32)
    print(f"向量维度d: {d}")
    print("原始数据形状:", xb.shape, xq.shape, gt.shape)
    
    # ---------------------- 解析 fvecs 格式（数据库/查询向量）----------------------
    # 验证数据长度是否符合格式（总长度必须是 (d+1) 的整数倍）
    assert xb.size % (d + 1) == 0, f"数据库文件损坏：总长度 {xb.size} 不是 (d+1)={d+1} 的整数倍"
    assert xq.size % (d + 1) == 0, f"查询文件损坏：总长度 {xq.size} 不是 (d+1)={d+1} 的整数倍"
    
    # 重构为 [向量数, d+1]，再去掉第一列（维度标识），得到实际向量
    xb = xb.reshape(-1, d + 1)[:, 1:]  # 数据库向量：(1000000, 128)
    xq = xq.reshape(-1, d + 1)[:, 1:]  # 查询向量：(10000, 128)
    
    # ---------------------- 解析 ivecs 格式（真实近邻）----------------------
    # ivecs 格式：每个近邻组 = [4字节近邻数k] + [k个4字节int32近邻ID]
    # 1. 获取每个查询的近邻数k（SIFT1M的groundtruth通常是每个查询100个近邻）
    k = int(gt[0])
    print(f"每个查询的近邻数k: {k}")
    
    # 验证数据长度是否符合格式
    assert gt.size % (k + 1) == 0, f"近邻文件损坏：总长度 {gt.size} 不是 (k+1)={k+1} 的整数倍"
    
    # 重构为 [查询数, k+1]，去掉第一列（近邻数标识），得到实际近邻ID
    gt = gt.reshape(-1, k + 1)[:, 1:]  # 真实近邻：(10000, 100)
    
    print("解析后数据形状:", xb.shape, xq.shape, gt.shape)
    return xb, xq, gt

# 测试加载（注意路径使用 raw 字符串或双反斜杠）
d = 128
xb, xq, gt = load_sift1m(r"C:\Users\xiong\Desktop\easy-vecdb\easy-vecdb\data\sift")
print(f"数据库向量：{xb.shape}，查询向量：{xq.shape}，真实结果：{gt.shape}")
```

运行结果

```
向量维度d: 128
原始数据形状: (129000000,) (1290000,) (1010000,)
每个查询的近邻数k: 100
解析后数据形状: (1000000, 128) (10000, 128) (10000, 100)
数据库向量：(1000000, 128)，查询向量：(10000, 128)，真实结果：(10000, 100)
```

### 1.4 指标计算代码实战

以“IVF100,Flat”索引为基准，计算五大核心指标。

```python
import time
from sklearn.metrics import precision_score

def get_gt_by_flat(xb, xq, k=10):
    d = xb.shape[1]
    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(xb)
    _, I_gt = index_flat.search(xq, k)
    return I_gt

def recall_at_k(pred, gt, k):
    hit = 0
    for p, g in zip(pred, gt):
        hit += len(set(p) & set(g))
    return hit / (len(pred) * k)

def benchmark(index, xq, k, n_round=50):
    nq = len(xq)
    total_time = 0.0

    # warm-up
    index.search(xq[:10], k)

    for _ in range(n_round):
        start = time.time()
        index.search(xq, k)
        total_time += time.time() - start

    latency_ms = (total_time / n_round) / nq * 1000
    qps = (n_round * nq) / total_time
    return latency_ms, qps

# 加载数据（请替换为你的SIFT1M数据集路径）
sift1m_path = r"C:\Users\xiong\Desktop\easy-vecdb\easy-vecdb\data\sift"
xb, xq, _ = load_sift1m(sift1m_path)
k=10
d=128

print("---使用暴力搜索生成 Ground Truth---")
I_gt = get_gt_by_flat(xb, xq,k)

print("---训练 IVF100---")
nlist = 100
nprobe = 5

quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

print("开始构建 IVF100 索引 ...")
build_start = time.time()
index_ivf.train(xb)
index_ivf.add(xb)
build_time = time.time() - build_start

index_ivf.nprobe = nprobe

print("---IVF 检索---")
_, I_ivf = index_ivf.search(xq, k)


recall = recall_at_k(I_ivf, I_gt, k)

latency, qps = benchmark(index_ivf, xq, k)

print("\n===== IVF100 四大核心指标 =====")
print(f"Recall@{k}:        {recall:.4f}")
print(f"Latency:          {latency:.3f} ms / query")
print(f"QPS:              {qps:.0f}")
print(f"Build Time:       {build_time:.2f} s")
```

运行结果

```
Recall@10:        0.9361
Latency:          0.351 ms / query
QPS:              2848
Build Time:       0.51 s
```

### 1.5 准确率-效率的核心权衡逻辑

FAISS的所有调优本质都是“准确率损失”与“效率提升”的权衡，核心规律如下：

- 提升准确率的代价：增加计算/存储开销（如增大nprobe、M，使用Flat索引）
- 提升效率的代价：降低检索准确率（如减小nprobe、使用PQ量化、降低ef值）
- 权衡原则：以“业务指标底线”为基准——如推荐系统需Recall@10≥95%，则在满足该条件的前提下，最小化延迟、最大化QPS。

## 2 核心参数调优：从“盲目试错”到“科学调参”

### 2.1 Faiss核心调参维度解析

不同索引的核心参数不同，但调参逻辑一致——围绕“聚类粒度”“量化精度”“搜索深度”三个维度，以下是高频索引的关键参数：

| 索引类型                 | 核心参数             | 参数作用                                   | 调优趋势（准确率↑/效率↑）                           |
| :----------------------- | :------------------- | :----------------------------------------- | :-------------------------------------------------- |
| IVF系列（如IVF+Flat/PQ） | nlist（聚类数）      | 控制数据库向量的分区粒度                   | nlist↑→聚类更细→准确率↑、效率↓（需配合nprobe调整）  |
| IVF系列                  | nprobe（查询聚类数） | 控制查询时的搜索范围                       | nprobe↑→搜索范围更广→准确率↑、效率↓（核心调优参数） |
| PQ系列（如IVF+PQ）       | M（子向量数）        | 将向量拆分为M个子向量分别量化              | M↑→量化精度更高→准确率↑、内存占用↑                  |
| HNSW系列                 | ef/efConstruction    | ef：查询深度；efConstruction：构建图的深度 | ef↑→搜索更充分→准确率↑、效率↓                       |

### 2.2 Faiss参数优化方法-HNSW 为例

**Auto-Tune（自动参数调优）** 是 Faiss 提供的工具链，旨在自动找到满足**检索精度（Recall）** 约束的最优参数配置，平衡精度与检索速度（QPS/Latency）。

**1. 调优目标**

在给定**Recall 阈值**（如 Recall@10 ≥ 0.95）的前提下，最小化检索延迟（或最大化 QPS），同时兼顾内存占用。

**2. 关键参数分类**

Faiss 的索引参数分为两类，调优策略不同：

| 类型         | 说明                                          | 示例                                | 调优成本 |
| ------------ | --------------------------------------------- | ----------------------------------- | -------- |
| **静态参数** | 索引创建时确定，需重新训练 / 构建索引才能修改 | IVF 的`nlist`、HNSW 的`M`、PQ 的`m` | 高       |
| **动态参数** | 运行时可直接修改，无需重新构建索引            | IVF 的`nprobe`、HNSW 的`efSearch`   | 低       |

**3. 前置准备**

- **Faiss 版本**：建议使用 `faiss-cpu/faiss-gpu ≥ 1.7.0`（Auto-Tune 支持更完善）。
- **Optuna 安装**：若未安装，执行`pip install optuna`即可。
- 数据集
  - 训练集（train set）：用于索引训练（如 IVF 聚类、HNSW 构建）。
  - 基础库（base set）：待索引的向量库。
  - 查询集（query set）：用于评估检索性能。
- **Ground Truth**：通过暴力索引（`IndexFlatL2/IndexFlatIP`）计算的真实近邻，用于计算 Recall。

现在你是公司的算法工程师，领导需要你对向量索引的参数进行调优，让准确度 Recall@10 ≥ 0.9的前提下，QPS尽可能的大，QPS越大意味着着向量数据库处理数据的能力越强。

为了方便学习和理解，本节使用HNSW参数调优做案例。

#### 2.2.1 HNSW 关键参数说明

- **`m`**：每个节点的邻居数（度）。越大，索引精度越高，但索引构建时间、内存占用、检索时间都会增加（常用范围：4~32）。
- **`ef_construction`**：构建索引时的探索范围。越大，索引质量越高（召回率越高），但构建时间越长（常用范围：100~400）。
- **`ef_search`**：检索时的探索范围。越大，召回率越高，但检索速度越慢（常用范围：20~200）。这是**检索阶段的核心调优参数**。

#### 2.2.2 三种搜索方法对比

| 方法       | 原理                                                         | 优点                                     | 缺点                                 |
| ---------- | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------ |
| 随机搜索   | 从参数空间中随机采样组合                                     | 实现简单、计算量小、不易陷入局部最优     | 精度较低，可能错过最优参数           |
| 网格搜索   | 遍历参数空间的笛卡尔积（穷举）                               | 结果稳定、能找到全局最优（参数范围小时） | 计算量随参数维度指数增长（维度灾难） |
| 贝叶斯搜索 | 基于贝叶斯概率模型，利用历史试验结果指导后续参数采样（Optuna 实现） | 效率高、精度高（尤其适合高维参数）       | 实现稍复杂，依赖优化框架             |

#### 2.2.3 实践代码

```python
import numpy as np
import faiss
import time
import optuna
import random
from itertools import product
from typing import Tuple, Dict

# ====================== 1. 基础函数定义 ======================
def load_sift1m(path):
    """加载SIFT1M数据集（用户原有代码，已保留）"""
    d = 128
    xb = np.fromfile(f"{path}/sift_base.fvecs", dtype=np.float32)
    xq = np.fromfile(f"{path}/sift_query.fvecs", dtype=np.float32)
    gt = np.fromfile(f"{path}/sift_groundtruth.ivecs", dtype=np.int32)
    
    # 验证数据格式
    assert xb.size % (d + 1) == 0, f"数据库文件损坏：总长度 {xb.size} 不是 (d+1)={d+1} 的整数倍"
    assert xq.size % (d + 1) == 0, f"查询文件损坏：总长度 {xq.size} 不是 (d+1)={d+1} 的整数倍"
    
    # 解析fvecs格式（首字节为维度，后续为向量值）
    xb = xb.reshape(-1, d + 1)[:, 1:]  # 数据库向量：(1000000, 128)
    xq = xq.reshape(-1, d + 1)[:, 1:]  # 查询向量：(10000, 128)
    
    # 解析ivecs格式（首字节为近邻数，后续为近邻索引）
    k_gt = int(gt[0])  # 每个查询的真实近邻数（SIFT1M中为100）
    assert gt.size % (k_gt + 1) == 0, f"近邻文件损坏：总长度 {gt.size} 不是 (k_gt+1)={k_gt+1} 的整数倍"
    gt = gt.reshape(-1, k_gt + 1)[:, 1:]  # 真实近邻：(10000, 100)
    
    print(f"解析后数据形状: 数据库{xb.shape}, 查询{xq.shape}, 真实近邻{gt.shape}")
    return xb, xq, gt

def get_gt_by_flat(xb, xq, k=10):
    """暴力搜索生成Ground Truth（用户原有代码，已保留）"""
    d = xb.shape[1]
    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(xb)
    _, I_gt = index_flat.search(xq, k)
    return I_gt

def recall_at_k(I_pred: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    """计算Recall@k（核心指标：预测近邻与真实近邻的交集比例）"""
    assert I_pred.shape == I_gt.shape, "预测结果与真实结果形状不一致"
    n_queries = I_pred.shape[0]
    total_correct = 0
    for i in range(n_queries):
        # 计算单个查询的交集数量
        total_correct += len(set(I_pred[i]) & set(I_gt[i]))
    # 总召回率 = 正确匹配数 / (查询数 * k)
    return total_correct / (n_queries * k)

def benchmark(index: faiss.Index, xq: np.ndarray, k: int) -> Tuple[float, float]:
    """基准测试：计算单查询延迟（ms）和QPS（每秒查询数）"""
    n_queries = xq.shape[0]
    # 预热：避免首次运行的额外开销（如缓存加载）
    index.search(xq[:100], k)
    # 正式测试
    start = time.time()
    index.search(xq, k)
    end = time.time()
    total_time = end - start
    # 计算指标
    latency = (total_time * 1000) / n_queries  # 延迟：ms/query
    qps = n_queries / total_time              # QPS：queries per second
    return latency, qps

def evaluate_hnsw(params: Dict[str, int], xb: np.ndarray, xq: np.ndarray, I_gt: np.ndarray, k: int) -> Tuple[float, float, float]:
    """
    给定HNSW参数，构建索引并返回Recall@k、QPS、构建时间
    :param params: HNSW参数字典，包含m、ef_construction、ef_search
    :param xb: 数据库向量
    :param xq: 查询向量
    :param I_gt: 真实近邻索引
    :param k: 检索近邻数
    :return: recall, qps, build_time
    """
    m = params["m"]
    ef_construction = params["ef_construction"]
    ef_search = params["ef_search"]
    d = xb.shape[1]

    # 构建HNSW索引（L2距离）
    index_hnsw = faiss.IndexHNSWFlat(d, m)
    # 设置构建和检索参数
    index_hnsw.hnsw.efConstruction = ef_construction
    index_hnsw.hnsw.efSearch = ef_search

    # 构建索引（HNSW无需train，直接add）
    build_start = time.time()
    index_hnsw.add(xb)
    build_time = time.time() - build_start

    # 检索并计算指标
    _, I_hnsw = index_hnsw.search(xq, k)
    recall = recall_at_k(I_hnsw, I_gt, k)
    _, qps = benchmark(index_hnsw, xq, k)

    return recall, qps, build_time

# ====================== 2. 数据加载与全局配置 ======================
# 加载数据（替换为用户的SIFT1M路径）
sift1m_path = r"C:\Users\xiong\Desktop\easy-vecdb\easy-vecdb\data\sift"
xb, xq, gt = load_sift1m(sift1m_path)

# 核心配置
d = 128  # SIFT向量维度
k = 10   # 检索近邻数
recall_threshold = 0.9  # Recall@10阈值（≥0.9才符合要求）
I_gt = get_gt_by_flat(xb, xq, k)  # 暴力搜索的真实近邻（Ground Truth）

# ====================== 3. 方法1：随机搜索（Random Search） ======================
print("\n" + "="*50)
print("开始随机搜索调优HNSW参数...")
# 定义参数搜索空间（根据经验设置合理范围）
param_space_random = {
    "m": [4, 8, 16, 32],  # 每个节点的邻居数（核心参数，影响精度和速度）
    "ef_construction": [100, 200, 300],  # 构建时的探索范围（影响索引质量和构建时间）
    "ef_search": [20, 30, 50, 100]  # 检索时的探索范围（直接影响召回率和检索速度）
}
n_random_samples = 20  # 随机采样的参数组合数（可根据时间调整）

# 随机生成参数组合
random_params_list = []
for _ in range(n_random_samples):
    params = {
        "m": random.choice(param_space_random["m"]),
        "ef_construction": random.choice(param_space_random["ef_construction"]),
        "ef_search": random.choice(param_space_random["ef_search"])
    }
    random_params_list.append(params)

# 遍历参数组合，筛选符合条件的最优解
best_random_params = None
best_random_qps = 0.0
random_results = []

for i, params in enumerate(random_params_list):
    print(f"\n随机搜索第{i+1}/{n_random_samples}组参数：{params}")
    recall, qps, build_time = evaluate_hnsw(params, xb, xq, I_gt, k)
    random_results.append({
        "params": params,
        "recall": recall,
        "qps": qps,
        "build_time": build_time
    })
    print(f"Recall@{k}: {recall:.4f}, QPS: {qps:.0f}, 构建时间: {build_time:.2f}s")

    # 筛选：Recall≥阈值且QPS最大
    if recall >= recall_threshold and qps > best_random_qps:
        best_random_qps = qps
        best_random_params = params

# 输出随机搜索结果
print("\n" + "="*30)
if best_random_params is not None:
    print(f"随机搜索最优参数：{best_random_params}")
    print(f"对应Recall@{k}: {recall_at_k(evaluate_hnsw(best_random_params, xb, xq, I_gt, k)[0], I_gt, k):.4f}")
    print(f"对应QPS: {best_random_qps:.0f}")
else:
    print("随机搜索中没有找到满足Recall≥0.9的参数组合")

# ====================== 4. 方法2：网格搜索（Grid Search） ======================
print("\n" + "="*50)
print("开始网格搜索调优HNSW参数...")
# 定义参数网格（为了减少计算量，选择比随机搜索更小的范围）
param_grid = {
    "m": [8, 16],  # 邻居数（选中间值，平衡精度和速度）
    "ef_construction": [100, 200],  # 构建探索范围
    "ef_search": [30, 50]  # 检索探索范围
}

# 生成所有参数组合（笛卡尔积）
grid_params_list = [
    {"m": m, "ef_construction": ef_con, "ef_search": ef_sea}
    for m, ef_con, ef_sea in product(
        param_grid["m"],
        param_grid["ef_construction"],
        param_grid["ef_search"]
    )
]

# 遍历参数组合，筛选符合条件的最优解
best_grid_params = None
best_grid_qps = 0.0
grid_results = []

for i, params in enumerate(grid_params_list):
    print(f"\n网格搜索第{i+1}/{len(grid_params_list)}组参数：{params}")
    recall, qps, build_time = evaluate_hnsw(params, xb, xq, I_gt, k)
    grid_results.append({
        "params": params,
        "recall": recall,
        "qps": qps,
        "build_time": build_time
    })
    print(f"Recall@{k}: {recall:.4f}, QPS: {qps:.0f}, 构建时间: {build_time:.2f}s")

    # 筛选：Recall≥阈值且QPS最大
    if recall >= recall_threshold and qps > best_grid_qps:
        best_grid_qps = qps
        best_grid_params = params

# 输出网格搜索结果
print("\n" + "="*30)
if best_grid_params is not None:
    print(f"网格搜索最优参数：{best_grid_params}")
    print(f"对应Recall@{k}: {recall_at_k(evaluate_hnsw(best_grid_params, xb, xq, I_gt, k)[0], I_gt, k):.4f}")
    print(f"对应QPS: {best_grid_qps:.0f}")
else:
    print("网格搜索中没有找到满足Recall≥0.9的参数组合")

# ====================== 5. 方法3：贝叶斯搜索（Optuna） ======================
print("\n" + "="*50)
print("开始贝叶斯搜索调优HNSW参数...")

def objective(trial: optuna.Trial) -> float:
    """
    Optuna目标函数：最大化QPS，约束Recall≥0.9
    :param trial: Optuna试验对象，用于采样参数
    :return: 若满足约束返回QPS，否则返回-1（惩罚）
    """
    # 定义参数搜索空间（Optuna支持离散/连续采样，这里用离散）
    m = trial.suggest_categorical("m", [4, 8, 16, 32])
    ef_construction = trial.suggest_categorical("ef_construction", [100, 200, 300])
    ef_search = trial.suggest_categorical("ef_search", [20, 30, 50, 100])

    params = {"m": m, "ef_construction": ef_construction, "ef_search": ef_search}
    recall, qps, _ = evaluate_hnsw(params, xb, xq, I_gt, k)

    # 约束：Recall≥0.9，否则返回-1（Optuna会自动避开这类参数）
    if recall < recall_threshold:
        return -1.0
    return qps  # 目标：最大化QPS

# 创建Optuna研究对象，设置方向为最大化（maximize）
study = optuna.create_study(direction="maximize", study_name="HNSW调优")
# 运行优化（n_trials为试验次数，可调整）
study.optimize(objective, n_trials=20)

# 输出贝叶斯搜索结果
print("\n" + "="*30)
best_bayes_params = study.best_params
best_bayes_qps = study.best_value
if best_bayes_qps > 0:  # 找到满足约束的参数
    best_bayes_recall, _, _ = evaluate_hnsw(best_bayes_params, xb, xq, I_gt, k)
    print(f"贝叶斯搜索最优参数：{best_bayes_params}")
    print(f"对应Recall@{k}: {best_bayes_recall:.4f}")
    print(f"对应QPS: {best_bayes_qps:.0f}")
else:
    print("贝叶斯搜索中没有找到满足Recall≥0.9的参数组合")

# ====================== 6. 三种方法结果对比 ======================
print("\n" + "="*50)
print("三种调优方法结果对比：")
print(f"1. 随机搜索：最优参数={best_random_params}, 最优QPS={best_random_qps:.0f}")
print(f"2. 网格搜索：最优参数={best_grid_params}, 最优QPS={best_grid_qps:.0f}")
print(f"3. 贝叶斯搜索：最优参数={best_bayes_params}, 最优QPS={best_bayes_qps:.0f}")
```

运行结果

```
解析后数据形状: 数据库(1000000, 128), 查询(10000, 128), 真实近邻(10000, 100)

==================================================
开始随机搜索调优HNSW参数...

随机搜索第1/20组参数：{'m': 8, 'ef_construction': 300, 'ef_search': 50}
Recall@10: 0.8997, QPS: 87635, 构建时间: 42.06s

随机搜索第2/20组参数：{'m': 16, 'ef_construction': 200, 'ef_search': 100}
Recall@10: 0.9869, QPS: 34691, 构建时间: 39.02s

随机搜索第3/20组参数：{'m': 32, 'ef_construction': 100, 'ef_search': 20}
Recall@10: 0.8672, QPS: 120929, 构建时间: 23.70s

随机搜索第4/20组参数：{'m': 16, 'ef_construction': 100, 'ef_search': 50}
Recall@10: 0.9423, QPS: 67679, 构建时间: 20.09s

随机搜索第5/20组参数：{'m': 4, 'ef_construction': 200, 'ef_search': 100}
Recall@10: 0.8066, QPS: 62569, 构建时间: 17.96s

随机搜索第6/20组参数：{'m': 32, 'ef_construction': 300, 'ef_search': 30}
Recall@10: 0.9425, QPS: 71508, 构建时间: 74.13s

随机搜索第7/20组参数：{'m': 32, 'ef_construction': 300, 'ef_search': 50}
Recall@10: 0.9761, QPS: 48076, 构建时间: 76.70s

随机搜索第8/20组参数：{'m': 8, 'ef_construction': 100, 'ef_search': 20}
Recall@10: 0.7448, QPS: 149946, 构建时间: 18.07s

随机搜索第9/20组参数：{'m': 4, 'ef_construction': 300, 'ef_search': 50}
Recall@10: 0.7132, QPS: 102353, 构建时间: 26.71s

随机搜索第10/20组参数：{'m': 8, 'ef_construction': 100, 'ef_search': 100}
Recall@10: 0.9464, QPS: 42868, 构建时间: 17.28s

随机搜索第11/20组参数：{'m': 32, 'ef_construction': 100, 'ef_search': 100}
Recall@10: 0.9877, QPS: 31869, 构建时间: 25.92s

随机搜索第12/20组参数：{'m': 16, 'ef_construction': 300, 'ef_search': 20}
Recall@10: 0.8557, QPS: 121627, 构建时间: 62.82s

随机搜索第13/20组参数：{'m': 4, 'ef_construction': 100, 'ef_search': 100}
Recall@10: 0.7811, QPS: 61929, 构建时间: 11.51s

随机搜索第14/20组参数：{'m': 32, 'ef_construction': 200, 'ef_search': 20}
Recall@10: 0.8910, QPS: 102534, 构建时间: 52.63s

随机搜索第15/20组参数：{'m': 16, 'ef_construction': 300, 'ef_search': 50}
Recall@10: 0.9580, QPS: 65223, 构建时间: 65.67s

随机搜索第16/20组参数：{'m': 32, 'ef_construction': 200, 'ef_search': 20}
Recall@10: 0.8920, QPS: 100173, 构建时间: 53.27s

随机搜索第17/20组参数：{'m': 32, 'ef_construction': 200, 'ef_search': 20}
Recall@10: 0.8917, QPS: 92801, 构建时间: 55.08s

随机搜索第18/20组参数：{'m': 4, 'ef_construction': 200, 'ef_search': 20}
Recall@10: 0.5399, QPS: 252027, 构建时间: 21.46s

随机搜索第19/20组参数：{'m': 8, 'ef_construction': 200, 'ef_search': 100}
Recall@10: 0.9569, QPS: 44680, 构建时间: 34.57s

随机搜索第20/20组参数：{'m': 16, 'ef_construction': 300, 'ef_search': 100}
Recall@10: 0.9887, QPS: 33965, 构建时间: 66.14s

==============================
随机搜索最优参数：{'m': 32, 'ef_construction': 300, 'ef_search': 30}
对应Recall@10: 0.9425
对应QPS: 71508

==================================================
开始网格搜索调优HNSW参数...

网格搜索第1/8组参数：{'m': 8, 'ef_construction': 100, 'ef_search': 30}
Recall@10: 0.8154, QPS: 131905, 构建时间: 17.94s

网格搜索第2/8组参数：{'m': 8, 'ef_construction': 100, 'ef_search': 50}
Recall@10: 0.8860, QPS: 97015, 构建时间: 17.73s

网格搜索第3/8组参数：{'m': 8, 'ef_construction': 200, 'ef_search': 30}
Recall@10: 0.8270, QPS: 122328, 构建时间: 32.42s

网格搜索第4/8组参数：{'m': 8, 'ef_construction': 200, 'ef_search': 50}
Recall@10: 0.8971, QPS: 63065, 构建时间: 33.30s

网格搜索第5/8组参数：{'m': 16, 'ef_construction': 100, 'ef_search': 30}
Recall@10: 0.8900, QPS: 98209, 构建时间: 22.98s

网格搜索第6/8组参数：{'m': 16, 'ef_construction': 100, 'ef_search': 50}
Recall@10: 0.9417, QPS: 72354, 构建时间: 23.04s

网格搜索第7/8组参数：{'m': 16, 'ef_construction': 200, 'ef_search': 30}
Recall@10: 0.9059, QPS: 97237, 构建时间: 43.91s

网格搜索第8/8组参数：{'m': 16, 'ef_construction': 200, 'ef_search': 50}
Recall@10: 0.9544, QPS: 62138, 构建时间: 44.24s

==============================
网格搜索最优参数：{'m': 16, 'ef_construction': 200, 'ef_search': 30}
对应Recall@10: 0.9059
网格搜索第8/8组参数：{'m': 16, 'ef_construction': 200, 'ef_search': 50}
Recall@10: 0.9544, QPS: 62138, 构建时间: 44.24s

==============================
网格搜索最优参数：{'m': 16, 'ef_construction': 200, 'ef_search': 30}
对应Recall@10: 0.9059
==============================
网格搜索最优参数：{'m': 16, 'ef_construction': 200, 'ef_search': 30}
对应Recall@10: 0.9059
网格搜索最优参数：{'m': 16, 'ef_construction': 200, 'ef_search': 30}
对应Recall@10: 0.9059
对应Recall@10: 0.9059
对应QPS: 97237

对应QPS: 97237


==================================================
开始贝叶斯搜索调优HNSW参数...
[I 2025-12-15 14:23:22,410] A new study created in memory with name: HNSW调优
[I 2025-12-15 14:24:28,533] Trial 0 finished with value: 33992.197111277885 and parameters: {'m': 16, 'ef_construction': 300, 'ef_search': 100}. Best is trial 0 with value: 33992.197111277885.
[I 2025-12-15 14:24:48,263] Trial 1 finished with value: -1.0 and parameters: {'m': 4, 'ef_construction': 200, 'ef_search': 20}. Best is trial 0 with value: 33992.197111277885.
[I 2025-12-15 14:25:15,991] Trial 2 finished with value: -1.0 and parameters: {'m': 4, 'ef_construction': 300, 'ef_search': 50}. Best is trial 0 with value: 33992.197111277885.
[I 2025-12-15 14:25:39,956] Trial 3 finished with value: -1.0 and parameters: {'m': 16, 'ef_construction': 100, 'ef_search': 30}. Best is trial 0 with value: 33992.197111277885.
[I 2025-12-15 14:26:00,279] Trial 4 finished with value: -1.0 and parameters: {'m': 4, 'ef_construction': 200, 'ef_search': 50}. Best is trial 0 with value: 33992.197111277885.
[I 2025-12-15 14:27:29,482] Trial 5 finished with value: 63128.72419683777 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 30}. Best is trial 5 with value: 63128.72419683777.
[I 2025-12-15 14:28:03,441] Trial 6 finished with value: -1.0 and parameters: {'m': 8, 'ef_construction': 200, 'ef_search': 30}. Best is trial 5 with value: 63128.72419683777.
[I 2025-12-15 14:28:15,106] Trial 7 finished with value: -1.0 and parameters: {'m': 4, 'ef_construction': 100, 'ef_search': 100}. Best is trial 5 with value: 63128.72419683777.
[I 2025-12-15 14:28:33,190] Trial 8 finished with value: -1.0 and parameters: {'m': 8, 'ef_construction': 100, 'ef_search': 30}. Best is trial 5 with value: 63128.72419683777.
[I 2025-12-15 14:29:18,769] Trial 9 finished with value: 35635.09386031614 and parameters: {'m': 16, 'ef_construction': 200, 'ef_search': 100}. Best is trial 5 with value: 63128.72419683777.
[I 2025-12-15 14:30:38,690] Trial 10 finished with value: 99322.83501858912 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 20}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:32:04,828] Trial 11 finished with value: -1.0 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 20}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:33:24,029] Trial 12 finished with value: -1.0 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 20}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:34:45,082] Trial 13 finished with value: -1.0 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 20}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:36:06,331] Trial 14 finished with value: 73448.59408846464 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 30}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:37:27,263] Trial 15 finished with value: 73154.46612796046 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 30}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:38:48,079] Trial 16 finished with value: -1.0 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 20}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:40:17,277] Trial 17 finished with value: 39483.16015596318 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 50}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:41:37,994] Trial 18 finished with value: -1.0 and parameters: {'m': 32, 'ef_construction': 300, 'ef_search': 20}. Best is trial 10 with value: 99322.83501858912.
[I 2025-12-15 14:41:55,980] Trial 19 finished with value: -1.0 and parameters: {'m': 8, 'ef_construction': 100, 'ef_search': 30}. Best is trial 10 with value: 99322.83501858912.

==============================
贝叶斯搜索最优参数：{'m': 32, 'ef_construction': 300, 'ef_search': 20}
对应Recall@10: 0.8992
对应QPS: 99323

==================================================
三种调优方法结果对比：
1. 随机搜索：最优参数={'m': 32, 'ef_construction': 300, 'ef_search': 30}, 最优QPS=71508
2. 网格搜索：最优参数={'m': 16, 'ef_construction': 200, 'ef_search': 30}, 最优QPS=97237
3. 贝叶斯搜索：最优参数={'m': 32, 'ef_construction': 300, 'ef_search': 20}, 最优QPS=99323
```

先通过表格直观展示三种调优方法的最优结果（关键指标：Recall@10、QPS、参数组合）：

| 调优方法   | 最优参数组合                              | Recall@10 | QPS（查询 / 秒） | 是否满足 Recall 约束（≥0.9） |
| ---------- | ----------------------------------------- | --------- | ---------------- | ---------------------------- |
| 随机搜索   | `m=32, ef_construction=300, ef_search=30` | 0.9425    | 71508            | ✅ 是                         |
| 网格搜索   | `m=16, ef_construction=200, ef_search=30` | 0.9059    | 97237            | ✅ 是（刚好达标）             |
| 贝叶斯搜索 | `m=32, ef_construction=300, ef_search=20` | 0.8992    | 99323            | ⚠️ 接近达标（仅差 0.0008）    |

从上述的表中我们可以看到

1. **贝叶斯搜索在速度上表现最优**：最优 QPS 达到 99323，远超随机搜索和网格搜索；
2. **网格搜索是 “精度 - 速度” 的平衡者**：Recall 刚好达标，QPS 仅次于贝叶斯搜索；
3. **随机搜索效率最低**：虽然 Recall 达标，但 QPS 显著偏低，体现了盲目采样的局限性。

**三种调优方法的对比与选型建议**

| 对比维度   | 随机搜索             | 网格搜索                  | 贝叶斯搜索（Optuna）             |
| ---------- | -------------------- | ------------------------- | -------------------------------- |
| 核心原理   | 随机采样参数组合     | 穷举参数网格的笛卡尔积    | 基于贝叶斯模型学习参数规律       |
| 调优效率   | 低（盲目采样）       | 中（穷举遍历）            | 高（智能学习）                   |
| 结果精度   | 低（易错过最优解）   | 中（仅覆盖指定范围）      | 高（快速收敛到全局最优）         |
| 实现复杂度 | 极低（几行代码即可） | 低（嵌套循环 /itertools） | 中（依赖 Optuna 框架，代码量少） |
| 适用场景   | 快速验证参数范围     | 小规模参数空间            | 中大规模参数空间、工业界项目     |

最后，没有 “绝对最优” 的参数，只有 “最适合场景” 的参数。

## 3 大规模向量检索优化：百万/亿级场景

当向量数据量达到百万级或更高时，传统单机索引容易面临以下问题：

- **内存压力大**：单索引存储上亿条128维向量，需要几十GB内存。
- **查询延迟高**：单节点检索大规模向量时，每次搜索都可能成为性能瓶颈。

核心解决思路：**分布式分片 + 内存优化**。

### 3.1 分片索引（Sharding）：突破单节点限制

**概念**：
 将大规模向量集合拆分为若干子索引（Shard），每个子索引独立检索，然后将结果合并。

**特点**：

- 减轻单节点内存压力
- 支持并行检索（线程/集群）
- 易于水平扩展

**示意流程**：

1. 数据拆分 → 分片索引训练
2. 查询 → 每个分片独立搜索
3. 合并 Top-K 结果 → 返回最终结果

在 FAISS 中可使用 `IndexShards` 来实现逻辑分片和结果合并：

```
# 示例：4个分片，逻辑合并
index_sharded = faiss.IndexShards(d, threaded=False)
for shard in shards:
    index_sharded.add_shard(shard)
```

> ⚠️ `threaded=False` 表示串行检索，仅用于理解结构。实际可开启多线程或在多机器分布式中并行执行。

**理论对比**：

- **单索引**：内存压力大，查询时间随向量数量线性增长
- **分片索引**：每个分片规模小，可并行检索 → 总延迟显著下降

### 3.2 分布式检索：FAISS Cluster

**概念**：
 FAISS官方提供的分布式集群解决方案，采用**主从架构**：

- **Master节点**：接收查询，合并结果
- **Worker节点**：存储分片索引，执行本地检索

**流程**：

1. Master启动，管理Worker分片
2. Worker接入，加载本地分片
3. 客户端查询 → Master分发到Worker → 汇总结果返回

> 优点：可扩展到亿级向量，查询延迟低且负载均衡。


### 3.3 内存占用优化

百万/亿级向量单纯用Flat索引存储，会消耗大量内存，例如：

- 128维 × 100万条 → 约 500 MB
- 128维 × 1亿条 → 约 50 GB

核心优化策略：

#### 3.3.1 向量降维（PCA）

- 将高维向量降维（如 128→64）
- 减少内存占用约 50%，对Recall损失小（2%-5%）
- FAISS通过 `PCAMatrix` 支持降维

**概念示意**：

```
原始向量 (128维) → PCA降维 → 降维向量 (64维) → 构建索引
```

> 适合对精度要求不是极端高的场景，性价比高。

#### 3.3.2 分桶存储（冷热数据分离）

- **热数据**：近期高频访问，加载到内存
- **冷数据**：低频访问，存储在磁盘
- 查询先查热数据，再查冷数据，减少内存消耗

**优势**：

- 内存占用降低
- 查询效率提升（大部分查询命中热数据）

**理论示意**：

```
查询向量 → 搜索热数据索引
            ↘ 若未命中 → 搜索冷数据索引
合并结果 → 返回Top-K
```

> 结合分片与冷热分层策略，可支持亿级向量检索。

## 4 总结与实战任务

### 4.1. 核心知识点回顾

- 评估指标：Recall@k（召回率）、QPS/延迟（效率）是调优的“指南针”
- 参数调优：nprobe、M、ef是核心，网格搜索入门，贝叶斯优化高效
- 大规模优化：分片索引（并行提速）、PCA降维（内存优化）、FAISS Cluster（分布式部署）

### 4.2. 实战任务

基于SIFT1M全量数据（100万条），完成以下任务并提交报告：

1. 构建HNSW索引，调优ef和M参数，目标Recall@10≥80%，延迟≤10ms
2. 任意选择一个你喜欢的索引评估在100万数据下的性能（Recall@10、延迟、内存）