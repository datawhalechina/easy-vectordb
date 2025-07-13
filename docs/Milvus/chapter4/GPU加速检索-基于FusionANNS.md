# 向量数据的存储优化与GPU加速检索

近似最近邻搜索（ANNS）已成为数据库和人工智能基础设施的关键组成部分。不断增长的向量数据集给 ANNS 服务在性能、成本和准确性方面带来了重大挑战。现有 ANNS 系统均无法同时解决这些问题。

高维空间中的近似最近邻搜索(ANNS)旨在找出与给定查询向量最相似的前 k 个向量。该技术在数据挖掘、搜索引擎以及 AI 驱动的推荐系统等诸多领域具有广泛应用。特别是在大型语言模型(LLMs)近期蓬勃发展的推动下，ANNS 系统已成为现代 AI 基础设施的关键组成部分。

检索增强生成(RAG)的典型框架：领域特定知识首先被嵌入为高维向量并存储于向量数据库中。当聊天机器人接收到查询时，会通过 ANNS 引擎从向量数据库中检索最相关的知识，使 LLM 能够将这些知识作为额外上下文进行更精准的推理。

![alt text](/docs/src/x1.png)

近似最近邻搜索（Approximate Nearest Neighbor Search, ANNS）是处理海量图片、文本特征等数据的核心技术。它的目标并非追求绝对最匹配的结果，**而是通过高效方法获取足够相似的候选集**。在实际应用中，这项技术面临着内存消耗巨大与计算强度极高两大严峻挑战。
### 1. 内存消耗
为了加速搜索过程，ANNS 系统需要预先构建类似 “目录” 或 “关系网” 的索引结构。详细内容请回到[Milvus 索引介绍](../chapter1/milvus%20索引介绍.md)中IVF系列和HNSW系列。


然而，当数据量攀升至十亿甚至千亿级别的规模时，这些索引自身就会占用 TB 级别（相当于数个大型硬盘）的内存空间。如此高昂的内存成本，极大地限制了 ANNS 技术向更大规模数据处理场景的扩展。
### 2. 计算强度
ANNS 的核心操作是进行距离计算，即计算数据点之间的相似度，例如比较两张图片特征向量的差异。在处理高维数据（每个数据点包含数百甚至数千个特征值）和超大基数数据时，系统需要执行海量的距离计算，这对算力形成了巨大压力，使得计算强度极高。

在当下热门的检索增强生成（RAG）场景中，ANNS 扮演着至关重要的角色，它是支撑大语言模型（Large Language Model, LLM）实时检索外部知识的核心环节。实际测试数据显示，在整个 LLM 查询流程中，ANNS 阶段所消耗的时间接近 50%，已然成为制约系统响应速度的关键瓶颈。

### 3. 优化方法
为降低近似最近邻搜索(ANNS)的内存开销，现有方法主要分为两类：层次化索引(Hierarchical Indexing, HI)与乘积量化(Product Quantization, PQ)。
1. 首先，层次化索引通过将索引存储于固态硬盘(SSD)来减少内存占用。以微软商用 ANNS 系统 SPANN为例，该系统将所有基于倒排文件(IVF)的索引（即倒排列表）存放于 SSD，仅通过导航图在内存中维护这些倒排列表的质心。虽然 SPANN 实现了低延迟，但我们发现其并发查询吞吐量存在明显瓶颈，在高端 SSD 上仅能支持最多四个 CPU 线程的峰值性能。这种有限的可扩展性制约了其在需要高吞吐量 AI 应用中的实用性。
2. 其次，乘积量化是另一种有效降低内存成本的技术。该向量压缩方法可将高维向量的内存占用量减少高达 95%，同时还能将 ANNS 速度提升数倍。但由于 PQ 属于有损压缩方案，更高的压缩率往往会导致查询精度下降。 对于某些要求高准确率的场景，这种情况通常是不可接受的。


## FusionANNS 系统介绍
### 1.系统架构
FusionANNS 的卓越之处在于：与最先进的 SSD 基 ANNS 系统 SPANN 及 GPU 加速内存 ANNS 系统 RUMMY 相比，实现了显著性能提升。实验数据显示，FusionANNS 相较 SPANN 取得 9.4-13.1 倍查询吞吐量(QPS)和 5.7-8.8 倍成本效益；相比 RUMMY 则提升 2-4.9 倍 QPS 和 2.3-6.8 倍成本效益。这种在保持高性能同时显著提升成本效益的特点，成为该研究最突出的创新点。
FusionANNS 的技术核心之一在于 CPU 与 GPU 协同过滤与重排序机制。该技术大幅减少了 CPU、GPU 和 SSD 之间的 I/O 操作，突破了 I/O 性能瓶颈。具体包含三项设计：1) 避免 CPU 与 GPU 数据交换的多层索引结构；2) 在保证高精度的同时消除冗余 I/O 与计算的启发式重排序；3) 通过冗余感知 I/O 去重技术改善重复 I/O 问题。


## 一、量化 (PQ) 核心流程详解

![alt text](../../src/PQ.jpg)

### 1.1 Codebook 构建（离线阶段）
在 FusionANNS 系统中，产品量化的第一步是构建 Codebook，这一过程在离线阶段完成。假定所有的向量都是512维，有一万个这样的向量，我们将其非为八组:
* 取一万个向量的前64维划为第一组
* 取一万个向量的64-128维划为第二组
* 取一万个向量的128-192维划为第三组
* ...
* 取一万个向量的448-512维划为第八组

对每一组进行k=256的k-means聚类，得到256个聚类中心，即Codebook。观察下图，可以看到codebook1为一个64x256的矩阵，其中256表示256个聚类中心，64表示每个聚类中心对应的维度。

![alt text](../../src/pq1.png)

最终我们可以得到8个子码本，即8个codebook，用于下一步的PQ量化。其实现逻辑可通过以下 Python 代码示例呈现：
```python
def build_codebook(vectors):
    codebooks = []
    for i in range(8):  # 8个子空间
        start_dim = i * 64
        end_dim = (i+1) * 64
        subspace = vectors[:, start_dim:end_dim]  # 提取64维子空间
        
        # 执行k-means聚类(k=256)
        centroids = kmeans(subspace, k=256)
        codebooks.append(centroids)  # 256x64矩阵
    return codebooks
```

从数学角度来看，子空间划分遵循公式

\(\text{Group}_i = [64i+1, 64(i+1)] \quad i \in [0,7]\)

而每个子空间生成的聚类中心矩阵表示为\(C_i \in \mathbb{R}^{256 \times 64}\)。通过这样的方式，为后续的向量量化奠定基础。

### 1.2 向量量化（在线阶段）
在线阶段，系统对输入的单个 512 维向量\(X = (x_1,x_2,...,x_{512})\)进行量化操作，输出 8 字节压缩编码。具体实现如下：

对于数据库中每一个512维度向量，取前1~64维为一组，切分为8组
```txt
X = (x1 ... ... x8)
```
对于其中子向量x1（64维）与8个子码本计算欧氏距离。
子码本Codebook1如下：

![pic](../../src/codebook1.png)

其中，欧氏距离的计算公式为：

![pic](../../src/欧氏距离.png)

若结果为：

![pic](../../src/8.png)

观察可得，子向量x1与CodeBook1中V256的距离最近，为0.1 记录为`(256 ... ...)`

同理，
* 子向量x2与CodeBook2的欧氏距离 d 为 xxx ,对应Vk (k∈R)
* 子向量x3与CodeBook3的欧氏距离 d 为 xxx ,对应Vk (k∈R)
* ...
* 子向量x8与CodeBook8的欧氏距离 d 为 xxx ,对应Vk (k∈R)

例如：
```text
 1    2 3  4  5  6  7   8
(256,78,3,41,25,97,62,187)
```
PS:上述量化编码表示了原始向量的离散化近似表示。

综上，经过这样的量化过程，原始向量的大小从 512×4=2048 字节大幅压缩至 8 字节，实现了256:1的惊人压缩率，有效降低了内存占用，提升了数据处理效率。

![pic](../../src/压缩.png)


代码如下：
```python
def quantize_vector(vector, codebooks):
    compressed = []
    for i in range(8):
        subvec = vector[i*64 : (i+1)*64]
        distances = [euclidean(subvec, centroid) for centroid in codebooks[i]]
        idx = np.argmin(distances)  # 找到最近聚类中心索引
        compressed.append(idx)      # 0-255整数
    return compressed  # 例：[256,78,3,41,25,97,62,187]
```

### 1.3 查询距离计算（GPU 加速）生成距离表
在查询距离计算环节，系统充分利用 GPU 加速能力。首先，对查询向量 Q 进行同样的子空间划分，得到\(Q = (q_1,q_2,...,q_8)\)。接着，计算子空间距离表，其计算公式为：

$
\text{DistanceTable}[i][j] = ||q_i - C_i[j]||^2
\quad \begin{array}{c} i \in [0,7] \\ j \in [0,255] \end{array}
$
然后，根据量化编码快速计算近似距离：
$
\text{Dist}(Q,X) \approx \sum_{k=1}^{8} \text{DistanceTable}[k][\text{code}_k]
$
为进一步提升计算效率，系统采用 2048 线程并行计算（8 子空间 ×256 中心），在 V100 GPU 上，每个查询的计算耗时仅为0.0046ms，极大地加快了查询响应速度。
如果不太理解，我们可以假设有一个query，对其进行PQ量化，得到PQ量化后的向量表示:
```txt
Q1 = (256,78,3,41,25,97,62,187)
```
同样的，在PQ量化部分，我们得到了Q的八个子空间：
```txt
Q = (q1 ... ... q8)
```
对于子空间q1，我们得到子空间q1的子码本：Codebook1，计算欧氏距离最终得到8x256的距离表：

![pic](../../src/距离表.png)

其中，
* 1.5表示子空间 q1 与 codebook1 中 1 号聚类中心距离为 1.5。
* 2.7表示子空间 q2 与 codebook2 中 1 号聚类中心距离为 2.7。
* ...
* 0.9表示子空间 q1 与 codebook1 中 2 号聚类中心距离为 0.9。

根据
```txt
Q1 = (256,78,3,41,25,97,62,187)
```
得到：
* [1][256] = 0.1
* [2][78] = xx1
* [3][3] = xx2
* ...
* [8][187] = xx8

所以，总距离为`0.1 + xx1 + xx2 + ... + xx8`。

至此，我们对前面的内容总结一下，以便加深概念与流程的理解。

对于子码表codebook，可后台提前巨酸。

对于新的query，生成距离表，需计算`8个子空间 x 256个中心 x 64维` = 131,072次float计算
而对于现代计算机来说:
* CPU:100亿次浮点运算/s --> 0.013ms
* GPU:20万亿次浮点运算/s --> 0.0006ms

可用`8 x 256= 2048`个线程并行计算距离表......反正就是多开几个线程，算的快一些。
## 二、分层索引与边界优化
### 2.1 多层级聚类结构
![pic](/docs/src/x12.png)
对于上图，我们对每一个部分进行更加详细的解释说明：

FusionANNS 采用独特的多层级聚类结构，如下图所示。

![pic](/docs/src/1.png)

系统将 10 亿向量依次进行 H1、H2、H3、H4 层聚类，最终形成 Posting List。以 H1 层为例，会聚类成 2 个质心，H2 层有 3 个质心，H3 层 5 个质心，H4 层 8 个质心，通过这种分层聚类方式，有效组织数据，缩小搜索空间。


### 2.2 边界向量分配算法
Specifically, when a vector lies on the boundary of multiple clusters, we assign this boundary vector to a cluster according to the following rule:
这句话的意思是：当一个向量位于多个集群的边界时，我们根据以下规则：

\(v \in C_i \iff \text{Dist}(v,C_i) \leq (1+\epsilon) \times \text{Dist}(v,C_1)\)

来决定向量的归属，其中\(C_1\)是最近质心。以一个具体示例来说明，假设
* \(\text{Dist}(v,C_1) = 0.7\)
* \(\text{Dist}(v,C_i) = 0.3\)
* \(\epsilon = 0.3\)

则\(0.3 \leq (1+0.3)\times0.7 = 0.91\)，满足条件，该向量会被包含在聚类\(C_i\)中。

To balance the query accuracy and efficiency,each vector is assigned to eight clusters at most。
通过这样的策略，单个向量可归属最多 8 个聚类，在实际测试中，相较于 SPANN，其召回率提升了 32%，显著提高了查询准确性。
## 三、启发式重排序算法
### 3.1 动态截断流程
由于 PQ 在距离计算中会导致一定的精度损失，FusionANNS 引入启发式重排序算法来优化查询结果。该算法的动态截断流程可通过以下 Python 代码实现：
```python
def heuristic_reranking(candidates, query_vec, k=10, batch_size=1000, ε=0.4, β=3):

    max_heap = MaxHeap(k)  # 维护Top-k最小堆
    stability_count = 0
    
    for i, batch in enumerate(batch_split(candidates, batch_size)):
        prev_topk = set(max_heap.get_ids())  # 保存前一批Top-k
        
        # 处理当前batch
        for vec_id in batch:
            raw_vec = ssd_read(vec_id)
            dist = distance(query_vec, raw_vec)
            max_heap.push(dist, vec_id)
        
        current_topk = set(max_heap.get_ids())
        Δ = len(current_topk - prev_topk) / k  # 计算变化率
        
        if Δ < ε:
            stability_count += 1
            if stability_count >= β:  # 连续β批变化小
                break
        else:
            stability_count = 0  # 重置计数器
    
    return max_heap.get_sorted()
```
在这个过程中，系统将重排序过程划分为多个小批量（mini-batch）依次处理，通过不断计算变化率\(\Delta\)来判断结果是否趋于稳定，从而决定是否终止重排序，避免不必要的计算和 I/O 操作。

总结的有些太简单了，这里再详细的来解释：
对于召回的十万Vectors我们将其分为100组，一组为1000个vector。

![pic](../../src/组.png)

依次进行处理，组1处理后，进入轻量反馈控制模型，判断是否需要进行组2的处理。（看3.2）
### 3.2 轻量反馈控制模型 变化率 Δ 的数学本质
思想：判断下一次的处理是否会对精度造成影响。若连续的β次检测后，变化率\(\Delta<ε\)，则安全终止重排序。其中，变化率\(\Delta\)的计算公式为

\(\Delta = \frac{|S_n \setminus S_{n-1}|}{k} = \frac{\text{新增Top-k向量数}}{k}\)
或者：

![pic](../../src/6.png)

最后，小于阈值β表示，当前连续β次满足\(\Delta<ε\)时的子批量对Top-k的优化贡献微弱。
更加详细的来说：其反映了每一批次处理后 Top-k 结果中新增向量的比例。当\(\Delta>ε\)时，说明结果仍在剧烈变化，需要继续处理；当\(\Delta<ε\)时，结果趋于稳定，系统开始准备终止；若连续\(\beta\)次\(\Delta<ε\)，则安全终止重排序。在实际应用中，该算法在不同数据集上均取得了显著效果，如在 SIFT1B 数据集上，全量重排序需要处理 40,000 个向量，而启发式重排序仅需处理 28,000 个，I/O 减少了 30%；在 DEEP1B 数据集上，同样实现了 30% 的 I/O 减少，有效提升了系统性能。
## 四、存储优化与 I/O 去重
### 4.1 物理存储布局优化
在存储方面，原始向量大小通常在 128 - 384 字节，而现代 NVMe SSD 的页大小为 4KB，这导致了严重的读放大问题，读放大倍数可达 10 - 32 倍。为解决这一问题，FusionANNS 对物理存储布局进行优化，如 所示，系统为每个质心创建专属 Bucket，

在启发式重排序阶段获取的组N，可保证与查询query vector 高度相似，构建导航图，为每一个质心分配存储桶Bucket,存放最接近该质心的若干向量ID。

![pic](/docs/src/存储2.png)

将向量按到质心距离排序填充到 Bucket 中，然后跨 Bucket 合并填充 4KB 页，以最小化碎片，从而提高存储效率和 I/O 性能。

![pic](/docs/src/存储布局.png)

### 4.2 二级 I/O 去重机制

 ![pic](/docs/src/x13.png)

FusionANNS 还设计了二级 I/O 去重机制，包括批内合并和批间复用。在批内合并方面，以 Mini-batch 请求 V2, V4, V6 为例，系统通过映射表获取每个向量对应的页 ID，发现 V2 和 V6 位于同一页 P0，V4 位于页 P2，最终实际 I/O 操作只需读取 2 次页面（P0 和 P2），替代了原本 3 次的 I/O 请求。在批间复用方面，若 Batch0 已加载 P0（包含 V2、V6）和 P2（包含 V4）到缓存中，当 Batch1 请求 V5（位于 P2）、V8（位于 P1）、V9（位于 P3）时，由于 P2 已在缓存中，Batch1 实际只需读取 P1 和 P3 两次页面即可。通过这样的 I/O 去重机制，在实际测试中，随机存储情况下 I/O 次数为 40,000 次，数据读取量为 160MB，而经过优化后，I/O 次数减少到 30,800 次，数据读取量降低至 123MB，I/O 次数和数据读取量均减少了 23%。
## 五、端到端查询流程
### 5.1 系统架构
FusionANNS 的端到端查询流程涉及多个组件协同工作，其流程如![系统架构序列](/docs/src/x11.png) 所示。当 Client 发送查询向量 Q 后，CPU 首先遍历导航图，获取 Top-64 候选列表，然后将向量 ID 列表发送给 GPU。GPU 从 HBM 加载 PQ 向量，进行并行距离计算，并将 Top-N 候选 ID 返回给 CPU。接着，CPU 按照优化后的存储策略从 SSD 读取原始向量，进行动态重排序，最后将最终的 Top-K 结果返回给 Client。整个过程高效有序，充分发挥了各组件的优势。
首先，利用 GPU 生成 query vector 的距离表用于后续 PQ 距离的计算
同时，CPU 遍历 in-memory 导航图定位到距离查询向量最近的 top-m 簇集
接着，CPU 查阅 metadata 获取候选的 Vector IDs
之后，CPU 将这些 IDs 传输给 GPU kernels 用于后续计算
GPU 获取 vec-IDs 后，通过 parallel vec-ID 去重相同 ID
对于每个 vec-ID，GPU 从 HBM 内读它们相应的 PQ 向量和计算它们与查询向量的距离，在这一步 GPU 给 PQ 向量的每个维度分配一个线程，用于访问该维度之前计算过存于 distance table 的值，接着，协调线程累加各维度值作为每个候选向量的 PQ 距离值
之后，GPU 以升序返回 top-n 个 vec ID 给 CPU
接着，CPU 根据 ID 从 SSD 内读取 raw vectors 用于后续 re-ranking，最后返回 top-k 个邻居向量。

**上面的一大串文字，可能看起来不太容易，下面将配合着图片来解释：**
对于我们所有存储在SSD中的数据，

![pi'c](/docs/src/ssd.png)

我们对其进行分层聚类得到Posting List。

![pic](/docs/src/postingList.png)

对于图中Posting List倒排索引列表，里面的ID为图的节点ID。
我们根据这个倒排索引列表，构建下图：

![pic](/docs/src/graph.png)

存放进入CPU，然后丢弃倒排索引列表，只保留图(注意：只存储向量的ID列表，不存实际的向量数据！)。对于新加入的点：
1. 计算与附近Top64个最近的点的Distance并连接
2. 类似于HNSW索引的多层小世界图
   ![pic](/docs/src/hnsw.png)
3. 取64个PL的Meta Data PQ量化存储到GPU HBM中
4. GPU 根据ID List 从HBM中获取PQ量化后的组，进行Dist表的构建
5. 其中，CPU -> SSD(根据IDs 从SSD中获取原始向量，计算精确距离，从而弥补PQ造成的精度损失，返回Top-k)

貌似有些抽象，我们回到原始的图中：

![pic](/docs/src/x12.png)

首先我们看离线层的逻辑操作：

![pic](/docs/src/22.png)

这是我们的SSD中存储的数据，包含了向量的ID和具体数据，当然，实际应用中，数据格式不可能是这么简单的，但100%包含ID 和 embedding这两个。
我们有两件事情要做的：

**首先**：将这些数据进行PQ量化后存储到GPU HBM中，
![pic](/docs/src/25.png)

**其次**我们对这些数据进行分组和倒排索引（关于这一部分，请看[索引部分](../chapter1/milvus%20索引介绍.md)），得到Posting List。

![pic](/docs/src/postingList.png)

然后构建图并丢弃Posting List：

![pic](/docs/src/24.png)

你可以看到，图中的节点ID就是Posting List中的ID。我们将这一部分数据都放到了CPU存储中，图中的每一个节点都包含：
1. 节点ID
2. 节点存储的向量的ID List

现在，我们离线层的数据构建就做好了，接下来，我们看在线层的逻辑操作：

![pic](/docs/src/16.png)

详细看过上面内容的小伙伴应该知道，这里query的计算过程，我们进入内存导航图，确定最近的PL，假设是上图中的那几个节点，我们可以组合这些ID List，可以得到：

![pic](/docs/src/17.png)

然后，GPU根据这些从CPU 传来的ID List，从自己的 HBM中获取PQ量化后的向量，得到每个V_ID (v0,v1,v3,v4 ... ... )的PQ向量。

构建距离表，

![pic](/docs/src/18.png)

然后根据距离表，计算精确距离，返回Top-k。

![pic](/docs/src/20.png)

我们可以看到 v2 v0 v5 是最近的，然后就到最后了，我们将这三个ID进入I/O Engine 取SSD查询。

## 六、性能实验与工业价值
### 6.1 性能对比实验
为验证 FusionANNS 的性能优势，我们在特定实验环境下进行测试。实验环境配置为：CPU 采用 2×Xeon 64-core，GPU 为 NVIDIA V100（32GB HBM），SSD 使用 Samsung 990Pro 2TB。在吞吐量对比（QPS）方面，不同数据集下的实验结果如 ![吞吐量对比图](/docs/src/x14.png) 所示，FusionANNS 在 SIFT1B、SPACEV1B、DEEP1B 等数据集上，相较于 SPANN 和 RUMMY，QPS 均有显著提升，展现出强大的处理能力。
### 6.2 工业应用场景
在工业应用中，FusionANNS 可有效优化 RAG 架构，如 [此处插入 RAG 架构优化示意图] 所示。当用户提问后，Query 经过嵌入处理进入 FusionANNS 引擎，引擎从知识库中快速检索相关向量，获取 Top-K 相关文档，为 LLM 生成回答提供准确信息。在实际应用中，原本占比 50% 的延迟降低至 10%，对于 10 亿级向量的检索，P99 延迟小于 100ms。其适用场景广泛，涵盖法律 AI（如 ChatLaw 千亿级法律条文检索）、家装设计（如 ChatHome 百万级 3D 模型检索）、金融风控（如 Xuanyuan 2.0 实时交易监测）、电商推荐（十亿级商品向量实时匹配）等多个领域，为各行业的智能化发展提供了有力支持。
## 七、结论与创新价值
### 7.1 核心突破
FusionANNS 在多个方面实现了核心突破。在存储方面，通过 PQ 压缩率达到 256:1，多层级索引将内存占用降低为 SPANN 的 1/8；计算范式上，采用 CPU-GPU 协同模式，CPU 负责导航，GPU 进行并行计算，使数据传输量减少 99%；在优化策略上，启发式重排序减少了 30% 的 I/O 操作，冗余感知存储降低了 23% 的读放大，全面提升了系统性能。
### 7.2 行业影响
从行业角度来看，FusionANNS 具有重要影响。在成本方面，实现千亿向量检索的硬件成本低于 $8,000；性能上，QPS 提升 13.1 倍，延迟小于 10ms P99；在生态建设中，有望成为 LLM+RAG 基础设施的标准组件，推动人工智能领域的进一步发展。
最终实现：在十亿级 ANNS 中首次同时满足：高吞吐 (10k+ QPS)｜低延迟 (<10ms)｜高精度 (Recall@10>95%)｜低成本 (<$10k)，为大数据和人工智能领域的向量搜索问题提供了极具价值的解决方案。


## 总结
### 三大核心技术
1. **多层级索引结构（Multi-tiered Indexing）**  
   - **存储策略**：
     | 设备          | 存储内容              | 关键优势                 |
     | ----------- | ----------------- | -------------------- |
     | **SSD**     | 原始向量（Raw Vectors） | 低成本存储海量数据            |
     | **GPU HBM** | PQ压缩向量（高压缩比）      | 显存容纳十亿级向量，避免数据交换     |
     | **CPU内存**   | 向量ID列表 + 导航图（无内容） | 传输量减少99%（仅传ID而非向量内容） |
   - **突破**：消除CPU-GPU间冗余数据传输，解决PCIe带宽瓶颈。

2. **启发式重排序（Heuristic Re-ranking）**  
   - **动态截断机制**：
     - 将重排序拆分为**小批次（Mini-batch）** 顺序执行。
     - 每批完成后计算**结果改进率**：
       \[
       \Delta = \frac{|S_n - S_n \cap S_{n-1}|}{k} \quad \text{(当前批与上批结果的差异率)}
       \]
     - 若连续β批的Δ < 阈值ε，则提前终止重排序。
   - **效果**：减少30% I/O和计算，精度损失<1%。

3. **冗余感知I/O去重（Redundancy-aware I/O Deduplication）**  
   - **优化策略**：
     - **存储布局**：相似向量紧凑存储（按聚类中心分桶）。
     - **去重机制**：
       - **批内合并**：同SSD页的I/O请求合并为单次读取。
       - **批间复用**：DRAM缓存复用已加载SSD页。
   - **解决痛点**：原始向量（128-384B）远小于SSD页（4KB），消除读放大。

---

### 性能突破（对比SOTA系统）
| 对比项          | vs. SSD方案 (SPANN)                        | vs. GPU内存方案 (RUMMY)  |
| ------------ | ---------------------------------------- | -------------------- |
| **吞吐量(QPS)** | ↑ **9.4–13.1倍**                          | ↑ **2.4–4.9倍**       |
| **成本效率**     | ↑ **5.7–8.8倍** (QPS/$)   | ↑ **2.3–6.8倍** (QPS/$) |                      |
| **内存效率**     | ↑ **13.1倍** (QPS/GB)                     | ↑ **32.4倍** (QPS/GB) |
| **硬件需求**     | 单GPU (如V100) + SSD                       | 需TB级内存 + 高端GPU       |

---

### 解决的核心挑战
| 挑战                   | 解决方案           | 关键效果            |
| -------------------- | -------------- | --------------- |
| GPU显存不足 → 频繁数据交换     | 多层级索引 + 仅传向量ID | 消除CPU-GPU数据传输瓶颈 |
| PQ压缩导致精度损失 → 需重排序    | 动态启发式重排序       | 最小化I/O+计算，保精度   |
| SSD小粒度I/O效率低 → 读放大严重 | 存储布局优化 + I/O去重 | 减少23% I/O操作     |

---
























### Reference
[1] [FusionANNS: An Efficient CPU/GPU Cooperative Processing Architecture for Billion-scale Approximate Nearest Neighbor Search](https://arxiv.org/html/2409.16576v1#bib.bib14)
[2] [FAST 25' FusionANNS (CPU+GPU+SSD)](https://zhuanlan.zhihu.com/p/1886009173860415240)