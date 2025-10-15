## 4. GPU加速

---
- **Faiss GPU环境配置**：详细说明Faiss GPU版的安装方法，包括Cond安装、源码编译和Docker方式，并提供验证方法（表格展示）。
- **Faiss GPU索引类型详解**：分类介绍Faiss GPU支持的各种索引类型，包括Flat索引、量化索引和层次导航图索引，并使用表格对比特性。
- **Faiss GPU实战代码解析**：通过完整的代码示例演示Faiss GPU的使用流程，包括数据准备、索引构建、搜索操作和结果分析（含代码示例）。
- **Faiss GPU性能优化技巧**：提供显存管理、并行计算和索引选择等方面的优化策略，帮助提升搜索性能（含表格总结）。
- **Faiss GPU技术前沿与总结**：介绍Faiss的最新发展和未来趋势，总结GPU加速的价值和应用建议。


### 4.1 安装准备

在安装Faiss GPU版本之前，需要确保系统满足基本的硬件和软件要求。首先，必须具备**NVIDIA显卡**，并安装合适版本的**CUDA工具包**。Faiss通常支持CUDA 11.4及以上版本，具体版本要求可能因Faiss发行版而异。其次，需要安装对应版本的**NVIDIA驱动**，建议使用较新的驱动版本以保证兼容性。软件方面，需要**Python环境**（通常为3.6及以上版本）和基本的科学计算库如numpy。

验证系统是否符合安装要求的方法很简单。对于GPU环境，可以使用`nvidia-smi`命令查看显卡信息和CUDA版本。对于Python环境，可以使用`python --version`检查Python版本，并使用`import numpy`验证numpy是否可用。如果这些基础依赖项没有问题，就可以开始安装Faiss GPU版本了。

### 4.2 安装方法

Faiss GPU版本有多种安装方式，根据用户的需求和技术背景可以选择最适合的方法。对于大多数用户，推荐使用**Conda安装**，这种方法简单快捷，能自动解决大部分依赖问题。对于有特殊需求的高级用户，可以考虑**源码编译**安装，以便进行特定优化或自定义功能。此外，还可以通过**Docker方式**安装，这种方式能提供完全隔离的环境，避免系统污染。

**pip安装**
上文提到的conda，适合于本地，但是本教程为了让学习者更好的操作，选择使用魔搭社区中GPU的noteBook上通过pip安装Faiss。
我使用的GPU环境为ubunitu22.04-cuda12.1.0-py311-torch2.3.1
```bash
pip install faiss-gpu-cu12
```
> 往往cpu和gpu版本的库不能共存，如果你下载了CPU版本的Faiss，那么使用GPU版本之前需要先卸载cpu版本的，才能下载gpu版本的



**Conda安装**是最简单的方法，只需执行以下命令即可：

```bash
# 安装基础GPU版本（包含CUDA支持）
conda install -c pytorch -c nvidia faiss-gpu=1.12.0

# 或者安装NVIDIA cuVS加速版（需要CUDA 12.4+）
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.12.0
```

Conda安装方式会自动处理所有依赖关系，包括CUDA工具包和必要的库文件。安装完成后，可以通过Python接口验证是否安装成功。

对于需要定制化功能或性能优化的用户，**源码编译**是更好的选择。源码编译虽然过程复杂，但可以针对特定硬件架构进行优化，如启用AVX2或AVX512指令集加速。以下是基本的源码编译步骤：

```bash
# 1. 克隆Faiss源码
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# 2. 配置编译环境（以CUDA 12.1为例）
cmake -B build . \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
  -DCMAKE_CUDA_ARCHITECTURES="75;86" \
  -DFAISS_OPT_LEVEL=avx2

# 3. 编译Faiss
make -C build -j faiss_gpu

# 4. 安装Python绑定
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
```

源码编译过程中，关键配置项包括`-DFAISS_ENABLE_GPU=ON`（启用GPU支持）、`-DCUDAToolkit_ROOT`（指定CUDA路径）和`-DCMAKE_CUDA_ARCHITECTURES`（指定GPU架构）。用户应根据自己的GPU架构调整这些参数，以获得最佳性能。

### 4.3 环境验证

安装完成后，需要验证Faiss GPU是否正常工作。可以通过简单的Python脚本来测试基础功能和GPU加速效果：

```python
import numpy as np
import faiss

# 测试GPU资源是否可用
res = faiss.StandardGpuResources()
print("GPU资源初始化成功")

# 创建测试数据
d = 128  # 向量维度
nb = 10000  # 数据库大小
nq = 100  # 查询数量
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 创建GPU索引
index = faiss.IndexFlatL2(d)  # 创建CPU索引
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 转换为GPU索引

# 添加数据并搜索
gpu_index.add(xb)
k = 4  # 返回最近邻数量
D, I = gpu_index.search(xq, k)

print("搜索完成，前5个查询结果：")
print("索引:", I[:5])
print("距离:", D[:5])
```

如果脚本正常运行并输出合理结果，说明Faiss GPU安装成功。此外，还可以通过比较CPU和GPU版本的搜索速度来直观感受性能提升。

## 5 Faiss GPU索引类型详解

### 5.1 索引分类与选择策略

Faiss提供了丰富多样的索引类型，每种索引都有其特定的适用场景和性能特点。根据索引结构和算法的不同，Faiss GPU索引大致可以分为三类：**Flat索引**、**量化索引**和**层次导航图索引**。Flat索引提供精确搜索，保证结果准确但内存消耗较大；量化索引采用向量压缩技术，显著减少内存占用但会损失一定精度；层次导航图索引基于图结构构建搜索路径，在保证较高准确率的同时提供快速的搜索速度。

选择合适的索引类型需要考虑多个因素，包括数据规模、向量维度、精度要求、搜索速度需求和硬件资源等。对于小规模数据集（如百万级别以下），Flat索引通常是理想选择，因为它能提供精确结果且实现简单。对于中到大规規数据集（百万到十亿级别），量化索引如IVF-PQ和层次导航图索引如HNSW更能平衡资源使用和搜索性能。此外，还需要考虑索引的训练需求，有些索引类型需要额外的训练阶段来构建内部数据结构。

### 5.2 Flat Indexes

Flat索引是最简单直接的索引类型，它**不加转换**地存储原始向量，并通过暴力比对的方式计算查询向量与数据库中所有向量的距离。Faiss GPU支持多种Flat索引，包括`GpuIndexFlatL2`（基于L2距离）、`GpuIndexFlatIP`（基于点积）和`GpuIndexFlatCos`（基于余弦相似度）。这些索引虽然简单，但在许多场景下非常实用，特别是当数据规模不大或需要精确结果时。

Flat索引的主要优点是**100%准确率**，因为它不采用任何近似算法。同时，由于不需要训练过程，Flat索引可以立即添加数据并进行搜索。然而，它的明显缺点是搜索速度与数据量成**线性关系**，当数据量很大时，搜索性能会显著下降在CPU下使用单线程或多线程顺序计算
在GPU下并行计算所有向量对的距离，利用数千个CUDA核心同时计算。此外，Flat索引的内存占用也最高，因为它需要存储所有原始向量。

**使用场景**：小到中等数据集、需要100%精度、基准测试和精度验证
以下是Flat索引的使用示例：

```python
import faiss
import numpy as np
import time

def benchmark_flat_l2():
    dimension = 768
    n_vectors = 100000
    n_queries = 1000
    k = 10
    
    np.random.seed(42)
    database = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    
    # CPU 版本
    cpu_index = faiss.IndexFlatL2(dimension)
    
    start_time = time.time()
    cpu_index.add(database)
    cpu_add_time = time.time() - start_time
    
    start_time = time.time()
    cpu_distances, cpu_indices = cpu_index.search(queries, k)
    cpu_search_time = time.time() - start_time
    
    # GPU 版本
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))
    
    start_time = time.time()
    gpu_index.add(database)
    gpu_add_time = time.time() - start_time
    
    start_time = time.time()
    gpu_distances, gpu_indices = gpu_index.search(queries, k)
    gpu_search_time = time.time() - start_time
    
    # 结果对比
    print(f"{'指标':<15} {'CPU':<12} {'GPU':<12} {'加速比':<10}")
    print("-" * 50)
    print(f"{'添加时间(s)':<15} {cpu_add_time:<12.4f} {gpu_add_time:<12.4f} {cpu_add_time/gpu_add_time:.2f}x")
    print(f"{'搜索时间(s)':<15} {cpu_search_time:<12.4f} {gpu_search_time:<12.4f} {cpu_search_time/gpu_search_time:.2f}x")
    # 修正 QPS 加速比的计算，应该用 GPU QPS 除以 CPU QPS
    print(f"{'QPS':<15} {n_queries/cpu_search_time:<12.1f} {n_queries/gpu_search_time:<12.1f} {(n_queries/gpu_search_time)/(n_queries/cpu_search_time):.2f}x")
    
    # 验证结果一致性
    accuracy = np.mean(cpu_indices == gpu_indices)
    print(f"\n结果一致性: {accuracy:.4f}")

benchmark_flat_l2()
```
**IndexIVFFlat**
**IndexIVFFlat** 是一种基于**倒排文件系统（Inverted File System）**的索引类型，它将向量空间划分为多个**Voronoi 单元**，并使用**乘积量化（Product Quantization）**对每个单元进行量化；具体来说，其搜索分为两个阶段：
1. 粗量化：将向量分配到最近的聚类中心
2. 精细搜索：在选定的聚类中进行精确搜索
在CPU、GPU两个场景下：

CPU：顺序处理聚类，GPU：并行处理多个聚类、同时搜索多个查询

**使用场景**：大规模数据集、平衡精度和速度的场景、需要可调节精度、速度权衡的应用

示例代码如下：
```python 
def benchmark_ivf_flat():
    
    dimension = 768
    n_vectors = 500000
    n_queries = 1000
    k = 10
    nlist = 100  # 聚类数量
    
    np.random.seed(42)
    database = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    
    # 测试不同 nprobe 参数
    nprobe_values = [1, 10, 50]
    
    for nprobe in nprobe_values:
        print(f"\n测试 nprobe = {nprobe}")
        
        # CPU 版本
        quantizer_cpu = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFFlat(quantizer_cpu, dimension, nlist)
        
        cpu_index.train(database)
        cpu_index.add(database)
        cpu_index.nprobe = nprobe
        
        start_time = time.time()
        cpu_distances, cpu_indices = cpu_index.search(queries, k)
        cpu_search_time = time.time() - start_time
        
        # GPU 版本
        res = faiss.StandardGpuResources()
        quantizer_gpu = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))
        gpu_index = faiss.GpuIndexIVFFlat(res, dimension, nlist, faiss.METRIC_L2)
        
        gpu_index.train(database)
        gpu_index.add(database)
        gpu_index.nprobe = nprobe
        
        start_time = time.time()
        gpu_distances, gpu_indices = gpu_index.search(queries, k)
        gpu_search_time = time.time() - start_time
        
        # 结果对比
        print(f"{'指标':<15} {'CPU':<12} {'GPU':<12} {'加速比':<10}")
        print("-" * 50)
        print(f"{'搜索时间(s)':<15} {cpu_search_time:<12.4f} {gpu_search_time:<12.4f} {cpu_search_time/gpu_search_time:.2f}x")
        print(f"{'QPS':<15} {n_queries/cpu_search_time:<12.1f} {n_queries/gpu_search_time:<12.1f} {(n_queries/gpu_search_time)/(n_queries/cpu_search_time):.2f}x")

benchmark_ivf_flat()
```
### 5.3 量化索引（Quantized Indexes）
> 再次强调，Faiss Chapter中的除了GPU的部分外，其他大部分的技术细节或者原理都在Milvus章节有详细的介绍，例如此处的量化原理。
量化索引是Faiss中处理大规模数据的关键技术，它通过**向量压缩**技术显著减少内存占用，从而支持十亿级别向量的搜索。量化索引的核心思想是将高维向量空间划分为多个子空间，并对每个子空间进行聚类量化，用压缩编码表示原始向量。Faiss GPU支持多种量化索引，其中最常用的是**IVF-PQ**（Inverted File System with Product Quantization）索引。

IVF-PQ索引结合了两种压缩技术：倒排文件系统（IVF）和乘积量化（PQ）。IVF通过**聚类**将向量空间划分为多个 Voronoi 单元，搜索时只需查询少数几个相关单元，大幅减少计算量。PQ则将高维向量分解为多个子向量，并对每个子向量进行独立量化，进一步压缩向量表示。这两种技术结合使IVF-PQ能在保持较高搜索准确率的同时，大幅提升搜索速度和减少内存占用。

**使用场景**：超大规模数据集、内存受限、需要内存效率的移动端或者边缘计算。

以下是IVF-PQ索引的创建和使用示例：

```python
def benchmark_ivf_pq():
    
    dimension = 768
    n_vectors = 1000000  # 1M 向量
    n_queries = 1000
    k = 10
    nlist = 100
    m = 8    # 子量化器数量
    bits = 8 # 每个量化器的比特数
    
    # 生成更大的数据集
    np.random.seed(42)
    database = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    
    print(f"数据集: {n_vectors} 个 {dimension} 维向量")
    print(f"PQ 配置: {m} 个子量化器, {bits} 比特")
    
    # CPU 版本
    quantizer_cpu = faiss.IndexFlatL2(dimension)
    cpu_index = faiss.IndexIVFPQ(quantizer_cpu, dimension, nlist, m, bits)
    
    start_time = time.time()
    cpu_index.train(database)
    cpu_train_time = time.time() - start_time
    
    start_time = time.time()
    cpu_index.add(database)
    cpu_add_time = time.time() - start_time
    
    cpu_index.nprobe = 10
    
    start_time = time.time()
    cpu_distances, cpu_indices = cpu_index.search(queries, k)
    cpu_search_time = time.time() - start_time
    
    # GPU 版本
    res = faiss.StandardGpuResources()
    
    # 注意: GPU IVFPQ 需要特定的配置
    config = faiss.GpuIndexIVFPQConfig()
    config.device = 0
    
    quantizer_gpu = faiss.GpuIndexFlatL2(res, dimension)
    gpu_index = faiss.GpuIndexIVFPQ(res, dimension, nlist, m, bits, faiss.METRIC_L2, config)
    
    start_time = time.time()
    gpu_index.train(database)
    gpu_train_time = time.time() - start_time
    
    start_time = time.time()
    gpu_index.add(database)
    gpu_add_time = time.time() - start_time
    
    gpu_index.nprobe = 10
    
    start_time = time.time()
    gpu_distances, gpu_indices = gpu_index.search(queries, k)
    gpu_search_time = time.time() - start_time
    
    # 内存使用估算：修正原代码中获取向量大小的方法，避免 AttributeError
    # 对于 IndexIVFPQ，我们可以通过计算大致估算内存使用
    # 这里估算每个向量的存储大小为 m * (bits/8) 字节
    cpu_memory = (n_vectors * m * (bits / 8)) / 1024**2
    print(f"\n内存使用估算: CPU ~{cpu_memory:.1f} MB")
    
    # 结果对比
    print(f"{'指标':<15} {'CPU':<12} {'GPU':<12} {'加速比':<10}")
    print("-" * 50)
    print(f"{'训练时间(s)':<15} {cpu_train_time:<12.4f} {gpu_train_time:<12.4f} {cpu_train_time/gpu_train_time:.2f}x")
    print(f"{'添加时间(s)':<15} {cpu_add_time:<12.4f} {gpu_add_time:<12.4f} {cpu_add_time/gpu_add_time:.2f}x")
    print(f"{'搜索时间(s)':<15} {cpu_search_time:<12.4f} {gpu_search_time:<12.4f} {cpu_search_time/gpu_search_time:.2f}x")
    # 修正 QPS 的加速比计算逻辑，应该是 GPU_QPS / CPU_QPS
    cpu_qps = n_queries / cpu_search_time
    gpu_qps = n_queries / gpu_search_time
    print(f"{'QPS':<15} {cpu_qps:<12.1f} {gpu_qps:<12.1f} {gpu_qps/cpu_qps:.2f}x")

benchmark_ivf_pq()
```

除了IVF-PQ，Faiss还支持其他量化方法，如**标量量化**（Scalar Quantization）和**残差量化**（Residual Quantization）。近年来，Faiss还引入了**RaBitQ**技术，这是一种基于优化的量化方法，能提供更好的准确率-压缩率平衡。RaBitQ在传统乘积量化的基础上，进一步优化了编码和距离计算方式，提升了检索的准确率和速度。

### 5.3.1 标量量化索引
Faiss对GPU上的标量量化支持有限，很多标量量化类型在GPU上没有原生实现，可以通过index_cpu_to_gpu()方法将CPU标量量化索引转移到GPU上，但性能提升有限，对于生产环境，建议使用IVFFlat、IVFPQ等GPU支持更好的索引类型。

下面将使用IndexScalarQuantizer使用标量量化：
1. 将每个向量维度独立量化为低精度表示
2. 大幅度减少内存使用，同时保持合理的精度
**使用场景**：内存敏感、精度要求不高
以下是一个使用cpu标量量化索引的示例：

```python 
def benchmark_scalar_quantizer():
    dimension = 256  
    n_vectors = 100000
    n_queries = 1000
    k = 10
    
    np.random.seed(42)
    database = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    
    
    quantizer_types = [
        (faiss.ScalarQuantizer.QT_8bit, "8-bit"),
        (faiss.ScalarQuantizer.QT_6bit, "6-bit"),
        (faiss.ScalarQuantizer.QT_4bit, "4-bit"),
    ]
    
    results = []
    
    for qtype, qname in quantizer_types:
        print(f"\n测试 {qname} 量化:")
        
        try:
            # CPU 版本 
            nlist = 100
            quantizer = faiss.IndexFlatL2(dimension)
            cpu_index = faiss.IndexIVFScalarQuantizer(quantizer, dimension, nlist, qtype, faiss.METRIC_L2)
            
            # 训练索引
            cpu_index.train(database)
            
            start_time = time.time()
            cpu_index.add(database)
            cpu_add_time = time.time() - start_time
            
            cpu_index.nprobe = 10
            
            start_time = time.time()
            cpu_distances, cpu_indices = cpu_index.search(queries, k)
            cpu_search_time = time.time() - start_time
            
            # 内存使用估算
            if hasattr(cpu_index, 'invlists') and hasattr(cpu_index.invlists, 'vector_size'):
                memory_usage = (cpu_index.invlists.vector_size * cpu_index.invlists.size) / 1024**2
            else:
                # 简单估算
                if qtype == faiss.ScalarQuantizer.QT_8bit:
                    memory_usage = n_vectors * dimension * 1 / 1024**2
                elif qtype == faiss.ScalarQuantizer.QT_6bit:
                    memory_usage = n_vectors * dimension * 0.75 / 1024**2
                else:  # 4bit
                    memory_usage = n_vectors * dimension * 0.5 / 1024**2
            
            # 与原始浮点数的内存对比
            original_memory = n_vectors * dimension * 4 / 1024**2  # float32
            compression_ratio = original_memory / memory_usage
            
            results.append({
                'name': qname,
                'search_time': cpu_search_time,
                'add_time': cpu_add_time,
                'qps': n_queries / cpu_search_time,
                'memory_usage': memory_usage,
                'compression_ratio': compression_ratio
            })
            
            print(f"搜索时间: {cpu_search_time:.4f}s, QPS: {n_queries/cpu_search_time:.1f}")
            print(f"内存使用: {memory_usage:.1f} MB, 压缩比: {compression_ratio:.1f}x")
            
        except Exception as e:
            print(f"创建 {qname} 量化索引失败: {e}")
            continue
    
    print(f"\n对比基准 - FlatL2:")
    cpu_index_flat = faiss.IndexFlatL2(dimension)
    
    start_time = time.time()
    cpu_index_flat.add(database)
    flat_add_time = time.time() - start_time
    
    start_time = time.time()
    flat_distances, flat_indices = cpu_index_flat.search(queries, k)
    flat_search_time = time.time() - start_time
    
    flat_memory = n_vectors * dimension * 4 / 1024**2
    
    results.append({
        'name': 'FlatL2',
        'search_time': flat_search_time,
        'add_time': flat_add_time,
        'qps': n_queries / flat_search_time,
        'memory_usage': flat_memory,
        'compression_ratio': 1.0
    })
    
    print(f"搜索时间: {flat_search_time:.4f}s, QPS: {n_queries/flat_search_time:.1f}")
    print(f"内存使用: {flat_memory:.1f} MB")
    
    # 输出对比结果
    print(f"\n{'量化类型':<10} {'搜索时间(s)':<12} {'QPS':<10} {'内存(MB)':<12} {'压缩比':<10}")
    print("-" * 65)
    for result in results:
        print(f"{result['name']:<10}     {result['search_time']:<12.4f}   {result['qps']:<10.1f}    {result['memory_usage']:<12.1f}   {result['compression_ratio']:<10.1f}")

benchmark_scalar_quantizer()
```
下面我们尝试转移到GPU上，看一下查询速度的提升
```python
def benchmark_gpu_scalar_quantizer():
    print("\n" + "=" * 60)
    print("GPU 标量量化尝试")
    print("=" * 60)
    
    dimension = 128
    n_vectors = 50000
    n_queries = 500
    
    np.random.seed(42)
    database = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    
    try:
        # 在CPU上创建标量量化索引
        nlist = 50
        quantizer_cpu = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFScalarQuantizer(quantizer_cpu, dimension, nlist, 
                                                 faiss.ScalarQuantizer.QT_8bit, 
                                                 faiss.METRIC_L2)
        cpu_index.train(database)
        cpu_index.add(database)
        cpu_index.nprobe = 10
        
        # 测试CPU性能
        start_time = time.time()
        cpu_distances, cpu_indices = cpu_index.search(queries, 10)
        cpu_time = time.time() - start_time
        
        # 尝试转移到GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        # 测试GPU性能
        start_time = time.time()
        gpu_distances, gpu_indices = gpu_index.search(queries, 10)
        gpu_time = time.time() - start_time
        
        print(f"CPU 搜索时间: {cpu_time:.4f}s")
        print(f"GPU 搜索时间: {gpu_time:.4f}s")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"GPU标量量化测试失败: {e}")
        print("Faiss对GPU标量量化的直接支持有限，建议使用IVFFlat或IVFPQ")

benchmark_gpu_scalar_quantizer()
```
### 5.4 层次导航图索引（HNSW Indexes）

层次可导航小世界（HNSW）索引是基于图结构的近似最近邻搜索算法，它通过构建一个**多层次**的图结构来加速搜索过程。HNSW索引结合了可导航小世界（NSW）图和跳表的思想，能够在高维空间中快速找到近似最近邻。Faiss GPU貌似不支持HNSW索引。
HNSW索引在Faiss中是通过IndexHNSW类实现的，属于CPU-only的索引类型，可以尝试使用`faiss.index_cpu_to_gpu(res, 0, cpu_index)`将HNSW索引转移到GPU上，但由于底层没有对应的GPU kernel，所以会失败。
HNSW索引的主要优点包括：**高搜索速度**、**高准确率**和**无需训练**。与量化索引不同，HNSW不需要单独的训练阶段，可以直接添加数据并构建索引。HNSW通过启发式方法构建图结构，确保图中节点具有良好的连接性，使得搜索过程可以快速收敛到最近邻区域。HNSW索引的性能主要通过`efConstruction`（控制图构建质量）和`efSearch`（控制搜索深度）参数调节。

**使用场景**：高召回率要求、动态数据集、支持增量添加、中等规模的高维数据。

先执行以下代码，检查当前Faiss版本是否支持GPU原生HNSW索引

```python
def benchmark_hnsw_gpu_native():
    """测试原生 GPU HNSW 支持"""

    dimension = 128
    n_vectors = 50000
    n_queries = 1000
    k = 10
    
    np.random.seed(42)
    database = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    
    # 参数设置
    M = 16  # 每个节点的连接数
    efConstruction = 200
    efSearch = 100
    
    try:
        # 尝试创建原生 GPU HNSW
        res = faiss.StandardGpuResources()
        
        config = faiss.GpuIndexHNSWFlatConfig()
        config.device = 0
        config.efConstruction = efConstruction
        config.efSearch = efSearch
        
        gpu_index = faiss.GpuIndexHNSWFlat(res, dimension, M, config)
        
        print("使用原生 GpuIndexHNSWFlat...")
        gpu_index.add(database)
        
        start_time = time.time()
        gpu_distances, gpu_indices = gpu_index.search(queries, k)
        gpu_time = time.time() - start_time
        
        print(f"GPU HNSW 搜索时间: {gpu_time:.4f}s")
        
    except AttributeError:
        print("当前 Faiss 版本不支持 GpuIndexHNSWFlat")
        return None
    
    return gpu_time

benchmark_hnsw_gpu_native()
```
### 5.5 总结
| 索引类型             | CPU优势            | GPU优势                 | 内存效率 | 精度     |
| -------------------- | ------------------ | ----------------------- | -------- | -------- |
| IndexFlatL2          | 小数据集，精确搜索 | 5 - 50x加速，大数据集   | 低       | 100%     |
| IndexIVFFlat         | 可调节精度         | 10 - 100x加速，并行聚类 | 中       | 95 - 99% |
| IndexIVFPQ           | 内存敏感场景       | 5 - 20x加速，批量量化   | 高       | 90 - 98% |
| IndexHNSW            | 高召回率           | 1 - 3x加速，有限并行    | 中       | 98 - 99% |
| IndexScalarQuantizer | 极致内存优化       | 3 - 10x加速，并行解压   | 极高     | 85 - 95% |


通过运行上文的代码，你可以发现:
1. Flat索引在GPU上加速最明显，适合精确搜索
2. IVF系列在GPU上表现优异，适合大规模数据
3. 量化索引在GPU上仍有不错加速，同时大幅节省内存
4. 图基索引(HNSW)适合CPU
5. 数据规模越大，GPU优势越明显

