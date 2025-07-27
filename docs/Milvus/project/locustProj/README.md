# Milvus Locust 性能测试实用指南

## 目录
- [Locust 实用功能](#locust-实用功能)
- [项目结构详解](#项目结构详解)
- [代码逐行解析](#代码逐行解析)
- [测试指标怎么看](#测试指标怎么看)
- [一步步跑起来](#一步步跑起来)
- [性能调优小技巧](#性能调优小技巧)
- [测试场景怎么设计](#测试场景怎么设计)


## Locust 实用功能

### 事件钩子：测试前后的准备与清理
Locust 可以在测试开始、结束等关键节点自动执行一些操作，比如初始化连接、统计结果。咱们提供的代码里就有这样的“钩子”，很实用：

```python
# 来自 locustfile.py
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时的准备工作"""
    logger.info("=" * 50)
    logger.info("🚀 Milvus性能测试开始")
    logger.info(f"目标: localhost:19530/default")
    logger.info(f"集合: locust_test_collection")
    logger.info("=" * 50)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时的清理工作"""
    # 统计总请求数、失败数、平均响应时间等
    if environment.stats and environment.stats.total:
        total = environment.stats.total
        logger.info(f"总请求数: {total.num_requests}")
        logger.info(f"失败请求数: {total.num_failures}")
        logger.info(f"平均响应时间: {total.avg_response_time:.2f}ms")
    # 断开Milvus连接
    if _connection_initialized and connections.has_connection(_shared_connection):
        connections.disconnect(_shared_connection)
```

这些钩子的作用很简单：测试开始时打印基本信息，确保连接正确；测试结束时统计结果并断开连接，避免资源浪费。


### 自定义用户行为：模拟真实操作
在 `locustfile.py` 里，`MilvusUser` 类定义了用户会做什么操作。比如搜索向量，这都是实际会用到的功能：

```python
# 来自 locustfile.py
class MilvusUser(User):
    # 每次操作后等待0.5-2秒，模拟真实用户操作间隔
    wait_time = between(0.5, 2.0)

    def on_start(self):
        # 每个用户启动时，确保连接已初始化
        init_shared_connection()
        self.collection = _shared_collection  # 用共享的集合对象
        self.dimension = _shared_dimension    # 向量维度

    def generate_random_vector(self):
        # 生成随机向量并归一化（和真实查询向量类似）
        vector = np.random.normal(0, 1, self.dimension).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    @task(10)  # 权重10，最常执行
    def single_vector_search(self):
        # 单向量搜索（最常见的操作）
        self._perform_search("single_search", batch_size=1, top_k=10, nprobe=16)

    @task(5)  # 权重5，执行频率次之
    def batch_vector_search(self):
        # 批量搜索（一次查2-5个向量）
        batch_size = random.randint(2, 5)
        self._perform_search("batch_search", batch_size=batch_size, top_k=10, nprobe=16)
```

这里的 `@task(10)` 表示这个操作的频率，数字越大，执行得越频繁。单向量搜索权重10，批量搜索5，符合实际使用中“单个查询更多”的情况。


## 项目架构深入解析

### 实际项目结构
根据提供的文件，项目结构很简单，没有复杂的目录，适合新手理解：

```
locustProj/
├── locustfile.py      # 核心测试代码（定义用户行为、连接Milvus）
├── run_test.py        # 启动脚本（自动找端口、打开浏览器）
├── start_test.py      # 简化启动器（和run_test类似，更简单）
├── setup_test_data.py # 生成测试数据（创建集合、插入向量）
├── requirements.txt   # 依赖列表（需要安装的库）
└── README.md          # 说明文档（怎么安装、怎么运行）
```

每个文件的作用：
- `locustfile.py`：测试的核心，告诉Locust要模拟什么操作。
- `run_test.py` 和 `start_test.py`：帮你启动测试，不用记复杂命令。
- `setup_test_data.py`：生成测试用的向量数据，不然Milvus里没数据可查。
- `requirements.txt`：列出需要安装的Python库（比如Locust、pymilvus）。


## 代码深度解析

### 连接Milvus的关键代码
测试前必须连接到Milvus，`locustfile.py` 里的 `init_shared_connection` 函数就是干这个的：

```python
# 来自 locustfile.py
def init_shared_connection():
    # 全局变量，确保所有用户共享一个连接（节省资源）
    global _connection_initialized, _shared_collection, _shared_dimension
    
    with _connection_lock:  # 加锁，避免多线程同时初始化
        if _connection_initialized:
            return  # 已经初始化过，直接返回
        
        try:
            # 连接到本地Milvus（localhost:19530）
            connections.connect(
                alias=_shared_connection,
                host="localhost",
                port="19530",
                timeout=10
            )
            
            # 检查有没有叫"locust_test_collection"的集合
            if not utility.has_collection("locust_test_collection", using=_shared_connection):
                raise Exception("集合 'locust_test_collection' 不存在，请先运行setup_test_data.py")
            
            # 获取集合对象，后续搜索要用
            _shared_collection = Collection("locust_test_collection", using=_shared_connection)
            
            # 确定向量维度（比如256维）
            schema = _shared_collection.schema
            for field in schema.fields:
                if field.name == "vector":
                    _shared_dimension = field.params.get('dim', 256)
                    break
            
            _connection_initialized = True
            print(f"连接成功！向量维度: {_shared_dimension}")
            
        except Exception as e:
            print(f"连接失败: {e}")
            raise  # 失败就报错，让用户知道
```

为什么要“共享连接”？因为如果每个用户都新建一个连接，Milvus可能扛不住，共享连接更符合实际场景（多个用户用同一个连接池）。


### 生成测试数据的代码
没有数据的话，搜索操作就没意义。`setup_test_data.py` 可以帮你生成数据：

```python
# 来自 setup_test_data.py
def create_test_collection(collection_name="locust_test_collection", 
                          dimension=256, 
                          num_vectors=100000,
                          batch_size=5000):
    # 连接Milvus
    connections.connect("default", host="localhost", port="19530")
    
    # 如果集合已存在，先删除（避免数据混乱）
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除旧集合: {collection_name}")
    
    # 定义集合结构（有哪些字段）
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),  # 向量字段
        FieldSchema(name="category", dtype=DataType.INT64),  # 分类字段（可选，用于过滤）
    ]
    schema = CollectionSchema(fields, f"测试集合，维度: {dimension}")
    collection = Collection(collection_name, schema)
    
    # 生成10万个向量（用聚类生成，更接近真实数据分布）
    print(f"生成 {num_vectors} 个 {dimension} 维向量...")
    vectors, labels = make_blobs(
        n_samples=num_vectors,
        centers=20,  # 20个聚类中心，让向量有聚簇性
        n_features=dimension,
        random_state=42  # 固定随机数种子，每次生成的 data 一样
    )
    
    # 归一化向量（很多实际场景中向量都是归一化的）
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    vectors = vectors.astype(np.float32)
    
    # 批量插入数据（一次插5000个，避免内存不够）
    print("开始插入数据...")
    for i in range(0, num_vectors, batch_size):
        batch_end = min(i + batch_size, num_vectors)
        batch_vectors = vectors[i:batch_end].tolist()
        collection.insert([batch_vectors])  # 插入向量
    
    # 创建索引（让搜索更快）
    index_params = {
        "metric_type": "L2",  # 用L2距离（欧氏距离）
        "index_type": "IVF_FLAT",  # 常用的索引类型
        "params": {"nlist": 2048}  # 索引参数，2048是比较合理的默认值
    }
    collection.create_index("vector", index_params)
    print("索引创建完成！测试数据准备好了")
```

这段代码的作用：
1. 连接Milvus，创建一个叫 `locust_test_collection` 的集合。
2. 生成10万个向量（带聚类，更像真实数据）。
3. 插入向量并创建索引（不然搜索会很慢）。


### 搜索操作的核心代码
`MilvusUser` 里的 `_perform_search` 方法实现了搜索的具体逻辑：

```python
# 来自 locustfile.py
def _perform_search(self, name, batch_size=1, top_k=10, nprobe=16):
    start_time = time.time()  # 记录开始时间（算响应时间用）
    exception = None
    result_count = 0
    
    try:
        # 生成要搜索的向量（比如批量搜索就生成多个）
        query_vectors = []
        for _ in range(batch_size):
            query_vectors.append(self.generate_random_vector())
        
        # 搜索参数（nprobe越大，搜索越准但越慢）
        search_params = {
            "metric_type": "L2",  # 用L2距离
            "params": {"nprobe": nprobe}
        }
        
        # 执行搜索
        results = self.collection.search(
            data=query_vectors,  # 要搜索的向量
            anns_field="vector",  # 搜索哪个字段（向量字段）
            param=search_params,
            limit=top_k,  # 返回前10个最像的
            timeout=30  # 30秒超时
        )
        
        # 统计结果数量
        result_count = sum(len(result) for result in results if result is not None)
        
    except Exception as e:
        exception = e  # 记录错误
        print(f"搜索失败: {e}")
    
    # 计算响应时间（毫秒）
    response_time = (time.time() - start_time) * 1000
    
    # 告诉Locust这次请求的结果（用于统计）
    events.request.fire(
        request_type="SEARCH",
        name=name,
        response_time=response_time,
        response_length=result_count,
        exception=exception
    )
```

这里的 `nprobe` 是个关键参数：值越大，搜索越精确，但速度越慢；值越小，越快但可能不准。测试时可以通过这个参数观察性能变化。


## 测试指标怎么看

### 关键指标解释
运行测试后，Locust的网页界面会显示几个重要指标，这些在 `README.md` 里也提到了：

| 指标名称       | 意思                          | 怎么看？                                  |
|---------------|-------------------------------|------------------------------------------|
| RPS           | 每秒请求数                    | 越大越好，说明Milvus处理能力强。          |
| Response Time | 响应时间（平均、P95、P99）     | 越小越好，P95表示95%的请求都比这个时间快。|
| Failures      | 失败率                        | 越低越好，0%最好，说明服务稳定。          |
| Users         | 当前模拟的用户数              | 用来观察不同并发下的性能（用户越多压力越大）。|

举个例子：如果RPS=50，说明Milvus每秒能处理50次搜索；P95响应时间=200ms，说明95%的搜索都能在200毫秒内返回。


### 指标之间的关系
- 当并发用户数增加时，RPS可能先增加（Milvus没满载），但超过某个点后会下降（Milvus扛不住了）。
- 响应时间通常随用户数增加而变长（Milvus忙不过来）。
- 失败率突然上升，可能是Milvus达到极限了（比如内存不够、CPU跑满）。


## 一步步跑起来

### 1. 安装依赖
首先要安装需要的库，打开终端，进入项目目录，运行：

```bash
pip install -r requirements.txt
```

`requirements.txt` 里的内容很简单：
```
locust>=2.14.0  # 性能测试工具
pymilvus>=2.3.0  # 操作Milvus的库
numpy>=1.21.0    # 生成向量用的库
```


### 2. 准备测试数据
Milvus里必须有数据才能测试，运行 `setup_test_data.py` 生成数据：

```bash
python setup_test_data.py
```

这个脚本会：
- 连接到本地Milvus（确保Milvus已经启动）。
- 创建 `locust_test_collection` 集合。
- 生成10万个256维向量并插入。
- 创建索引（让搜索更快）。

如果想改向量数量或维度，可以加参数：
```bash
# 生成5万个128维向量
python setup_test_data.py --num_vectors 50000 --dimension 128
```


### 3. 启动测试
有三种启动方式（`README.md` 里有详细说明）：

#### 方式一：用 `start_test.py`（推荐新手）
```bash
python start_test.py
```

这个脚本会自动找一个可用的端口（比如8090），并打开浏览器，直接进入Locust的网页界面。

#### 方式二：用 `run_test.py`
```bash
python run_test.py
```

和 `start_test.py` 类似，功能更全一点（比如更详细的启动信息）。

#### 方式三：直接用Locust命令
如果想自己指定端口，可以用：
```bash
locust -f locustfile.py --web-port 8089 --host http://localhost:19530
```

`--web-port 8089` 表示网页界面在8089端口，访问 `http://localhost:8089` 就能看到。


### 4. 设置测试参数
在Locust的网页界面，需要设置两个参数：
- **Number of users**：总共要模拟多少用户（建议从5开始，慢慢增加）。
- **Spawn rate**：每秒启动多少个用户（建议1-2，启动太快Milvus可能直接崩）。

设置好后点“Start swarming”就开始测试了。
> PS:目前代码还有点问题，是关于Pymilvus的连接复用问题，所以，当你启动项目后，看到网页中没有任何的动态变化，那么请你返回到终端中，手动运行一次Ctrl+C就可以了。

## 性能调优小技巧

### 遇到问题怎么办？
`README.md` 里提到了一些常见问题的解决方法：

1. **连接失败**：
   - 检查Milvus是不是真的启动了（在 `localhost:19530`）。
   - 有没有运行 `setup_test_data.py` 创建 `locust_test_collection` 集合。

2. **端口被占用**：
   - 用 `run_test.py` 或 `start_test.py` 会自动找可用端口。
   - 手动指定端口：`locust -f locustfile.py --web-port 8090`（换个端口号）。

3. 测试时响应时间太长、失败率高：
   - 减少并发用户数（比如从5降到3）。
   - 降低Spawn rate（每秒少启动几个用户）。
   - 检查Milvus服务器的资源（CPU、内存是不是快满了）。


### 优化Milvus的小建议
根据 `setup_test_data.py` 里的索引设置，可以简单调整来优化性能：

- **索引类型**：脚本里用的是 `IVF_FLAT`，这是最常用的索引（平衡速度和精度）。如果数据量特别大，可以试试 `IVF_SQ8`（占用内存小，速度更快，但精度略低）。
- **nlist参数**：创建索引时的 `nlist=2048`，这个值约等于 `sqrt(数据量)` 比较合适（比如10万数据，sqrt(10万)≈316，2048比它大，适合更大的数据量）。
- **nprobe参数**：搜索时的 `nprobe=16`，如果想更快，改小一点（比如8）；如果想更准，改大一点（比如32）。


## 测试场景怎么设计

### 内置的4种测试场景
`locustfile.py` 里定义了4种场景，覆盖了常见的搜索操作：

1. **单向量搜索**（权重10）：最常见的情况，一次搜1个向量，占比最高。
2. **批量向量搜索**（权重5）：一次搜2-5个向量，比单向量搜索少一点。
3. **高精度搜索**（权重3）：返回50个结果（top_k=50），nprobe=32（更精确但慢）。
4. **快速搜索**（权重2）：只返回5个结果（top_k=5），nprobe=8（更快但可能没那么准）。

这些场景的权重（10、5、3、2）表示它们的执行频率，越常见的操作权重越高，更贴近真实使用情况。
