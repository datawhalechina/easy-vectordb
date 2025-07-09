import time
import random
import threading
import statistics
import argparse
import os
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from prettytable import PrettyTable
import matplotlib.pyplot as plt

class MilvusIndexTester:
    def __init__(self, host='localhost', port='19530', 
                 collection_name='perf_test', dim=128, 
                 num_vectors=100000, warmup_queries=1000):
        """
        初始化性能测试器
        
        参数:
            host: Milvus 服务器地址
            port: Milvus 端口
            collection_name: 测试集合名称
            dim: 向量维度
            num_vectors: 测试数据量
            warmup_queries: 预热查询次数
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.num_vectors = num_vectors
        self.warmup_queries = warmup_queries
        
        # 连接 Milvus
        connections.connect("default", host=host, port=port)
        print(f"✅ 已连接到 Milvus 服务器: {host}:{port}")
        
        # 检查并删除已存在的同名集合
        self._prepare_collection()
    
    def _prepare_collection(self):
        """准备测试集合"""
        # 如果集合存在，先卸载然后删除
        if utility.has_collection(self.collection_name):
            print(f"♻️ 发现现有集合: {self.collection_name}")
            
            # 获取集合实例
            collection = Collection(self.collection_name)
            
            # 如果集合已加载，先释放
            try:
                if hasattr(collection, 'loaded') and collection.loaded:
                    print("  释放已加载的集合...")
                    collection.release()
            except Exception:
                pass
            
            # 删除集合
            utility.drop_collection(self.collection_name)
            print(f"  已删除现有集合")
        
        # 创建新集合
        self._create_collection()
        print(f"🆕 已创建新集合: {self.collection_name} (维度={self.dim})")
        
        # 插入测试数据
        self._insert_test_data()
        print(f"📊 已插入 {self.num_vectors} 条测试数据")
    
    def _create_collection(self):
        """创建测试集合"""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, description="性能测试集合")
        self.collection = Collection(self.collection_name, schema)
    
    def _insert_test_data(self):
        """插入随机测试数据"""
        # 生成随机向量
        vectors = [[random.random() for _ in range(self.dim)] 
                  for _ in range(self.num_vectors)]
        
        # 插入数据（分批进行）
        batch_size = 5000
        total_batches = (self.num_vectors + batch_size - 1) // batch_size
        
        for i in range(0, self.num_vectors, batch_size):
            batch_vectors = vectors[i:i+batch_size]
            self.collection.insert([batch_vectors])
            progress = min(i + batch_size, self.num_vectors)
            print(f"  插入进度: {progress}/{self.num_vectors} ({progress/self.num_vectors*100:.1f}%)", end='\r')
        
        # 刷新数据确保可见
        self.collection.flush()
        print("\n✅ 数据插入完成")
    
    def create_index(self, index_type, index_params):
        """
        创建索引
        
        参数:
            index_type: 索引类型 (HNSW, IVF_FLAT, IVF_SQ8, etc.)
            index_params: 索引参数
        """
       # 确保集合未加载
        if hasattr(self.collection, 'loaded') and self.collection.loaded:
            print("  释放已加载的集合...")
            self.collection.release()
            # 等待集合释放完成
            start_time = time.time()
            while True:
                try:
                    if utility.load_state(self.collection_name) != "Loaded":
                        print("✅ 集合已成功释放")
                        break
                    time.sleep(0.5)
                    if time.time() - start_time > 10:
                        print("⚠️ 集合释放超时，尝试强制继续")
                        break
                except Exception as e:
                    print(f"  检查集合状态失败: {str(e)}")
                    time.sleep(1)
        
        # 删除现有索引（如果存在）
        if self.collection.has_index():
            print("  删除现有索引...")
            try:
                self.collection.drop_index()
            except Exception as e:
                print(f"⚠️ 删除索引时出错: {str(e)}")
                # 尝试再次释放后重试
                if "collection is loaded" in str(e):
                    print("  再次尝试释放集合...")
                    self.collection.release()
                    time.sleep(2)
                    self.collection.drop_index()
                    print("✅ 成功删除索引")
        
        # 创建新索引
        print(f"🛠️ 创建 {index_type} 索引，参数: {index_params}")
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": "L2",
                "params": index_params
            }
        )
        
        # 等待索引构建完成
        start_time = time.time()
        while True:
            time.sleep(2)
            try:
                index_info = utility.index_building_progress(self.collection_name)
                indexed = index_info['indexed_rows']
                total = index_info['total_rows']
                
                if indexed == total:
                    print("✅ 索引构建完成")
                    break
                
                print(f"  索引构建进度: {indexed}/{total} ({indexed/total*100:.1f}%)")
                
                # 超时检查（10分钟）
                if time.time() - start_time > 600:
                    print("⚠️ 索引构建超时！")
                    break
            except Exception as e:
                print(f"  获取索引进度失败: {str(e)}")
                time.sleep(5)
        
        # 加载集合到内存
        print("📥 加载集合到内存...")
        self.collection.load()
        
        # 新的进度检查逻辑
        start_time = time.time()
        last_progress = -1
        
        while True:
            time.sleep(1)
            try:
                # 获取加载状态（返回格式：{'loading_progress': 'X%'}）
                progress_info = utility.loading_progress(self.collection_name)
                
                # 解析百分比值
                progress_str = progress_info.get('loading_progress', '0%')
                progress_percent = int(progress_str.strip('%'))
                
                # 打印进度（仅在进度变化时打印）
                if progress_percent != last_progress:
                    print(f"  加载进度: {progress_percent}%")
                    last_progress = progress_percent
                
                # 检查是否完成加载
                if progress_percent >= 100:
                    print("✅ 集合已加载到内存")
                    break
                    
            except Exception as e:
                print(f"  获取加载进度失败: {str(e)}")
                # 直接检查集合加载状态作为后备方案
                try:
                    if self.collection.loaded:
                        print("✅ 集合已加载到内存 (通过loaded属性验证)")
                        break
                except Exception as e2:
                    print(f"  验证集合加载状态失败: {str(e2)}")
            
            # 超时检查（10分钟）
            if time.time() - start_time > 600:
                print("⚠️ 集合加载超时！")
                break

    def _run_test(self, index_config, threads, duration):
        """执行单个索引配置的性能测试"""
        print(f"\n🚀 开始测试配置: {index_config['name']}")
        print("-" * 60)
        
        # 创建索引
        self.create_index(index_config["type"], index_config["params"])
        
        # 预热阶段
        print(f"🔥 预热查询 ({self.warmup_queries} 次)...")
        for i in range(self.warmup_queries):
            vector = [[random.random() for _ in range(self.dim)]]
            self.collection.search(
                vector, 
                "embedding", 
                index_config["search_params"], 
                limit=10
            )
            if (i + 1) % 100 == 0:
                print(f"  预热进度: {i+1}/{self.warmup_queries}", end='\r')
        print("\n✅ 预热完成")
        
        # 性能指标
        latencies = []
        query_count = 0
        running = True
        start_time = None
        
        def query_worker(worker_id):
            """查询工作线程"""
            nonlocal query_count
            local_count = 0
            local_latencies = []
            
            while running:
                try:
                    # 生成随机查询向量
                    vector = [[random.random() for _ in range(self.dim)]]
                    
                    # 执行查询
                    start_time = time.perf_counter()
                    self.collection.search(
                        vector, 
                        "embedding", 
                        index_config["search_params"], 
                        limit=10
                    )
                    
                    # 记录延迟(毫秒)
                    latency = (time.perf_counter() - start_time) * 1000
                    local_latencies.append(latency)
                    local_count += 1
                except Exception as e:
                    print(f"⚠️ 线程 {worker_id} 查询失败: {str(e)}")
                    continue
            
            # 更新全局统计
            nonlocal latencies
            with threading.Lock():
                query_count += local_count
                latencies.extend(local_latencies)
        
        # 创建工作线程
        workers = []
        print(f"🛠️ 启动 {threads} 个查询线程...")
        for i in range(threads):
            t = threading.Thread(target=query_worker, args=(i+1,))
            t.daemon = True
            t.start()
            workers.append(t)
        
        # 运行指定时长
        print(f"⏱️ 运行测试 {duration} 秒...")
        start_time = time.perf_counter()
        last_report = start_time
        
        while time.perf_counter() - start_time < duration:
            time.sleep(1)
            elapsed = time.perf_counter() - start_time
            if time.perf_counter() - last_report >= 5:  # 每5秒报告一次
                current_qps = query_count / elapsed if elapsed > 0 else 0
                print(f"  已运行: {elapsed:.1f}s, 当前QPS: {current_qps:.2f}, 总查询数: {query_count}", end='\r')
                last_report = time.perf_counter()
        
        running = False
        print("\n🛑 测试结束，等待线程退出...")
        
        # 等待线程结束
        for t in workers:
            t.join(timeout=1)
        
        # 计算实际运行时间
        elapsed = time.perf_counter() - start_time
        
        # 计算性能指标
        if query_count == 0:
            print("⚠️ 警告: 没有成功执行的查询!")
            return {
                "name": index_config["name"],
                "qps": 0,
                "avg_latency": 0,
                "p95_latency": 0,
                "max_latency": 0,
                "total_queries": 0
            }
        
        qps = query_count / elapsed
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=100)[94] if len(latencies) > 1 else avg_latency
        max_latency = max(latencies)
        
        print("\n📊 测试结果:")
        print(f"  总查询次数: {query_count}")
        print(f"  实际测试时间: {elapsed:.2f} 秒")
        print(f"  QPS: {qps:.2f} 次/秒")
        print(f"  平均延迟: {avg_latency:.2f} ms")
        print(f"  P95延迟: {p95_latency:.2f} ms")
        print(f"  最大延迟: {max_latency:.2f} ms")
        print(f"  最小延迟: {min(latencies):.2f} ms")
        # 测试完成后释放集合
        print("♻️ 测试完成，释放集合...")
        try:
            if utility.load_state(self.collection_name) == "Loaded":
                self.collection.release()
                # 确认集合已释放
                start_time = time.time()
                while utility.load_state(self.collection_name) == "Loaded":
                    time.sleep(0.5)
                    if time.time() - start_time > 10:
                        print("⚠️ 集合释放确认超时")
                        break
                print("✅ 集合已释放")
        except Exception as e:
            print(f"⚠️ 释放集合时出错: {str(e)}")
        return {
            "name": index_config["name"],
            "qps": qps,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "max_latency": max_latency,
            "total_queries": query_count
        }
    
    def run_comparison(self, index_configs, threads=100, duration=60):
        """
        运行多索引配置对比测试
        
        参数:
            index_configs: 索引配置列表
            threads: 并发线程数
            duration: 每个测试运行时长(秒)
        """
        results = []
        
        # 运行所有配置的测试
        for i, config in enumerate(index_configs):
            print(f"\n{'=' * 60}")
            print(f"🏁 开始测试 {i+1}/{len(index_configs)}: {config['name']}")
            print(f"{'=' * 60}")
            
            result = self._run_test(config, threads, duration)
            results.append(result)
            
            # 测试完成后释放集合
            print("♻️ 测试完成，释放集合...")
            if self.collection.load():
                self.collection.release()
        
        # 打印对比结果
        self._print_results_table(results)
        
        # 生成性能图表
        self._plot_results(results)
        
        return results
    
    def _print_results_table(self, results):
        """打印结果表格"""
        table = PrettyTable()
        table.field_names = ["索引配置", "QPS", "平均延迟(ms)", "P95延迟(ms)", "最大延迟(ms)", "总查询数"]
        table.align["索引配置"] = "l"
        table.align["QPS"] = "r"
        table.align["平均延迟(ms)"] = "r"
        table.align["P95延迟(ms)"] = "r"
        table.align["最大延迟(ms)"] = "r"
        table.align["总查询数"] = "r"
        
        for res in results:
            table.add_row([
                res["name"],
                f"{res['qps']:.2f}",
                f"{res['avg_latency']:.2f}",
                f"{res['p95_latency']:.2f}",
                f"{res['max_latency']:.2f}",
                f"{res['total_queries']:,}"
            ])
        
        print("\n" + "=" * 80)
        print("🔥 Milvus 索引性能对比结果")
        print("=" * 80)
        print(table)
        print("=" * 80)
    
    def _plot_results(self, results):
        """生成性能对比图表"""
        if not results:
            print("⚠️ 没有结果可生成图表")
            return
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        plt.suptitle('Milvus 索引性能对比', fontsize=16)
        
        # QPS 对比图
        plt.subplot(2, 2, 1)
        names = [res["name"] for res in results]
        qps_values = [res["qps"] for res in results]
        plt.bar(names, qps_values, color='skyblue')
        plt.title('QPS 对比 (越高越好)')
        plt.ylabel('QPS (次/秒)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 平均延迟对比图
        plt.subplot(2, 2, 2)
        avg_latencies = [res["avg_latency"] for res in results]
        plt.bar(names, avg_latencies, color='lightgreen')
        plt.title('平均延迟对比 (越低越好)')
        plt.ylabel('延迟 (ms)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # P95延迟对比图
        plt.subplot(2, 2, 3)
        p95_latencies = [res["p95_latency"] for res in results]
        plt.bar(names, p95_latencies, color='salmon')
        plt.title('P95延迟对比 (越低越好)')
        plt.ylabel('延迟 (ms)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 最大延迟对比图
        plt.subplot(2, 2, 4)
        max_latencies = [res["max_latency"] for res in results]
        plt.bar(names, max_latencies, color='gold')
        plt.title('最大延迟对比 (越低越好)')
        plt.ylabel('延迟 (ms)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"results/milvus_perf_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"📈 性能图表已保存至: {filename}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Milvus 索引性能对比测试工具')
    parser.add_argument('--host', default='localhost', help='Milvus 服务器地址')
    parser.add_argument('--port', default='19530', help='Milvus 端口')
    parser.add_argument('--dim', type=int, default=128, help='向量维度')
    parser.add_argument('--data-size', type=int, default=100000, help='测试数据量')
    parser.add_argument('--threads', type=int, default=100, help='并发线程数')
    parser.add_argument('--duration', type=int, default=60, help='每个测试时长(秒)')
    parser.add_argument('--warmup', type=int, default=1000, help='预热查询次数')
    args = parser.parse_args()
    
    # 定义要测试的索引配置
    index_configs = [
        {
            "name": "HNSW (ef=32)",
            "type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
            "search_params": {"metric_type": "L2", "params": {"ef": 32}}
        },
        {
            "name": "HNSW (ef=64)",
            "type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
            "search_params": {"metric_type": "L2", "params": {"ef": 64}}
        },
        {
            "name": "HNSW (ef=128)",
            "type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
            "search_params": {"metric_type": "L2", "params": {"ef": 128}}
        },
        {
            "name": "IVF_FLAT (nprobe=16)",
            "type": "IVF_FLAT",
            "params": {"nlist": 1024},
            "search_params": {"metric_type": "L2", "params": {"nprobe": 16}}
        },
        {
            "name": "IVF_FLAT (nprobe=32)",
            "type": "IVF_FLAT",
            "params": {"nlist": 1024},
            "search_params": {"metric_type": "L2", "params": {"nprobe": 32}}
        },
        {
            "name": "IVF_SQ8 (nprobe=32)",
            "type": "IVF_SQ8",
            "params": {"nlist": 1024},
            "search_params": {"metric_type": "L2", "params": {"nprobe": 32}}
        }
    ]
    
    print("=" * 80)
    print("🚀 Milvus 多索引性能对比测试")
    print("=" * 80)
    print(f"测试参数:")
    print(f"  Milvus 地址: {args.host}:{args.port}")
    print(f"  向量维度: {args.dim}")
    print(f"  测试数据量: {args.data_size}")
    print(f"  并发线程数: {args.threads}")
    print(f"  每个测试时长: {args.duration} 秒")
    print(f"  预热查询次数: {args.warmup}")
    print(f"  测试配置数: {len(index_configs)}")
    print("=" * 80)
    
    # 初始化测试器
    tester = MilvusIndexTester(
        host=args.host,
        port=args.port,
        dim=args.dim,
        num_vectors=args.data_size,
        warmup_queries=args.warmup
    )
    
    # 运行对比测试
    results = tester.run_comparison(
        index_configs=index_configs,
        threads=args.threads,
        duration=args.duration
    )
    
    # 保存结果到文件
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"results/milvus_perf_results_{timestamp}.txt"
    
    with open(result_file, "w") as f:
        f.write(f"Milvus 性能测试结果 ({timestamp})\n")
        f.write("=" * 80 + "\n")
        f.write(f"Milvus 地址: {args.host}:{args.port}\n")
        f.write(f"向量维度: {args.dim}\n")
        f.write(f"测试数据量: {args.data_size}\n")
        f.write(f"并发线程数: {args.threads}\n")
        f.write(f"每个测试时长: {args.duration} 秒\n")
        f.write(f"预热查询次数: {args.warmup}\n\n")
        
        f.write("性能对比结果:\n")
        for res in results:
            f.write(f"\n配置: {res['name']}\n")
            f.write(f"  QPS: {res['qps']:.2f}\n")
            f.write(f"  平均延迟: {res['avg_latency']:.2f} ms\n")
            f.write(f"  P95延迟: {res['p95_latency']:.2f} ms\n")
            f.write(f"  最大延迟: {res['max_latency']:.2f} ms\n")
            f.write(f"  总查询次数: {res['total_queries']}\n")
    
    print(f"📝 详细结果已保存至: {result_file}")
    print("✅ 测试完成!")

if __name__ == "__main__":
    main()


# python milvus_perf_test.py \
#   --host your-milvus-host \
#   --data-size 500000 \
#   --threads 200 \
#   --duration 120 \
#   --warmup 2000

# 参数	默认值	说明
# --host	localhost	Milvus 服务器地址
# --port	19530	Milvus 端口
# --dim	128	向量维度
# --data-size	100000	测试数据集大小
# --threads	100	并发查询线程数
# --duration	60	每个索引测试时长(秒)
# --warmup	1000	预热查询次数
