import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
class VisualizableLSH:
    """
    可可视化的局部敏感哈希（LSH）实现，基于随机超平面投影
    适用于余弦相似度搜索
    """
    
    def __init__(self, hash_size=4, num_tables=3, dimension=2):
        """
        初始化LSH参数
        
        参数:
        - hash_size: 每个哈希表的位数（超平面数量）
        - num_tables: 哈希表数量
        - dimension: 数据维度
        """
        self.hash_size = hash_size
        self.num_tables = num_tables
        self.dimension = dimension
        self.hash_tables = [{} for _ in range(num_tables)]
        self.random_planes_list = []
        self.data_points = []
        self.is_trained = False
        
        # 生成随机超平面（法向量）
        self._generate_random_planes()
        
    def _generate_random_planes(self):
        """生成随机超平面法向量"""
        for i in range(self.num_tables):
            # 生成随机超平面法向量，以原点为中心
            planes = np.random.randn(self.hash_size, self.dimension) - 0.5
            self.random_planes_list.append(planes)
        self.is_trained = True
    
    def _hash_vector(self, vector, plane_norms):
        """计算单个向量的哈希值（二进制编码）"""
        # 计算向量与每个超平面法向量的点积
        projections = np.dot(vector, plane_norms.T)
        # 根据点积符号生成二进制编码
        hash_bits = (projections > 0).astype(int)
        # 转换为二进制字符串作为哈希键
        return ''.join(hash_bits.astype(str))
    
    def add_vector(self, vector, vector_id=None):
        """向LSH索引中添加向量"""
        if vector_id is None:
            vector_id = len(self.data_points)
        
        self.data_points.append(vector)
        
        # 将向量添加到所有哈希表
        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(vector, self.random_planes_list[table_idx])
            
            if hash_key in self.hash_tables[table_idx]:
                self.hash_tables[table_idx][hash_key].append(vector_id)
            else:
                self.hash_tables[table_idx][hash_key] = [vector_id]
        
        return vector_id
    
    def query(self, query_vector, max_results=5):
        """查询相似向量"""
        candidates = set()
        
        # 在所有哈希表中查找候选向量
        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(query_vector, self.random_planes_list[table_idx])
            if hash_key in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_key])
        
        # 如果没有找到精确匹配，查找汉明距离最近的桶
        if not candidates:
            print("未找到精确匹配，正在搜索邻近桶...")
            for table_idx in range(self.num_tables):
                original_key = self._hash_vector(query_vector, self.random_planes_list[table_idx])
                # 查找汉明距离为1的邻近桶
                for i in range(len(original_key)):
                    neighbor_key = list(original_key)
                    neighbor_key[i] = '1' if neighbor_key[i] == '0' else '0'
                    neighbor_key = ''.join(neighbor_key)
                    if neighbor_key in self.hash_tables[table_idx]:
                        candidates.update(self.hash_tables[table_idx][neighbor_key])
        
        candidate_ids = list(candidates)[:max_results]
        candidate_vectors = [self.data_points[i] for i in candidate_ids]
        
        return candidate_ids, candidate_vectors
    
    def get_hash_stats(self):
        """获取哈希表统计信息"""
        stats = {
            'total_vectors': len(self.data_points),
            'total_buckets': 0,
            'table_details': []
        }
        
        for i, table in enumerate(self.hash_tables):
            num_buckets = len(table)
            vectors_in_table = sum(len(bucket) for bucket in table.values())
            avg_size = vectors_in_table / num_buckets if num_buckets > 0 else 0
            
            stats['table_details'].append({
                'table_index': i,
                'num_buckets': num_buckets,
                'total_vectors': vectors_in_table,
                'average_bucket_size': avg_size
            })
        
        stats['total_buckets'] = sum(detail['num_buckets'] for detail in stats['table_details'])
        return stats

def visualize_lsh_process(lsh, query_vector=None, highlight_vector=None):
    """可视化LSH哈希过程"""
    
    if len(lsh.data_points) == 0:
        print("没有数据可可视化")
        return
    
    # 如果数据维度大于2，使用PCA降维
    if lsh.dimension > 2:
        pca = PCA(n_components=2)
        all_vectors = np.array(lsh.data_points + ([query_vector] if query_vector is not None else []))
        vectors_2d = pca.fit_transform(all_vectors)
        
        data_2d = vectors_2d[:len(lsh.data_points)]
        if query_vector is not None:
            query_2d = vectors_2d[-1]
        else:
            query_2d = None
    else:
        data_2d = np.array(lsh.data_points)
        query_2d = query_vector
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LSH算法可视化', fontsize=16, fontweight='bold')
    
    # 子图1: 显示数据点和超平面
    ax1 = axes[0, 0]
    
    # 绘制数据点
    colors = plt.cm.Set1(np.linspace(0, 1, len(data_2d)))
    for i, point in enumerate(data_2d):
        ax1.scatter(point[0], point[1], c=[colors[i]], s=100, alpha=0.7, label=f'向量 {i}')
    
    # 绘制超平面（只显示第一个哈希表的前两个超平面）
    if len(lsh.random_planes_list) > 0:
        planes = lsh.random_planes_list[0]
        for j, plane in enumerate(planes[:2]):  # 只显示前两个超平面
            # 在2D空间中，超平面是直线
            if lsh.dimension > 2:
                # 对于高维数据，显示投影后的超平面方向
                plane_2d = pca.transform([plane])[0]
            else:
                plane_2d = plane
            
            # 计算超平面的法线方向
            norm = np.linalg.norm(plane_2d)
            if norm > 0:
                # 绘制超平面法线
                ax1.quiver(0, 0, plane_2d[0], plane_2d[1], 
                          angles='xy', scale_units='xy', scale=1, 
                          color='red', alpha=0.7, width=0.01)
                
                # 绘制超平面（垂直于法线）
                if abs(plane_2d[0]) > 1e-10:  # 避免除零
                    slope = -plane_2d[0] / plane_2d[1] if abs(plane_2d[1]) > 1e-10 else 1e10
                    x_vals = np.array(ax1.get_xlim())
                    y_vals = slope * (x_vals - 0) + 0
                    ax1.plot(x_vals, y_vals, 'r--', alpha=0.5, label=f'超平面 {j+1}')
    
    # 标记查询点（如果有）
    if query_2d is not None:
        ax1.scatter(query_2d[0], query_2d[1], c='yellow', marker='*', 
                   s=300, edgecolors='black', linewidth=2, label='查询点')
    
    ax1.set_title('数据点和超平面划分')
    ax1.set_xlabel('特征 1')
    ax1.set_ylabel('特征 2')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: 显示哈希桶分布
    ax2 = axes[0, 1]
    
    # 统计每个桶的向量数量
    bucket_sizes = []
    bucket_labels = []
    for table_idx, table in enumerate(lsh.hash_tables):
        for hash_key, vectors in table.items():
            bucket_sizes.append(len(vectors))
            bucket_labels.append(f'T{table_idx}_{hash_key[:4]}...')
    
    if bucket_sizes:
        ax2.bar(range(len(bucket_sizes)), bucket_sizes, alpha=0.7)
        ax2.set_title('各哈希桶中的向量数量分布')
        ax2.set_xlabel('哈希桶')
        ax2.set_ylabel('向量数量')
        ax2.tick_params(axis='x', rotation=45)
    
    # 子图3: 显示哈希编码
    ax3 = axes[1, 0]
    
    # 显示前几个向量的哈希编码
    display_count = min(5, len(lsh.data_points))
    hash_codes = []
    vector_labels = []
    
    for i in range(display_count):
        hash_code = lsh._hash_vector(lsh.data_points[i], lsh.random_planes_list[0])
        hash_codes.append([int(bit) for bit in hash_code])
        vector_labels.append(f'向量{i}')
    
    if hash_codes:
        im = ax3.imshow(hash_codes, cmap='Blues', aspect='auto')
        ax3.set_xticks(range(len(hash_code)))
        ax3.set_xticklabels([f'位{i+1}' for i in range(len(hash_code))])
        ax3.set_yticks(range(display_count))
        ax3.set_yticklabels(vector_labels)
        
        # 添加数值标注
        for i in range(len(hash_codes)):
            for j in range(len(hash_codes[0])):
                ax3.text(j, i, hash_codes[i][j], 
                        ha='center', va='center', fontweight='bold')
        
        ax3.set_title('向量哈希编码（第一个哈希表）')
        plt.colorbar(im, ax=ax3, label='比特值')
    
    # 子图4: 显示查询结果（如果有查询）
    ax4 = axes[1, 1]
    
    if query_2d is not None:
        # 执行查询
        candidate_ids, candidate_vectors = lsh.query(query_vector)
        
        # 绘制所有数据点
        for i, point in enumerate(data_2d):
            if i in candidate_ids:
                # 高亮候选向量
                ax4.scatter(point[0], point[1], c='red', s=150, 
                           alpha=0.8, label='候选向量' if i == candidate_ids[0] else "")
            else:
                ax4.scatter(point[0], point[1], c='blue', s=80, alpha=0.3)
        
        # 标记查询点
        ax4.scatter(query_2d[0], query_2d[1], c='yellow', marker='*', 
                   s=300, edgecolors='black', linewidth=2, label='查询点')
        
        ax4.set_title('LSH查询结果')
        ax4.set_xlabel('特征 1')
        ax4.set_ylabel('特征 2')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        # 如果没有查询，显示哈希表统计信息
        stats = lsh.get_hash_stats()
        ax4.text(0.1, 0.9, f"LSH索引统计:", fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.8, f"向量总数: {stats['total_vectors']}", fontsize=10)
        ax4.text(0.1, 0.7, f"哈希表数量: {lsh.num_tables}", fontsize=10)
        ax4.text(0.1, 0.6, f"总桶数: {stats['total_buckets']}", fontsize=10)
        
        for i, detail in enumerate(stats['table_details']):
            ax4.text(0.1, 0.5 - i*0.1, 
                    f"表{detail['table_index']}: {detail['num_buckets']}个桶", 
                    fontsize=9)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('LSH索引统计信息')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def demonstrate_lsh():
    """演示LSH算法的完整流程"""
    print("=" * 60)
    print("LSH算法完整演示")
    print("=" * 60)
    
    # 1. 创建示例数据（2维便于可视化）
    np.random.seed(42)
    n_samples = 20
    dim = 2
    
    # 生成具有聚类结构的数据
    cluster1 = np.random.normal(loc=[2, 3], scale=0.5, size=(n_samples//2, dim))
    cluster2 = np.random.normal(loc=[-1, -1], scale=0.3, size=(n_samples//2, dim))
    data = np.vstack([cluster1, cluster2])
    
    print(f"生成{len(data)}个{dim}维数据点")
    
    # 2. 创建并初始化LSH索引
    lsh = VisualizableLSH(hash_size=4, num_tables=2, dimension=dim)
    
    # 3. 向LSH索引中添加数据
    for i, vector in enumerate(data):
        lsh.add_vector(vector, i)
    
    print("LSH索引构建完成!")
    
    # 4. 显示统计信息
    stats = lsh.get_hash_stats()
    print(f"\nLSH索引统计:")
    print(f"向量总数: {stats['total_vectors']}")
    print(f"哈希表数量: {lsh.num_tables}")
    print(f"总桶数: {stats['total_buckets']}")
    
    for detail in stats['table_details']:
        print(f"表{detail['table_index']}: {detail['num_buckets']}个桶, "
              f"平均每个桶{detail['average_bucket_size']:.2f}个向量")
    
    # 5. 创建查询点
    query_point = np.array([1.5, 2.0])
    print(f"\n查询点: {query_point}")
    
    # 6. 执行查询
    candidate_ids, candidate_vectors = lsh.query(query_point, max_results=3)
    
    print(f"找到{len(candidate_ids)}个候选向量:")
    for i, vec_id in enumerate(candidate_ids):
        similarity = cosine_similarity([query_point], [data[vec_id]])[0][0]
        print(f"候选向量{vec_id}: 相似度={similarity:.4f}")
    
    # 7. 可视化整个过程
    print("\n生成可视化图表...")
    visualize_lsh_process(lsh, query_point)
    
    return lsh, data, query_point, candidate_ids

# 运行演示
lsh, data, query, candidates = demonstrate_lsh()