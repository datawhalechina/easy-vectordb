import faiss
import numpy as np

# 参数设置
d = 128                         # 向量维度
num_vectors = 100000            # 索引中的向量数量
num_queries = 10000             # 查询数量
k = 4                           # 返回最近邻的数量

# 模拟数据：随机生成索引向量和查询向量 (正态分布)
np.random.seed(42)              # 固定随机种子以便复现
xb = np.random.rand(num_vectors, d).astype('float32')  # 索引数据 (100000, 128)
xq = np.random.rand(num_queries, d).astype('float32')  # 查询数据 (10000, 128)

# 构建索引并添加数据
index = faiss.IndexFlatL2(d)    # 使用L2距离（欧氏距离）
index.add(xb)                   # 添加索引数据
print("索引中的向量数量:", index.ntotal)  # 应输出 100000

# 执行搜索
D, I = index.search(xq, k)      # D: 距离矩阵 (10000, 4), I: 索引矩阵 (10000, 4)

# 输出前5个查询的最近邻ID
print("\n前5个查询的最近邻ID:")
print(I[:5])

# 可选：输出前5个查询的距离
print("\n前5个查询的距离:")
print(D[:5])