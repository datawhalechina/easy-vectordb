import faiss
import numpy as np

# 参数设置
d = 128                         # 向量维度
nlist = 100                     # 聚类中心数量（Voronoi Cells数量）
m = 8                           # 分段数量（乘积量化参数）
bits = 8                        # 每段的编码位数（总字节数 = m * bits / 8）
k = 4                           # 返回最近邻的数量
nprobe = 10                     # 搜索的聚类中心数量

# 生成模拟数据
np.random.seed(1234)            # 固定随机种子保证可复现
xb = np.random.rand(100000, d).astype('float32')  # 数据库向量 (100000, 128)
xq = np.random.rand(100, d).astype('float32')     # 查询向量 (100, 128)

# 构建索引
quantizer = faiss.IndexFlatL2(d)  # 用于分配向量的量化器（使用L2距离）
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)  # 乘积量化索引

# 训练阶段（聚类 + PQ码本学习）
print("开始训练索引...")
index.train(xb)                  # 同时训练聚类和PQ码本
print("训练完成！")

# 添加数据到索引
print("添加数据到索引...")
index.add(xb)                    # 将数据编码为PQ并分配到对应的Cell
print(f"索引中的向量数量: {index.ntotal}")

# 设置搜索范围
index.nprobe = nprobe            # 搜索的Cell数量（默认nprobe=1）

# 执行搜索
print(f"\n执行搜索（nprobe={nprobe}）...")
D, I = index.search(xq, k)       # D: 近似距离矩阵 (100, 4), I: 索引矩阵 (100, 4)

# 输出结果
print("\n最后5个查询的最近邻ID:")
print(I[-5:])                    # 最后5次查询的top-k结果

print("\n最后5个查询的近似距离:")
print(D[-5:])                    # 对应的近似距离

# 验证索引类型和参数
print("\n索引类型:", index)
print("PQ参数: m={}, bits={} (总字节数={})".format(m, bits, m * bits // 8))