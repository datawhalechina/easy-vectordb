import faiss
import numpy as np

d = 128                           # 向量维度
nlist = 100                       # Cell 数量
k = 4                             # 返回最近邻数
nprobe = 10                       # 搜索的 Cell 数

# 生成随机数据
np.random.seed(42)
xb = np.random.rand(100000, d).astype('float32')  # 数据库向量
xq = np.random.rand(100, d).astype('float32')    # 查询向量

# 构建索引
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.train(xb)                    # 训练聚类
index.add(xb)                      # 添加数据
index.nprobe = nprobe              # 设置搜索范围

# 查询
D, I = index.search(xq, k)
print("最近邻 IDs (最后 5 个查询):\n", I[-5:])