import faiss
import numpy as np
import time

print("FAISS version:", faiss.__version__)

# 1. 数据准备
dim = 128
db_size = 10000
query_size = 50

np.random.seed(42)
db_vectors = np.random.random((db_size, dim)).astype('float32')
query_vectors = np.random.random((query_size, dim)).astype('float32')

# 2. 创建索引：L2、IP
index_l2 = faiss.IndexFlatL2(dim)
index_ip = faiss.IndexFlatIP(dim)

# 3. 余弦相似度：手动归一化 + IP
db_vectors_norm = db_vectors.copy()
query_vectors_norm = query_vectors.copy()
faiss.normalize_L2(db_vectors_norm)
faiss.normalize_L2(query_vectors_norm)

index_cos = faiss.IndexFlatIP(dim)

# 添加向量
index_l2.add(db_vectors)
index_ip.add(db_vectors)
index_cos.add(db_vectors_norm)

print("三个索引均已添加向量：", index_l2.ntotal)

# 4. 检索
k = 10

# ---- L2 ----
t0 = time.time()
dist_l2, idx_l2 = index_l2.search(query_vectors, k)
t1 = time.time()
print(f"IndexFlatL2 检索时间：{t1 - t0:.4f} 秒")

# ---- IP ----
t0 = time.time()
dist_ip, idx_ip = index_ip.search(query_vectors, k)
t1 = time.time()
print(f"IndexFlatIP 检索时间：{t1 - t0:.4f} 秒")

# ---- COSINE ----
t0 = time.time()
dist_cos, idx_cos = index_cos.search(query_vectors_norm, k)
t1 = time.time()
print(f"Cosine（归一化+IP）检索时间：{t1 - t0:.4f} 秒")

# 5. Top-K 重合率对比（第 0 个查询向量）
print("\n=== 第 0 个查询向量的 Top-K 对比 ===")
print("L2 Top-10：", idx_l2[0])
print("IP  Top-10：", idx_ip[0])
print("COS Top-10：", idx_cos[0])

def overlap(a, b):
    return len(set(a) & set(b))

print(f"\nL2 vs IP   重合度：{overlap(idx_l2[0], idx_ip[0])}/10")
print(f"L2 vs COS  重合度：{overlap(idx_l2[0], idx_cos[0])}/10")
print(f"IP vs COS  重合度：{overlap(idx_ip[0], idx_cos[0])}/10")