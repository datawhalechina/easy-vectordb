import faiss
import numpy as np

# 1. 构造测试数据（3 个数据库向量，2 个查询向量，模拟语义相似性）
xb = np.array([[1,2,3], [2,4,6], [3,3,3]], dtype=np.float32)  # 向量0与向量1方向完全一致（COSINE=1），向量2方向独特
xq = np.array([[0.5,1,1.5], [1,1,1]], dtype=np.float32)  # 查询0与向量0/1方向一致，查询1与向量2方向一致
d = xb.shape[1]  # 向量维度：3
k = 1  # 检索Top-1结果（只取最相似的1个）

# 方案 1：未归一化，用内积度量（错误方式）
index_no_norm = faiss.IndexFlatIP(d)  # 内积索引
index_no_norm.add(xb)  # 加入未归一化的数据库向量
D_no_norm, I_no_norm = index_no_norm.search(xq, k)  # 检索

# 方案 2：归一化后，用内积度量（正确方式）
xb_norm = xb.copy()
xq_norm = xq.copy()
faiss.normalize_L2(xb_norm)  # 数据库向量归一化（按行）
faiss.normalize_L2(xq_norm)  # 查询向量必须同步归一化（关键！）

index_norm = faiss.IndexFlatIP(d)
index_norm.add(xb_norm)
D_norm, I_norm = index_norm.search(xq_norm, k)

# 输出结果对比
print("=== 检索结果对比（目标：COSINE 相似度 Top-1）===")
print("数据库向量：\n", xb)
print("查询向量 0：", xq[0], "（与数据库向量0/1方向一致，COSINE=1）")
print("查询向量 1：", xq[1], "（与数据库向量2方向一致）")
print("\n【未归一化+内积】检索结果（错误，受向量长度干扰）：")
print(f"查询0 最相似向量索引：{I_no_norm[0][0]}，内积：{D_no_norm[0][0]:.2f}")  
print(f"查询1 最相似向量索引：{I_no_norm[1][0]}，内积：{D_no_norm[1][0]:.2f}")  

print("\n【归一化+内积】检索结果（正确，内积=COSINE相似度）：")
print(f"查询0 最相似向量索引：{I_norm[0][0]}，内积（COSINE）：{D_norm[0][0]:.2f}")  
print(f"查询1 最相似向量索引：{I_norm[1][0]}，内积（COSINE）：{D_norm[1][0]:.2f}")  