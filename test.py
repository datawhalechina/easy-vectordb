import time
import numpy as np
import faiss

# -------------------------- 1. 实验配置与数据准备 --------------------------
# 基础配置
d = 128  # 向量维度
nb = 100000  # 基础向量数量
nq = 1000  # 查询向量数量
k = 10  # 检索的近邻数
np.random.seed(1234)  # 固定随机种子，保证结果可复现

# 生成测试数据（float32类型是FAISS的要求）
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 构建精确索引作为召回率基准
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D_flat, I_flat = index_flat.search(xq, k)  # 精确结果，用于计算召回率

# -------------------------- 2. 定义性能评估函数 --------------------------
def calculate_recall(I_pred, I_true, k):
    """
    计算召回率：预测结果中命中真实近邻的比例
    参数：I_pred-模型预测的索引矩阵，I_true-精确结果的索引矩阵，k-近邻数
    返回：平均召回率
    """
    recall_list = []
    for i in range(len(I_pred)):
        pred_set = set(I_pred[i])
        true_set = set(I_true[i])
        hit = len(pred_set & true_set)
        recall = hit / k
        recall_list.append(recall)
    return np.mean(recall_list)

# -------------------------- 3. LSH索引构建与性能测试 --------------------------
print("=== 测试IndexLSH性能 ===")
# 初始化LSH索引（n_bits设为32，平衡精度与速度）
index_lsh = faiss.IndexLSH(d, 32)

# 索引构建时间
start_time = time.time()
index_lsh.add(xb)
lsh_index_time = time.time() - start_time
print(f"LSH索引构建时间：{lsh_index_time:.4f} 秒")

# 检索性能测试
start_time = time.time()
D_lsh, I_lsh = index_lsh.search(xq, k)
lsh_search_time = (time.time() - start_time) 
print(f"LSH查询时间：{lsh_search_time:.6f} 秒")

# 计算召回率
lsh_recall = calculate_recall(I_lsh, I_flat, k)
print(f"LSH召回率：{lsh_recall:.4f}")

# -------------------------- 4. IVF-PQ索引构建与性能测试 --------------------------
print("\n=== 测试IndexIVFPQ性能 ===")
# IVF-PQ需要先定义量化器（通常用Flat索引）
quantizer = faiss.IndexFlatL2(d)
nlist = 100  # 聚类桶数量
m = 16  # 乘积量化的分段数（需整除向量维度d）
nbits_per_idx = 8  # 每个分段的编码位数

# 初始化IVF-PQ索引
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits_per_idx)

# IVF-PQ需先训练（聚类过程）
start_time = time.time()
index_ivfpq.train(xb)  # 训练聚类中心
index_ivfpq.add(xb)    # 加入向量构建索引
ivfpq_index_time = time.time() - start_time
print(f"IVF-PQ索引构建（含训练）时间：{ivfpq_index_time:.4f} 秒")

# 设置查询时的探测桶数量（nprobe越大，召回率越高但速度越慢）
index_ivfpq.nprobe = 10

# 检索性能测试
start_time = time.time()
D_ivfpq, I_ivfpq = index_ivfpq.search(xq, k)
ivfpq_search_time = (time.time() - start_time)
print(f"IVF-PQ查询时间：{ivfpq_search_time:.6f} 秒")

# 计算召回率
ivfpq_recall = calculate_recall(I_ivfpq, I_flat, k)
print(f"IVF-PQ召回率：{ivfpq_recall:.4f}")

print("\n" + "="*80)
print("📋 性能对比汇总表（LSH vs IVF-PQ）")
print("="*80)
# 表头（左对齐指标名，右对齐数值，宽度固定）
header = f"{'指标':<15} {'LSH':<18} {'IVF-PQ':<18}"
print(header)
print("-"*80)  # 分隔线
# 每行数据（统一格式：时间4位小数，召回率4位小数，内存2位小数）
rows = [
    (f"构建时间", f"{lsh_index_time:.4f} 秒", f"{ivfpq_index_time:.4f} 秒"),
    (f"查询时间", f"{lsh_search_time:.4f} 秒", f"{ivfpq_search_time:.4f} 秒"),
    (f"召回率", f"{lsh_recall:.4f}", f"{ivfpq_recall:.4f}")
]
# 格式化输出（确保列对齐）
for metric, lsh_val, ivfpq_val in rows:
    print(f"{metric:<15} {lsh_val:<18} {ivfpq_val:<18}")
print("="*80)