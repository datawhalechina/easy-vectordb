# FAISS 专题课程大纲

## Chapter 1：FAISS 入门与环境搭建
小节	核心内容	学习要求
7.1 FAISS 核心定位与生态	1. FAISS 的定义、研发背景与核心优势
   FAISS 与 Milvus/Chroma 等向量库的对比
   FAISS 的适用场景（大规模向量检索、低延迟检索等）	理解 FAISS 的核心价值，明确其适用边界
7.2 FAISS 环境搭建	1. Python/C++ 版本安装（pip / 源码编译）
   依赖配置（OpenBLAS/CUDA）
   安装验证与常见问题解决（版本兼容、CUDA 适配）	独立完成本地 / 实验室服务器的 FAISS 环境搭建
7.3 FAISS 核心数据结构	1. Index 类体系（基类 Index 与派生类）
   向量存储基础（Numpy/Tensor 适配）
   ID 映射与向量维度约束	掌握 FAISS 的核心数据载体，理解 Index 的核心作用
7.4 第一个 FAISS 示例	1. 生成测试向量（随机向量 / SIFT 小样本）
   创建 IndexFlatL2（精确检索）
   向量添加、检索、结果解析
   基础 API 解读（add/search/reset）	独立运行基础检索案例，理解检索流程

---

## Chapter 2：FAISS 数据结构与索引类型
小节	核心内容	学习要求
8.1 精确检索索引：IndexFlat 系列	1. IndexFlatL2/IP/COSINE 原理回顾（暴力检索）
   API 使用与参数说明
1. 精确检索的性能瓶颈（时间 / 内存）
2. 实战：SIFT10k 数据集精确检索对比	理解精确检索的适用场景，能区分不同距离度量的 Index
8.2 IVF 系列索引：量化检索基础	1. IVF 原理回顾（倒排文件）
1. IndexIVF_FLAT/IndexIVF_PQ 的 API
2. 核心参数：nlist（聚类数）、nprobe（检索聚类数）
3. 实战：IVF 索引检索 SIFT100k，对比 nprobe 对 Recall 的影响	掌握 IVF 索引调参逻辑，理解 “聚类数 - 检索效率” 权衡
8.3 PQ/OPQ 量化索引：压缩检索	1. PQ（乘积量化）原理回顾
1. IndexPQ/IndexIVF_PQ 的使用
2. 码本训练、压缩率与精度权衡
3. 实战：OPQ 优化 PQ 检索精度	能基于 PQ 实现向量压缩，平衡内存占用与检索精度
8.4 HNSW 索引：图结构近邻检索	1. HNSW 原理回顾（分层图）
1. IndexHNSWFlat 核心参数（M/efConstruction/efSearch）
2. 实战：HNSW vs IVF-PQ 性能对比
3. 高维向量检索优化	掌握 HNSW 索引调参，理解图结构检索的优势
8.5 LSH 索引：哈希检索	1. IndexLSH 原理与 API
1. 适用场景（低维 / 近似检索）
2. 实战：LSH 检索文本嵌入向量	了解 LSH 在 FAISS 中的落地方式，区分其适用场景
8.6 复合索引设计	1. 索引组合逻辑（如 IVF_HNSW_PQ）
1. 自定义复合索引的构建流程
2. 实战：大规模数据集（SIFT1M）复合索引检索	能根据场景选择 / 组合索引，解决 “高精度 + 高效率

---

## Chapter 3：FAISS 搜索与相似度计算
- 基本搜索接口  
  - `index.search()`  
  - Top-K 查询  
- 相似度度量  
  - 内积（Dot Product）  
  - L2 距离  
- 批量搜索与性能优化  
  - `index.search_batch()`  
  - GPU 加速搜索  

---

## Chapter 4：FAISS 高级功能
- 向量归一化与度量优化  
- 支持 ID 与元数据  
  - `IndexIDMap`  
- 多索引组合（IndexShards / IndexIVF + PQ）  
- 逐步构建自定义检索管道  

---

## Chapter 5：FAISS 实战项目
- **项目 1：小规模文本向量检索**  
  - 使用 Transformer Embedding  
  - 构建 Flat 索引与查询  
- **项目 2：大规模向量搜索**  
  - 使用 IVF + PQ 索引  
  - 批量添加、训练与搜索  
- **项目 3：FAISS 与实际应用结合**  
  - 文档搜索  
  - 图像相似度搜索  
  - 嵌入向量可视化分析

---

## Chapter 6：FAISS 性能优化与调优
- 索引参数调优  
  - nlist、nprobe、m 参数的意义  
- 内存优化与 GPU 使用  
- 高并发查询场景下的索引分片策略  
- 结合 Python 多线程 / 异步操作优化搜索性能

---

## Chapter 7：FAISS 与其他工具结合
- FAISS + Milvus / Pinecone 对比  
- FAISS + LangChain / RAG 系统实践  
- 向量数据库全栈架构示例  

---

## Chapter 8：课程总结与学习路径
- 总结 FAISS 核心概念  
- 实战经验与最佳实践  
- 学生自选小项目  
  - 文本检索系统  
  - 图像相似度搜索  
  - 多模态向量检索
