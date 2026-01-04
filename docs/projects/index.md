# 实战项目

本教程包含多个实战项目，涵盖向量数据库的各个应用场景。

## 项目列表

### 🔗 [URL处理实践](./url-process/)
基于RAG技术的问答系统，支持视频链接提取和智能回答。

**技术栈**: ZhipuAI + Milvus + Gradio
**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/url_process

### 🚀 [Cre_milvus 主项目](./cre-milvus/)
通用向量化处理器，支持多种文件格式的向量化存储。

**技术栈**: Milvus + Streamlit + FastAPI
**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/Cre_milvus

### 🚀 [graph_rag](./graph_rag)
文搜图

**技术栈**: Milvus + langgraph
**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/graph_rag

### 🚀 [HDBSCAN](./HDBSCAN)
聚类数据可视化

**技术栈**: Milvus + HDBSCAN + umap
**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/HDBSCAN

### 📊 [K8s+Loki 监控](./k8s-loki/)
基于Kubernetes部署的Milvus日志监控系统。

**技术栈**: Kubernetes + Grafana + Loki
**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/k8s+loki

### 🧠 [Meta-chunking](./meta-chunking/)
Meta-chunking论文的代码实现demo。

**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/Meta_chunking

### 🧠 [Limit](../Milvus/chapter4/向量/code/Meta_limit/code/startup.md)
Meta:Limit论文的代码实现demo。

**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/docs/Milvus/chapter4/%E5%90%91%E9%87%8F/code/Meta_limit

### ⚡ [Locust性能测试](./locust/)
Milvus性能测试工具和基准测试。

**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/locustProj

### 🚀 [Faiss](./faissSear)
基于Faiss的问答系统实战。

**源码位置**: https://github.com/datawhalechina/easy-vectordb/tree/main/src/faissSear
## 学习路径

1. **入门**: 从URL处理实践开始
2. **使用**: 尝试HDBSCAN聚类
3. **进阶**：学习Cre_milvus主项目(开发ing)
4. **部署**: 掌握K8s+Loki监控
5. **优化**: 研究Meta-chunking和性能测试
6. **扩展**: 探索Meta-limit和其他项目