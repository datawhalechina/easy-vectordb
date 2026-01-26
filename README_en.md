<div align='center'>
    <img src="./logo.png" alt="alt text" width="100%">
    <h1>Easy-vecDB (⚠️ Alpha Internal Test)</h1>
</div>

> [!CAUTION]
> ⚠️ Alpha version warning: This is an early internal build. It is not yet complete and may contain bugs. Issues and suggestions are very welcome via GitHub Issues.

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/datawhalechina/easy-vecdb?style=flat-square)](https://github.com/datawhalechina/easy-vecdb/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/datawhalechina/easy-vecdb?style=flat-square)](https://github.com/datawhalechina/easy-vecdb/network/members)
[![GitHub issues](https://img.shields.io/github/issues/datawhalechina/easy-vecdb?style=flat-square)](https://github.com/datawhalechina/easy-vecdb/issues)
[![GitHub license](https://img.shields.io/github/license/datawhalechina/easy-vecdb?style=flat-square)](https://github.com/datawhalechina/easy-vecdb/blob/main/LICENSE)

[中文](./README.md) | [English](./README_en.md)

[📚 Online Documentation](https://datawhalechina.github.io/easy-vecdb/)

📚 A Hands-on Tutorial on Vector Database Principles and Practices — From Zero to Production

</div>

## 🧭 Project Overview

**EasyVecDB** is a **systematic learning project on vector databases** designed for developers and researchers.  
It covers the full lifecycle from foundational concepts and algorithmic principles to production-level application deployment, focusing on three main directions:

- 🧩 **Theory Fundamentals**: Understand the principles, architecture, and indexing mechanisms of vector databases  
- ⚙️ **Hands-on Practice**: Master the usage and optimization of Milvus / Faiss / Annoy  
- 💡 **Project Cases**: Build complete projects from scratch, including RAG systems, embedding-based retrieval, and clustering visualization  

---

## 📖 Content Navigation

The project is divided into **Fundamentals** and **Practice** sections, corresponding to the navigation structure below:

| Section | Key Content | Status |
| --- | --- | --- |
| <strong>Part I: Fundamentals (Base)</strong> | Vector DB principles, embeddings, and search basics |  |
| [Chapter 1 Project Introduction](./docs/base/chapter1/项目介绍.md) | Project goals and learning roadmap | ✅ |
| [Chapter 2 Why Vector Databases](./docs/base/chapter2/为什么需要向量数据库.md) | Retrieval bottlenecks and similarity search | ✅ |
| [Chapter 3 Vector Embedding Basics](./docs/base/chapter3/向量嵌入算法基础.md) | Word2Vec, Transformer embeddings | ✅ |
| [Chapter 4 Vector Search Basics](./docs/base/chapter4/向量搜索算法基础.md) | Brute-force search and similarity metrics | ✅ |
| [Chapter 5 ANN Search Algorithms](./docs/base/chapter5/ANN搜索算法.md) | IVF, PQ, HNSW, LSH, Annoy principles & code | ✅ |
| [Chapter 6 Build Your Own Vector Database](./docs/base/chapter6/实现你自己的向量数据库.md) | Minimal vector DB implementation | ✅ |
| <strong>Part II: Faiss Tutorial</strong> | High-performance vector search engine |  |
| [Chapter 1 FAISS Introduction & Setup](./docs/faiss/chapter1/FAISS入门与环境搭建.md) | Installation and core concepts | ✅ |
| [Chapter 2 FAISS Core Indexes](./docs/faiss/chapter2/FAISS数据结构与索引.md) | Flat, IVF, PQ, HNSW indexes | ✅ |
| [Chapter 3 Advanced FAISS Features](./docs/faiss/chapter3/FAISS核心功能进阶.md) | Composite indexes, GPU, batch search | ✅ |
| [Chapter 4 FAISS Performance Tuning](./docs/faiss/chapter4/FAISS性能调优与评估.md) | Recall, latency, memory optimization | ✅ |
| [Chapter 5 FAISS Engineering Practices](./docs/faiss/chapter5/FAISS工程化落地实战.md) | Service deployment and real-world cases | ✅ |
| <strong>Part III: Milvus Tutorial</strong> | Distributed vector database & engineering |  |
| [Chapter 1 Milvus Introduction: Concepts & Architecture](./docs/milvus/chapter1/Milvus向量数据库入门.md) | Architecture and core components | ✅ |
| [Chapter 2 Milvus Core Concepts](./docs/milvus/chapter2/Milvus核心概念.md) | Collection, Partition, Index | ✅ |
| [Chapter 3 Milvus Basic Operations](./docs/milvus/chapter3/PyMilvus核心API实战.md) | Data ingestion, query, index management | ✅ |
| [Chapter 4 Milvus AI Applications: Hybrid Search with BM25](./docs/milvus/chapter4/Milvus的AI应用开发.md) | RAG and hybrid retrieval | ✅ |
| [Chapter 5 Milvus AI Applications: Image Retrieval](./docs/milvus/docs/Milvus/chapter5/Milvus的AI应用开发.md) | Image retrieval system | ✅ |
| [Chapter 6 Milvus Advanced Topics](./docs/milvus/chapter6/Milvus底层架构详解.md) | Internal architecture, reranker, Milvus Lite, MinerU | ✅ |
| <strong>Part IV: AI Applications Based on Vector Databases</strong> |  |  |
| [Project 1 RAG with FAISS](./docs/projects/project1/README.md) | RAG using FAISS | ✅ |
| [Project 2 Agent with Milvus](./docs/projects/project2/README.md) | Agent system using Milvus | ✅ |
| [Project 3 RAG with Milvus & ArangoDB](./docs/projects/project3/README.md) | Hybrid RAG system | ✅ |
| <strong>Part V: Supplementary Topics</strong> | Related advanced topics |  |
| [Vector Fundamentals](./docs/more/chapter5/向量.md) | Vector math and basics | ✅ |
| [FusionANNS Architecture](./docs/more/chapter1/GPU加速检索-基于FusionANNS.md) | GPU-accelerated retrieval | ✅ |
| [Meta-Chunking Strategy](./docs/more/chapter2/Meta-Chunking：一种新的文本切分策略.md) | Intelligent text chunking | ✅ |
| [Theoretical Limits of Retrieval](./docs/more/chapter3/Limit基于嵌入检索的理论极限.md) | Performance boundaries | ✅ |
| [RabitQ Indexing](./docs/more/chapter4/RabitQ：用于近似最近邻搜索的带理论误差界的高维向量量化.md) | High-dimensional quantization | ✅ |
| [Clustering Algorithms](./docs/more/chapter6/聚类算法介绍.md) | Clustering overview | ✅ |

If you’d like to add more specific documentation items, feel free to tell me what to include.  
If you want to modify the existing JSON navigation structure, you can also specify where to add them.

⏳ **Continuously updating...**

> 📘 This project aims to help you master vector databases from **principles → practice → deployment**.

## 🛠️ Project Structure

```
.
├── docs Vector database tutorials and documentation
├── data Common example datasets
├── src Project-related source code
└── tmp Temporary files
```


## 📄 Additional Resources

- 📚 [Introduction to the Datawhale Community](./docs/Datawhale%E7%A4%BE%E5%8C%BA%E4%BB%8B%E7%BB%8D.pdf)
- 🌐 [Online Documentation](https://datawhalechina.github.io/easy-vecdb/)
- 💻 [Source Code](https://github.com/datawhalechina/easy-vecdb/tree/main/src)

**Related Competition**
- 🚩 [2025 National College Student Computer Systems Ability Competition — 2nd PolarDB Database Innovation Design Contest](https://tianchi.aliyun.com/competition/entrance/532409)

## 🤝 Contributing

- If you find any issues, feel free to open an Issue. If there is no response, you may contact the [Support Team](https://github.com/datawhalechina/DOPMC/blob/main/OP.md).
- If you’d like to contribute, submit a Pull Request. If there is no response, you may also contact the Support Team.
- If you are interested in starting a new Datawhale project, please follow the [Datawhale Open Source Project Guide](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md).

### Core Contributors
- [Muxiaoxiong – Project Lead](https://github.com/muxiaoxiong) (Datawhale Member)
- [Liu Xiao – Contributor](https://github.com/Halukisan) (Datawhale Teaching Assistant)
- [Ke Muling – Contributor](https://github.com/1985312383) (Datawhale Member)
- [Zhao Xinlong – Contributor](https://github.com/xiaoming910) (Datawhale Teaching Assistant)
- [Chen Fuyuan – Contributor](https://github.com/Fyuan0206) (Datawhale Member)

### Special Thanks
- Thanks to [@Sm1les](https://github.com/Sm1les) for the support and help
- Thanks to all contributors who made this project possible ❤️

<div align="left">
<a href="https://github.com/datawhalechina/easy-vecdb/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=datawhalechina/easy-vecdb" />
</a>
</div>

## Follow Us

<div align=center>
<p>Scan the QR code below to follow the Datawhale official account</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width="180" height="180">
</div>

## 📊 Star History

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=datawhalechina/easy-vecdb&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=datawhalechina/easy-vecdb&type=Date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=datawhalechina/easy-vecdb&type=Date" />
</picture>
</div>

## 📜 License

<div align="left">
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
  <img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" />
</a>

This work is licensed under the  
[Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

**Made with ❤️ by Datawhale**
</div>
