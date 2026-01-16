<div align='center'>
    <img src="./logo.png" alt="alt text" width="100%">
    <h1>Easy-vectorDB（⚠️ Alpha内测版）</h1>
</div>

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/stargazers) [![GitHub forks](https://img.shields.io/github/forks/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/network/members) [![GitHub issues](https://img.shields.io/github/issues/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/issues) [![GitHub license](https://img.shields.io/github/license/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/blob/main/LICENSE)

[中文](./README.md) | [English](./README_en.md)

[📚 在线阅读地址](https://datawhalechina.github.io/easy-vecdb/)

📚 从零开始的向量数据库原理与实践教程

</div>

## 🧭 项目简介

**EasyVectorDB** 是一个面向开发者与研究者的 **向量数据库系统性学习项目**。  
项目内容覆盖从基础概念、算法原理到生产级应用部署的全流程，聚焦以下三个方向：

- 🧩 **理论入门**：理解向量数据库的原理、架构与索引机制  
- ⚙️ **实战教程**：掌握 Milvus / Faiss / Annoy 的使用与优化技巧  
- 💡 **项目案例**：从零构建 RAG、嵌入检索、聚类可视化等完整项目  

---


## 📖 内容导航

项目共分为 **基础学习篇** 与 **实践篇** 两个部分，对应导航栏配置如下：

| 章节 | 关键内容 | 状态 |
| --- | --- | --- |
| <strong>第一部分：基础学习篇（Base）</strong> | 向量数据库原理、嵌入与搜索基础 |  |
| [Chapter 1 项目介绍](./docs/base/chapter1/项目介绍.md) | 项目目标、整体学习路径 | ✅ |
| [Chapter 2 为什么需要向量数据库](./docs/base/chapter2/为什么需要向量数据库.md) | 检索瓶颈、相似度搜索原理 | ✅ |
| [Chapter 3 向量嵌入算法基础](./docs/base/chapter3/向量嵌入算法基础.md) | Word2Vec、Transformer Embedding | ✅ |
| [Chapter 4 向量搜索算法基础](./docs/base/chapter4/向量搜索算法基础.md) | 暴力检索、向量相似度 | ✅ |
| [Chapter 5 ANN 搜索算法](./docs/base/chapter5/ANN搜索算法.md) | IVF、PQ、HNSW、LSH、Annoy算法原理与代码实战 | ✅ |
| [Chapter 6 实现你自己的向量数据库](./docs/base/chapter6/实现你自己的向量数据库.md) | 向量数据库最小实现 | ✅ |
| <strong>第二部分：Faiss 教程（Faiss）</strong> | 高性能向量检索引擎实战 |  |
| [Chapter 1 FAISS 入门与环境搭建](./docs/faiss/chapter1/FAISS入门与环境搭建.md) | 安装配置、基础概念 | ✅ |
| [Chapter 2 FAISS 核心索引实战](./docs/faiss/chapter2/FAISS数据结构与索引.md) | Flat、IVF、PQ、HNSW 等索引 | ✅ |
| [Chapter 3 FAISS 核心功能进阶](./docs/faiss/chapter3/FAISS核心功能进阶.md) | 复合索引、GPU、批量检索 | ✅ |
| [Chapter 4 FAISS 性能调优与评估](./docs/faiss/chapter4/FAISS性能调优与评估.md) | Recall、延迟、内存调优 | ✅ |
| [Chapter 5 FAISS 工程化落地实战](./docs/faiss/chapter5/FAISS工程化落地实战.md) | 工程结构、服务化、实战案例 | ✅ |
| <strong>第三部分：Milvus 教程（Milvus）</strong> | 分布式向量数据库与工程实践 |  |
| [Chapter 1 Milvus 向量数据库入门：认知与架构](./docs/milvus/chapter1/Milvus向量数据库入门.md) | 架构设计、核心组件 | ✅ |
| [Chapter 2 Milvus 核心概念：数据模型与索引体系](./docs/milvus/chapter2/Milvus核心概念.md) | Collection、Partition、Index | ✅ |
| [Chapter 3 Milvus 基础操作：PyMilvus核心API实战](./docs/milvus/chapter3/PyMilvus核心API实战.md) | Milvus数据写入、查询、索引管理 | ✅ |
| [Chapter 4 Milvus的AI应用开发：基于BM25的混合搜索向量数据库开发实战](./docs/milvus/chapter4/Milvus的AI应用开发.md) | RAG、混合向量检索应用 | ✅ |
| [Chapter 5 Milvus的AI应用开发：图像检索应用实战](./docs/milvus/docs/Milvus/chapter5/Milvus的AI应用开发.md) | 图像检索应用 | ✅ |
| [Chapter 6 Milvus 选学部分](./docs/milvus/chapter6/Milvus底层架构详解.md) |Milvus底层架构详解、Milvus reranker、Milvus Lite部署与应用、MinerU部署教程 | ✅ |
| <strong>第四部分：基于向量数据库的AI应用开发</strong> |  |  |
| [project 1 基于FAISS框架RAG实战项目](./docs/projects/project1/README.md)  | RAG with FAISS    |✅  |
| [project 2 基于Milvus框架的Agent项目](./docs/projects/project2/README.md) | Agent with Milvus     |✅   |
| [project 3 基于Milvus和ArangoDB的RAG系统](./docs/projects/project3/README.md) | RAG with Milvus & ArangoDB     |✅  |
| <strong>第五部分：补充内容</strong> | 与向量数据库有关的内容 |  |
| [向量基础知识](./docs/more/chapter5/向量.md) | 向量基础概念与数学原理 | ✅ |
| [FusionANNS架构设计](./docs/more/chapter1/GPU加速检索-基于FusionANNS.md) | GPU加速检索系统架构 | ✅ |
| [Meta-Chunking策略](./docs/more/chapter2/Meta-Chunking：一种新的文本切分策略.md) | 智能文本切分算法 | ✅ |
| [检索理论极限](./docs/more/chapter3/Limit基于嵌入检索的理论极限.md) | 向量检索性能边界分析 | ✅ |
| [RabitQ索引技术](./docs/more/chapter4/RabitQ：用于近似最近邻搜索的带理论误差界的高维向量量化.md) | 高维向量量化方法 | ✅ |
| [聚类算法](./docs/more/chapter6/聚类算法介绍.md) | 聚类算法介绍 | ✅ |
或者你想要添加更多具体的文档项？可以告诉我具体要补充哪些内容。

如果你是想在原来的JSON导航结构中添加，也可以告诉我具体要加在哪个位置。
⏳ **持续更新中...** 

> 📘 本项目旨在让你从 **原理 → 实践 → 部署** 全流程掌握向量数据库核心知识与实战能力。


## 🛠️ 项目目录结构说明

```
.
├── docs 向量数据库学习指南与项目文档
├── data 通用示例数据目录
├── src  项目相关代码
└── tmp  临时文件目录
```

## 📄 补充资源

- 📚 [Datawhale社区介绍](./docs/Datawhale%E7%A4%BE%E5%8C%BA%E4%BB%8B%E7%BB%8D.pdf)
- 🌐 [在线文档站点](https://datawhalechina.github.io/easy-vectordb/)
- 💻 [项目源码](https://github.com/datawhalechina/easy-vectordb/tree/main/src)

【相关竞赛】
- 🚩[2025 全国大学生计算机系统能力大赛——第2届PolarDB数据库创新设计赛](https://tianchi.aliyun.com/competition/entrance/532409)

## 🤝 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

### 核心贡献者
- [牧小熊-项目负责人](https://github.com/muxiaoxiong)(Datawhale成员)
- [刘晓-项目贡献者](https://github.com/Halukisan)(Datawhale鲸英助教)
- [柯慕灵-项目贡献者](https://github.com/1985312383)(Datawhale成员)
- [赵鑫龙-项目贡献者](https://github.com/xiaoming910)(Datawhale精英助教)
- [陈辅元-项目贡献者](https://github.com/Fyuan0206)(Datawhale成员)
### 特别感谢

- 感谢 [@Sm1les](https://github.com/Sm1les) 对本项目的帮助与支持
- 感谢所有为本项目做出贡献的开发者们 ❤️

<div align="left">

<a href="https://github.com/datawhalechina/easy-vectordb/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=datawhalechina/easy-vectordb" />
</a>

</div>


## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>


## 📊 Star History

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=datawhalechina/easy-vectordb&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=datawhalechina/easy-vectordb&type=Date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=datawhalechina/easy-vectordb&type=Date" />
</picture>

---
</div>

## 📜 开源协议

<div align="left">

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
  <img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" />
</a>

本作品采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。

**Made with ❤️ by Datawhale**

</div>



















