<div align='center'>
    <img src="./logo.png" alt="alt text" width="100%">
    <h1>Easy-vectorDB</h1>
</div>
<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/stargazers) [![GitHub forks](https://img.shields.io/github/forks/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/network/members) [![GitHub issues](https://img.shields.io/github/issues/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/issues) [![GitHub license](https://img.shields.io/github/license/datawhalechina/easy-vectordb?style=flat-square)](https://github.com/datawhalechina/easy-vectordb/blob/main/LICENSE)

[中文](./README.md) | [English](./README_en.md)

[📚 在线阅读地址](https://datawhalechina.github.io/easy-vectordb/)

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


### 🏁 一、基础学习篇（Base）

> 💡 了解向量数据库的设计哲学、嵌入算法与搜索原理，为后续实践打下坚实基础。

| 章节 | 内容 | 关键词 |
|------|------|--------|
| **Chapter 1** | [项目介绍](./doc/base/chapter1/项目介绍.md) | 项目目标 / 学习路径 |
| **Chapter 2** | [为什么需要向量数据库](./doc/base/chapter2/为什么需要向量数据库.md) | 检索瓶颈 / 相似度搜索原理 |
| **Chapter 3** | [向量嵌入算法基础](./doc/base/chapter3/向量嵌入算法基础.md) | Word2Vec / Transformer Embedding |
| **Chapter 4** | [向量搜索算法基础](./doc/base/chapter4/向量搜索算法基础.md) | 暴力检索 / 向量相似度 |
| **Chapter 5** | [ANN 搜索算法](./doc/base/chapter5/ANN搜索算法.md) | IVF / PQ / HNSW / LSH 算法原理/代码实战|
| **Chapter 6** | [实现你自己的向量数据库](./doc/base/chapter6/实现你自己的向量数据库.md) | 代码实战|


### 🔧 二、Faiss 教程（Faiss）

> ⚙️ 从零构建高性能向量检索系统，掌握 Faiss 的索引机制与加速策略。

| 章节 | 内容 | 关键词 |
|------|------|--------|
| **Chapter 1** | [Faiss 引言](./docs/Faiss/引言.md) | 向量检索基础 / Faiss 概览 |
|                | [基础使用](./docs/Faiss/基础使用.md) | 索引构建 / 向量检索流程 |
| **Chapter 2** | [索引](./docs/Faiss/索引.md) | CPU 索引结构 / IVF / Flat |
|                | [GPU 加速](./docs/Faiss/GPU加速.md) | GPU 索引 / 性能对比 / 批量检索 |
| **Chapter 3** | [问答实战](./docs/Faiss/问答实战.md) | 基于 Faiss 的问答检索系统构建 |
|                | [总结](./docs/Faiss/总结.md) | 经验总结 / 性能优化技巧 |
| ⏳ **持续更新中...** |  |  |

> 📘 本项目旨在让你从 **原理 → 实践 → 部署** 全流程掌握向量数据库核心知识与实战能力。

---


## 📄 补充资源

- 📚 [Datawhale社区介绍](./docs/Datawhale%E7%A4%BE%E5%8C%BA%E4%BB%8B%E7%BB%8D.pdf)
- 🌐 [在线文档站点](https://datawhalechina.github.io/easy-vectordb/)
- 💻 [项目源码](https://github.com/datawhalechina/easy-vectordb/tree/main/src)

## 🤝 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

### 核心贡献者
- [牧小熊-项目负责人](https://github.com/muxiaoxiong) (Datawhale成员-武汉社区负责人)
- [刘晓-项目贡献者](https://github.com/Halukisan)(Datawhale鲸英助教)

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







