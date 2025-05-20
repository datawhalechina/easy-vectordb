# 项目名称

向量数据库部署与实践指南

# 项目意义

在人工智能技术高速发展的今天，向量数据库已成为支撑大模型、推荐系统、图像检索等场景的核心基础设施。然而，开发者面临部署复杂、概念抽象、实战案例不足等痛点。本教程旨在通过开源、模块化、实战驱动的方式，降低向量数据库的学习门槛，推动技术普及与社区协作。其意义体现在：

1.填补知识鸿沟：系统梳理向量数据库的核心概念与部署流程，帮助开发者快速掌握这一新兴技术

2.加速AI应用落地：结合真实案例（如RAG框架、大模型知识库），解决非结构化数据处理难题

# 项目介绍

1. 什么是向量数据库？
   向量数据库是专为高维向量数据设计的数据库系统，能够将文本、图像、音频等非结构化数据转化为数学向量（即高维数组），并基于相似性度量（如余弦相似度、欧氏距离）实现高效检索。与传统关系型数据库相比，其核心优势在于：

     - 处理非结构化数据：通过嵌入模型（Embedding）将复杂数据转化为向量，突破传统数据库的行列限制。

     - 近似最近邻搜索（ANN）：利用索引技术（如HNSW、LSH）实现毫秒级海量数据检索，支持模糊匹配。

     - 多模态支持：统一管理文本、图像、视频等多源数据，构建跨模态检索能力

2. 向量数据库有什么用？
   向量数据库是AI时代的“数据枢纽”，其应用场景包括但不限于：

     - 大模型知识库增强：通过RAG（检索增强生成）框架，为ChatGPT等模型提供实时外部知识，减少“幻觉”问题。

     - 推荐系统：基于用户行为向量与商品特征的相似性匹配，实现个性化推荐（如电商、流媒体平台）。

     - 图像/音视频检索：将多媒体内容向量化，支持以图搜图、跨模态搜索（如医疗影像分析、版权检测）。

     - 物联网与边缘计算：高效处理传感器数据流，实现实时异常检测与预测性维护。

3. 教程内容概览
   本教程将涵盖以下核心模块：

   基础部署：手把手搭建开源向量数据库（如Milvus、Chroma），配置环境与集群。

   实战案例：

     - 使用RAG框架构建企业级知识库

     - 图像检索系统开发：从数据嵌入到服务端API封装

     - 性能优化：索引选择策略、GPU加速、分布式扩展方案

# 友情链接

- [开源大模型食用指南](https://github.com/datawhalechina/self-llm)

# 教程

向量数据库软件
- [milvus](https://github.com/milvus-io/milvus)
  - [x] [milvus lite版部署](./docs/Milvus%20Lite部署与应用.md)
  - [x] [milvus Standalone版部署](./docs/Milvus%20Standalone部署.md)
  - [x] [milvus 文本嵌入实战](./docs/milvus%20%E6%96%87%E6%9C%AC%E5%B5%8C%E5%85%A5%E5%AE%9E%E6%88%98.md)
  - [x] [milvus pdf嵌入实战](./docs/milvus%20pdf%20%E5%B5%8C%E5%85%A5%E5%AE%9E%E6%88%98.md)
 
文档解析软件
- [MinerU](https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md)
  - [x] [MinerU部署](./docs/MinerU%E9%83%A8%E7%BD%B2%E6%95%99%E7%A8%8B.md)
 
pdf文档
- [Datawhale社区介绍](./docs/Datawhale%E7%A4%BE%E5%8C%BA%E4%BB%8B%E7%BB%8D.pdf)

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>


## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
