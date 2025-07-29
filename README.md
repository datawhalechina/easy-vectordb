# 项目名称

向量数据库部署与实践指南


# 教程
## 📖 内容导航

| **Milvus**     | 章节                                       | 关键内容                                     | 状态   |
|----------|-------------------------------------------|---------------------------------------------|--------|
| 前言 | [前言](./docs/Milvus/chapter0/前言.md)    | 项目目标与大纲                                |   ✅    |
|    第一章  | [向量数据库介绍](./docs/Milvus/chapter1/Milvus%20介绍.md) | 核心概念/发展历程/应用场景/技术选型            |   ⌛    |
|      | [ 索引介绍](./docs/Milvus/chapter1/milvus%20索引介绍.md) | 索引原理/类型选择/优化策略/重建方法           | ✅     |
|          | [ 聚类介绍](./docs/Milvus/chapter1/聚类算法介绍.md) | 聚类原理/算法选择/优化策略                  | ✅     |
|    第二章    | [Milvus Lite部署与应用](./docs/Milvus/chapter2/Milvus%20Lite部署与应用.md) | Lite部署方案             | ✅     |
|          | [Milvus Standalone部署](./docs/Milvus/chapter2/Milvus%20Standalone部署.md) | Standalone部署方案                            | ✅     |
|          | [ MinerU部署](./docs/Milvus/chapter2/MinerU部署教程.md) | MinerU部署方案                            | ✅     |
|    第三章   | [ Milvus 文本嵌入实战](./docs/Milvus/chapter3/milvus%20文本嵌入实战.md) | 文本嵌入实战案例                            | ✅     |
|          | [Milvus pdf嵌入实战](./docs/Milvus/chapter3/milvus%20pdf%20嵌入实战.md) | pdf嵌入实战案例                            | ✅     |
|          | [Milvus pdf多模型嵌入实战](./docs/Milvus/chapter3/milvus%20pdf%20多模型嵌入实战.md) | pdf多模型嵌入实战案例                            | ✅     |
|          | [Milvus 数据切分总结](./docs/Milvus/chapter3/milvus%20数据切分总结.md) | 数据切分场景和应用总结                            | ✅     |
|      第四章| [ Milvus 存储优化](./docs/Milvus/chapter4/milvus%20存储优化.md) | mmp理论与实践，数据切分策略                            | ✅     |
|          | [ FunsionAnns](./docs/Milvus/chapter4/GPU加速检索-基于FusionANNS.md) | FusionANNS论文解读                 |    ✅    |
|          | [K-means](./docs/Milvus/chapter4/K-mean算法详解.md) | An efficient K-means clustering algorithm for massive data论文解读                 |    ⌛    |
|          | [Meta-Chunking：一种新的文本切分策略](./docs//Milvus/chapter4/Meta-Chunking：一种新的文本切分策略.md) | Learning Text Segmentation and Semantic Completion via Logical Perception论文解读                 |    ⌛    |
|      第五章   | [url分割处理](./docs/Milvus/project/url_process/README.md) | 视频数据存储过程中url分割与存放                 |   ✅     |
|          | [Cre_Milvus](./docs/Milvus/project/Cre_milvus/PROJECT_STRUCTURE.md) | 综合实践                 |     ⌛   |
|          | [Meta-chunking](./docs/Milvus/project/Meta_chunking/README.md) | Meta-chunking论文实现demo                 |     ✅   |
|          | [Locust](./docs/Milvus/project/locustProj/README.md) | 基于Locust的Milvus性能测试工具                 |     ✅   |
|     第六章     | [k8s部署Milvus监控](./docs/Milvus/project/k8s+loki/README.md) | 基于loki与Grafana的Milvus监控系统                 |     ✅   |

| **Faiss**     | 章节                                       | 关键内容                                     | 状态   |
|----------|-------------------------------------------|---------------------------------------------|--------|
|          | *待补充*                                  | Faiss核心原理与架构                          | ⌛     |
|          | *待补充*                                  | 索引构建与参数调优                            | ⌛     |
|          | *待补充*                                  | GPU加速方案                                 | ⌛     |
|          | *待补充*                                  | 大规模向量检索实践                            | ⌛     |
|          | *待补充*                                  | 文本嵌入实战                            | ⌛     |
|          | *待补充*                                  | 不同数据库比较                      | ⌛     |

---

## 📄 补充文档
- [Datawhale社区介绍](./docs/Datawhale%E7%A4%BE%E5%8C%BA%E4%BB%8B%E7%BB%8D.pdf)
## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

### 核心贡献者
- [牧小熊-项目负责人](https://github.com/muxiaoxiong) (Datawhale成员-武汉社区负责人)
- [刘晓-项目贡献者](https://github.com/Halukisan)(Datawhale鲸英助教)


### 特别感谢
- 感谢 [@Sm1les](https://github.com/Sm1les) 对本项目的帮助与支持
- 感谢所有为本项目做出贡献的开发者们 ❤️


## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>


## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
