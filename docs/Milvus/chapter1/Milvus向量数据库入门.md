# Chapter 1 Milvus向量数据库入门：认知与架构

## 1.1 学习目标

- 理解向量数据库的核心价值与应用场景
- 掌握Milvus的定位、发展历程与核心优势
- 拆解Milvus的核心架构，理解各组件的功能与协同逻辑
- 完成Milvus的环境部署（单机版），搭建基础学习环境
- 区分Milvus Lite、Standalone与集群版的差异，明确各版本适用场景

## 1.2 向量数据库基础认知

### 1.2.1 什么是向量与向量数据库？

在AI领域，文本、图像、音频、视频等非结构化数据经过模型（如BERT、ResNet）编码后，会转化为一组具有固定维度的数值数组，这组数组就是“向量”（也叫嵌入向量/Embedding）。向量的核心价值是通过数值维度刻画数据的语义/特征信息，两个向量的距离（如欧氏距离、余弦距离）越小，代表对应的原始数据语义/特征越相似。

向量数据库是专门用于存储、管理、检索向量数据的数据库系统，核心能力是“相似性检索”——即快速从海量向量中找到与目标向量最相似的Top K个向量。相比传统关系型数据库（如MySQL）和键值数据库（如Redis），向量数据库在处理高维向量数据、支持高效相似性检索场景中具备不可替代的优势。

### 1.2.2 向量数据库的典型应用场景

- 智能检索：图文检索、语音检索、文档相似度检索（如论文查重、知识库问答）
- AI生成式应用：大语言模型（LLM）知识库增强（RAG）、AI绘画风格匹配
- 推荐系统：商品推荐、内容推荐（基于用户行为向量与物品特征向量匹配）
- 计算机视觉：图像分类、目标识别、人脸匹配
- 其他：异常检测（如网络攻击识别）、药物分子匹配、基因序列分析

## 1.3 Milvus向量数据库概述

### 1.3.1 Milvus的定位与发展

Milvus是一款开源的、云原生的向量数据库，专注于海量高维向量数据的高效相似性检索与管理。由Zilliz公司主导开发，2019年首次发布，目前最新稳定版本为Milvus 2.x（相较于1.x版本，架构全面升级，支持分布式部署、多租户管理等企业级特性）。

Milvus的核心定位：为AI应用提供“海量向量存储+快速相似检索”的基础能力，降低AI应用开发中向量管理的复杂度。

### 1.3.2 Milvus的核心优势

- 高效检索性能：支持多种索引算法，单台机器可支撑亿级向量的毫秒级检索
- 云原生架构：基于微服务设计，支持容器化部署（Docker、K8s），可弹性扩容
- 多维度兼容：支持多模态向量（文本、图像等），兼容多种客户端（Python、Java、Go等）
- 数据安全可靠：支持数据备份、恢复，提供访问权限控制
- 开源易用：开源免费，文档完善，社区活跃，降低学习与使用成本

### 1.3.3 Milvus各部署版本差异

Milvus提供三种核心部署版本，分别为Milvus Lite、Milvus Standalone（单机版）和Milvus Distributed（集群版），各版本围绕“向量存储与检索”核心能力，在架构复杂度、部署成本、性能规模及适用场景上形成差异化，满足从原型开发到大规模生产的全流程需求。各版本核心差异如下：

#### 1. 核心定义与架构特点

- **Milvus Lite**：轻量级版本，本质是Python库，无需容器或复杂依赖，可直接导入应用程序使用。架构极简，数据以本地文件形式持久化（如生成.db文件），支持Milvus核心API，与其他部署版本的客户端代码完全兼容。
- **Milvus Standalone（单机版）**：将所有核心组件打包整合，通过Docker容器化部署，仅需单台服务器即可运行。架构保留微服务核心逻辑，但组件集中部署，兼顾易用性与基础性能，是平衡开发效率与服务稳定性的过渡方案。
- **Milvus Distributed（集群版）**：云原生分布式架构，基于Kubernetes实现多节点部署，各组件（Proxy、Query Node、Data Node等）独立扩容、弹性伸缩。采用计算与存储分离设计，通过协调器统一调度任务，保障大规模场景下的高可用性与高并发处理能力。

#### 2. 关键能力与限制对比

- **数据规模支撑**：Milvus Lite适用于百万级向量数据集；Milvus Standalone可扩展至1亿级向量；Milvus Distributed支持百亿级甚至更大规模的向量存储与检索。
- **功能支持**：Milvus Lite仅支持FLAT索引类型，不支持分区、用户权限管理等高级功能；Standalone支持多种索引算法与基础数据管理功能；集群版新增多租户管理、负载均衡、故障自动恢复等企业级特性。
- **部署与运维成本**：Milvus Lite通过`pip install pymilvus`即可安装，运维成本几乎为零；Standalone依赖Docker环境，部署步骤简单，运维难度低；集群版需依赖Kubernetes生态，需专业团队进行集群管理与运维。

#### 3. 适用场景推荐

- Milvus Lite：Jupyter Notebook原型开发、RAG演示、边缘设备本地搜索（如专有文档离线检索）、小规模测试验证场景。
- Milvus Standalone：早期生产环境、中小规模应用部署、开发/测试环境搭建、对运维资源要求较低的场景。
- Milvus Distributed：大规模生产环境、高并发检索需求（如大型电商推荐系统）、海量多模态数据检索（如全网图文检索平台）、对系统可用性与扩展性要求极高的企业级应用。

补充说明：三种版本支持统一API，基于Milvus Lite开发的应用可无缝迁移至Standalone或集群版，数据也可通过工具直接导出迁移，保障开发流程的连贯性。

> 需要注意的是，milvus_lite版本【***\*不支持windows系统\****】

## 1.4 Milvus核心架构解析

Milvus 2.x采用微服务架构，各组件独立部署、协同工作，核心组件分为“数据处理层”“元数据管理层”“存储层”三大模块，具体组件及功能如下：

### 1.4.1 数据处理层（核心业务组件）

- Proxy：接入层组件，负责接收客户端请求（如插入、查询、删除），进行请求校验、路由分发，是客户端与Milvus的交互入口
- Query Node：查询节点，负责加载索引并执行相似性检索任务，返回检索结果
- Data Node：数据节点，负责处理异步数据写入任务，将原始数据转化为结构化数据并写入存储层
- Index Node：索引节点，负责异步构建索引（基于用户创建的索引类型），提升检索性能

### 1.4.2 元数据管理层

Meta Service：元数据服务，基于ETCD实现，负责存储Milvus的核心元数据，如数据库/集合信息、用户权限信息、组件节点信息等，保障整个系统的正常协同。

### 1.4.3 存储层

- MinIO/S3：对象存储，用于存储原始向量数据、索引文件等大文件数据
- Pulsar/Kafka：消息队列，用于存储数据写入的日志信息，保障数据传输的可靠性与异步处理能力

### 1.4.4 架构核心逻辑总结

客户端通过Proxy发送请求 → Proxy将请求分发至对应组件（Data Node处理写入、Index Node构建索引、Query Node处理查询） → 元数据由Meta Service管理 → 数据最终存储在对象存储与消息队列中，实现“读写分离、异步处理”的高效架构模式。

## 1.5 实操：Milvus单机版环境部署（Docker方式）

### 1.5.1 系统要求

- **操作系统**：Linux（推荐 Ubuntu 20.04+ / CentOS 7+）或 Windows（需 Docker Desktop）
- **硬件要求**：内存：≥ 8GB（建议 16GB 以上）
- 硬盘：≥ 50GB SSD（存储向量数据和日志）

**依赖项**：Docker 19.03+ 和 Docker Compose 1.25.1+

> Milvus支持docker部署，因此在不同平台上具有较好的通用性，如果不了解docker 可以参考 [Docker+万字教程](https://github.com/datawhalechina/daily-interview/blob/master/开发/Docker+万字教程：从入门到掌握.pdf)

### 1.5.2 Milvus运行

在 Microsoft Windows 上安装 Docker Desktop 后，就可以在管理员模式下通过 PowerShell 或 Windows 命令提示符访问 Docker CLI。你可以在 PowerShell、Windows Command Prompt 或 WSL 2 中运行 Docker Compose 来启动 Milvus。

#### 从 PowerShell 或 Windows 命令提示符

1. 在管理员模式下右击并选择**以管理员身份运行**，打开 Docker Desktop。
2. 在 PowerShell 或 Windows Command Prompt 中运行以下命令，为 Milvus Standalone 下载 Docker Compose 配置文件并启动 Milvus。

```powershell
# Download the configuration file and rename it as docker-compose.yml
C:\>Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.4.15/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml

# Start Milvus
C:\>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

![milvus启动](/images/fig2.png)

根据网络连接情况，下载用于安装 Milvus 的映像可能需要一段时间。名为**milvus-standalone**、**milvus-minio** 和**milvus-etcd**的容器启动后，你可以看到：

- **milvus-etcd**容器不向主机暴露任何端口，并将其数据映射到当前文件夹中的**volumes/etcd**。
- **milvus-minio**容器使用默认身份验证凭据在本地为端口**9090**和**9091**提供服务，并将其数据映射到当前文件夹中的**volumes/minio**。
- **milvus-standalone**容器使用默认设置为本地**19530**端口提供服务，并将其数据映射到当前文件夹中的**volumes/milvus**。

如果安装了 WSL 2，还可以调用 Linux 版本的 Docker Compose 命令。

#### 从 WSL 2

1. 启动 WSL 2

```bash
C:\>wsl --install
Ubuntu already installed.
Starting Ubuntu...
```

1. 下载 Milvus 配置文件。

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.17/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

1. 启动 Milvus。

```bash
sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

启动成功后可看到相关容器创建完成的提示（如上述命令输出所示）。

#### Linux 环境部署补充（延续原步骤）

1. 创建Milvus部署目录：

```bash
mkdir -p ~/milvus && cd ~/milvus
```

1. 下载Milvus单机版Docker Compose配置文件：

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.5/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

（注：若无法访问GitHub，可通过国内镜像获取，或直接在Milvus官网下载）

1. 启动Milvus服务：

```bash
docker-compose up -d
```

1. 验证服务是否启动成功：

```bash
docker-compose ps
```

若看到“milvus-standalone”“etcd”“minio”三个容器状态均为“Up”，则部署成功。

### 1.5.3 Attu安装

Attu 是一款一体化的 Milvus 管理工具，旨在管理和与 Milvus 交互，提供以下功能：

- **数据库、集合和分区管理：**有效地组织和管理您的 Milvus 设置。
- **向量嵌入的插入、索引和查询：**轻松处理 Milvus 向量数据操作。
- **执行矢量搜索：**使用矢量搜索功能快速验证您的结果。
- **用户和角色管理：**轻松管理 Milvus 权限和安全性。
- **查看系统拓扑：**可视化 Milvus 系统架构，以便更好地管理和优化。

Attu 提供了2种安装的方法：

#### 方法一、Docker 方式安装

1. 打开cmd命令行窗口，并执行以下代码获取IP地址：

```cmd
ipconfig
```

![fig3](/images/fig3.png)

找到 WSL (Hyper-V firewall) IPv4地址。

1. 执行安装Attu命令：将以下命令中的{milvus server IP}替换成确保 Attu 容器可以访问 Milvus 的 IP 地址

```bash
docker run -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:v2.5.6
```

示例：

```bash
docker run -p 8000:3000 -e MILVUS_URL=172.X.X.1:19530 zilliz/attu:v2.5.6  # 根据实际的版本进行替换
```

需要注意的是，Attu和Milvus之间有兼容性问题，最好按照官方推荐的版本来：

| Milvus 版本 | 推荐 Attu 版本                                               |
| :---------- | :----------------------------------------------------------- |
| 2.5.x       | [v2.5.6](https://github.com/zilliztech/attu/releases/tag/v2.5.6) |
| 2.4.x       | [v2.4.12](https://github.com/zilliztech/attu/releases/tag/v2.4.12) |
| 2.3.x       | [v2.3.5](https://github.com/zilliztech/attu/releases/tag/v2.3.5) |
| 2.2.x       | [v2.2.8](https://github.com/zilliztech/attu/releases/tag/v2.2.8) |
| 2.1.x       | [v2.2.2](https://github.com/zilliztech/attu/releases/tag/v2.2.2) |

![fig4](/images/fig4.png)

等待执行完毕后，在浏览器打开：http://127.0.0.1:8000/ 即可访问Attu管理界面。

![fig5](/images/fig5.png)

#### 方法二、Windows 独立安装包方式

1. 下载Attu Windows独立安装包，下载链接：https://github.com/zilliztech/attu/releases/tag/v2.5.8

![fig6](/images/fig6.png)

1. 运行安装包完成安装，安装完成后启动Attu，输入Milvus服务的IP地址和端口（默认19530）即可连接使用。

![fig7](/images/fig7.png)

### 1.5.4 停止与卸载Milvus

- 停止Milvus服务：docker-compose down
- 删除Milvus数据（如需重新部署）：rm -rf ~/milvus

## 1.6 本章小结

本章核心掌握：
- 向量数据库的核心价值与应用场景；
- Milvus的定位与核心优势；
- Milvus 2.x的核心架构及各组件功能；
- 完成Milvus单机版部署；
  
其中，各部署版本的差异认知可帮助后续根据项目规模与需求精准选择部署方案，而单机版部署则为基础实操能力的核心载体。
后续章节将基于此基础，深入学习Milvus的核心概念与实操技能。
