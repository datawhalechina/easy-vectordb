# Cre_milvus - 智能向量数据库管理系统

## 项目简介

Cre_milvus是一个集成了向量数据库、智能分块、语义搜索和聚类分析的综合性系统。经过重构后，项目具有清晰的架构和统一的启动入口。

## 功能特性

- 🔍 **智能搜索**: 基于向量相似度的语义搜索
- ✂️ **智能分块**: 支持多种文本分块策略（传统、PPL、MSP、语义分块等）
- 📊 **聚类分析**: 自动聚类和可视化搜索结果
- 🎨 **Web界面**: 基于Streamlit的用户友好界面
- 🚀 **API服务**: 基于FastAPI的RESTful API
- 🗄️ **向量存储**: 基于Milvus的高性能向量数据库

## 系统要求

- Python 3.8+
- Milvus服务器 (推荐使用Docker)
- 8GB+ 内存 (用于模型加载) 
- 注意，一定是空闲内存大于等于8GB
- **如果你的电脑内存是16GB，请保证运行程序之前，电脑内存占用在5GB-6GB之间**

## 快速开始

### 1. 安装依赖

```bash
cd Cre_milvus
pip install -r requirements.txt
```

### 2. 启动Milvus服务器

使用Docker启动Milvus服务器：

```bash
docker run -p 19530:19530 milvusdb/milvus:v2.4.17
```

或者参考[Milvus官方文档](https://milvus.io/docs/install_standalone-docker.md)进行安装。

### 3. 启动系统

```bash
python simple_startup.py
```

系统将自动：
- 加载配置文件
- 初始化Milvus连接
- 初始化向量模型
- 初始化Qwen模型（用于PPL分块）
- 启动后端API服务 (端口12089)
- 启动前端界面 (端口12088)
- 测试前后端连接

### 4. 访问系统

- **前端界面**: http://localhost:12088
- **后端API**: http://localhost:12089
- **API文档**: http://localhost:12089/docs

## 配置说明

主要配置文件为 `config.yaml`：

```yaml
milvus:
  host: "127.0.0.1"        # Milvus服务器地址
  port: "19530"            # Milvus服务器端口
  collection_name: "Test_one"  # 集合名称

system:
  backend_port: 12089       # 后端API端口
  frontend_port: 12088      # 前端界面端口

chunking:
  strategy: "traditional"  # 分块策略
  chunk_length: 512        # 块长度
```

## 使用流程

1. **上传文档**: 通过前端界面上传PDF、TXT、MD等文档
2. **自动处理**: 系统自动进行文本分块、向量化、存储
3. **语义搜索**: 输入查询文本，获得相关文档片段
4. **结果分析**: 查看聚类结果和可视化分析

## 支持的文件格式

- PDF文档 (.pdf)
- 文本文件 (.txt)
- Markdown文件 (.md)
- CSV文件 (.csv)
- 图像文件 (.jpg, .png, .bmp) - 需要启用多模态功能

## 分块策略

- **traditional**: 传统固定长度分块
- **meta_ppl**: 基于困惑度的智能分块
- **msp**: 基于边际采样的分块
- **semantic**: 基于语义相似度的分块

## 故障排除

### 常见问题

1. **Milvus连接失败**
   - 确保Milvus服务器正在运行
   - 检查端口19530是否可访问
   - 查看防火墙设置

2. **模型加载失败**
   - 确保网络连接正常（需要下载模型）
   - 检查磁盘空间是否充足
   - 等待模型下载完成

3. **端口占用**
   - 修改config.yaml中的端口配置
   - 或者停止占用端口的其他程序
   - 在作者测试的情况下发现，请勿频繁重启项目，系统的向量化或者qwen模型并不会因为你项目ctrl+c终止而终止，**这会导致端口的占用问题，请使用命令行kill掉端口占用**

### 日志查看

系统启动日志保存在 `system_startup.log` 文件中。

## 项目结构

```
Cre_milvus/
├── start_simple.py         # 统一启动入口和Milvus连接管理
├── backend_api.py          # FastAPI后端服务
├── frontend.py             # Streamlit前端界面
├── config.yaml             # 主配置文件
├── simple_startup.py       # 服务启动管理
├── config_loader.py        # 配置加载器
├── dataBuilder/            # 数据处理模块
├── Search/                 # 搜索与检索
├── System/                 # 系统核心
└── data/upload/            # 用户上传数据
```

## 开发说明

- **前端代码**: `frontend.py` (基于Streamlit)
- **后端代码**: `backend_api.py` (基于FastAPI)
- **核心逻辑**: `System/start.py`
- **数据处理**: `dataBuilder/data.py`


## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请通过GitHub Issues联系。