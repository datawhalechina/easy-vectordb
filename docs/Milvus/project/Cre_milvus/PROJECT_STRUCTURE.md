# CreMilvus 项目结构说明

## 📁 核心目录结构

```
Cre_milvus/
├── 🎯 核心应用文件
│   ├── backend_api.py          # FastAPI后端服务
│   ├── frontend.py             # Streamlit前端界面
│   ├── config.yaml             # 主配置文件
│   └── requirements.txt        # 依赖包列表
│
├── 📊 数据处理模块
│   └── dataBuilder/
│       ├── data.py             # 主数据处理逻辑
│       ├── __init__.py         # 模块初始化
│       ├── chunking/           # 文本分块策略
│       │   ├── __init__.py
│       │   ├── chunk_strategies.py    # 分块策略管理
│       │   ├── meta_chunking.py       # 高级分块实现
│       │   └── perplexity_chunking.py # PPL分块核心
│       └── tools/              # 文件处理工具
│           ├── csvmake.py      # CSV处理
│           ├── mdmake.py       # Markdown处理
│           ├── pdfmake.py      # PDF处理
│           ├── txtmake.py      # 文本处理
│           └── imgmake.py      # 图像处理
│
├── 🔍 搜索与检索
│   └── Search/
│       ├── embedding.py        # 嵌入向量生成
│       ├── search.py           # 搜索逻辑
│       ├── milvusSer.py        # Milvus服务
│       ├── redisSer.py         # Redis服务
│       └── ES/                 # Elasticsearch相关
│
├── 🗄️ 数据库构建
│   ├── milvusBuilder/
│   │   └── milvus.py           # Milvus数据库操作
│   └── IndexParamBuilder/
│       ├── indexparam.py       # 索引参数构建
│       └── searchparam.py      # 搜索参数构建
│
├── 🎨 多模态处理
│   └── multimodal/
│       ├── __init__.py
│       ├── clip_encoder.py     # CLIP编码器
│       ├── image_processor.py  # 图像处理器
│       └── text_processor.py   # 文本处理器
│
├── 🔧 系统核心
│   └── System/
│       ├── start.py            # 系统启动逻辑
│       ├── init.py             # 初始化模块
│       ├── monitor.py          # 监控日志
│       ├── Retry.py            # 重试机制
│       ├── eval.py             # 评估工具
│       └── cluster_utils.py    # 聚类工具
│
├── 📈 可视化与重排序
│   ├── ColBuilder/
│   │   └── visualization.py    # 数据可视化
│   └── reorder/
│       └── reo_clu.py          # 结果重排序
│
├── 🧪 测试与工具
│   ├── testing/                # 性能测试模块
│   │   ├── __init__.py
│   │   ├── locust_test.py      # 负载测试
│   │   ├── performance_monitor.py # 性能监控
│   │   └── test_data_generator.py # 测试数据生成
│   ├── diagnostic_tool.py      # 系统诊断工具
│   ├── test_vectorization.py   # 向量化测试
│   └── config_manager.py       # 配置管理器
│
├── 📚 文档
│   ├── PROJECT_FIXES.md        # 修复记录
│   ├── CHUNKING_STRATEGIES.md  # 分块策略说明
│   ├── QUICK_START.md          # 快速开始
│   ├── readme.md               # 项目说明
│   └── introduction.md         # 项目介绍
│
└── 📁 数据目录
    ├── data/upload/            # 用户上传数据
    └── test/data/              # 测试数据
```

## 🔗 模块串联关系

### 1. 数据处理流程
```
用户上传 → frontend.py → backend_api.py → System/start.py → dataBuilder/data.py → tools/*.py
```

### 2. 分块策略流程
```
dataBuilder/data.py → dataBuilder/chunking/chunk_strategies.py → dataBuilder/chunking/meta_chunking.py
```

### 3. 向量化流程
```
dataBuilder/data.py → Search/embedding.py → milvusBuilder/milvus.py
```

### 4. 搜索流程
```
frontend.py → backend_api.py → System/start.py → Search/search.py → reorder/reo_clu.py
```

### 5. 多模态流程
```
dataBuilder/tools/imgmake.py → multimodal/clip_encoder.py → multimodal/image_processor.py
```

## ⚙️ 核心配置文件

### config.yaml
主配置文件，包含所有模块的配置参数：
- Milvus连接配置
- 分块策略配置
- 多模态配置
- 系统配置

## 🚀 启动方式

### 1. 后端API服务
```bash
python -m uvicorn backend_api:app --reload --port 8506
```

### 2. 前端界面
```bash
python -m streamlit run frontend.py
```

### 3. 系统诊断
```bash
python diagnostic_tool.py
```

### 4. 向量化测试
```bash
python test_vectorization.py
```

## 📋 模块依赖关系

### 核心依赖
- `backend_api.py` ← `System/start.py` ← `dataBuilder/data.py`
- `frontend.py` → `backend_api.py`
- `dataBuilder/data.py` → `dataBuilder/chunking/*` + `dataBuilder/tools/*`

### 可选依赖
- `multimodal/*` (CLIP功能)
- `testing/*` (性能测试)
- `Search/ES/*` (Elasticsearch支持)

## 🔧 关键接口

### 1. 数据处理接口
- `dataBuilder.data.data_process()` - 主数据处理函数
- `dataBuilder.chunking.ChunkingManager.chunk_text()` - 文本分块

### 2. API接口
- `POST /upload` - 文件上传和向量化
- `POST /search` - 文本搜索
- `GET /chunking/strategies` - 获取分块策略
- `GET /system/status` - 系统状态检查

### 3. 配置接口
- `config_manager.load_config()` - 配置加载
- `config_manager.update_config()` - 配置更新

## 🎯 使用流程

1. **系统初始化**: 启动后端和前端服务
2. **配置设置**: 通过前端界面配置参数
3. **数据上传**: 上传文档文件
4. **自动处理**: 系统自动进行分块、向量化、存储
5. **搜索查询**: 通过前端进行语义搜索
6. **结果展示**: 查看搜索结果和可视化

## 🔍 故障排除

1. **系统诊断**: 运行 `python diagnostic_tool.py`
2. **向量化测试**: 运行 `python test_vectorization.py`
3. **日志查看**: 检查控制台输出和日志文件
4. **配置检查**: 验证 `config.yaml` 配置正确性