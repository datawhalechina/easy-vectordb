# 说明文档

本项目旨在构建一个基于 RAG（Retrieval-Augmented Generation）技术的问答系统，结合 ZhipuAI 的语言模型与 Milvus 向量数据库，实现对用户查询的智能响应，**并在回答中附带相关视频链接。**

---

## 📁 项目结构

```
url_process/
├── video_Test_Data/            # 测试用的 Markdown 文件，包含文本和视频链接
│   └── testData_One.md
├── front.py                    # 前端交互界面（Gradio）
├── test.css                    # 样式文件（前端样式）
├── video_trueData.py           # 数据处理脚本：提取文本、生成 embedding 并插入 Milvus
└── README.md                   # 项目说明文档
```
---

## 🔧 功能模块说明

### 1. video_trueData.py

- 读取 `video_Test_Data` 目录下的md文件；
- 使用正则表达式提取文本中的 URL；
- 将文本内容分段处理后，调用 ZhipuAI 的 Embedding 接口生成向量；
- 将每段文本及其对应的 URL 存储到 Milvus 向量数据库中；
- 支持多 URL 提取与字段扩展。

### 2. front.py
- 构建 Gradio Web 界面；
- 提供用户输入框和提交按钮；
- 调用 RAG 模型处理用户问题：
  - 使用 ZhipuAI 获取用户问题的 embedding；
  - 在 Milvus 中搜索最相似的视频描述；
  - 结合相似内容生成最终回答，并附加推荐的视频链接。

### 3. test.css

- 自定义 Gradio 界面样式，提升用户体验。

---

##  依赖安装

```bash
pip install zhipuai pymilvus gradio
```

> 确保已启动 Milvus 服务并配置好环境变量 `ZHIPUAI_API_KEY`。

---

##  运行步骤

### 1. 初始化数据库并导入数据

运行以下命令以提取 Markdown 内容并将其插入 Milvus：

```bash
python video_trueData.py
```

### 2. 启动前端应用

运行以下命令启动 Gradio 应用：

```bash
python front.py
```

访问本地 Web 地址（通常是 http://localhost:7860），即可开始使用。

---

##  示例输入输出

### 输入示例：

```
RAG技术是什么？
```

### 输出示例：

```
RAG技术（Reinforcement Learning with Augmented GANs）是一种结合强化学习和生成对抗网络的技术，用于优化决策过程。它在游戏 AI、机器人导航等领域有广泛应用。

For more information, please refer to the following URLs:
https://example.com/video/1000004
```

---

##  注意事项

- 确保 Milvus 已正确部署并运行在默认地址 `localhost:19530`。
- 若更换模型或参数，请注意调整 embedding 维度等设置。
- Markdown 文件需遵循特定格式（包含 URL 注释）以便正确提取内容。

---
如有疑问或需要帮助，请联系项目维护者。