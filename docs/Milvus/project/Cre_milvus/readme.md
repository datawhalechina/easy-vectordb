# TodoList
**HalukiSan**：
1. （noraml）对于召回回来的数据，可能是存在url或者图片地址的，总之都是一个url，现在需要在前端做一个展示，如果数据中包含url（代码已经做了提取处理了）就在前端中展示出来，这个url可能是网络链接，也可能是图片的地址，如果是图片，需要渲染出来。
2. （important）search中，关于milvus检索部分，需要一个是否切分url的status（true\false）,然后在milvus_search.py中进行判断，如果有url，返回的数据就要包含url的一列，否则不包含。
3. （important） 检查ES和Redis中是否有根据url进行判断决定存储结构中是否应该包含url一列。（es_utils.py/redisSer.py）存储和查询的都要检查，可能会遇到，存储里面的有url，但查询返回的数据里面设置了没有url！
   

**NN**：
1. （important） 文本内容审核正确性，Cre_milvus\System\eval.py中


# 🚀 项目启动教程

欢迎使用本项目！以下是详细的启动教程，帮助你快速上手。

---

## 1. 克隆或下载项目代码

首先，获取项目代码：

```bash
git clone <项目仓库地址>
cd <项目目录>
```

或者直接从 GitHub 下载 ZIP 文件并解压。

---

## 2. 安装依赖

确保你的系统已安装 **Python 3.8+**。推荐使用虚拟环境来管理依赖。

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
# 图片向量化模型下载
modelscope download --model openai-mirror/clip-vit-base-patch16 --local_dir /model

```

---

## 3. 配置 Milvus、Elasticsearch、Redis 服务

### 3.1 Milvus
- 参考 [Milvus 官方文档](https://milvus.io/docs) 启动服务。
- 记下以下信息：
  - `host`
  - `port`
  - 用户名和密码（如有）。

### 3.2 Elasticsearch
- 推荐使用 **7.x 版本**。
- 启动后记下以下信息：
  - `host`
  - `port`。

### 3.3 Redis
- 可以使用本地或远程 Redis 服务。
- 记下以下信息：
  - `host`
  - `port`。

---

## 4. 配置 `config.yaml`

在项目根目录下创建或修改 `config.yaml` 文件，填写你的服务信息和参数。例如：

```yaml
milvus:
  host: "localhost"
  port: 19530
  username: ""
  password: ""

elasticsearch:
  host: "localhost"
  port: 9200

redis:
  host: "localhost"
  port: 6379

data_path: "./data/upload"  # 数据文件路径
```

---

## 5. 上传你的数据

将你的数据文件（支持 `csv`、`md`、`pdf`、`txt` 格式）放入 `./data/upload` 文件夹（或你在 `config.yaml` 中指定的路径）。

---

## 6. 构建向量数据库

运行主流程，自动处理数据并构建 Milvus 向量库：

```bash
python main.py
```

如需交互式体验，可以使用 Streamlit 前端：

```bash
streamlit run app.py
```

---

## 7. 检索与聚类

你可以通过以下方式使用检索与聚类功能：

### 7.1 前端页面
- 打开前端页面，输入检索问题。
- 选择聚类和重排序方式。
- 查看聚类和重排序后的结果。

### 7.2 Python 交互环境
在 Python 环境中调用相关函数：

```python
from your_module import search_and_cluster

results = search_and_cluster(query="你的问题", cluster_method="kmeans")
print(results)
```

---

## 8. （可选）启动后端 API 服务

如需前后端分离体验，可以运行 FastAPI 后端服务：

```bash
uvicorn api:app --reload
```

访问 `http://localhost:8000/docs` 查看 API 文档。

---

## 9. 常见问题

### 9.1 服务未启动
请确保 **Milvus**、**Elasticsearch** 和 **Redis** 均已启动并正确配置。

### 9.2 依赖缺失
检查是否完整安装了 `requirements.txt` 中的依赖。

### 9.3 数据格式问题
确保上传的数据文件格式正确，目前支持 `csv`、`md`、`pdf` 和 `txt`。

---

## 10. 参考

- [Milvus 官方文档](https://milvus.io/docs)
- [Elasticsearch 官方文档](https://www.elastic.co/guide/index.html)
- [Redis 官方文档](https://redis.io/documentation)

---

如有其他问题，欢迎提交 issue 或联系开发者！ 😊