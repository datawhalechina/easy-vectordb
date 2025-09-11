## 项目说明

本项目使用本地 `bge-small-zh-v1.5` 模型对 `news_data_dedup.csv` 文本进行向量化，写入 Milvus（127.0.0.1:19530），随后对向量执行 HDBSCAN 聚类，构建余弦距离矩阵，并用 UMAP 进行可视化。

### 环境准备（Windows PowerShell）

```bash
# 1. 进入项目目录
cd D:\Github\MindCode\HDBSCAN

# 2. 创建并激活虚拟环境（可自定义名称 my_env）
python -m venv my_env
.\nmy_env\Scripts\Activate.ps1

# 3. 升级基础工具并安装依赖
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

- 确保本地 Milvus 已运行并监听 `127.0.0.1:19530`。
- 模型文件放置在 `model/` 目录（`bge-small-zh-v1.5` 权重与 tokenizer）。
- 注意，你需要去[modelscope](https://www.modelscope.cn/models/BAAI/bge-small-zh-v1.5/files)下载5个文件放到model文件夹下，分别是:
    1. config.json
    2. pytorch_model.bin
    3. special_tokens_map.json
    4. tokenizer_config.json
    5. vocab.txt

### 启动运行

```bash
# 仍在虚拟环境中
python .\pipeline_hdbscan_umap.py
```

运行完成后，将在项目根目录生成：
- `hdbscan_labels.csv`：每条 `guid` 的聚类标签与置信度（-1 表示噪声）。
- `distance_matrix.npy`：全部样本的余弦距离矩阵（注意内存消耗为 O(n^2)）。
- `umap_hdbscan.png`：UMAP 降维后的聚类可视化图。

如需清空 Milvus 集合（重新插入）可参考：
```python
from pymilvus import connections, utility, Collection
connections.connect(alias='default', host='127.0.0.1', port='19530')
if utility.has_collection('news_bge_small_zh_v15'):
    Collection('news_bge_small_zh_v15').drop()
```

---

## 代码流程概述

以下流程在 `pipeline_hdbscan_umap.py` 中实现：

1. **读取数据**：加载 `news_data_dedup.csv`，使用列 `guid/title/description/venue/url/published_at`。
2. **文本拼接**：将 `title` 与 `description` 拼接为待编码文本。
3. **加载本地模型**：从 `model/` 读取 `bge-small-zh-v1.5`，使用 Transformers 编码得到 512 维向量，并做 L2 归一化。
4. **连接 Milvus 并建表**：创建集合 `news_bge_small_zh_v15`，字段包括主键 `pk`、`vector` 及若干元数据字段，并为向量字段创建索引（`IVF_FLAT`，`IP`）。
5. **安全截断与插入**：对字符串字段执行保守截断（例如 `description` 截到 4000）以满足 Milvus `VARCHAR` 的 `max_length`，再批量写入。
6. **读取向量**：从 Milvus 读取全部向量用于聚类（也可直接用内存中的向量）。
7. **HDBSCAN 聚类**：对向量执行 HDBSCAN（默认 `min_cluster_size=10`），输出 `labels` 与 `probabilities`。
8. **距离矩阵**：计算余弦距离矩阵并保存为 `distance_matrix.npy`。
9. **UMAP 可视化**：以余弦度量将向量降至 2D，按聚类结果着色并保存图像。
10. **结果导出**：将 `guid/label/probability` 保存为 `hdbscan_labels.csv`。

可调参数：
- HDBSCAN：`min_cluster_size`、`min_samples`、`metric` 等。
- UMAP：`n_neighbors`、`min_dist`、`metric` 等。
- 向量批大小：`BATCH_SIZE`（GPU/CPU 可按需调大或调小）。

---

## 什么是 HDBSCAN？有什么用？

**HDBSCAN（Hierarchical Density-Based Spatial Clustering of Applications with Noise）** 是一种基于密度的层次聚类算法，是 DBSCAN 的改进：

- **无需预先指定簇数**：不同于 KMeans 需要 `k`，HDBSCAN 自动发现聚类数量。
- **鲁棒处理噪声**：能将无法归入任何簇的点标记为噪声（标签为 -1）。
- **适应非凸形簇**：能够发现任意形状的聚类结构，不局限于球形簇。
- **层次结构**：在不同密度阈值下形成层次树，并以稳定性原则选取最佳簇。

在文本向量场景中，HDBSCAN 适合：
- **主题发现/新闻聚合**：自动将相近语义的文本归为同一簇，无需人工设定簇数。
- **异常检测**：标签为 -1 的样本可视为离群点或噪声。
- **数据探索与可视化**：结合 UMAP 将高维向量映射到 2D，直观查看簇结构与边界。

---

## 常见问题（简要）

- 安装报错与 Python 版本兼容：已在 `requirements.txt` 固定了兼容版本（如 `hdbscan==0.8.40`、`marshmallow>=3.13,<4`、`environs>=9.5.0`）。若仍失败，请先 `pip install -U pip setuptools wheel` 再重试。
- Milvus 写入报 `VARCHAR` 超长：项目已在写入前执行保守截断，如仍有旧数据导致失败，可先删除集合后重跑。
- 距离矩阵过大：`distance_matrix.npy` 内存为 O(n^2)。当样本量很大时，建议采样或跳过距离矩阵生成。

---

## 目录结构（简要）

- `model/`：本地 `bge-small-zh-v1.5` 模型文件
- `news_data_dedup.csv`：新闻数据集
- `pipeline_hdbscan_umap.py`：主流程脚本
- `requirements.txt`：依赖列表
- 运行后生成：`hdbscan_labels.csv`、`distance_matrix.npy`、`umap_hdbscan.png`
