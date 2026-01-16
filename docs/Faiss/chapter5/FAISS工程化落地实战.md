# 5 FAISS工程化落地实战

本节聚焦FAISS在实际业务中的工程化应用。

通过文本语义检索、图像相似检索两大核心场景的实战演练，以及工程化部署关键技术的讲解，帮助学习者掌握高维向量检索的完整流程，具备独立开发与部署向量检索系统的能力。

学习前置要求：

1. 掌握Python基础语法与数据处理能力；
2. 了解Transformer/CNN基本原理；
3. 熟悉NumPy等基础数据科学库；

## 5.1 文本语义检索实战

文本语义检索是FAISS最典型的应用场景，核心是将非结构化文本转化为结构化向量后，通过相似性搜索实现"语义匹配"而非传统关键词匹配。本小节将完成从文本嵌入生成到检索接口开发的全流程实战。

### 5.1.1 环境准备

首先创建独立虚拟环境并安装依赖包，推荐使用Python 3.8-3.10版本：

```bash
# 虚拟环境创建（conda示例）
conda create -n faiss-env python=3.10 -y
conda activate faiss-env

# 核心依赖安装
pip install faiss-cpu  # CPU版本，GPU版本需安装faiss-gpu
pip install sentence-transformers  # 文本嵌入模型库
pip install fastapi uvicorn  # API开发框架
pip install numpy pydantic  # 数据处理与校验
pip install modelscope #魔塔社区库
```

### 5.1.2 Transformer生成文本嵌入

文本嵌入（Embedding）是将文本转化为高维向量的过程，国内模型下载推荐使用魔塔社区：https://www.modelscope.cn/my/overview

#### 模型下载

```python
from modelscope import snapshot_download
model_dir = snapshot_download('iic/nlp_gte_sentence-embedding_chinese-base',cache_dir='./model')
```

#### 实操代码

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 加载模型（首次运行自动下载，约4GB，建议指定缓存目录）
model = SentenceTransformer("./model/iic/nlp_gte_sentence-embedding_chinese-base")

# 2. 定义文本数据（实战中可替换为课程文档/论文摘要）
documents = [
    {
        "text": "机器学习是人工智能的核心技术，通过算法使计算机从数据中自动学习规律",
        "metadata": {"id": "doc_001", "category": "课程笔记", "source": "AI导论"}
    },
    {
        "text": "BERT模型采用双向Transformer架构，能捕捉文本上下文依赖关系",
        "metadata": {"id": "doc_002", "category": "论文摘要", "source": "NAACL2019"}
    },
    {
        "text": "FAISS支持多种索引类型，IndexIVFFlat适用于大规模向量的近似搜索",
        "metadata": {"id": "doc_003", "category": "工具手册", "source": "FAISS官方文档"}
    }
]

# 3. 生成文本嵌入
prefix = "为这个句子生成表示以用于检索相关句子："  # 任务引导前缀
texts = [prefix + doc["text"] for doc in documents]
embeddings = model.encode(
    texts,
    normalize_embeddings=True  # 向量归一化
).astype(np.float32)  # FAISS要求输入为float32类型

print(f"生成向量维度：{embeddings.shape[1]}")  # 输出：生成向量维度：768
print(f"向量数量：{embeddings.shape[0]}")  # 输出：向量数量：3
```

运行结果

```
生成向量维度：768
向量数量：3
```

### 5.1.3 嵌入向量接入FAISS构建检索库

FAISS通过"索引"结构实现高效向量搜索，需根据数据规模选择合适的索引类型。

#### 实操代码：构建与保存检索库

```python
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ===================== 1. 配置参数与路径 =====================
# 模型路径（本地路径，若不存在会自动从HuggingFace下载）
model_path = "./model/iic/nlp_gte_sentence-embedding_chinese-base"
# FAISS索引与元数据保存路径
data_dir = Path("./text_search_db")
data_dir.mkdir(parents=True, exist_ok=True)
index_path = data_dir / "faiss_index.index"
metadata_path = data_dir / "metadata.json"

# 检索参数
top_k = 3  # 改为返回Top-3相似结果（数据更多，便于学生观察排名）

# ===================== 2. 加载模型 =====================
print("正在加载模型...")
model = SentenceTransformer(model_path)
print("模型加载完成！")

# ===================== 3. 直接定义FAISS示例数据（无需TXT文件） =====================
# 扩充的FAISS知识点数据（大模型随机生成）
sample_data = [
    "维基百科::FAISS（Facebook AI Similarity Search）是由Facebook AI研究院开发的高效向量相似性检索库，专为大规模高维向量的近邻搜索场景设计，开源且支持CPU和GPU加速。",
    "Facebook官方文档::FAISS的核心特点包括支持多种索引类型（精确索引如IndexFlatIP/IndexFlatL2、近似索引如IVF、HNSW、PQ等）、能处理亿级别的向量数据，且提供了距离计算、向量归一化等配套工具。",
    "技术博客::FAISS的架构主要包含向量存储模块、距离计算模块和索引优化模块，这三大模块协同工作，支撑了高维向量的高效检索。",
    "入门教程::FAISS的安装方式十分灵活，可通过pip安装（pip install faiss-cpu/faiss-gpu），也可从源码编译，源码编译支持更多自定义配置。",
    "行业报告::FAISS广泛应用于推荐系统、语义检索、计算机视觉中的特征匹配、语音识别中的向量检索等场景，是工业界处理高维向量检索的主流工具。",
    "学术论文::FAISS的性能优化手段包括向量量化（PQ/OPQ）、倒排索引（IVF）、分层导航小世界（HNSW）等，这些技术大幅降低了检索的时间和内存开销。",
    "GPU教程::FAISS的GPU版本能够利用显卡的并行计算能力，将大规模向量检索的速度提升数十倍，尤其适合百亿级向量的检索场景。",
    "技术对比::与Milvus、Pinecone等向量数据库相比，FAISS更专注于向量检索的核心算法优化，轻量化且性能优异，但缺乏分布式存储和管理的原生支持。",
    "使用指南::FAISS构建索引的基本步骤为：初始化索引→添加向量→保存索引→加载索引→执行检索，整个流程简洁且易于集成到业务代码中。",
    "常见问题::FAISS处理低维向量时性能优势不明显，此时可直接使用线性检索；而处理128维以上的高维向量时，其优化效果会显著体现。"
]

# 解析示例数据为文档元数据列表（保持原有结构：source + text）
documents = []
for item in sample_data:
    if "::" in item:
        source, text = item.split("::", 1)
        documents.append({"source": source, "text": text})
    else:
        documents.append({"source": "未知", "text": item})

# 校验数据
if not documents:
    raise ValueError("无有效示例数据！")
print(f"加载完成，共{len(documents)}条FAISS相关知识点数据")

# ===================== 4. 生成文本嵌入向量 =====================
print("\n正在生成文本嵌入向量...")
# 提取文本内容
texts = [doc["text"] for doc in documents]
# 生成嵌入向量（归一化，适配内积索引）
embeddings = model.encode(
    texts,  # 简化写法，与原逻辑一致
    normalize_embeddings=True,
    convert_to_numpy=True  # 转换为numpy数组
).astype(np.float32)  # FAISS要求float32类型
print(f"嵌入向量生成完成，维度：{embeddings.shape}")

# ===================== 5. 创建并保存FAISS索引 =====================
# 初始化内积索引（归一化后等价于余弦相似度）
d = embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatIP(d)
# 添加向量到索引
index.add(embeddings)
print(f"索引已添加向量数量：{index.ntotal}")

# 保存索引与元数据
faiss.write_index(index, str(index_path))
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)
print(f"索引已保存至：{index_path}")
print(f"元数据已保存至：{metadata_path}")

# ===================== 6. 加载FAISS索引（模拟实际应用场景） =====================
print("\n正在加载索引与元数据...")
loaded_index = faiss.read_index(str(index_path))
with open(metadata_path, "r", encoding="utf-8") as f:
    loaded_metadata = json.load(f)
print("索引与元数据加载完成！")

# ===================== 7. 执行相似性检索（学生可替换不同查询文本测试） =====================
# 示例查询文本（学生可修改为其他问题，如"FAISS如何安装？"、"FAISS与Milvus的区别？"等）
query_text = "FAISS的架构主要包含哪些模块？"
print(f"\n查询文本：{query_text}")

# 生成查询向量
query_embedding = model.encode(
    query_text,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype(np.float32).reshape(1, -1)  # FAISS要求2D数组（batch_size, dim）

# 执行搜索
distances, indices = loaded_index.search(query_embedding, top_k)

# ===================== 8. 解析并展示结果 =====================
print("\n检索结果：")
# IndexFlatIP是内积计算，值越大相似度越高（归一化后为余弦相似度，范围[-1,1]）
for i in range(top_k):
    idx = indices[0][i]
    if idx == -1:  # 无结果时的处理
        print(f"排名{i+1}：无匹配结果")
        continue
    similarity = distances[0][i]
    print(f"排名{i+1}：相似度{similarity:.4f}")
    print(f"文本：{loaded_metadata[idx]['text']}")
    print(f"来源：{loaded_metadata[idx]['source']}\n")
```

运行结果

```
检索结果：
排名1：相似度0.8594
文本：FAISS的架构主要包含向量存储模块、距离计算模块和索引优化模块，这三大模块协同工作，支撑了高维向量的高效检索。
来源：技术博客

排名2：相似度0.7816
文本：FAISS的安装方式十分灵活，可通过pip安装（pip install faiss-cpu/faiss-gpu），也可从源码编译，源码编译支持更多自定义配置。
来源：入门教程

排名3：相似度0.7813
文本：FAISS（Facebook AI Similarity Search）是由Facebook AI研究院开发的高效向量相似性检索库，专为大规模高维向量的近邻搜索场景设计，开源且支持CPU和GPU加速。       
来源：维基百科
```

### 5.1.4 FastAPI搭建检索接口

FastAPI是高性能Python Web框架，支持自动生成API文档，便于快速开发和测试检索接口。

#### 实操代码：检索接口开发

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import faiss
import json
import numpy as np
import re  # 用于句子切分的正则表达式
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uvicorn

# ===================== 1. 配置路径与参数 =====================
app = FastAPI(title="文本语义检索API", version="1.0")

# 路径配置
txt_corpus_path = Path("./text_corpus.txt")  # 待处理的TXT文本文件路径
data_dir = Path("./text_search_db")  # 索引和元数据保存目录
index_path = data_dir / "faiss_index.index"  # FAISS索引路径
metadata_path = data_dir / "metadata.json"  # 元数据路径

# 模型与检索配置（换回指定的gte中文模型）
model_path = "./model/iic/nlp_gte_sentence-embedding_chinese-base"  # 自定义模型路径
prefix = ""  # gte中文模型无需前缀，可根据需求自定义
# 中文句子分隔符（正则表达式：匹配。！？；中的任意一个，后面可能跟换行/空格）
sentence_sep_pattern = re.compile(r'[。！？；]+')

# ===================== 2. 定义数据模型（请求/响应） =====================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3  # 默认返回3条结果

class SearchResult(BaseModel):
    rank: int
    similarity: float
    text: str
    metadata: Dict[str, str]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# ===================== 3. 工具函数 =====================
def read_and_split_sentences(txt_path: Path) -> List[str]:
    """
    读取TXT文件并按中文句子切分，返回非空句子列表
    """
    # 检查文件是否存在
    if not txt_path.exists():
        # 生成示例文本（首次运行时自动创建）
        sample_text = """FAISS是由Facebook AI研究院开发的高效向量相似性检索库。它专为大规模高维向量的近邻搜索场景设计，支持CPU和GPU加速。
FAISS的核心特点包括支持多种索引类型，比如精确索引IndexFlatIP、IndexFlatL2，以及近似索引IVF、HNSW等。它能处理亿级别的向量数据，还提供了距离计算、向量归一化等工具。
FAISS的架构主要包含向量存储模块、距离计算模块和索引优化模块。它广泛应用于推荐系统、语义检索、计算机视觉中的特征匹配等场景。"""
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"未找到{txt_path}，已自动生成示例文本！")
        text_content = sample_text
    else:
        # 读取TXT文件内容
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    
    # 按句子分隔符切分文本
    sentences = sentence_sep_pattern.split(text_content)
    # 过滤空句子和仅含空白字符的句子
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    if not sentences:
        raise ValueError("TXT文件中无有效句子可切分！")
    return sentences

def build_faiss_index():
    """
    构建FAISS索引：读取TXT→切分句子→生成嵌入→创建并保存索引/元数据
    """
    # 1. 读取并切分句子
    sentences = read_and_split_sentences(txt_corpus_path)
    # 2. 加载指定的gte中文模型（本地路径，若不存在会自动从HuggingFace下载）
    model = SentenceTransformer(model_path)
    # 3. 生成句子嵌入（归一化，适配内积索引）
    print("正在生成句子嵌入向量...")
    embeddings = model.encode(
        [prefix + sent for sent in sentences],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)  # FAISS要求float32类型
    # 4. 构建元数据（包含句子ID、来源、文本内容）
    metadata = []
    for idx, sent in enumerate(sentences):
        metadata.append({
            "text": sent,
            "metadata": {
                "sentence_id": str(idx + 1),
                "source": str(txt_corpus_path),
                "total_sentences": str(len(sentences))
            }
        })
    # 5. 创建FAISS索引（内积索引，归一化后等价于余弦相似度）
    d = embeddings.shape[1]  # 向量维度
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    # 6. 保存索引和元数据
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"FAISS索引构建完成，共添加{index.ntotal}个句子向量")
    return index, model, metadata

# ===================== 4. 初始化资源（启动时仅加载一次） =====================
print("正在初始化资源...")
try:
    # 尝试加载已存在的索引和元数据
    if index_path.exists() and metadata_path.exists():
        model = SentenceTransformer(model_path)  # 加载指定模型
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"成功加载已有索引，包含{index.ntotal}个句子向量")
    else:
        # 索引不存在时，重新构建
        index, model, metadata = build_faiss_index()
except Exception as e:
    raise RuntimeError(f"资源初始化失败：{str(e)}")

# ===================== 5. 接口定义 =====================
@app.post("/text-search", response_model=SearchResponse, summary="文本语义检索")
def text_search(request: SearchRequest):
    try:
        # 生成查询向量（与句子嵌入使用相同的前缀和归一化）
        query_embedding = model.encode(
            prefix + request.query,
            normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)
        
        # 执行检索（IndexFlatIP：内积值越大，相似度越高）
        distances, indices = index.search(query_embedding, request.top_k)
        results = []
        for i in range(request.top_k):
            idx = indices[0][i]
            if idx == -1:  # 无匹配结果时的处理
                continue
            # 内积索引的相似度直接使用距离值（归一化后为余弦相似度，范围[-1,1]）
            similarity = round(distances[0][i], 4)
            results.append(SearchResult(
                rank=i+1,
                similarity=similarity,
                text=metadata[idx]["text"],
                metadata=metadata[idx]["metadata"]
            ))
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")

@app.get("/health", summary="服务健康检查")
def health_check():
    return {
        "status": "healthy",
        "index_vector_count": index.ntotal,
        "corpus_source": str(txt_corpus_path),
        "total_sentences": len(metadata),
        "model_path": model_path
    }

if __name__ == "__main__":
    # 配置启动参数（可根据需求调整）
    uvicorn.run(
        app="main:app",  # 格式：文件名:FastAPI实例名（若你的文件不是main.py，替换为实际文件名，如api:app）
        host="0.0.0.0",  # 允许局域网/公网访问
        port=8000,       # 服务端口
        reload=True      # 开发环境热重载（生产环境关闭）
    )
```

#### 接口启动与测试

```bash
import requests

# 配置项
API_URL = "http://127.0.0.1:8000/text-search"
# 可修改查询文本测试不同场景
QUERY_TEXT = "FAISS的架构包含哪些模块？"
TOP_K = 3

def test_search():
    # 构造请求体
    data = {
        "query": QUERY_TEXT,
        "top_k": TOP_K
    }
    try:
        response = requests.post(API_URL, json=data, timeout=5)
        response.raise_for_status()  # 主动触发HTTP状态码异常（如500/400）
        result = response.json()
        
        # 格式化打印结果
        print("===== 检索结果 =====")
        print(f"查询文本：{result['query']}\n")
        for item in result['results']:
            print(f"排名：{item['rank']}")
            print(f"相似度：{item['similarity']}")
            print(f"文本：{item['text']}")
            print(f"元数据：{item['metadata']}\n")
    except requests.exceptions.HTTPError as e:
        print(f"请求失败，状态码：{response.status_code}")
        print(f"错误详情：{response.text}")
        # 针对500错误给出提示
        if response.status_code == 500 and "'metadata'" in response.text:
            print("\n【解决方案】：删除text_search_db目录下的faiss_index.index和metadata.json，重启FastAPI服务后重试！")
    except requests.exceptions.ConnectionError:
        print("连接失败：请确认FastAPI服务已启动，且地址/端口正确！")
    except Exception as e:
        print(f"未知异常：{str(e)}")

if __name__ == "__main__":
    test_search()
```

运行结果

```
查询文本：FAISS的架构包含哪些模块？

排名：1
相似度：0.8575000166893005
文本：FAISS的架构主要包含向量存储模块、距离计算模块和索引优化模块
元数据：{'sentence_id': '5', 'source': 'text_corpus.txt', 'total_sentences': '6'}

排名：2
相似度：0.7817999720573425
文本：FAISS是由Facebook AI研究院开发的高效向量相似性检索库
元数据：{'sentence_id': '1', 'source': 'text_corpus.txt', 'total_sentences': '6'}

排名：3
相似度：0.7763000130653381
文本：FAISS的核心特点包括支持多种索引类型，比如精确索引IndexFlatIP、IndexFlatL2，以及近似索引IVF、HNSW等
元数据：{'sentence_id': '3', 'source': 'text_corpus.txt', 'total_sentences': '6'}
```

### 5.1.5 实战任务：课程问答检索系统

#### 任务目标

基于某门课程的PPT文本或课件资料，构建课程问答检索系统，实现"输入问题返回相关课程内容"的功能。

#### 实施步骤

1. **数据准备**：将课程资料按段落拆分，整理为"text+metadata"格式（metadata包含章节、页码等信息）；
2. **向量生成**：使用本小节5.1.2的方法生成所有段落的嵌入向量；
3. **检索库构建**：通过FAISS创建索引并保存，确保元数据与向量一一对应；
4. **接口优化**：在FastAPI接口中增加文本长度校验、结果排序优化；
5. **系统测试**：设计10个课程相关问题，测试检索结果的准确率，调整top_k参数优化体验。

#### 学习检验

完成后需提交：1. 数据预处理代码；2. 检索库构建脚本；3. API服务代码；4. 测试报告（含问题、结果及准确率分析）。

### 5.1.6 学习总结

看到这里，你已经初步掌握了文本检索并做出服务的相关实战代码，接下来我们将核心知识点与实践思路进行梳理，帮你快速沉淀本次学习的关键收获。

本次实战以 FAISS+FastAPI 为核心技术栈，完成了文本检索服务从本地实现到线上部署的全流程，不仅是工具使用的练习，更是对语义检索工程化落地思路的掌握。

你已经掌握了 FAISS 的核心使用逻辑，包括将文本通过预训练模型转化为向量，完成向量库构建、相似性检索及索引的保存与加载，同时理解了不同索引类型的适用场景，懂得在检索速度与精度间做平衡，这是语义检索的核心基础。

在 FastAPI 的使用上，你学会了将 FAISS 检索功能封装为 HTTP 接口，掌握了请求参数校验、接口设计、服务启动及利用自动文档调试的关键步骤，理解了如何将本地功能转化为可对外调用的标准化服务。

本次实战形成了 “文本向量化→向量检索→服务封装→接口调用” 的完整闭环，这是工业界文本检索服务的最小可行架构，每个环节相互依存，向量检索是核心价值，服务封装是落地关键。

总的来说，你完成了文本检索服务的从 0 到 1 落地，核心收获是掌握了 FAISS 与 FastAPI 的协同使用逻辑，以及 “算法功能→工程化服务” 的转化思路，后续围绕性能优化与生产环境适配持续深入，就能将入门级服务升级为工业级解决方案。

# 5.2 图像相似检索实战

图像相似检索通过CNN提取图像视觉特征，将特征向量接入FAISS实现相似图像匹配，广泛应用于图片检索、人脸比对等场景。

本小节以ResNet模型为例，完成全流程实战。

## 5.2.1 环境准备

在5.1的虚拟环境基础上补充安装计算机视觉相关依赖：

```bash
pip install torch torchvision  # 深度学习框架
pip install pillow opencv-python  # 图像处理库
```

数据集：https://www.modelscope.cn/datasets/wufaxianshi/wukong

> 本数据集是笔者在魔塔社区随机搜索，也可以用自己的图片，本案例只用于学习使用。

## 5.2.2 CNN（ResNet）提取图像特征

ResNet通过残差连接解决深层网络梯度消失问题，是提取图像特征的经典模型。我们使用预训练的ResNet50模型，去除分类层后获取图像特征向量。

### 实操代码：图像特征提取

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
from pathlib import Path

# 1. 设备配置（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载预训练ResNet模型并修改为特征提取器（修复兼容性问题）
weights = ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights).to(device)
# 去除最后两层（全局平均池化层后直接输出特征，无需分类层）
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()  # 进入评估模式，禁用Dropout等

# 3. 图像预处理（与预训练模型要求一致，修复缩放问题）
transform = transforms.Compose([
    transforms.Resize(256),  # 先缩放到256（短边），保持比例
    transforms.CenterCrop(224),  # 中心裁剪到224×224（ResNet标准输入）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                         std=[0.229, 0.224, 0.225])   # ImageNet标准差
])

# 4. 定义特征提取函数（完善类型注解、边界条件）
def extract_image_feature(image_path: str | Path) -> np.ndarray | None:
    """提取单张图像的特征向量"""
    try:
        image = Image.open(str(image_path)).convert("RGB")  # 支持Path对象，转为RGB格式
        input_tensor = transform(image).unsqueeze(0).to(device)  # 增加批次维度
        
        # 无梯度计算加速
        with torch.no_grad():
            feature = feature_extractor(input_tensor)
        
        # 特征向量处理（展平为1D向量并归一化）
        feature_vector = feature.squeeze().cpu().numpy()
        # L2归一化（增加边界条件，避免除以0）
        norm = np.linalg.norm(feature_vector)
        feature_vector = feature_vector / norm if norm > 1e-6 else feature_vector
        return feature_vector.astype(np.float32)
    except Exception as e:
        print(f"特征提取失败：{image_path} -> {e}")
        return None

# 5. 测试特征提取（优化路径处理）
image_dir = Path("C:/Users/xiong/Desktop/easy-vecdb/valid")  # 去掉./，规范绝对路径
image_paths = [p for p in image_dir.iterdir() if p.is_file()]

# 提取文件夹内所有图像的特征
image_features = []
image_metadata = []
for path in image_paths[:5]:  # 先处理5张图像测试
    vec = extract_image_feature(path)
    if vec is not None:
        image_features.append(vec)
        # 构建元数据（图像路径、论坛图片ID等，实际中可从文件名解析）
        image_metadata.append({
            "image_path": str(path),
            "product_id": path.stem,  # Path对象直接调用stem，更简洁
            "category": "electronics"  # 可根据实际分类修改
        })

if image_features:  # 增加判断，避免空数组报错
    image_features = np.array(image_features)
    print(f"提取特征向量维度：{image_features.shape[1]}")  # 输出：2048
    print(f"成功提取特征的图像数量：{image_features.shape[0]}")
else:
    print("未提取到任何特征向量")
```

运行结果

```
提取特征向量维度：2048
成功提取特征的图像数量：5
```

## 5.2.3 特征向量接入FAISS实现检索

图像特征向量维度通常为2048维（ResNet系列），需结合数据规模选择索引类型。此处采用IndexIVFFlat实现高效近似搜索。

### 实操代码：图像检索库构建与检索

```python
import faiss
import json
import numpy as np
from pathlib import Path

# 假设以下变量已在之前的代码中定义（需确保存在）
# image_features: np.ndarray (n, d)，float32类型
# image_metadata: list，存储图片元数据
# extract_image_feature: 特征提取函数

# 1. 配置路径
db_dir = Path("./image_search_db")
db_dir.mkdir(parents=True, exist_ok=True)
index_path = db_dir / "image_index.index"
metadata_path = db_dir / "image_metadata.json"

# 2. 边界条件：检查特征向量是否为空
if len(image_features) == 0:
    print("错误：没有可处理的特征向量，终止索引构建")
else:
    # 确保特征向量为float32类型（FAISS的硬性要求）
    image_features = np.array(image_features).astype(np.float32)
    d = image_features.shape[1]  # 特征维度（如2048）
    n = len(image_features)     # 特征数量

    # 3. 动态计算nlist（经验值：数据量的平方根，且不小于1、不大于数据量）
    nlist = int(np.sqrt(n)) if n > 0 else 1
    nlist = max(nlist, 1)  # 至少1个聚类
    nlist = min(nlist, n)  # 不超过数据量（避免聚类数大于数据量）

    # 4. 构建FAISS索引（根据数据量选择索引类型）
    if n < 10000:  # 小规模数据：使用精确检索的IndexFlatL2（无需训练）
        print("小规模数据，使用IndexFlatL2精确检索")
        index = faiss.IndexFlatL2(d)
    else:  # 大规模数据：使用IndexIVFFlat近似检索
        print("大规模数据，使用IndexIVFFlat近似检索")
        quantizer = faiss.IndexFlatL2(d)  # 量化器（基于L2距离）
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

    # 5. 训练索引（仅IVF类索引需要训练）
    if isinstance(index, faiss.IndexIVFFlat) and not index.is_trained:
        index.train(image_features)
        print("索引训练完成")

    # 6. 添加向量并保存
    index.add(image_features)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, ensure_ascii=False, indent=2)
    print(f"图像检索库构建完成，包含{index.ntotal}个特征向量")

    # 7. 加载索引与元数据
    loaded_index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded_img_metadata = json.load(f)

    # 8. 相似图像检索测试
    test_image_path = "./test_image.jpg"  # 测试查询图像
    test_vec = extract_image_feature(test_image_path)

    # 边界条件：检查测试向量是否提取成功
    if test_vec is not None:
        test_vec = test_vec.reshape(1, -1).astype(np.float32)  # 转为float32并增加batch维度

        # 调整检索精度（仅IVF类索引有效，nprobe越大精度越高，速度越慢）
        if isinstance(loaded_index, faiss.IndexIVFFlat):
            loaded_index.nprobe = 3  # 经验值，可根据需求调整

        k = 3  # 返回Top-3相似图像
        distances, indices = loaded_index.search(test_vec, k)

        # 解析结果（L2距离越小相似度越高）
        print("\n相似图像检索结果：")
        for i in range(k):
            idx = indices[0][i]
            # 防止索引越界（比如数据量不足k个）
            if idx < 0 or idx >= len(loaded_img_metadata):
                print(f"排名{i+1}：无匹配结果")
                continue
            print(f"排名{i+1}：L2距离{distances[0][i]:.4f}")
            print(f"论坛图片ID：{loaded_img_metadata[idx]['product_id']}")
            print(f"图像路径：{loaded_img_metadata[idx]['image_path']}\n")
    else:
        print(f"错误：测试图片{test_image_path}特征提取失败")
```

运行结果

```
===== 相似图片检索结果 =====
排名1 | L2距离：0.0000
产品ID：0_13
图片路径：C:\Users\xiong\Desktop\easy-vecdb\valid\0_13.png

排名2 | L2距离：0.5698
产品ID：0_323660
图片路径：C:\Users\xiong\Desktop\easy-vecdb\valid\0_323660.jpeg

排名3 | L2距离：0.6224
产品ID：0_223551
图片路径：C:\Users\xiong\Desktop\easy-vecdb\valid\0_223551.jpg
```

## 5.2.4 实战任务：论坛图片相似检索系统

### 任务目标

构建一个论坛图片相似检索系统，支持输入一张论坛图片图像，返回数据库中视觉特征最相似的论坛图片信息。

### 实施步骤

1. **数据准备**：收集不同类别的论坛图片图像；
2. **特征提取**：使用ResNet50模型提取所有论坛图片图像的特征向量，确保向量归一化；
3. **检索库构建**：训练FAISS IndexIVFFlat索引并保存，元数据包含论坛图片类别、价格（模拟）、图像路径；
4. **交互功能开发**：基于Streamlit开发简单前端界面，支持图像上传与检索结果展示；
5. **系统优化**：测试不同nprobe值对检索精度和速度的影响，确定最优参数。

> 前端界面可以借助AI大模型帮忙生成

# 5.3 工程化部署注意事项

FAISS检索系统从开发环境走向生产环境，需解决索引更新、性能监控、容错处理三大核心问题，确保系统稳定高效运行。

## 5.3.1 索引增量更新策略

实际业务中，检索库的向量数据会持续新增（如新增图片图像、课程资料），需设计增量更新策略避免全量重建索引。

### 1. 小规模增量：直接添加向量

适用于每次新增向量数量较少（万级以内）的场景，直接在原有索引上添加新向量，无需重新训练。

```python
import faiss
import numpy as np

# 加载原有索引
index = faiss.read_index("./text_search_db/faiss_index.index")

# 模拟新增数据的向量（实际中从新文本/图像提取）
new_embeddings = np.random.random((100, 1024)).astype(np.float32)  # 新增100个向量

# 直接添加到索引
index.add(new_embeddings)
print(f"增量更新后向量总数：{index.ntotal}")

# 保存更新后的索引（建议先备份旧索引）
faiss.write_index(index, "./text_search_db/faiss_index_updated.index")
```

### 2. 大规模增量：索引合并

当新增向量数量达到原有规模的50%以上时，建议单独构建新索引，再与旧索引合并，提升检索效率。

```python
import faiss

# 1. 加载旧索引和新构建的增量索引
old_index = faiss.read_index("./old_index.index")
new_index = faiss.read_index("./new_index.index")  # 基于新增数据构建的索引

# 2. 合并索引（需确保两个索引类型一致）
merged_index = faiss.IndexFlatIP(old_index.d)  # 与原索引维度一致
merged_index.add(old_index.reconstruct_n(0, old_index.ntotal))  # 读取旧索引所有向量
merged_index.add(new_index.reconstruct_n(0, new_index.ntotal))  # 读取新索引所有向量

# 3. 保存合并后的索引
faiss.write_index(merged_index, "./merged_index.index")
```

### 3. 增量更新注意事项

- IVF类索引增量添加向量后，检索精度可能下降，可定期（如每周）重新训练索引；
- 更新时需加锁机制，避免检索服务读取残缺索引；
- 元数据需与向量同步更新，建议使用数据库存储元数据，通过向量索引关联ID查询。

## 5.3.2 接口性能监控（QPS/延迟）

性能监控是保障服务稳定的核心，需重点关注查询吞吐量（QPS）、响应延迟、服务器资源占用三大指标。

### 1. 指标采集：Prometheus+FastAPI

通过prometheus-fastapi-instrumentator库采集API性能指标，结合Prometheus和Grafana实现可视化监控。

```bash
# 安装依赖
pip install prometheus-fastapi-instrumentator==6.1.0 prometheus-client==0.20.0
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# 初始化监控器
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health"]  # 排除健康检查接口
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# 后续添加检索接口...

# 启动服务后，访问http://localhost:8000/metrics即可获取指标数据
```

### 2. 监控指标核心关注项

| 指标名称                      | 含义                   | 合理阈值                     |
| :---------------------------- | :--------------------- | :--------------------------- |
| http_requests_total           | API请求总数            | 无固定值，需结合业务峰值     |
| http_request_duration_seconds | 请求响应延迟（分位值） | P95<500ms                    |
| process_cpu_usage             | CPU使用率              | <80%                         |
| process_memory_usage_bytes    | 内存占用               | 根据索引大小调整，无内存泄漏 |

### 3. 性能优化方向

- **索引优化**：大规模数据使用HNSW索引（兼顾速度与精度），如IndexHNSWFlat；
- **缓存机制**：对高频查询结果使用Redis缓存，减少重复检索计算；
- **分布式部署**：使用FAISS的分布式索引（DistributedIndex），实现负载均衡；
- **模型优化**：使用轻量化模型（如Sentence-BERT的mini版本、MobileNet替代ResNet）降低特征提取耗时。

## 5.3.3 容错处理（索引损坏/向量缺失）

生产环境中需应对索引文件损坏、向量与元数据不匹配、服务异常中断等问题，确保系统容错能力。

### 1. 索引损坏应对策略

- **定期备份**：使用定时任务（如crontab）每日备份索引文件，保留近7天的备份版本；
- **校验机制**：加载索引时校验索引完整性，捕获faiss.Error异常并自动切换至备份索引；
- **示例代码**：

```python
import faiss
import os

def load_index_safely(main_path: str, backup_path: str):
    try:
        # 尝试加载主索引
        index = faiss.read_index(main_path)
        # 简单校验：索引向量数量大于0
        if index.ntotal == 0:
            raise Exception("主索引向量数量异常")
        return index, "main"
    except Exception as e:
        print(f"主索引加载失败：{e}，切换至备份索引")
        # 加载备份索引
        if not os.path.exists(backup_path):
            raise Exception("备份索引不存在，服务启动失败")
        index = faiss.read_index(backup_path)
        return index, "backup"
```

### 2. 向量缺失与元数据不一致处理

- **数据校验**：构建检索库时，确保向量数量与元数据数量严格一致，添加校验步骤；
- **异常捕获**：检索时若索引返回的indices超出元数据范围，返回友好错误提示并记录日志；
- **示例代码**：

```python
def safe_search(query_embedding, index, metadata, top_k):
    try:
        distances, indices = index.search(query_embedding, top_k)
        # 校验indices有效性
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(metadata)]
        if len(valid_indices) == 0:
            return {"error": "未找到相似结果"}
        # 过滤无效结果
        results = []
        for idx in valid_indices[:top_k]:
            results.append({
                "similarity": 1 - distances[0][list(indices[0]).index(idx)],
                "text": metadata[idx]["text"]
            })
        return results
    except Exception as e:
        print(f"检索异常：{e}")
        return {"error": "检索服务临时不可用，请稍后重试"}
```

### 3. 服务高可用保障

- **进程守护**：使用Supervisor或systemd管理API服务，确保服务异常退出后自动重启；
- **日志记录**：使用logging模块记录详细日志（请求参数、响应结果、错误堆栈），便于问题排查；
- **降级策略**：当服务负载过高时，自动降级为"只返回缓存结果"或"减少top_k数量"，保障核心功能可用。



以上是faiss课程的全部内容，如果你喜欢我们的项目，可以给我们点一个star，完结，✿✿ヽ(°▽°)ノ✿