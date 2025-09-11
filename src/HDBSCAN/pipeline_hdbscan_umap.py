import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from rich import print
from rich.progress import track

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import torch
from transformers import AutoTokenizer, AutoModel

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

import umap
import matplotlib.pyplot as plt
import hdbscan


# -----------------------------
# # 创建虚拟环境（命名为 milvus_env，你可以自定义）
# python -m venv modelscope_env

# 激活虚拟环境（Windows）
#  modelscope_env\Scripts\activate
#  pip install -r requirements.txt
# -----------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), 'news_data_dedup.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')  # local bge-small-zh-v1.5
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'news_bge_small_zh_v15'
BATCH_SIZE = 64
TEXT_COLUMNS = ['title', 'description']  
EMBEDDING_DIM = 512  



@torch.no_grad()
def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@torch.no_grad()
def encode_texts(tokenizer: AutoTokenizer, model: AutoModel, texts: List[str], batch_size: int = 64, device: str = None) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    embeddings: List[np.ndarray] = []
    for i in track(range(0, len(texts), batch_size), description='[cyan]Encoding texts'):
        batch_texts = texts[i:i + batch_size]
        # BGE 推荐在句首加指令；中文短文本可选。这里保持最简。
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        sentence_embeddings = mean_pooling(outputs, encoded['attention_mask'])
        # BGE 常见做法进行归一化以便余弦相似度
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.append(sentence_embeddings.detach().cpu().numpy())
    return np.vstack(embeddings)



def ensure_milvus_collection(collection_name: str, dim: int) -> Collection:
    connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        return coll

    fields = [
        FieldSchema(name='pk', dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name='venue', dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name='published_at', dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields=fields, description='News embeddings by bge-small-zh-v1.5')
    coll = Collection(name=collection_name, schema=schema)
    # 建索引（可选，HDBSCAN不需要检索；但保留以便后续查询）
    coll.create_index(
        field_name='vector',
        index_params={
            'index_type': 'IVF_FLAT',
            'metric_type': 'IP',  # 因为我们向量已归一化，内积≈余弦
            'params': {'nlist': 1024}
        }
    )
    return coll


def _truncate(s: str, max_len: int) -> str:
    if s is None:
        return ''
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len]


def insert_embeddings(coll: Collection, ids: List[str], vectors: np.ndarray, meta: pd.DataFrame) -> None:
    assert len(ids) == len(vectors) == len(meta)

    
    TITLE_MAX = 1000
    DESC_MAX = 4000
    VENUE_MAX = 256
    URL_MAX = 1000
    PUB_MAX = 64

    title_list = [_truncate(v, TITLE_MAX) for v in meta['title'].fillna('')]
    desc_list = [_truncate(v, DESC_MAX) for v in meta['description'].fillna('')]
    venue_list = [_truncate(v, VENUE_MAX) for v in meta['venue'].fillna('')]
    url_list = [_truncate(v, URL_MAX) for v in meta['url'].fillna('')]
    pub_list = [_truncate(v, PUB_MAX) for v in meta['published_at'].fillna('')]

    entities = [
        ids,
        vectors.tolist(),
        title_list,
        desc_list,
        venue_list,
        url_list,
        pub_list,
    ]
    coll.insert(entities)
    coll.flush()


def load_all_vectors(coll: Collection, limit: int = None) -> Tuple[List[str], np.ndarray]:
    coll.load()
    num_entities = coll.num_entities
    batch_size = 4096
    offset = 0
    all_ids: List[str] = []
    all_vecs: List[np.ndarray] = []

    while offset < num_entities:
        size = batch_size
        if limit is not None:
            size = min(size, max(0, limit - len(all_ids)))
        if size == 0:
            break
        # 通过 pk 顺序分页
        results = coll.query(expr=f"pk >= ''", output_fields=['pk', 'vector'], limit=size, offset=offset)
        if not results:
            break
        # results 是 List[Dict]
        all_ids.extend([r['pk'] for r in results])
        all_vecs.extend([np.array(r['vector'], dtype=np.float32) for r in results])
        offset += len(results)

    if len(all_vecs) == 0:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return all_ids, np.vstack(all_vecs)



def run_hdbscan(vectors: np.ndarray, min_cluster_size: int = 10, metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = clusterer.fit_predict(vectors)
    probabilities = clusterer.probabilities_
    return labels, probabilities


def build_distance_matrix(vectors: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    # 余弦距离矩阵（小心内存：n^2）
    return pairwise_distances(vectors, vectors, metric=metric)


def visualize_umap(vectors: np.ndarray, labels: np.ndarray, out_path: str = 'umap_hdbscan.png') -> None:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    emb2d = reducer.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    uniq = np.unique(labels)
    for lab in uniq:
        mask = labels == lab
        if lab == -1:
            plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=8, c='#999999', label='noise', alpha=0.6)
        else:
            plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=8, label=f'cluster {lab}', alpha=0.8)
    plt.legend(loc='best', fontsize=8, markerscale=2)
    plt.title('UMAP projection with HDBSCAN labels')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main():
    if not os.path.exists(CSV_PATH):
        print(f"[red]CSV 不存在: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    required_cols = ['guid', 'title', 'description', 'venue', 'url', 'published_at']
    for col in required_cols:
        if col not in df.columns:
            print(f"[red]CSV 缺少列: {col}")
            sys.exit(1)

    # 准备文本：title + description
    texts = (df['title'].fillna('') + '。' + df['description'].fillna('')).astype(str).tolist()

    # 加载本地模型
    print('[green]加载本地 bge-small-zh-v1.5...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    model.eval()

    # 编码
    vectors = encode_texts(tokenizer, model, texts, batch_size=BATCH_SIZE)
    assert vectors.shape[1] == EMBEDDING_DIM, f"向量维度不符: {vectors.shape}"

    # 连接 Milvus 并建表
    coll = ensure_milvus_collection(COLLECTION_NAME, EMBEDDING_DIM)

    # 判断是否已存在数据，避免重复插入
    if coll.num_entities == 0:
        print('[green]写入向量到 Milvus...')
        insert_embeddings(coll, ids=df['guid'].astype(str).tolist(), vectors=vectors, meta=df)
    else:
        print(f"[yellow]Milvus 集合已有数据: {coll.num_entities}，跳过插入。")

    # 读取向量用于聚类（也可直接用内存 vectors）
    ids, vecs = load_all_vectors(coll)
    if len(vecs) == 0:
        print('[red]未从 Milvus 读取到向量。')
        sys.exit(1)

    print(f"[green]进行 HDBSCAN 聚类，样本数: {len(vecs)}")
    labels, probs = run_hdbscan(vecs, min_cluster_size=10, metric='euclidean')

    # 距离矩阵（可能占用较多内存）
    print('[green]构建余弦距离矩阵...')
    dist_matrix = build_distance_matrix(vecs, metric='cosine')
    np.save(os.path.join(os.path.dirname(__file__), 'distance_matrix.npy'), dist_matrix)

    # 可视化
    print('[green]UMAP 可视化...')
    visualize_umap(vecs, labels, out_path=os.path.join(os.path.dirname(__file__), 'umap_hdbscan.png'))

    # 导出聚类结果
    out_df = pd.DataFrame({'guid': ids, 'label': labels, 'probability': probs})
    out_df.to_csv(os.path.join(os.path.dirname(__file__), 'hdbscan_labels.csv'), index=False)
    print('[bold green]完成。输出: hdbscan_labels.csv, distance_matrix.npy, umap_hdbscan.png')


if __name__ == '__main__':
    main()
