import os
import numpy as np
import pickle
import uuid
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SimpleVectorDB:
    def __init__(self, vector_dim, db_path="vector_db.pkl"):
        """
        初始化向量数据库
        :param vector_dim: 向量维度（所有入库向量必须保持一致）
        :param db_path: 数据持久化存储路径
        """
        self.vector_dim = vector_dim  # 向量维度
        self.db_path = db_path        # 持久化路径
        
        # 数据存储核心结构
        self.vectors = np.array([])               # 存储所有向量（numpy数组，形状为[N, vector_dim]）
        self.vector_ids = []                      # 存储向量唯一ID，与vectors下标一一对应
        self.id_to_index = dict()                 # 映射：向量ID → 向量在vectors中的下标
        self.metadata = dict()                    # 存储向量元数据（key: 向量ID, value: 元数据字典）
        
        # 索引相关（后续初始化）
        self.ivf_index = None                     # IVF索引结构
        self.ivf_kmeans = None
        
    def _check_vector_dim(self, vector):
        """校验向量维度是否符合要求"""
        if len(vector) != self.vector_dim:
            raise ValueError(f"向量维度错误，需为{self.vector_dim}维，当前为{len(vector)}维")
        
    def insert(self, vector, metadata=None):
        """
        插入向量数据
        :param vector: 向量数据（list或numpy.ndarray）
        :param metadata: 元数据（可选，如文本描述、来源等）
        :return: 向量唯一ID（用于后续查询/更新）
        """
        # 类型转换与维度校验
        vector = np.array(vector, dtype=np.float32).flatten()
        self._check_vector_dim(vector)
        
        # 生成唯一ID
        vector_id = str(uuid.uuid4())
        
        # 插入数据
        if len(self.vectors) == 0:
            self.vectors = np.expand_dims(vector, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        self.vector_ids.append(vector_id)
        self.id_to_index[vector_id] = len(self.vector_ids) - 1
        if metadata:
            self.metadata[vector_id] = metadata
        
        return vector_id

    def get_by_id(self, vector_id):
        """通过ID查询向量及元数据"""
        if vector_id not in self.id_to_index:
            raise KeyError(f"未找到ID为{vector_id}的向量")
        index = self.id_to_index[vector_id]
        return {
            "vector_id": vector_id,
            "vector": self.vectors[index].tolist(),
            "metadata": self.metadata.get(vector_id, {})
        }

    def update(self, vector_id, new_vector=None, new_metadata=None):
        """更新向量数据或元数据"""
        if vector_id not in self.id_to_index:
            raise KeyError(f"未找到ID为{vector_id}的向量")
        index = self.id_to_index[vector_id]
        
        # 更新向量（若提供新向量）
        if new_vector is not None:
            new_vector = np.array(new_vector, dtype=np.float32).flatten()
            self._check_vector_dim(new_vector)
            self.vectors[index] = new_vector
        
        # 更新元数据（若提供新元数据）
        if new_metadata is not None:
            self.metadata[vector_id] = new_metadata

    def delete(self, vector_id):
        """删除向量数据"""
        if vector_id not in self.id_to_index:
            raise KeyError(f"未找到ID为{vector_id}的向量")
        index = self.id_to_index[vector_id]
        
        # 删除核心数据
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.vector_ids.pop(index)
        del self.id_to_index[vector_id]
        if vector_id in self.metadata:
            del self.metadata[vector_id]
        
        # 重新构建ID与下标的映射（因删除后下标发生变化）
        self.id_to_index = {vid: idx for idx, vid in enumerate(self.vector_ids)}

    def brute_force_search(self, query_vector, top_k=5):
        """
        暴力检索（精确匹配）：计算查询向量与所有向量的余弦相似度
        :param query_vector: 查询向量
        :param top_k: 返回相似度最高的前k个结果
        :return: 检索结果（按相似度降序排列）
        """
        if len(self.vectors) == 0:
            return []
        
        # 预处理查询向量
        query_vector = np.array(query_vector, dtype=np.float32).flatten()
        self._check_vector_dim(query_vector)
        
        # 计算余弦相似度（利用sklearn简化实现，也可手动实现：(a·b)/(||a||·||b||)）
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        
        # 按相似度降序排序，取前top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 组装结果
        results = []
        for idx in top_indices:
            vector_id = self.vector_ids[idx]
            results.append({
                "vector_id": vector_id,
                "similarity": float(similarities[idx]),
                "vector": self.vectors[idx].tolist(),
                "metadata": self.metadata.get(vector_id, {})
            })
        return results

    def build_ivf_index(self, n_clusters=8):
        """
        构建IVF索引（基于KMeans聚类）
        核心逻辑：将向量聚类到n_clusters个桶中，检索时先找查询向量所属的桶，再在桶内暴力检索
        """
        if len(self.vectors) == 0:
            raise ValueError("数据库中无向量数据，无法构建索引")
        
        # 转换为 float64
        vectors_for_kmeans = self.vectors.astype(np.float64)
        
        # KMeans 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors_for_kmeans)
        
        # 构建IVF索引
        self.ivf_index = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(cluster_labels):
            self.ivf_index[label].append(idx)
        
        self.ivf_kmeans = kmeans

    def ivf_search(self, query_vector, top_k=5):
        """基于IVF索引的近似检索"""
        if self.ivf_index is None:
            raise ValueError("请先调用build_ivf_index()构建IVF索引")
        if len(self.vectors) == 0:
            return []
        
        # 1. 预处理查询向量，确定其所属的聚类（桶）
        query_vector = np.array(query_vector, dtype=np.float64).flatten()
        self._check_vector_dim(query_vector)
        cluster_id = self.ivf_kmeans.predict([query_vector])[0]
        
        # 2. 获取该聚类下的所有向量下标，提取对应向量
        cluster_indices = self.ivf_index[cluster_id]
        if not cluster_indices:
            return []
        cluster_vectors = self.vectors[cluster_indices]
        
        # 3. 在聚类内计算相似度并排序
        similarities = cosine_similarity([query_vector], cluster_vectors)[0]
        top_cluster_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. 组装结果（映射回原数据库的向量ID）
        results = []
        for idx in top_cluster_indices:
            original_idx = cluster_indices[idx]
            vector_id = self.vector_ids[original_idx]
            results.append({
                "vector_id": vector_id,
                "similarity": float(similarities[idx]),
                "vector": self.vectors[original_idx].tolist(),
                "metadata": self.metadata.get(vector_id, {}),
                "cluster_id": int(cluster_id)  # 标注所属聚类，便于理解
            })
        return results
    
    def save(self):
        """将数据库数据与索引持久化到本地文件"""
        data = {
            "vector_dim": self.vector_dim,
            "vectors": self.vectors,
            "vector_ids": self.vector_ids,
            "id_to_index": self.id_to_index,
            "metadata": self.metadata,
            "ivf_index": self.ivf_index,
            "ivf_kmeans": self.ivf_kmeans  # 存储聚类模型
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"数据库已保存至{self.db_path}")

    @classmethod
    def load(cls, db_path="vector_db.pkl"):
        """从本地文件加载数据库"""
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        
        # 重建数据库实例
        db = cls(vector_dim=data["vector_dim"], db_path=db_path)
        db.vectors = data["vectors"]
        db.vector_ids = data["vector_ids"]
        db.id_to_index = data["id_to_index"]
        db.metadata = data["metadata"]
        db.ivf_index = data["ivf_index"]
        db.ivf_kmeans = data.get("ivf_kmeans")
        
        print(f"已从{db_path}加载数据库，共包含{len(db.vectors)}个向量")
        return db

def read_txt_and_split(file_path):
    """读取txt文件内容并切分为句子列表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定的txt文件不存在：{file_path}")
    # 读取文件（默认UTF-8编码，若有乱码可尝试gbk）
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 句子切分
    sentences = content.split("。")
    print(f"成功读取txt文件，共切分出 {len(sentences)} 个句子")
    return sentences

def text_to_vector(text, model):
    """将文本转换为向量（基于SentenceTransformer模型）"""
    # encode方法直接返回向量，convert_to_numpy=True确保输出为numpy数组
    return model.encode(text, convert_to_numpy=True)

if __name__ == "__main__":
    # 1. 配置参数（请根据实际情况修改txt文件路径和模型路径）
    txt_file = "./data/Datawhale社区介绍大模型改写版.txt"  # 你的txt文件路径
    model_dir = "./model/iic/nlp_gte_sentence-embedding_chinese-base"  # 预训练模型本地存储路径
    VECTOR_DIM = 768  # 主流中文语义模型输出维度多为768，可根据实际模型调整
    
    # 2. 加载中文语义向量模型（两种加载方式可选）
    print("正在加载中文文本嵌入模型...")
    model = SentenceTransformer(model_dir)
    print(f"成功加载本地模型：{model_dir}")

    # 3. 初始化向量数据库
    db = SimpleVectorDB(vector_dim=VECTOR_DIM)
    
    # 4. 读取txt文件并切分句子
    print(f"\n正在读取并处理txt文件：{txt_file}")
    sentences = read_txt_and_split(txt_file)
    if not sentences:
        raise ValueError("未从txt文件中提取到有效句子")
    
    # 5. 句子转向量并插入数据库（附带元数据：原始句子）
    print("\n正在将句子转向量并插入数据库...")
    embeddings = model.encode(sentences)  # 批量生成向量，效率更高
    for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings), 1):
        # 元数据包含句子内容和序号，便于后续查看
        metadata = {"sentence": sentence, "sequence": idx}
        db.insert(embedding, metadata=metadata)
    
    # 6. 持久化数据库
    db.save()
    
    # 7. 测试暴力检索（查询与向量数据库相关的内容）
    print("\n=== 暴力检索结果（查询：'Datawhale有多个学习者参与活动'）===")
    query_text = "Datawhale有多个学习者参与活动"
    query_vector = text_to_vector(query_text, model)
    brute_results = db.brute_force_search(query_vector, top_k=3)
    for res in brute_results:
        print(f"相似度：{res['similarity']:.4f} | 句子：{res['metadata']['sentence']}")
    
    # 8. 构建IVF索引并测试近似检索
    print("\n=== IVF近似检索结果（查询：'Datawhale有多个学习者参与活动'）===")
    # 根据句子数量调整聚类数（一般为数据量的平方根左右）
    n_clusters = max(2, int(len(sentences)**0.5))
    db.build_ivf_index(n_clusters=n_clusters)
    ivf_results = db.ivf_search(query_vector, top_k=3)
    for res in ivf_results:
        print(f"相似度：{res['similarity']:.4f} | 聚类ID：{res['cluster_id']} | 句子：{res['metadata']['sentence']}")
    
    # 9. 测试数据更新与查询
    print("\n=== 数据更新与查询测试 ===")
    # 获取第一个向量的ID（即第一个句子对应的向量）
    first_vector_id = db.vector_ids[0]
    first_sentence = db.get_by_id(first_vector_id)['metadata']['sentence']
    print(f"待更新的原始句子：{first_sentence}")
    # 更新其元数据（模拟句子修正）
    new_metadata = {"sentence": f"【修正】{first_sentence}", "sequence": 1, "updated": True}
    db.update(first_vector_id, new_metadata=new_metadata)
    # 按ID查询更新结果
    updated_res = db.get_by_id(first_vector_id)
    print(f"更新后的数据：{updated_res['metadata']['sentence']}")
    
    # 10. 测试数据删除
    db.delete(first_vector_id)
    print(f"\n删除后数据库向量总数：{len(db.vectors)}")
    
    # 11. 从本地加载数据库验证持久化功能
    print("\n=== 从本地加载数据库 ===")
    loaded_db = SimpleVectorDB.load()
    print(f"加载的数据库向量总数：{len(loaded_db.vectors)}")
    # 验证加载的数据
    if loaded_db.vector_ids:
        sample_id = loaded_db.vector_ids[0]
        sample_data = loaded_db.get_by_id(sample_id)
        print(f"加载数据示例：{sample_data['metadata']['sentence']}")