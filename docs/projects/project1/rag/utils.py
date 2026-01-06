import os
from glob import glob
from typing import List, Dict, Any
from .Embeddings import QwenEmbedding
from .faiss_db import FaissVectorStore
from .llm import load_model, stream_chat
from .prompt import PROMPT
import re

def read_documents(directory: str) -> List[str]:
    """读取目录下所有 .md 文件的内容"""
    documents = []
    for file_path in glob(os.path.join(directory, "**", "*.md"), recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(f.read())
    return documents

def split_markdown(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Markdown 切分：按标题分段，段内保留标题并按长度切分"""
    # 使用正则按标题行拆分文档
    parts = re.split(r'(^#{1,6}\s+.*)', text, flags=re.MULTILINE)
    
    chunks, header = [], ""
    for part in parts:
        part = part.strip()
        if not part: continue
        
        if part.startswith('#'):
            header = part
        else:
            prefix = f"{header}\n" if header else ""
            available = max(50, chunk_size - len(prefix))
            # 按长度切分内容并加上标题前缀
            for i in range(0, len(part), available - chunk_overlap):
                chunks.append(prefix + part[i:i + available])
    return chunks

def initialize_rag(directory: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """初始化 RAG 系统：加载模型、读取文档并构建索引"""
    # 1. 加载嵌入模型
    embedding_model = QwenEmbedding(
        model_path=config["embedding"]["model"],
        device=config["embedding"]["device"]
    )
    
    # 2. 读取和切分文档
    docs = read_documents(directory)
    all_chunks = []
    for doc in docs:
        # 仅处理 Markdown 切分
        all_chunks.extend(split_markdown(doc))
    
    # 检查是否有文档
    if not all_chunks:
        print(f"警告: 在目录 {directory} 中未找到任何 Markdown 文档(.md)")
        # 创建一个空的向量数据库
        test_emb = embedding_model.get_embedding("test")
        dim = len(test_emb)
        print(f"嵌入模型维度: {dim}")
        db = FaissVectorStore(dimension=dim)
        
        persist_path = config["storage"].get("persist_dir", os.path.join(os.getcwd(), "db", "faiss_db"))
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        print(f"创建空的索引文件: {persist_path}")
        db.save(persist_path)
    else:
        # 3. 初始化向量数据库并添加文档
        # 获取嵌入模型的实际维度
        test_emb = embedding_model.get_embedding("test")
        dim = len(test_emb)
        print(f"嵌入模型维度: {dim}")
        db = FaissVectorStore(dimension=dim)
        
        persist_path = config["storage"].get("persist_dir", os.path.join(os.getcwd(), "db", "faiss_db"))
        
        # 检查是否存在已有的索引文件
        if os.path.exists(f"{persist_path}.index") and os.path.exists(f"{persist_path}.pkl"):
            print(f"检测到现有索引文件，正在直接加载: {persist_path}")
            db.load(persist_path)
        else:
            print(f"未找到索引，正在为 {len(all_chunks)} 个文本块生成向量向量（模型较大，请耐心等待）...")
            embeddings = embedding_model.get_embeddings(all_chunks)
            db.add_vectors(embeddings, all_chunks)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(persist_path), exist_ok=True)
            print(f"保存索引到: {persist_path}")
            db.save(persist_path)
    
    # 4. 加载 LLM（如果配置了）
    llm = None
    with_llm = config["llm"]["api_key"] != "YOUR_API_KEY" and config["llm"]["api_key"] != ""
    if with_llm:
        llm = load_model(config)
    
    return {
        "embedding_model": embedding_model,
        "db": db,
        "llm": llm,
        "config": config
    }

def query_rag(question: str, rag_context: Dict[str, Any], top_k: int = 3, stream: bool = False) -> Dict[str, Any]:
    """使用已初始化的 RAG 上下文进行查询"""
    embedding_model = rag_context["embedding_model"]
    db = rag_context["db"]
    llm = rag_context["llm"]
    
    # 检查数据库是否为空
    if db.index is None or db.index.ntotal == 0:
        return {
            "contexts": "数据库中没有文档内容",
            "answer": "数据库中没有文档内容，无法回答您的问题。"
        }
    
    # 1. 检索相关上下文
    query_embedding = embedding_model.get_embedding(question)
    search_results = db.search(query_embedding, k=top_k)
    contexts = "\n".join([res[0] for res in search_results])
    
    # 2. 生成答案
    answer = None
    if llm:
        prompt = PROMPT.format(contexts=contexts, question=question)
        answer = stream_chat(llm, prompt)  
    return {
        "contexts": contexts,
        "answer": answer
    }

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    return config