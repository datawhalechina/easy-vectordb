from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
from tqdm import tqdm

class QwenEmbedding:
    def __init__(self, model_path: str, device: str):
        """初始化 Qwen 嵌入模型"""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def get_embeddings(self, texts: List[str], batch_size: int = 4) -> List[List[float]]:
        """获取文本列表的嵌入向量，增加批处理防止内存溢出"""
        all_embeddings = []
        
        # 增加进度条显示
        for i in tqdm(range(0, len(texts), batch_size), desc="生成向量"):
            batch_texts = texts[i : i + batch_size]
            
            # 对当前批次进行分词
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用最后一层的平均值作为嵌入
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())
                
            # 及时释放显存/内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        return all_embeddings

    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        return self.get_embeddings([text])[0]
