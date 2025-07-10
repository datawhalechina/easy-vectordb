# Search/embedding.py
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingGenerator:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        """最可靠的模型加载方法"""
        try:
            logger.info(f"尝试加载模型: {self.model_name}")
            
            # 1. 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 2. 加载模型 - 强制在 CPU 上
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            # 3. 确保模型在 CPU 上
            self.model.to("cpu")
            self.model.eval()
            
            logger.info(f"模型加载成功: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def get_embedding(self, text):
        """获取文本嵌入向量"""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return []
        
        try:
            # 分词并转换为模型输入
            inputs = self.tokenizer(
                [text], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
            
            # 归一化处理
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"生成嵌入失败: {str(e)}")
            return []
    
    def batch_get_embedding(self, texts):
        """批量获取文本嵌入向量（高效）"""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return []
        
        if not texts:
            return []
        
        try:
            # 分词并转换为模型输入
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            )
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
            
            # 归一化处理
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"批量生成嵌入失败: {str(e)}")
            return []
    
    def check_status(self):
        """检查模型状态"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.device),
            "model_name": self.model_name
        }

# 全局实例
embedder = SimpleEmbeddingGenerator()
