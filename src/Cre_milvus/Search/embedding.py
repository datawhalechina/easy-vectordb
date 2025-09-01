# Search/embedding.py
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SimpleEmbeddingGenerator()
    return embedder

class SimpleEmbeddingGenerator:
    def __init__(self, model_name="AI-ModelScope/bge-large-zh-v1.5"):
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.use_modelscope = True  # 标记使用ModelScope
        self.load_model()
    
    # 设置递归限制在类级别
    import sys
    original_limit = sys.getrecursionlimit()
    if original_limit < 2000:
        sys.setrecursionlimit(2000)
    
    def load_model(self):
        if self.model is not None:
            return True
        """从ModelScope加载BGE模型"""
        try:
            import os
            import socket
            
            logger.info(f"开始加载模型: {self.model_name}")
            
            # 设置网络超时以避免递归问题
            socket.setdefaulttimeout(30)
            
            # 设置环境变量
            os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
            
            # 尝试使用ModelScope加载
            try:
                from modelscope import AutoTokenizer, AutoModel
                logger.info("使用ModelScope加载模型...")
                
                # 检查本地是否已有模型
                local_model_exists = os.path.exists(os.path.expanduser(f"~/.cache/modelscope/hub/models/{self.model_name}"))
                
                # 加载分词器和模型
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=local_model_exists
                )
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    local_files_only=local_model_exists
                )
                
                logger.info("✅ ModelScope模型加载成功")
                
            except (ImportError, Exception) as modelscope_error:
                # 回退到transformers + HuggingFace
                logger.warning(f"ModelScope加载失败: {modelscope_error}")
                logger.warning("使用HuggingFace...")
                
                from transformers import AutoTokenizer, AutoModel
                
                hf_model_name = "BAAI/bge-large-zh-v1.5"
                local_hf_exists = os.path.exists(os.path.expanduser(f"~/.cache/huggingface/transformers/{hf_model_name}"))
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                    local_files_only=local_hf_exists
                )
                
                self.model = AutoModel.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    local_files_only=local_hf_exists
                )
                
                logger.info("✅ HuggingFace模型加载成功")
            
            # 设置为CPU并进入评估模式
            self.model.to("cpu")
            self.model.eval()
            
            logger.info("✅ 模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.info("将跳过向量模型初始化")
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
    
    def get_embedding_dimension(self):
        """获取嵌入向量维度"""
        if self.model is None:
            if not self.load_model():
                return 1024  # 默认维度
        
        try:
            # 使用测试文本获取维度
            test_embedding = self.get_embedding("测试")
            return len(test_embedding) if test_embedding else 1024
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {e}")
            return 1024  # 默认维度
    
    def check_status(self):
        """检查模型状态"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.device),
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension() if self.model else 1024
        }

# 向后兼容的别名
class EmbeddingModel(SimpleEmbeddingGenerator):
    """向后兼容的EmbeddingModel类"""
    
    def encode(self, texts):
        """兼容旧接口的encode方法"""
        if isinstance(texts, str):
            return [self.get_embedding(texts)]
        elif isinstance(texts, list):
            return self.batch_get_embedding(texts)
        else:
            return []

# 全局实例
embedder = SimpleEmbeddingGenerator()
