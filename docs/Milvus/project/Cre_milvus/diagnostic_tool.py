"""
CreMilvus诊断工具

用于检测和诊断向量化存储过程中的问题
"""

import os
import yaml
import logging
import traceback
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreMilvusDiagnostic:
    """CreMilvus诊断工具"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.issues = []
        self.warnings = []
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.issues.append(f"配置文件加载失败: {e}")
            return {}
    
    def check_dependencies(self):
        """检查依赖包"""
        logger.info("检查依赖包...")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("pymilvus", "Milvus客户端"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("fastapi", "FastAPI"),
            ("streamlit", "Streamlit")
        ]
        
        optional_packages = [
            ("clip", "CLIP模型"),
            ("hdbscan", "HDBSCAN聚类"),
            ("umap", "UMAP降维"),
            ("jieba", "中文分词"),
            ("nltk", "自然语言处理")
        ]
        
        for package, description in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✅ {package} ({description}) - 已安装")
            except ImportError:
                self.issues.append(f"❌ {package} ({description}) - 未安装")
        
        for package, description in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✅ {package} ({description}) - 已安装")
            except ImportError:
                self.warnings.append(f"⚠️ {package} ({description}) - 未安装（可选）")
    
    def check_config(self):
        """检查配置文件"""
        logger.info("检查配置文件...")
        
        if not self.config:
            self.issues.append("配置文件为空或无效")
            return
        
        # 检查必需的配置项
        required_configs = {
            "milvus.host": "Milvus主机地址",
            "milvus.port": "Milvus端口",
            "milvus.collection_name": "集合名称",
            "data.data_location": "数据位置"
        }
        
        for config_key, description in required_configs.items():
            keys = config_key.split('.')
            value = self.config
            
            try:
                for key in keys:
                    value = value[key]
                if not value:
                    self.issues.append(f"配置项 {config_key} ({description}) 为空")
                else:
                    logger.info(f"✅ {config_key}: {value}")
            except KeyError:
                self.issues.append(f"缺少配置项 {config_key} ({description})")
    
    def check_data_directory(self):
        """检查数据目录"""
        logger.info("检查数据目录...")
        
        data_location = self.config.get("data", {}).get("data_location")
        if not data_location:
            self.issues.append("数据位置未配置")
            return
        
        if not os.path.exists(data_location):
            self.issues.append(f"数据目录不存在: {data_location}")
            return
        
        # 检查文件类型
        folder = Path(data_location)
        files = list(folder.rglob("*"))
        
        if not files:
            self.warnings.append(f"数据目录为空: {data_location}")
            return
        
        supported_extensions = {".csv", ".md", ".pdf", ".txt", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        valid_files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]
        
        logger.info(f"✅ 数据目录: {data_location}")
        logger.info(f"✅ 总文件数: {len(files)}")
        logger.info(f"✅ 有效文件数: {len(valid_files)}")
        
        # 按类型统计
        file_types = {}
        for file in valid_files:
            ext = file.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        for ext, count in file_types.items():
            logger.info(f"  {ext}: {count} 个文件")
    
    def check_embedding_model(self):
        """检查嵌入模型"""
        logger.info("检查嵌入模型...")
        
        try:
            from Search.embedding import embedder
            status = embedder.check_status()
            
            if status["model_loaded"] and status["tokenizer_loaded"]:
                logger.info(f"✅ 嵌入模型已加载: {status['model_name']}")
                logger.info(f"✅ 设备: {status['device']}")
                
                # 测试嵌入生成
                test_text = "这是一个测试文本"
                embedding = embedder.get_embedding(test_text)
                if embedding:
                    logger.info(f"✅ 嵌入生成测试成功，维度: {len(embedding)}")
                else:
                    self.issues.append("嵌入生成测试失败")
            else:
                self.issues.append("嵌入模型未正确加载")
                
        except Exception as e:
            self.issues.append(f"嵌入模型检查失败: {e}")
    
    def check_milvus_connection(self):
        """检查Milvus连接"""
        logger.info("检查Milvus连接...")
        
        try:
            from pymilvus import connections, utility
            
            milvus_config = self.config.get("milvus", {})
            host = milvus_config.get("host", "127.0.0.1")
            port = milvus_config.get("port", "19530")
            
            # 尝试连接
            connections.connect(alias="diagnostic", host=host, port=port)
            logger.info(f"✅ Milvus连接成功: {host}:{port}")
            
            # 检查集合
            collections = utility.list_collections()
            logger.info(f"✅ 现有集合: {collections}")
            
            collection_name = milvus_config.get("collection_name")
            if collection_name in collections:
                logger.info(f"✅ 目标集合存在: {collection_name}")
            else:
                self.warnings.append(f"目标集合不存在: {collection_name} (将自动创建)")
            
            connections.disconnect("diagnostic")
            
        except Exception as e:
            self.issues.append(f"Milvus连接失败: {e}")
    
    def check_file_processing(self):
        """检查文件处理功能"""
        logger.info("检查文件处理功能...")
        
        try:
            # 检查各种文件处理工具
            from dataBuilder.tools import csvmake, mdmake, pdfmake, txtmake, imgmake
            logger.info("✅ 文件处理工具导入成功")
            
            # 检查分块策略
            try:
                from dataBuilder.chunking import ChunkingManager, get_available_strategies
                strategies = get_available_strategies()
                strategy_names = [s['name'] for s in strategies]
                logger.info(f"✅ 可用分块策略: {strategy_names}")
                
                # 检查每种策略的可用性
                chunking_manager = ChunkingManager()
                test_text = "这是一个测试文本，用于验证分块策略的功能。"
                
                for strategy in strategy_names:
                    try:
                        chunks = chunking_manager.chunk_text(test_text, strategy)
                        if chunks:
                            logger.info(f"  ✅ {strategy}: 测试通过，生成 {len(chunks)} 个块")
                        else:
                            self.warnings.append(f"分块策略 {strategy} 测试失败：返回空结果")
                    except Exception as e:
                        self.warnings.append(f"分块策略 {strategy} 测试失败: {e}")
                        
            except ImportError:
                self.warnings.append("高级分块策略不可用，将使用传统方法")
            
        except Exception as e:
            self.issues.append(f"文件处理功能检查失败: {e}")
    
    def run_full_diagnostic(self):
        """运行完整诊断"""
        logger.info("开始CreMilvus系统诊断...")
        logger.info("=" * 50)
        
        self.check_dependencies()
        self.check_config()
        self.check_data_directory()
        self.check_embedding_model()
        self.check_milvus_connection()
        self.check_file_processing()
        
        # 输出诊断结果
        logger.info("=" * 50)
        logger.info("诊断结果:")
        
        if not self.issues and not self.warnings:
            logger.info("🎉 系统状态良好，没有发现问题！")
        else:
            if self.issues:
                logger.error(f"发现 {len(self.issues)} 个问题:")
                for issue in self.issues:
                    logger.error(f"  {issue}")
            
            if self.warnings:
                logger.warning(f"发现 {len(self.warnings)} 个警告:")
                for warning in self.warnings:
                    logger.warning(f"  {warning}")
        
        return {
            "issues": self.issues,
            "warnings": self.warnings,
            "status": "healthy" if not self.issues else "issues_found"
        }

def main():
    """主函数"""
    diagnostic = CreMilvusDiagnostic()
    result = diagnostic.run_full_diagnostic()
    
    if result["status"] == "healthy":
        print("\n✅ 系统诊断完成，状态良好！")
    else:
        print(f"\n❌ 系统诊断完成，发现 {len(result['issues'])} 个问题需要解决")
        print("请根据上述诊断结果修复问题后重试")

if __name__ == "__main__":
    main()