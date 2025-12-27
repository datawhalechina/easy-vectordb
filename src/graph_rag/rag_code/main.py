"""
基于图RAG的智能烹饪助手 - 主程序
整合传统检索和图RAG检索，实现真正的图数据优势
"""

import os
import sys
import time
import logging
from typing import List, Optional
from typing import List
from langchain_core.documents import Document
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig
# 在main.py中修改导入
from rag_modules import (
    GraphDataPreparationModule,
    FAISSIndexConstructionModule,  # 替换MilvusIndexConstructionModule
    GenerationIntegrationModule,
    AnnoyIndexConstructionModule,
    MilvusIndexConstructionModule
)
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis

# 加载环境变量
# 尝试指定编码
try:
    load_dotenv()
except UnicodeDecodeError:
    # 尝试其他编码
    load_dotenv(encoding='utf-16')  # 如果是 UTF-16

class AdvancedGraphRAGSystem:
    """
    图RAG系统
    
    核心特性：
    1. 智能路由：自动选择最适合的检索策略
    2. 双引擎检索：传统混合检索 + 图RAG检索
    3. 图结构推理：多跳遍历、子图提取、关系推理
    4. 查询复杂度分析：深度理解用户意图
    5. 自适应学习：基于反馈优化系统性能
    """
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # 核心模块
        self.data_module = None
        self.index_module = None
        self.generation_module = None
        
        # 检索引擎
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.query_router = None
        
        # 系统状态
        self.system_ready = False
        
    def initialize_system(self):
        """初始化高级图RAG系统"""
        logger.info("启动高级图RAG系统...")
        
        try:
            # 1. 数据准备模块
            print("初始化数据准备模块...")
            self.data_module = GraphDataPreparationModule(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
            
            # # 2. 向量索引模块
            # print("初始化Milvus向量索引...")
            # self.index_module = MilvusIndexConstructionModule(
            #     host=self.config.milvus_host,
            #     port=self.config.milvus_port,
            #     collection_name=self.config.milvus_collection_name,
            #     dimension=self.config.milvus_dimension,
            #     model_name=self.config.embedding_model
            # )
            # 2. 向量索引模块（使用FAISS替换Milvus）
            # 根据配置动态选择向量数据库模块
            vector_db_type = self.config.vector_db.lower()
            
            if vector_db_type == "milvus":
                logger.info("正在初始化 Milvus 索引模块...")
                self.index_module = MilvusIndexConstructionModule(
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    collection_name=self.config.milvus_collection_name,
                    dimension=self.config.milvus_dimension,
                    model_name=self.config.embedding_model,
                    index_type=self.config.milvus_index_type,
                    metric_type=self.config.milvus_metric_type,
                    embedding_api_key=self.config.embedding_api_key,
                    embedding_base_url=self.config.embedding_base_url
                )
                
            elif vector_db_type == "annoy":
                logger.info("正在初始化 Annoy 索引模块...")
                self.index_module = AnnoyIndexConstructionModule(
                    index_path=self.config.annoy_index_path,
                    dimension=self.config.annoy_dimension,
                    model_name=self.config.embedding_model,
                    metric=self.config.annoy_metric_type,
                    n_trees=self.config.annoy_n_trees,
                    embedding_api_key=self.config.embedding_api_key,
                    embedding_base_url=self.config.embedding_base_url
                )
                
            elif vector_db_type == "faiss":
                logger.info("正在初始化 FAISS 索引模块...")
                self.index_module = FAISSIndexConstructionModule(
                    index_path=self.config.faiss_index_path,
                    dimension=self.config.faiss_dimension,
                    model_name=self.config.embedding_model,
                    index_type=self.config.faiss_index_type,
                    nlist=self.config.faiss_nlist,
                    embedding_api_key=self.config.embedding_api_key,
                    embedding_base_url=self.config.embedding_base_url
                )
                
            else:
                raise ValueError(f"不支持的向量数据库类型: {vector_db_type}，请检查配置文件中的 VECTOR_DB。")
            # 3. 生成模块
            print("初始化生成模块...")
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.llm_api_key,
                llm_base_url=self.config.llm_base_url
            )
            
            # 4. 传统混合检索模块
            print("初始化传统混合检索...")
            self.traditional_retrieval = HybridRetrievalModule(
                config=self.config,
                vector_module=self.index_module,
                data_module=self.data_module,
                llm_client=self.generation_module.client
            )
            
            # 5. 图RAG检索模块
            print("初始化图RAG检索引擎...")
            self.graph_rag_retrieval = GraphRAGRetrieval(
                config=self.config,
                llm_client=self.generation_module.client
            )
            
            # 6. 智能查询路由器
            print("初始化智能查询路由器...")
            self.query_router = IntelligentQueryRouter(
                traditional_retrieval=self.traditional_retrieval,
                graph_rag_retrieval=self.graph_rag_retrieval,
                llm_client=self.generation_module.client,
                config=self.config
            )
            
            print("✅ 高级图RAG系统初始化完成！")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    # def build_knowledge_base(self):
    #     """构建知识库（如果需要）"""
    #     print("\n检查知识库状态...")
        
    #     try:
    #         # 检查Milvus集合是否存在
    #         if self.index_module.has_collection():
    #             print("✅ 发现已存在的知识库，尝试加载...")
    #             if self.index_module.load_collection():
    #                 print("知识库加载成功！")
                    
    #                 # 重要：即使从已存在的知识库加载，也需要加载图数据以支持图索引
    #                 print("加载图数据以支持图检索...")
    #                 self.data_module.load_graph_data()
    #                 print("构建菜谱文档...")
    #                 self.data_module.build_recipe_documents()
    #                 print("进行文档分块...")
    #                 chunks = self.data_module.chunk_documents(
    #                     chunk_size=self.config.chunk_size,
    #                     chunk_overlap=self.config.chunk_overlap
    #                 )
                    
    #                 self._initialize_retrievers(chunks)
    #                 return
    #             else:
    #                 print("❌ 知识库加载失败，开始重建...")
            
    #         print("未找到已存在的集合，开始构建新的知识库...")
            
    #         # 从Neo4j加载图数据
    #         print("从Neo4j加载图数据...")
    #         self.data_module.load_graph_data()
            
    #         # 构建菜谱文档
    #         print("构建菜谱文档...")
    #         self.data_module.build_recipe_documents()
            
    #         # 进行文档分块
    #         print("进行文档分块...")
    #         chunks = self.data_module.chunk_documents(
    #             chunk_size=self.config.chunk_size,
    #             chunk_overlap=self.config.chunk_overlap
    #         )
            
    #         # 构建Milvus向量索引
    #         print("构建Milvus向量索引...")
    #         if not self.index_module.build_vector_index(chunks):
    #             raise Exception("构建向量索引失败")
            
    #         # 初始化检索器
    #         self._initialize_retrievers(chunks)
            
    #         # 显示统计信息
    #         self._show_knowledge_base_stats()
            
    #         print("✅ 知识库构建完成！")
            
    #     except Exception as e:
    #         logger.error(f"知识库构建失败: {e}")
    #         raise
    def build_knowledge_base(self):
        """
        构建知识库：适配 Milvus, Annoy, FAISS 多后端
        """
        db_type = self.config.vector_db.upper()
        print(f"\n" + "="*20 + f" 检查 {db_type} 知识库 " + "="*20)
        
        try:
            # 1. 尝试从现有索引加载
            # has_collection 检查文件是否存在(FAISS/Annoy)或集合是否存在(Milvus)
            if self.index_module.has_collection():
                print(f"✅ 发现已存在的 {db_type} 索引，正在加载...")
                if self.index_module.load_collection():
                    print(f"✨ {db_type} 索引加载成功！")
                    
                    # 即使加载了向量索引，也需要获取 chunks 来初始化 BM25 检索器
                    # 因为 BM25 需要原始文本数据
                    chunks = self._prepare_essential_data()
                    self._initialize_retrievers(chunks)
                    return
                else:
                    print(f"⚠️ {db_type} 索引文件损坏或加载失败，准备重新构建...")
            else:
                print(f"ℹ️ 未找到 {db_type} 索引，准备执行首次构建...")

            # 2. 全量构建流程
            # 第一步：数据准备 (从 Neo4j 获取数据 -> 转换文档 -> 分块)
            chunks = self._prepare_essential_data()
            
            if not chunks:
                logger.warning("未发现有效文档分块，跳过向量索引构建。")
                return

            # 第二步：构建向量索引
            # 对于 Annoy：这会创建树并保存文件
            # 对于 Milvus：这会创建 Collection 并导入数据
            # 对于 FAISS：这会训练并添加向量
            print(f"🚀 正在向 {db_type} 写入向量并构建索引 (数量: {len(chunks)})...")
            if not self.index_module.build_vector_index(chunks):
                raise Exception(f"{db_type} 索引核心构建任务返回失败")
            
            # 第三步：初始化其他检索组件（BM25, GraphRAG等）
            self._initialize_retrievers(chunks)
            
            # 3. 统计展示
            self._show_knowledge_base_stats()
            print(f"✅ {db_type} 知识库构建并初始化完成！")
            
        except Exception as e:
            logger.error(f"知识库构建流程中断: {e}")
            # 给出具体的排查建议
            if "Annoy" in str(e):
                print("💡 建议：检查 ANNOY_INDEX_PATH 路径是否存在且可写。")
            elif "Milvus" in str(e):
                print("💡 建议：检查 Milvus 服务是否启动且网络端口 19530 可达。")
            raise

    def _prepare_essential_data(self) -> List[Document]:
        """
        内部辅助方法：统一数据准备流程
        """
        print("🔍 正在从 Neo4j 同步图数据并生成文档分块...")
        # 从图数据库加载节点和关系
        self.data_module.load_graph_data()
        
        # 将图节点转换为结构化文档
        self.data_module.build_recipe_documents()
        
        # 文本分块
        chunks = self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        return chunks

   
    def _initialize_retrievers(self, chunks: List = None):
        """初始化检索器"""
        print("初始化检索引擎...")
        
        # 如果没有chunks，从数据模块获取
        if chunks is None:
            chunks = self.data_module.chunks or []
        
        # 初始化传统检索器
        self.traditional_retrieval.initialize(chunks)
        
        # 初始化图RAG检索器
        self.graph_rag_retrieval.initialize()
        
        self.system_ready = True
        print("✅ 检索引擎初始化完成！")
    
    # def _show_knowledge_base_stats(self):
    #     """显示知识库统计信息"""
    #     print(f"\n知识库统计:")
        
    #     # 数据统计
    #     stats = self.data_module.get_statistics()
    #     print(f"   菜谱数量: {stats.get('total_recipes', 0)}")
    #     print(f"   食材数量: {stats.get('total_ingredients', 0)}")
    #     print(f"   烹饪步骤: {stats.get('total_cooking_steps', 0)}")
    #     print(f"   文档数量: {stats.get('total_documents', 0)}")
    #     print(f"   文本块数: {stats.get('total_chunks', 0)}")
        
    #     # Milvus统计
    #     milvus_stats = self.index_module.get_collection_stats()
    #     print(f"   向量索引: {milvus_stats.get('row_count', 0)} 条记录")
        
    #     # 图RAG统计
    #     route_stats = self.query_router.get_route_statistics()
    #     print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")
        
    #     if stats.get('categories'):
    #         categories = list(stats['categories'].keys())[:10]
    #         print(f"   🏷️ 主要分类: {', '.join(categories)}")
    

    def _show_knowledge_base_stats(self):
        """显示知识库统计信息：通用版本"""
        print(f"\n================ 知识库统计 ================ ")
        
        # 1. 基础数据统计 (DataModule)
        stats = self.data_module.get_statistics()
        print(f"📊 数据规模:")
        print(f"   菜谱/节点数量: {stats.get('total_recipes', 0)}")
        print(f"   食材/属性数量: {stats.get('total_ingredients', 0)}")
        print(f"   烹饪步骤:     {stats.get('total_cooking_steps', 0)}")
        print(f"   文本块总数:   {stats.get('total_chunks', 0)}")
        
        # 2. 向量库统计 (通用接口)
        # 这里的 index_module 可能是 Milvus, Annoy 或 FAISS
        db_type = self.config.vector_db.upper()
        vector_stats = self.index_module.get_collection_stats()
        
        print(f"\n🗄️ 向量引擎 ({db_type}):")
        # row_count 是我们在所有模块中统一的键名
        print(f"   记录总数: {vector_stats.get('row_count', 0)} 条")
        
        # 如果有特定引擎的信息也可以打印
        if 'index_type' in vector_stats:
            print(f"   索引类型: {vector_stats.get('index_type', 'N/A')}")
        if 'dimension' in vector_stats:
            print(f"   向量维度: {vector_stats.get('dimension', 'N/A')}")

        # 3. 检索路由统计
        route_stats = self.query_router.get_route_statistics()
        print(f"\n🧠 检索系统状态:")
        print(f"   总查询次数: {route_stats.get('total_queries', 0)} 次")
        
        # 4. 业务元数据展示
        if stats.get('categories'):
            categories = list(stats['categories'].keys())[:10]
            print(f"🏷️ 覆盖领域: {', '.join(categories)}")
        
        print(f"============================================")
    def ask_question_with_routing(self, question: str, stream: bool = False, explain_routing: bool = False):
        """
        智能问答：自动选择最佳检索策略
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
            
        print(f"\n❓ 用户问题: {question}")
        
        # 显示路由决策解释（可选）
        if explain_routing:
            explanation = self.query_router.explain_routing_decision(question)
            print(explanation)
        
        start_time = time.time()
        
        try:
            # 1. 智能路由检索
            print("执行智能查询路由...")
            relevant_docs, analysis = self.query_router.route_query(question, self.config.top_k)
            
            # 2. 显示路由信息
            strategy_icons = {
                "hybrid_traditional": "🔍",
                "graph_rag": "🕸️", 
                "combined": "🔄"
            }
            strategy_icon = strategy_icons.get(analysis.recommended_strategy.value, "❓")
            print(f"{strategy_icon} 使用策略: {analysis.recommended_strategy.value}")
            print(f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")
            
            # 3. 显示检索结果信息
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    recipe_name = doc.metadata.get('recipe_name', '未知内容')
                    search_type = doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown'))
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    doc_info.append(f"{recipe_name}({search_type}, {score:.3f})")
                
                print(f"📋 找到 {len(relevant_docs)} 个相关文档: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等 {len(relevant_docs)} 个结果...")
            else:
                return "抱歉，没有找到相关的烹饪信息。请尝试其他问题。"
            
            # 4. 生成回答
            print("🎯 智能生成回答...")
            
            if stream:
                try:
                    for chunk_text in self.generation_module.generate_adaptive_answer_stream(question, relevant_docs):
                        print(chunk_text, end="", flush=True)
                    print("\n")
                    result = "流式输出完成"
                except Exception as stream_error:
                    logger.error(f"流式输出过程中出现错误: {stream_error}")
                    print(f"\n⚠️ 流式输出中断，切换到标准模式...")
                    # 使用非流式作为后备
                    result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            else:
                result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            
            # 5. 性能统计
            end_time = time.time()
            print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")
            
            return result, analysis
            
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return f"抱歉，处理问题时出现错误：{str(e)}", None
    

    

    
    def run_interactive(self):
        """运行交互式问答"""
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return
            
        print("\n欢迎使用尝尝咸淡RAG烹饪助手！")
        print("可用功能：")
        print("   - 'stats' : 查看系统统计")
        print("   - 'rebuild' : 重建知识库")
        print("   - 'quit' : 退出系统")
        print("\n" + "="*50)
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                elif user_input.lower() == 'rebuild':
                    self._rebuild_knowledge_base()
                    continue
                
                # 普通问答 - 使用默认设置
                use_stream = True  # 默认使用流式输出
                explain_routing = False  # 默认不显示路由决策

                print("\n回答:")
                
                result, analysis = self.ask_question_with_routing(
                    user_input, 
                    stream=use_stream, 
                    explain_routing=explain_routing
                )
                if analysis:
                    print(f"\n💡 路由决策: {analysis.recommended_strategy.value}")
                if not use_stream and result:
                    print(f"{result}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n👋 感谢使用尝尝咸淡RAG烹饪助手！")
        self._cleanup()
    
    def _show_system_stats(self):
        """显示系统统计信息"""
        print("\n系统运行统计")
        print("=" * 40)
        
        # 路由统计
        route_stats = self.query_router.get_route_statistics()
        total_queries = route_stats.get('total_queries', 0)
        
        if total_queries > 0:
            print(f"总查询次数: {total_queries}")
            print(f"传统检索: {route_stats.get('traditional_count', 0)} ({route_stats.get('traditional_ratio', 0):.1%})")
            print(f"图RAG检索: {route_stats.get('graph_rag_count', 0)} ({route_stats.get('graph_rag_ratio', 0):.1%})")
            print(f"组合策略: {route_stats.get('combined_count', 0)} ({route_stats.get('combined_ratio', 0):.1%})")
        else:
            print("暂无查询记录")
        
        # 知识库统计
        self._show_knowledge_base_stats()
    
    # def _rebuild_knowledge_base(self):
    #     """重建知识库"""
    #     print("\n准备重建知识库...")
        
    #     # 确认操作
    #     confirm = input("⚠️  这将删除现有的向量数据并重新构建，是否继续？(y/N): ").strip().lower()
    #     if confirm != 'y':
    #         print("❌ 重建操作已取消")
    #         return
        
    #     try:
    #         print("删除现有的Milvus集合...")
    #         if self.index_module.delete_collection():
    #             print("✅ 现有集合已删除")
    #         else:
    #             print("删除集合时出现问题，继续重建...")
            
    #         # 重新构建知识库
    #         print("开始重建知识库...")
    #         self.build_knowledge_base()
            
    #         print("✅ 知识库重建完成！")
            
    #     except Exception as e:
    #         logger.error(f"重建知识库失败: {e}")
    #         print(f"❌ 重建失败: {e}")
    #         print("建议：请检查Milvus服务状态后重试")
    def _rebuild_knowledge_base(self):
        """重建知识库：支持 Milvus, Annoy, FAISS 多后端"""
        db_type = self.config.vector_db.upper()
        print(f"\n准备重建 {db_type} 知识库...")
        
        # 确认操作
        confirm = input(f"⚠️  这将删除现有的 {db_type} 向量数据并重新构建，是否继续？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 重建操作已取消")
            return
        
        try:
            print(f"正在删除现有的 {db_type} 索引/集合...")
            # 这里的 delete_collection 会根据具体模块执行删除操作：
            # Milvus 会 drop_collection，FAISS/Annoy 会删除本地磁盘文件
            if self.index_module.delete_collection():
                print(f"✅ 现有 {db_type} 索引已成功删除")
            else:
                print(f"⚠️ 未能完全清理现有的 {db_type} 索引，尝试继续重建...")
            
            # 重新构建知识库
            # 这一步会调用我们之前修改过的 build_knowledge_base()，它是后端无关的
            print("开始执行全量构建流程...")
            self.build_knowledge_base()
            
            print(f"✅ {db_type} 知识库重建完成！")
            
        except Exception as e:
            logger.error(f"重建知识库失败: {e}")
            print(f"❌ 重建失败: {e}")
            # 通用建议：涵盖了云端数据库连接和本地文件权限两种可能性
            print(f"建议：请检查 {db_type} 的连接状态或文件读写权限后重试")
    def _cleanup(self):
        """清理资源"""
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()

def main():
    """主函数"""
    try:
        print("启动高级图RAG系统...")
        
        # 创建高级图RAG系统
        rag_system = AdvancedGraphRAGSystem()
        
        # 初始化系统
        rag_system.initialize_system()
        
        # 构建知识库
        rag_system.build_knowledge_base()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统错误: {e}")

if __name__ == "__main__":
    main() 