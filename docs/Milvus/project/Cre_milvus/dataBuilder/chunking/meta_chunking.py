"""
Meta-chunking实现模块

集成PPL困惑度分块和边际采样分块策略
"""

import logging
from typing import List, Dict, Optional, Any
import re
import math

logger = logging.getLogger(__name__)

class DependencyChecker:
    """PPL分块依赖检查器"""
    
    def __init__(self):
        self._dependency_cache = {}
        self._last_check_time = 0
        self._cache_duration = 60  # 缓存60秒
    
    def check_ppl_dependencies(self) -> Dict[str, bool]:
        """检查PPL分块所需依赖"""
        import time
        current_time = time.time()
        
        # 检查缓存
        if (current_time - self._last_check_time) < self._cache_duration and self._dependency_cache:
            return self._dependency_cache
        
        dependencies = {
            "torch": False,
            "torch_functional": False,
            "chunking_class": False,
            "nltk": False,
            "jieba": False,
            "perplexity_chunking": False
        }
        
        # 检查PyTorch
        try:
            import torch
            import torch.nn.functional as F
            dependencies["torch"] = True
            dependencies["torch_functional"] = True
        except ImportError:
            pass
        
        # 检查Chunking类
        try:
            from .perplexity_chunking import Chunking
            dependencies["chunking_class"] = True
            dependencies["perplexity_chunking"] = True
        except ImportError:
            try:
                from perplexity_chunking import Chunking
                dependencies["chunking_class"] = True
                dependencies["perplexity_chunking"] = True
            except ImportError:
                pass
        
        # 检查NLP库
        try:
            from nltk.tokenize import sent_tokenize
            dependencies["nltk"] = True
        except ImportError:
            pass
        
        try:
            import jieba
            dependencies["jieba"] = True
        except ImportError:
            pass
        
        self._dependency_cache = dependencies
        self._last_check_time = current_time
        return dependencies
    
    def get_missing_dependencies(self) -> List[str]:
        """获取缺失的依赖"""
        deps = self.check_ppl_dependencies()
        missing = []
        
        if not deps["torch"]:
            missing.append("torch")
        if not deps["chunking_class"]:
            missing.append("perplexity_chunking模块")
        if not deps["nltk"]:
            missing.append("nltk")
        if not deps["jieba"]:
            missing.append("jieba")
        
        return missing
    
    def suggest_installation_commands(self) -> List[str]:
        """建议安装命令"""
        missing = self.get_missing_dependencies()
        commands = []
        
        if "torch" in missing:
            commands.append("pip install torch")
        if "nltk" in missing:
            commands.append("pip install nltk")
        if "jieba" in missing:
            commands.append("pip install jieba")
        if "perplexity_chunking模块" in missing:
            commands.append("确保perplexity_chunking.py文件存在于当前目录")
        
        return commands
    
    def validate_glm_api_availability(self) -> bool:
        """验证GLM API可用性"""
        try:
            # 这里应该检查GLM API配置
            # 暂时返回True，实际实现需要检查API配置
            return True
        except Exception:
            return False
    
    def is_ppl_chunking_available(self) -> bool:
        """检查PPL分块是否可用"""
        deps = self.check_ppl_dependencies()
        # PPL分块需要torch和chunking_class
        return deps["torch"] and deps["chunking_class"]
    
    def get_dependency_status_message(self) -> str:
        """获取依赖状态消息"""
        if self.is_ppl_chunking_available():
            return "✅ PPL分块依赖已满足"
        else:
            missing = self.get_missing_dependencies()
            return f"❌ PPL分块依赖缺失: {', '.join(missing)}"

class ChunkingStrategyResolver:
    """分块策略智能选择器"""
    
    def __init__(self, dependency_checker: DependencyChecker):
        self.dependency_checker = dependency_checker
        self._strategy_fallbacks = {
            "meta_ppl": ["semantic", "traditional"],
            "msp": ["margin_sampling", "semantic", "traditional"],
            "margin_sampling": ["semantic", "traditional"],
            "semantic": ["traditional"],
            "traditional": []  # 传统方法是最后的fallback
        }
    
    def resolve_strategy(self, requested: str, config: Dict) -> str:
        """
        智能选择可用的分块策略
        
        参数:
            requested: 用户请求的策略
            config: 配置信息（包含GLM API等）
        
        返回:
            实际可用的策略名称
        """
        # 检查请求的策略是否可用
        if self._is_strategy_available(requested, config):
            return requested
        
        # 如果不可用，尝试fallback策略
        fallbacks = self._strategy_fallbacks.get(requested, ["traditional"])
        
        for fallback in fallbacks:
            if self._is_strategy_available(fallback, config):
                logger.info(f"策略 {requested} 不可用，使用 {fallback} 作为替代")
                return fallback
        
        # 最后的fallback
        logger.warning(f"所有策略都不可用，使用传统分块")
        return "traditional"
    
    def get_fallback_strategy(self, failed_strategy: str) -> str:
        """获取失败策略的fallback"""
        fallbacks = self._strategy_fallbacks.get(failed_strategy, ["traditional"])
        return fallbacks[0] if fallbacks else "traditional"
    
    def log_strategy_decision(self, decision: Dict) -> None:
        """记录策略决策日志"""
        requested = decision.get("requested")
        actual = decision.get("actual")
        reason = decision.get("reason", "")
        
        if requested == actual:
            logger.info(f"使用请求的策略: {actual}")
        else:
            logger.warning(f"策略降级: {requested} -> {actual}, 原因: {reason}")
    
    def _is_strategy_available(self, strategy: str, config: Dict) -> bool:
        """检查策略是否可用"""
        if strategy == "traditional":
            return True  # 传统策略总是可用
        
        if strategy == "semantic":
            return True  # 语义分块不依赖外部库
        
        if strategy in ["meta_ppl", "msp", "margin_sampling"]:
            # 这些策略需要GLM API或本地模型
            has_glm_api = config.get("glm_configured", False)
            has_local_deps = self.dependency_checker.is_ppl_chunking_available()
            
            # 检查是否有API客户端可用
            has_api_client = False
            try:
                from .glm_config import get_glm_config_service
                service = get_glm_config_service()
                glm_config = service.get_active_config()
                has_api_client = glm_config is not None
            except Exception:
                pass
            
            return has_glm_api or has_local_deps or has_api_client
        
        return False
    
    def get_strategy_requirements(self, strategy: str) -> Dict[str, Any]:
        """获取策略的依赖要求"""
        requirements = {
            "traditional": {
                "dependencies": [],
                "description": "无特殊依赖，基于固定长度切分"
            },
            "semantic": {
                "dependencies": [],
                "description": "基于语义相似度切分，无外部依赖"
            },
            "meta_ppl": {
                "dependencies": ["torch", "perplexity_chunking", "GLM API"],
                "description": "基于困惑度的智能切分，需要模型支持"
            },
            "msp": {
                "dependencies": ["GLM API"],
                "description": "边际采样分块，需要GLM API支持"
            },
            "margin_sampling": {
                "dependencies": ["GLM API"],
                "description": "边际采样切分，需要GLM API支持"
            }
        }
        
        return requirements.get(strategy, {
            "dependencies": ["未知"],
            "description": "未知策略"
        })

try:
    from .perplexity_chunking import Chunking
except ImportError:
    try:
        from perplexity_chunking import Chunking
    except ImportError:
        logger.warning("无法导入Chunking类，PPL分块功能将不可用")
        Chunking = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch不可用，某些分块策略将降级")
    TORCH_AVAILABLE = False
    torch = None
    F = None

try:
    from nltk.tokenize import sent_tokenize
    import jieba 
    NLP_AVAILABLE = True
except ImportError:
    logger.warning("NLTK或jieba不可用，将使用简单的文本分割")
    sent_tokenize = None
    jieba = None
    NLP_AVAILABLE = False


def split_text_by_punctuation(text, language): 
    """
    按标点符号将文本分割为句子
    
    这是文本预处理的第一步，将长文本分割为句子级别的片段。
    不同语言使用不同的分割策略。
    
    参数:
        text: 输入的原始文本
        language: 语言类型 ('zh'中文 或 'en'英文)
    
    返回:
        sentences: 分割后的句子列表
    """
    if language == 'zh' and jieba: 
        # 中文文本处理
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        
        # 重新组合句子：遇到句末标点符号时结束一个句子
        for word in sentences_list:  
            if word in ["。", "！", "？", "；"]:  # 中文句末标点
                sentences.append(temp_sentence.strip() + word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        
        # 处理最后一个句子（可能没有标点结尾）
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        return sentences
    else:
        # 英文文本处理或fallback处理
        if sent_tokenize:
            full_segments = sent_tokenize(text)
        else:
            # 简单的句子分割fallback
            full_segments = re.split(r'[.!?]+', text)
            full_segments = [s.strip() for s in full_segments if s.strip()]
        
        ret = []
        
        # 限制句子长度，防止单个句子过长影响模型处理
        for item in full_segments:
            item_l = item.strip().split(' ')  # 按空格分割单词
            
            # 如果句子太长，进行截断
            if len(item_l) > 512:
                if len(item_l) > 1024:
                    # 超长句子截断到256个单词
                    item = ' '.join(item_l[:256]) + "..."
                else:
                    # 长句子截断到512个单词
                    item = ' '.join(item_l[:512]) + "..."
            ret.append(item)
        return ret


def find_minima(values, threshold):  
    """
    在困惑度序列中寻找局部极小值点
    
    这是PPL分块算法的核心：通过识别困惑度的局部极小值来确定文本的语义边界。
    
    参数:
        values: 困惑度值序列（每个句子的平均困惑度）
        threshold: 阈值，用于过滤微小的波动，只保留显著的极小值点
    
    返回:
        minima_indices: 局部极小值点的索引列表
    """
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        # 情况1：标准的局部极小值（两边都比当前值大）
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            # 检查下降幅度是否显著（左边或右边的下降超过阈值）
            if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                minima_indices.append(i)  
        # 情况2：左边下降，右边持平的情况
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            # 只要左边的下降超过阈值就认为是有效的分割点
            if values[i - 1] - values[i] >= threshold:
                minima_indices.append(i) 
    return minima_indices


def find_minima_dynamic(values, threshold, threshold_zlist):  
    """
    动态阈值版本的局部极小值检测函数
    
    与find_minima函数类似，但增加了动态阈值调整机制。
    通过记录历史的下降幅度，动态调整阈值以适应不同文本的特性。
    
    参数:
        values: 困惑度值序列
        threshold: 当前阈值
        threshold_zlist: 历史阈值记录列表
    
    返回:
        minima_indices: 局部极小值点的索引列表
        threshold: 更新后的阈值
        threshold_zlist: 更新后的历史阈值记录
    """
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        # 标准局部极小值检测
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                minima_indices.append(i)
                # 记录较小的下降幅度（更保守的估计）
                threshold_zlist.append(min(values[i - 1] - values[i], values[i + 1] - values[i]))  
        # 左边下降，右边持平的情况
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1] - values[i] >= threshold:
                minima_indices.append(i) 
                threshold_zlist.append(values[i - 1] - values[i])
        
        # 动态阈值调整：当有足够的历史数据时
        if len(threshold_zlist) >= 100:
            # 使用历史记录中的最小值作为新阈值（更严格的标准）
            avg = min(threshold_zlist)
            threshold = avg
    
    return minima_indices, threshold, threshold_zlist


class MetaChunking:
    """
    Meta-chunking主类，提供多种文本分块策略
    """
    
    def __init__(self, model=None, tokenizer=None, api_client=None):
        """
        初始化Meta-chunking
        
        参数:
            model: 语言模型（用于PPL分块）
            tokenizer: 分词器
            api_client: API客户端（用于API调用方式）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.api_client = api_client
        self.dependency_checker = DependencyChecker()
        
        # 如果没有提供API客户端，尝试从GLM配置创建
        if not self.api_client:
            try:
                from .glm_config import create_glm_api_client
                self.api_client = create_glm_api_client()
                if self.api_client:
                    logger.info("成功创建GLM API客户端")
            except Exception as e:
                logger.warning(f"创建GLM API客户端失败: {e}")
        
        if model and tokenizer:
            self.chunking = Chunking(model, tokenizer)
    
    def smart_chunking(self, text: str, strategy: str, config: Dict, **kwargs) -> List[str]:
        """
        智能分块方法，自动选择最佳可用策略
        
        参数:
            text: 输入文本
            strategy: 请求的策略
            config: 配置信息
            **kwargs: 策略特定参数
        
        返回:
            分块后的文本列表
        """
        resolver = ChunkingStrategyResolver(self.dependency_checker)
        actual_strategy = resolver.resolve_strategy(strategy, config)
        
        # 记录策略决策
        decision = {
            "requested": strategy,
            "actual": actual_strategy,
            "reason": "依赖检查" if strategy != actual_strategy else "策略可用"
        }
        resolver.log_strategy_decision(decision)
        
        # 执行实际策略
        if actual_strategy == "meta_ppl":
            # PPL分块参数
            ppl_kwargs = {k: v for k, v in kwargs.items() if k in ['threshold', 'language']}
            return self.ppl_chunking(text, **ppl_kwargs)
        elif actual_strategy == "msp":
            # MSP分块参数
            msp_kwargs = {k: v for k, v in kwargs.items() if k in ['threshold', 'language']}
            return self.msp_chunking(text, **msp_kwargs)
        elif actual_strategy == "margin_sampling":
            # 边际采样分块参数
            margin_kwargs = {k: v for k, v in kwargs.items() if k in ['language', 'chunk_length']}
            return self.margin_sampling_chunking(text, **margin_kwargs)
        elif actual_strategy == "semantic":
            # 语义分块参数
            semantic_kwargs = {k: v for k, v in kwargs.items() if k in ['similarity_threshold', 'min_chunk_size', 'max_chunk_size']}
            return self.semantic_chunking(text, **semantic_kwargs)
        else:  # traditional
            # 传统分块参数
            traditional_kwargs = {k: v for k, v in kwargs.items() if k in ['chunk_size', 'overlap']}
            return self.traditional_chunking(text, **traditional_kwargs)
    
    def ppl_chunking(self, text: str, threshold: float = 0.3, language: str = 'zh') -> List[str]:
        """
        PPL困惑度分块
        
        参数:
            text: 输入文本
            threshold: 困惑度阈值
            language: 语言类型
        
        返回:
            分块后的文本列表
        """
        try:
            # 检查依赖
            if not self.dependency_checker.is_ppl_chunking_available():
                missing_deps = self.dependency_checker.get_missing_dependencies()
                install_commands = self.dependency_checker.suggest_installation_commands()
                
                logger.warning(f"PPL分块依赖缺失: {', '.join(missing_deps)}")
                logger.info(f"建议安装命令: {'; '.join(install_commands)}")
                
                # 如果有API客户端，优先使用API方式
                if self.api_client:
                    logger.info("使用API方式进行PPL分块")
                    return self._extract_by_ppl_api(text, threshold, language)
                else:
                    logger.warning("降级到语义分块")
                    return self.semantic_chunking(text, similarity_threshold=0.7)
            
            if self.api_client:
                # 使用API方式进行分块
                return self._extract_by_ppl_api(text, threshold, language)
            elif self.model and self.tokenizer and TORCH_AVAILABLE and Chunking:
                # 使用本地模型方式
                return self._extract_by_ppl(text, threshold, language)
            else:
                logger.warning("PPL分块所需依赖不可用，降级到语义分块")
                return self.semantic_chunking(text, similarity_threshold=0.7)
        except Exception as e:
            logger.error(f"PPL分块失败: {e}")
            # 降级到传统分块
            return self.traditional_chunking(text, chunk_size=512, overlap=50)
    
    def margin_sampling_chunking(self, text: str, language: str = 'zh', chunk_length: int = 512) -> List[str]:
        """
        边际采样分块
        
        参数:
            text: 输入文本
            language: 语言类型
            chunk_length: 最大块长度
        
        返回:
            分块后的文本列表
        """
        try:
            if self.api_client:
                # 使用API方式进行分块
                return self._extract_by_margin_sampling_api(text, language, chunk_length)
            elif self.model and self.tokenizer and TORCH_AVAILABLE:
                # 使用本地模型方式
                return self._extract_by_margin_sampling(text, language, chunk_length)
            else:
                logger.warning("边际采样分块所需依赖不可用，降级到语义分块")
                return self.semantic_chunking(text, similarity_threshold=0.7, max_chunk_size=chunk_length)
        except Exception as e:
            logger.error(f"边际采样分块失败: {e}")
            # 降级到传统分块
            return self.traditional_chunking(text, chunk_size=chunk_length, overlap=50)
    
    def traditional_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        传统分块方法
        
        参数:
            text: 输入文本
            chunk_size: 块大小
            overlap: 重叠大小
        
        返回:
            分块后的文本列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    def msp_chunking(self, text: str, language: str = 'zh', chunk_length: int = 512, confidence_threshold: float = 0.7) -> List[str]:
        """
        MSP (Margin Sampling Partitioning) 切分策略
        
        基于边际概率的高级切分策略，结合置信度阈值进行更精确的切分
        
        参数:
            text: 输入文本
            language: 语言类型
            chunk_length: 最大块长度
            confidence_threshold: 置信度阈值
        
        返回:
            分块后的文本列表
        """
        if not self.model or not self.tokenizer:
            # 降级到语义切分方法（比传统方法更智能）
            return self.semantic_chunking(
                text, 
                similarity_threshold=0.7, 
                min_chunk_size=chunk_length // 4, 
                max_chunk_size=chunk_length
            )
        
        return self._extract_by_msp(text, language, chunk_length, confidence_threshold)
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.8, 
                         min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
        """
        语义切分策略
        
        基于语义相似度的智能切分，保持语义完整性
        
        参数:
            text: 输入文本
            similarity_threshold: 语义相似度阈值
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        
        返回:
            分块后的文本列表
        """
        return self._extract_by_semantic(text, similarity_threshold, min_chunk_size, max_chunk_size)
    
    def _extract_by_ppl(self, sub_text: str, threshold: float, language: str) -> List[str]:
        """
        PPL分块的内部实现
        """
        try:
            # 检查必要的依赖
            if not TORCH_AVAILABLE or not self.model or not self.tokenizer or not Chunking:
                raise ValueError("PPL分块所需依赖不可用")
            
            # 初始化chunking对象
            if not hasattr(self, 'chunking') or not self.chunking:
                self.chunking = Chunking(self.model, self.tokenizer)
            
            # 句子分割
            segments = split_text_by_punctuation(sub_text, language)
            segments = [item for item in segments if item.strip()]
            
            if len(segments) <= 1:
                return [sub_text]
            
            # 编码所有句子
            len_sentences = []
            input_ids = torch.tensor([[]], device=self.model.device, dtype=torch.long)  
            attention_mask = torch.tensor([[]], device=self.model.device, dtype=torch.long)  
            
            for context in segments:
                try:
                    tokenized_text = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                    input_id = tokenized_text["input_ids"].to(self.model.device)
                    input_ids = torch.cat([input_ids, input_id], dim=-1)
                    len_sentences.append(input_id.shape[1])
                    attention_mask_tmp = tokenized_text["attention_mask"].to(self.model.device)
                    attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
                except Exception as e:
                    logger.error(f"句子编码失败: {e}")
                    # 跳过有问题的句子
                    continue

            if len(len_sentences) == 0:
                return [sub_text]

            # 计算困惑度
            loss, past_key_values = self.chunking.get_ppl_batch( 
                input_ids,
                attention_mask,
                past_key_values=None,
                return_kv=True
            )
            
            # 计算句子级平均困惑度
            first_cluster_ppl = []
            index = 0
            for i in range(len(len_sentences)):
                try:
                    if i == 0:
                        if len_sentences[i] > 1:
                            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
                            index += len_sentences[i] - 1
                        else:
                            first_cluster_ppl.append(0.0)
                    else:
                        if len_sentences[i] > 0:
                            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
                            index += len_sentences[i]
                        else:
                            first_cluster_ppl.append(0.0)
                except Exception as e:
                    logger.warning(f"困惑度计算失败，使用默认值: {e}")
                    first_cluster_ppl.append(0.5)
            
            # 寻找分割点
            minima_indices = find_minima(first_cluster_ppl, threshold)
            
            # 生成文本块
            split_points = [0] + minima_indices + [len(segments)-1]    
            final_chunks = []
            
            for i in range(len(split_points)-1):
                chunk_sentences = []
                start_idx = split_points[i]
                end_idx = split_points[i+1]
                
                for sp_index in range(start_idx, end_idx+1):
                    if sp_index < len(segments):
                        chunk_sentences.append(segments[sp_index])
                
                if chunk_sentences:
                    final_chunks.append(''.join(chunk_sentences))
            
            return final_chunks if final_chunks else [sub_text]
            
        except Exception as e:
            logger.error(f"PPL分块内部实现失败: {e}")
            # 降级到语义分块
            return self.semantic_chunking(sub_text, similarity_threshold=0.7)
    
    def _extract_by_margin_sampling(self, text: str, language: str, chunk_length: int) -> List[str]:
        """
        边际采样分块的内部实现
        """
        try:
            full_segments = split_text_by_punctuation(text, language)
            if not full_segments:
                return [text]
            
            tmp = ''
            threshold = 0
            threshold_list = []
            final_chunks = []
            
            for i, sentence in enumerate(full_segments):
                if tmp == '':
                    tmp += sentence
                else:
                    try:
                        prob_subtract = self._get_prob_subtract(tmp, sentence, language)    
                        threshold_list.append(prob_subtract)
                        
                        if prob_subtract > threshold:
                            separator = ' ' if language == 'en' else ''
                            tmp += separator + sentence
                        else:
                            if tmp.strip():
                                final_chunks.append(tmp)
                            tmp = sentence
                    except Exception as e:
                        logger.warning(f"处理句子 {i} 时出错: {e}")
                        # 出错时默认合并
                        separator = ' ' if language == 'en' else ''
                        tmp += separator + sentence
                        
                # 动态阈值更新
                if len(threshold_list) >= 5:
                    last_five = threshold_list[-5:]  
                    avg = sum(last_five) / len(last_five)
                    threshold = avg
                    
            if tmp.strip():
                final_chunks.append(tmp)
            
            # 如果没有生成任何块，返回原文本
            if not final_chunks:
                return [text]
                
            # 合并块以遵守长度限制
            return self._merge_chunks_by_length(final_chunks, chunk_length, language)
            
        except Exception as e:
            logger.error(f"边际采样分块内部实现失败: {e}")
            # 降级到语义分块
            return self.semantic_chunking(text, similarity_threshold=0.7, max_chunk_size=chunk_length)
    
    def _get_prob_subtract(self, sentence1: str, sentence2: str, language: str) -> float:
        """
        计算边际采样的概率差值
        """
        try:
            # 优先使用API客户端
            if self.api_client:
                return self._get_prob_subtract_api(sentence1, sentence2, language)
            
            # 使用本地模型
            if not TORCH_AVAILABLE or not self.model or not self.tokenizer:
                logger.warning("本地模型不可用，使用默认概率")
                return 0.0
            
            if language == 'zh':
                query = '''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
                1. 将"{}"分割成"{}"与"{}"两部分；
                2. 将"{}"不进行分割，保持原形式；
                请回答1或2。'''.format(sentence1+sentence2, sentence1, sentence2, sentence1+sentence2)
            else:
                query = '''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
                1. Split "{}" into "{}" and "{}" two parts;
                2. Keep "{}" unsplit in its original form;
                Please answer 1 or 2.'''.format(sentence1+' '+sentence2, sentence1, sentence2, sentence1+' '+sentence2)
            
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
            output_ids = self.tokenizer.encode(['1','2'], return_tensors='pt').to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(prompt_ids)
                next_token_logits = outputs.logits[:, -1, :]
                token_probs = F.softmax(next_token_logits, dim=-1)
            
            next_token_id_0 = output_ids[:, 0].unsqueeze(0)
            next_token_prob_0 = token_probs[:, next_token_id_0].item()      
            
            next_token_id_1 = output_ids[:, 1].unsqueeze(0)
            next_token_prob_1 = token_probs[:, next_token_id_1].item()  
            
            prob_subtract = next_token_prob_1 - next_token_prob_0
            return prob_subtract
            
        except Exception as e:
            logger.error(f"概率计算失败: {e}")
            return 0.0
    
    def _get_prob_subtract_api(self, sentence1: str, sentence2: str, language: str) -> float:
        """
        使用API计算边际采样的概率差值
        """
        try:
            if language == 'zh':
                prompt = f'''这是一个文本分块任务。你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：

1. 将"{sentence1+sentence2}"分割成"{sentence1}"与"{sentence2}"两部分；
2. 将"{sentence1+sentence2}"不进行分割，保持原形式；

请只回答数字1或2：'''
            else:
                prompt = f'''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:

1. Split "{sentence1+' '+sentence2}" into "{sentence1}" and "{sentence2}" two parts;
2. Keep "{sentence1+' '+sentence2}" unsplit in its original form;

Please answer only 1 or 2:'''
            
            # 使用二元选择方法
            prob_split = self.api_client.binary_choice(prompt, "1", "2")
            
            # 转换为概率差值（选择2的概率 - 选择1的概率）
            prob_subtract = (1 - prob_split) - prob_split
            return prob_subtract
            
        except Exception as e:
            logger.error(f"API概率计算失败: {e}")
            return 0.0
    
    def _extract_by_msp(self, text: str, language: str, chunk_length: int, confidence_threshold: float) -> List[str]:
        """
        MSP切分的内部实现
        
        结合置信度阈值和动态调整机制的高级边际采样切分
        """
        try:
            full_segments = split_text_by_punctuation(text, language)
            if len(full_segments) <= 1:
                return [text]
            
            tmp = ''
            threshold = 0
            threshold_list = []
            confidence_scores = []
            final_chunks = []
            
            for i, sentence in enumerate(full_segments):
                if tmp == '':
                    tmp += sentence
                    continue
                
                try:
                    # 计算边际概率
                    prob_subtract = self._get_prob_subtract(tmp, sentence, language)
                    threshold_list.append(prob_subtract)
                    
                    # 计算置信度分数（基于概率差值的绝对值）
                    confidence = abs(prob_subtract)
                    confidence_scores.append(confidence)
                    
                    # MSP决策逻辑：结合概率差值和置信度
                    should_split = False
                    
                    if confidence >= confidence_threshold:
                        # 高置信度情况下，使用概率差值决策
                        should_split = prob_subtract <= threshold
                    else:
                        # 低置信度情况下，倾向于不切分（保持语义连贯性）
                        should_split = prob_subtract < (threshold - 0.1)
                    
                    if should_split and tmp.strip():
                        final_chunks.append(tmp)
                        tmp = sentence
                    else:
                        separator = ' ' if language == 'en' else ''
                        tmp += separator + sentence
                    
                    # 动态阈值更新（考虑置信度权重）
                    if len(threshold_list) >= 5:
                        # 使用加权平均，高置信度的样本权重更大
                        recent_thresholds = threshold_list[-5:]
                        recent_confidences = confidence_scores[-5:]
                        
                        weighted_sum = sum(t * c for t, c in zip(recent_thresholds, recent_confidences))
                        weight_sum = sum(recent_confidences)
                        
                        if weight_sum > 0:
                            threshold = weighted_sum / weight_sum
                        else:
                            threshold = sum(recent_thresholds) / len(recent_thresholds)
                            
                except Exception as e:
                    logger.warning(f"MSP处理句子 {i} 时出错: {e}")
                    # 出错时默认合并
                    separator = ' ' if language == 'en' else ''
                    tmp += separator + sentence
            
            if tmp.strip():
                final_chunks.append(tmp)
            
            # 如果没有生成任何块，返回原文本
            if not final_chunks:
                return [text]
            
            # 长度控制和合并
            return self._merge_chunks_by_length(final_chunks, chunk_length, language)
            
        except Exception as e:
            logger.error(f"MSP分块内部实现失败: {e}")
            # 降级到边际采样分块
            return self._extract_by_margin_sampling(text, language, chunk_length)
    
    def _extract_by_semantic(self, text: str, similarity_threshold: float, 
                           min_chunk_size: int, max_chunk_size: int) -> List[str]:
        """
        语义切分的内部实现
        
        基于句子间语义相似度进行切分
        """
        # 简单的语义切分实现（不依赖复杂的语义模型）
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # 检查是否需要开始新的块
            should_start_new_chunk = False
            
            if current_length == 0:
                # 第一个句子
                current_chunk.append(sentence)
                current_length += sentence_length
            elif current_length + sentence_length > max_chunk_size:
                # 超过最大长度，必须切分
                should_start_new_chunk = True
            elif current_length >= min_chunk_size:
                # 达到最小长度，检查语义相似度
                semantic_similarity = self._calculate_semantic_similarity(
                    ' '.join(current_chunk), sentence
                )
                
                if semantic_similarity < similarity_threshold:
                    should_start_new_chunk = True
            
            if should_start_new_chunk and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子
        """
        # 简单的句子分割
        import re
        
        # 中文句子分割
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本片段的语义相似度
        
        这里使用简单的词汇重叠度作为语义相似度的近似
        """
        # 简单的词汇相似度计算
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _merge_chunks_by_length(self, chunks: List[str], max_length: int, language: str) -> List[str]:
        """
        根据长度限制合并文本块
        """
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if language == 'zh':
                chunk_len = len(chunk)
                current_len = len(current_chunk)
            else:
                chunk_len = len(chunk.split())
                current_len = len(current_chunk.split())
            
            if current_len == 0:
                current_chunk = chunk
            elif current_len + chunk_len <= max_length:
                separator = ' ' if language == 'en' else ''
                current_chunk += separator + chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _extract_by_ppl_api(self, text: str, threshold: float, language: str) -> List[str]:
        """
        使用API的PPL分块实现
        
        通过API调用大语言模型来判断文本的语义边界
        """
        # 句子分割
        segments = split_text_by_punctuation(text, language)
        segments = [item for item in segments if item.strip()]
        
        if len(segments) <= 1:
            return [text]
        
        # 使用API判断分割点
        split_points = [0]
        
        for i in range(1, len(segments)):
            # 构建前文和当前句子
            context = ''.join(segments[:i])
            current_sentence = segments[i]
            
            # 构建提示
            if language == 'zh':
                prompt = f"""请判断以下文本是否应该在指定位置分割：

前文：{context}
当前句子：{current_sentence}

如果在前文和当前句子之间分割能够保持语义完整性，请回答"1"；
如果不应该分割，请回答"2"。

请只回答数字1或2："""
            else:
                prompt = f"""Please determine if the following text should be split at the specified position:

Context: {context}
Current sentence: {current_sentence}

If splitting between the context and current sentence maintains semantic integrity, answer "1";
If it should not be split, answer "2".

Please answer only 1 or 2:"""
            
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.api_client.chat_completion(messages)
                
                if response and response.strip() == "1":
                    split_points.append(i)
                    
            except Exception as e:
                logger.warning(f"API调用失败，跳过分割点 {i}: {e}")
        
        split_points.append(len(segments))
        
        # 生成文本块
        final_chunks = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            chunk_sentences = segments[start_idx:end_idx]
            final_chunks.append(''.join(chunk_sentences))
        
        return final_chunks
    
    def _extract_by_margin_sampling_api(self, text: str, language: str, chunk_length: int) -> List[str]:
        """
        使用API的边际采样分块实现
        
        通过API调用来进行概率决策
        """
        full_segments = split_text_by_punctuation(text, language)
        if len(full_segments) <= 1:
            return [text]
        
        tmp = ''
        final_chunks = []
        threshold = 0.5  # 初始阈值
        threshold_history = []
        
        for sentence in full_segments:
            if tmp == '':
                tmp += sentence
            else:
                # 使用API获取分割概率
                prob_split = self._get_split_probability_api(tmp, sentence, language)
                threshold_history.append(prob_split)
                
                if prob_split > threshold:
                    # 分割
                    final_chunks.append(tmp)
                    tmp = sentence
                else:
                    # 不分割
                    separator = ' ' if language == 'en' else ''
                    tmp += separator + sentence
                
                # 动态调整阈值
                if len(threshold_history) >= 3:
                    recent_probs = threshold_history[-3:]
                    threshold = sum(recent_probs) / len(recent_probs)
        
        if tmp:
            final_chunks.append(tmp)
        
        # 长度控制
        return self._merge_chunks_by_length(final_chunks, chunk_length, language)
    
    def _get_split_probability_api(self, text1: str, text2: str, language: str) -> float:
        """
        使用API获取分割概率
        """
        try:
            if language == 'zh':
                prompt = f"""请判断以下两个文本片段是否应该分开处理：

文本1：{text1}
文本2：{text2}

如果这两个文本片段在语义上应该分开处理，请回答"分开"；
如果应该合并在一起，请回答"合并"。

请只回答"分开"或"合并"："""
                
                prob = self.api_client.binary_choice(prompt, "分开", "合并")
            else:
                prompt = f"""Please determine if the following two text segments should be processed separately:

Text 1: {text1}
Text 2: {text2}

If these two text segments should be processed separately semantically, answer "separate";
If they should be merged together, answer "merge".

Please answer only "separate" or "merge":"""
                
                prob = self.api_client.binary_choice(prompt, "separate", "merge")
            
            return prob
            
        except Exception as e:
            logger.warning(f"API概率获取失败: {e}")
            return 0.5  # 默认概率