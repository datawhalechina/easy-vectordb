"""
Meta-chunking实现模块

集成PPL困惑度分块和边际采样分块策略
"""

try:
    from .perplexity_chunking import Chunking
except ImportError:
    from perplexity_chunking import Chunking
from typing import List, Dict
import re
import math 
import torch
import torch.nn.functional as F

try:
    from nltk.tokenize import sent_tokenize
    import jieba 
except ImportError:
    print("Warning: NLTK or jieba not installed. Some chunking strategies may not work.")
    sent_tokenize = None
    jieba = None


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
    
    def __init__(self, model=None, tokenizer=None):
        """
        初始化Meta-chunking
        
        参数:
            model: 语言模型（用于PPL分块）
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        if model and tokenizer:
            self.chunking = Chunking(model, tokenizer)
    
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
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer are required for PPL chunking")
        
        return self._extract_by_ppl(text, threshold, language)
    
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
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer are required for margin sampling chunking")
        
        return self._extract_by_margin_sampling(text, language, chunk_length)
    
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
            # 降级到传统方法
            return self.traditional_chunking(text, chunk_length, chunk_length // 10)
        
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
            tokenized_text = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
            input_id = tokenized_text["input_ids"].to(self.model.device)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            len_sentences.append(input_id.shape[1])
            attention_mask_tmp = tokenized_text["attention_mask"].to(self.model.device)
            attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)

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
            if i == 0:
                first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
                index += len_sentences[i] - 1
            else:
                first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
                index += len_sentences[i]
        
        # 寻找分割点
        minima_indices = find_minima(first_cluster_ppl, threshold)
        
        # 生成文本块
        split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
        final_chunks = []
        
        for i in range(len(split_points)-1):
            chunk_sentences = []
            if i == 0:
                chunk_sentences.append(segments[0])
            
            for sp_index in range(split_points[i]+1, split_points[i+1]+1):
                chunk_sentences.append(segments[sp_index])
            
            final_chunks.append(''.join(chunk_sentences))
        
        return final_chunks
    
    def _extract_by_margin_sampling(self, text: str, language: str, chunk_length: int) -> List[str]:
        """
        边际采样分块的内部实现
        """
        full_segments = split_text_by_punctuation(text, language)
        tmp = ''
        threshold = 0
        threshold_list = []
        final_chunks = []
        
        for sentence in full_segments:
            if tmp == '':
                tmp += sentence
            else:
                prob_subtract = self._get_prob_subtract(tmp, sentence, language)    
                threshold_list.append(prob_subtract)
                
                if prob_subtract > threshold:
                    tmp += ' ' + sentence
                else:
                    final_chunks.append(tmp)
                    tmp = sentence
                    
            # 动态阈值更新
            if len(threshold_list) >= 5:
                last_five = threshold_list[-5:]  
                avg = sum(last_five) / len(last_five)
                threshold = avg
                
        if tmp != '':
            final_chunks.append(tmp)
            
        # 合并块以遵守长度限制
        merged_paragraphs = []
        current_paragraph = ""  
        
        if language == 'zh':
            for paragraph in final_chunks:  
                if len(current_paragraph) + len(paragraph) <= chunk_length:  
                    current_paragraph += paragraph  
                else:  
                    merged_paragraphs.append(current_paragraph)  
                    current_paragraph = paragraph    
        else:
            for paragraph in final_chunks:  
                if len(current_paragraph.split()) + len(paragraph.split()) <= chunk_length:  
                    current_paragraph += ' ' + paragraph  
                else:  
                    merged_paragraphs.append(current_paragraph)   
                    current_paragraph = paragraph 
                
        if current_paragraph:  
            merged_paragraphs.append(current_paragraph) 
            
        return merged_paragraphs
    
    def _get_prob_subtract(self, sentence1: str, sentence2: str, language: str) -> float:
        """
        计算边际采样的概率差值
        """
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
    
    def _extract_by_msp(self, text: str, language: str, chunk_length: int, confidence_threshold: float) -> List[str]:
        """
        MSP切分的内部实现
        
        结合置信度阈值和动态调整机制的高级边际采样切分
        """
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
            
            if should_split:
                final_chunks.append(tmp)
                tmp = sentence
            else:
                tmp += (' ' if language == 'en' else '') + sentence
            
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
        
        if tmp:
            final_chunks.append(tmp)
        
        # 长度控制和合并
        return self._merge_chunks_by_length(final_chunks, chunk_length, language)
    
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