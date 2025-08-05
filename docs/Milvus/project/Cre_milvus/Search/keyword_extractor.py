"""
聚类关键词提取模块

提供从聚类文档中提取代表性关键词的功能
"""

import re
import math
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
import jieba
import jieba.analyse

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """关键词提取器"""
    
    def __init__(self):
        # 初始化jieba
        jieba.initialize()
        
        # 停用词列表
        self.stop_words = self._load_stop_words()
        
        # 标点符号和特殊字符
        self.punctuation = set('！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿！"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        # 基础中文停用词
        basic_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '这个', '那个', '这些', '那些', '这里', '那里', '这样', '那样', '什么', '怎么', '为什么', '哪里', '哪个', '哪些', '多少', '几个', '第一', '第二', '第三', '可以', '应该', '需要', '必须', '能够', '可能', '或者', '但是', '然后', '因为', '所以', '如果', '虽然', '虽说', '尽管', '不过', '而且', '并且', '以及', '还有', '包括', '比如', '例如', '等等', '之类', '方面', '问题', '情况', '时候', '地方', '方式', '方法', '内容', '结果', '原因', '目的', '意思', '作用', '影响', '关系', '差别', '不同', '相同', '类似', '一样', '差不多', '大概', '可能', '也许', '或许', '估计', '应该', '肯定', '一定', '必然', '当然', '显然', '明显', '清楚', '知道', '了解', '认为', '觉得', '感觉', '发现', '注意', '观察', '分析', '研究', '讨论', '考虑', '思考', '判断', '决定', '选择', '确定', '建议', '推荐', '提出', '提供', '给出', '得出', '产生', '形成', '创造', '建立', '设立', '成立', '开始', '进行', '实现', '完成', '结束', '停止', '继续', '保持', '维持', '改变', '变化', '发展', '增加', '减少', '提高', '降低', '改善', '改进', '优化', '加强', '减弱', '扩大', '缩小', '增长', '下降', '上升', '提升', '下滑', '波动', '稳定', '平稳', '快速', '缓慢', '突然', '逐渐', '立即', '马上', '很快', '不久', '最近', '目前', '现在', '以前', '过去', '将来', '未来', '今天', '昨天', '明天', '今年', '去年', '明年', '本月', '上月', '下月', '本周', '上周', '下周', '早上', '上午', '中午', '下午', '晚上', '夜里', '白天', '夜晚'
        }
        
        # 添加英文停用词
        english_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
        }
        
        return basic_stop_words.union(english_stop_words)
    
    def extract_keywords_from_cluster(self, documents: List[str], max_keywords: int = 5) -> List[str]:
        """
        从聚类文档中提取关键词
        
        Args:
            documents: 文档内容列表
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        if not documents:
            return []
        
        # 合并所有文档内容
        combined_text = ' '.join(documents)
        
        # 使用多种方法提取关键词
        tfidf_keywords = self._extract_tfidf_keywords(combined_text, max_keywords * 2)
        textrank_keywords = self._extract_textrank_keywords(combined_text, max_keywords * 2)
        frequency_keywords = self._extract_frequency_keywords(combined_text, max_keywords * 2)
        
        # 合并和排序关键词
        keyword_scores = defaultdict(float)
        
        # TF-IDF权重
        for i, keyword in enumerate(tfidf_keywords):
            keyword_scores[keyword] += (len(tfidf_keywords) - i) / len(tfidf_keywords) * 0.4
        
        # TextRank权重
        for i, keyword in enumerate(textrank_keywords):
            keyword_scores[keyword] += (len(textrank_keywords) - i) / len(textrank_keywords) * 0.4
        
        # 频率权重
        for i, keyword in enumerate(frequency_keywords):
            keyword_scores[keyword] += (len(frequency_keywords) - i) / len(frequency_keywords) * 0.2
        
        # 按分数排序并返回前N个
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [keyword for keyword, score in sorted_keywords[:max_keywords]]
    
    def _extract_tfidf_keywords(self, text: str, max_keywords: int) -> List[str]:
        """使用TF-IDF提取关键词"""
        try:
            # 使用jieba的TF-IDF
            keywords = jieba.analyse.extract_tags(
                text, 
                topK=max_keywords,
                withWeight=False,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an')
            )
            
            # 过滤停用词和短词
            filtered_keywords = []
            for keyword in keywords:
                if (len(keyword) >= 2 and 
                    keyword not in self.stop_words and 
                    not self._is_punctuation(keyword) and
                    not keyword.isdigit()):
                    filtered_keywords.append(keyword)
            
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"TF-IDF关键词提取失败: {e}")
            return []
    
    def _extract_textrank_keywords(self, text: str, max_keywords: int) -> List[str]:
        """使用TextRank提取关键词"""
        try:
            # 使用jieba的TextRank
            keywords = jieba.analyse.textrank(
                text,
                topK=max_keywords,
                withWeight=False,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an')
            )
            
            # 过滤停用词和短词
            filtered_keywords = []
            for keyword in keywords:
                if (len(keyword) >= 2 and 
                    keyword not in self.stop_words and 
                    not self._is_punctuation(keyword) and
                    not keyword.isdigit()):
                    filtered_keywords.append(keyword)
            
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"TextRank关键词提取失败: {e}")
            return []
    
    def _extract_frequency_keywords(self, text: str, max_keywords: int) -> List[str]:
        """使用词频提取关键词"""
        try:
            # 分词
            words = jieba.lcut(text)
            
            # 过滤和统计
            word_freq = Counter()
            for word in words:
                word = word.strip()
                if (len(word) >= 2 and 
                    word not in self.stop_words and 
                    not self._is_punctuation(word) and
                    not word.isdigit() and
                    self._is_meaningful_word(word)):
                    word_freq[word] += 1
            
            # 返回频率最高的词
            return [word for word, freq in word_freq.most_common(max_keywords)]
            
        except Exception as e:
            logger.error(f"频率关键词提取失败: {e}")
            return []
    
    def _is_punctuation(self, text: str) -> bool:
        """判断是否为标点符号"""
        return all(char in self.punctuation for char in text)
    
    def _is_meaningful_word(self, word: str) -> bool:
        """判断是否为有意义的词"""
        # 过滤纯数字、纯字母、纯符号
        if word.isdigit() or word.isalpha() and len(word) == 1:
            return False
        
        # 过滤常见的无意义词
        meaningless_patterns = [
            r'^[a-zA-Z]$',  # 单个字母
            r'^\d+$',       # 纯数字
            r'^[^\w\s]+$',  # 纯符号
            r'^(啊|呀|哦|嗯|哈|呵|嘿|喂|哎|唉)$',  # 语气词
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, word):
                return False
        
        return True
    
    def extract_keywords_with_context(self, documents: List[str], max_keywords: int = 5) -> List[Dict[str, Any]]:
        """
        提取关键词并包含上下文信息
        
        Args:
            documents: 文档内容列表
            max_keywords: 最大关键词数量
            
        Returns:
            包含关键词和上下文的字典列表
        """
        keywords = self.extract_keywords_from_cluster(documents, max_keywords)
        
        result = []
        for keyword in keywords:
            # 找到包含该关键词的文档和上下文
            contexts = []
            for doc in documents:
                if keyword in doc:
                    # 提取关键词周围的上下文
                    context = self._extract_context(doc, keyword)
                    if context:
                        contexts.append(context)
            
            result.append({
                "keyword": keyword,
                "frequency": sum(1 for doc in documents if keyword in doc),
                "contexts": contexts[:3]  # 最多3个上下文
            })
        
        return result
    
    def _extract_context(self, text: str, keyword: str, context_length: int = 50) -> str:
        """提取关键词的上下文"""
        try:
            # 找到关键词的位置
            index = text.find(keyword)
            if index == -1:
                return ""
            
            # 提取前后文本
            start = max(0, index - context_length)
            end = min(len(text), index + len(keyword) + context_length)
            
            context = text[start:end]
            
            # 如果不是从开头开始，添加省略号
            if start > 0:
                context = "..." + context
            
            # 如果不是到结尾，添加省略号
            if end < len(text):
                context = context + "..."
            
            return context
            
        except Exception as e:
            logger.error(f"提取上下文失败: {e}")
            return ""


def create_keyword_extractor() -> KeywordExtractor:
    """创建关键词提取器实例"""
    return KeywordExtractor()