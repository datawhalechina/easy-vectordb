from transformers import AutoTokenizer, AutoModel
import torch
import jieba.analyse

class KeywordExtractor:
    """先用jieba初筛，再用BERT精细化处理的关键词提取器"""
    def __init__(self, model_name="bert-base-chinese", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_keywords(self, text, top_k=5, jieba_k=15):
        # 1. 先用jieba初筛，获得候选关键词
        jieba_keywords = jieba.analyse.extract_tags(text, topK=jieba_k)
        if not jieba_keywords:
            return []

        # 2. 用BERT对候选关键词做精细化排序
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            attn_mean = attn.mean(dim=0)[0, 1:]  # [CLS]对每个token的注意力
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]

        # 计算每个jieba关键词在token中的最大注意力得分
        keyword_scores = []
        for word in jieba_keywords:
            # 找到所有与word匹配的token索引
            indices = [i for i, t in enumerate(tokens) if word in t or t in word]
            if indices:
                score = max([attn_mean[i].item() for i in indices])
                keyword_scores.append((word, score))
            else:
                keyword_scores.append((word, 0))

        # 按得分排序，取top_k
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        keywords = [w for w, s in keyword_scores[:top_k]]
        return keywords