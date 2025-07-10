from Search.ES.keyword_extractor import KeywordExtractor
from System.monitor import log_event


def es_search(es_client, question, topK, es_index="your_index"):
    keyword_extractor = KeywordExtractor()
    keywords = keyword_extractor.extract_keywords(question)
    try:
        resp = es_client.search(
            index=es_index,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"keywords": keywords}},
                            {"match": {"content": question}}
                        ]
                    }
                },
                "size": topK
            }
        )
        # 返回一个列表，列表中的每个元素都是一个字典，字典中包含id、content、embedding、keywords和distance等字段
        return [
            {
                "id": hit["_id"],
                "content": hit["_source"].get("content", ""),
                "embedding": hit["_source"].get("embedding", []),
                "keywords": hit["_source"].get("keywords", []),
                "distance": hit["_score"]
            }
            # 这里的hit是ES返回的每个文档，_id是文档的唯一标识符，_source是文档的内容，_score是匹配得分
            for hit in resp["hits"]["hits"]
        ]
    except Exception as e:
        log_event(f"ES检索异常: {e}")
        return []