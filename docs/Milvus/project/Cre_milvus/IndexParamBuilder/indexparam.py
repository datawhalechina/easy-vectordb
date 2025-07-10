"""
索引构建，选择CPU还是GPU来构建索引
"""
def indexParamBuilder(Choice, IndexName, metric_type="IP", nlist=1024, M=16, efConstruction=200):
    """
    Choice: "cpu" 或 "gpu"
    IndexName: "IVF_FLAT"、"HNSW"等IP
    metric_type: "L2" 或 "IP"
    nlist/M/efConstruction: 可选参数，按索引类型自动使用
    """
    if IndexName.upper() == "HNSW":
        index_params = {
            "metric_type": metric_type,
            "index_type": "HNSW",
            "params": {
                "M": int(M),
                "efConstruction": int(efConstruction)
            }
        }
    elif IndexName.upper() == "IVF_FLAT":
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": int(nlist)
            }
        }
    elif Choice == "gpu":
        # 这里可以根据需要扩展GPU专用参数
        index_params = {
            "metric_type": metric_type,
            "index_type": IndexName,
            "params": {
                'intermediate_graph_degree': 64,
                'graph_degree': 32
            },
            "build_algo": "IVF_PQ",
            "cache_data_set_on_device": "true"
        }
    else:
        # 默认CPU IVF_FLAT
        index_params = {
            "metric_type": metric_type,
            "index_type": IndexName,
            "params": {
                "nlist": int(nlist)
            }
        }
    return index_params