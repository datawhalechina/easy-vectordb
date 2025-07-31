# DataBuilder模块介绍

1. data.py作为整个模块的主启动函数，控制不同类型数据的流向以及分块算法的选择
2. chunking块下包含着Meta-Chunking论文实现，内涵MSP以及PPL算法
3. tools块下包含通用的默认处理工具类，分别为txt pdf md csv(已舍弃) img格式数据处理