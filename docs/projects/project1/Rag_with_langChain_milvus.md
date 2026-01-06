# Chapter1 RAG with LangChain & Milvus

打开modelscope CPU环境，

新建两个终端，
其中一个终端里面运行ollama serve启动ollama服务

在另外一个终端里面运行ollama --version查看ollama是否正确启动



**依赖安装**
```bash
! python -m venv venv
! ./venv/bin/pip install langchain pymilvus sentence-transformers ollama
```