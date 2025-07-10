# 向量数据库Milvus

>声明：该文档主要解决学习问题，具体技术细节请参考代码。

参考文章：

[Windows安装部署向量数据库（Milvus）_milvus windows-CSDN博客](https://blog.csdn.net/m0_54345753/article/details/136738293?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-136738293-blog-132702155.235^v43^pc_blog_bottom_relevance_base5&spm=1001.2101.3001.4242.1&utm_relevant_index=1)

在`docker compose up -d`这一步会遇到pulling不下来的问题，这时候需要修改配置文件：

## [Daemon configuration](https://docs.docker.com/config/daemon/proxy/#daemon-configuration)

You may configure proxy behavior for the daemon in the `daemon.json` file, or using CLI flags for the `--http-proxy` or `--https-proxy` flags for the `dockerd` command. Configuration using `daemon.json` is recommended.

```
{
  "proxies": {
    "http-proxy": "http://proxy.example.com:3128",
    "https-proxy": "https://proxy.example.com:3129",
    "no-proxy": "*.test.example.com,.example.org,127.0.0.0/8"
  }
}
```

After changing the configuration file, restart the daemon for the proxy configuration to take effect:

```
$ sudo systemctl restart docker
```

参考[Configure the daemon to use a proxy | Docker Docs](https://docs.docker.com/config/daemon/proxy/#httphttps-proxy)

然后手动重启docker desktop就可以了。

# Milvus学习

首先启动魔法，然后打开docker，在docker中启动milvus，

最后打开attu。

## 配置环境

- 需要 Python 3.7 或更高版本。

- 需要安装 Google protobuf。你可以使用命令 `pip3 install protobuf==3.20.0` 来安装。

- 需要安装 grpcio-tools。你可以使用命令 `pip3 install grpcio-tools` 来安装。

- 通过pip安装pyMilvus：`pip install pymilvus`

  > [pymilvus · PyPI](https://pypi.org/project/pymilvus/)

- 通过`python3 -c "from pymilvus import Collection`来验证PyMilvus安装正确，运行该指令时不会触发任何异常。


## Milvus基础使用

milvus基础使用参考[Index – Milvus向量库中文文档 (milvus-io.com)](https://www.milvus-io.com/)

# 数据处理

参考了DataWhale的[llm-universe/docs/C3/3.数据处理.md at main · Halukisan/llm-universe (github.com)](https://github.com/Halukisan/llm-universe/blob/main/docs/C3/3.%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.md)

我们将知识库源数据放置在../data_base/knowledge_db 目录下。

详细内容参考注释























