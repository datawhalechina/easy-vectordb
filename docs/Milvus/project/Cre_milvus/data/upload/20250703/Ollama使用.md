# Ollama使用

## 什么是Ollama

## 下载

## 配置模型下载地址

首先，把启动着的ollama关闭，然后在用户环境变量中点击新建环境变量

![环境变量配置](https://gitee.com/Liuxiaomj/ollama/raw/master/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20240713001853.png)

OLLAMA_MODELS的值为你希望模型所在的地址。

OLLAMA_ORIGINS = "*"

设置后需要**重启ollama服务和vscode**

> ollama版本需要>0.2.0
>
> 你可以通过在终端中输入`ollama -v`来检查你的版本

启动ollama服务`ollama serve`

再打开一个新的终端，在终端输入`ollama run codegeex4`

在本地模式中配置接口地址

在插件的配置页配置模型地址`http://localhost:11434/v1/chat/completions`

打开模型高级模式，在模型名称栏中填写：`codegeex4`



## 提高ollama下载模型速度

## 可能存在的问题

1. `Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address`

   windows中默认开机启动了ollama，首先退出ollama，然后打开任务管理器，*启动应用*中 禁用掉ollama，并在进程中结束掉ollama的任务,参考[Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address-CSDN博客](https://blog.csdn.net/MurphyStar/article/details/138966626)

2. ​

## Reference

[‍‍﻿‬‍⁠‬‬﻿﻿⁠‬⁠‬‌﻿‌‬‍﻿‬‍‌‬⁠‍‍‌‬‍CodeGeeX 本地模式使用指南 - 飞书云文档 (feishu.cn)](https://zhipu-ai.feishu.cn/wiki/DAtfwkaqniX9erkxvIScCkSonKh)

[THUDM/GLM-4: GLM-4 series: Open Multilingual Multimodal Chat LMs | 开源多语言多模态对话模型 (github.com)](https://github.com/THUDM/GLM-4)

[glm4 (ollama.com)](https://ollama.com/library/glm4)

[常见问题解答 - Ollama中文网](https://ollama.fan/resources/faq/#how-do-i-configure-ollama-server)

[CodeGeeX4/README_zh.md at main · THUDM/CodeGeeX4 (github.com)](https://github.com/THUDM/CodeGeeX4/blob/main/README_zh.md)

[codegeex4 (ollama.com)](https://ollama.com/library/codegeex4)

[CodeGeeX4/guides/Local_mode_guideline_zh.md at main · THUDM/CodeGeeX4 (github.com)](https://github.com/THUDM/CodeGeeX4/blob/main/guides/Local_mode_guideline_zh.md)

[Ollama教程——入门：开启本地大型语言模型开发之旅_ollama中文说明书-CSDN博客](https://blog.csdn.net/walkskyer/article/details/137255596)