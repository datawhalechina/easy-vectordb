# 			关于fork项目同步问题



最近在GitHub上fork了别人的一个项目，是关大模型学习的一个项目，Git地址如下：https://github.com/Halukisan/llm-universe.git，但是如果在这个项目上改动了，比如加了一些知识点，我fork到我自己GitHub上的项目如何保持同步的更新呢？
其实只需要下面三步：

### 把fork的项目克隆到本地仓库中

下面操练一遍：

把fork的项目克隆到本地仓库中

```shell
git clone
```



后面这个地址是我fork项目在我GitHub的地址。这样完成之后，就会发现我的本地仓库中多了一个[llm-universe](https://github.com/Halukisan/llm-universe)的文件夹。说明你已经把仓库克隆到本地了。

### Configuring a remote for a fork

给 fork 配置一个 remote
主要使用 git remote -v查看远程状态。
克隆了fork项目到本地之后，进入那个项目文件夹，我就进入Data-Science-Notes的文件夹，然后输入命令

git remote -v

查看远程状态如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191124185048253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)

添加一个将被同步给 fork 远程的上游仓库

```shell
git remote add upstream https://github.com/datawhalechina/self-llm.git
```

这里的地址是你fork的项目的源地址。

再次查看状态确认是否配置成功。

```shell
git remote -v
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191124185449409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)

有了下面那两个upstream，说明第二步已经成功了。这就相当于有一个管道在源项目和你fork的项目之间建立了，下面看看如何通信更新。

### Syncing a fork

从上游仓库 fetch 分支和提交点，传送到本地，并会被存储在一个本地分支 upstream/master

```shell
git fetch upstream
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112418570384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)

切换到本地主分支

```shell
git checkout master
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191124185824132.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)

把 upstream/master 分支合并到本地 master 上，这样就完成了同步，并且不会丢掉本地修改的内容。

```
git merge upstream/master
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112418592036.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)

由于我是刚刚fork的，肯定是最新的项目了，所以提示Already up to date. 如果源项目有更新而你fork的项目没更新的话，这里就会显示不同了。
这样，就把源项目同步更新到你的本地仓库中了。 如果再想更新到远程仓库fork，只需要：

```shell
git push origin master
```




原文链接：https://blog.csdn.net/wuzhongqiang/article/details/103227170