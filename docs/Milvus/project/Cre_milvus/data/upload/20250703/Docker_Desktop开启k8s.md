# Docker_Desktop开启k8s

原文地址：[在 Docker Desktop 中启用 K8s 服务 - 墨墨墨墨小宇 - 博客园](https://www.cnblogs.com/danvic712/p/enable-k8s-in-docker-desktop.html)

### 开启k8s服务

打开docker的设置界面，选择Docker Engine，修改如下：

```json
{
  "debug": false,
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "insecure-registries": [],
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://registry.docker-cn.com"
  ]
}
```

镜像配置完后，去github下载https://github.com/AliyunContainerService/k8s-for-docker-desktop，找到里面的load_images文件，还有kubernetes-dashboard.yaml文件，首先执行load_images脚本文件，

然后去Docker_desktop中选择Kubeadm和show system containers，然后选择Apply&restart。

### 启动Dashboard

```shell
-- 创建 dashboard 资源
kubectl apply -f kubernetes-dashboard.yaml

-- 查看 Deployment 的运行状态 
kubectl get deployment -n kuberenetes-dashboard

-- 查看 Pod 的运行状态
kubectl get pods -n kuberenetes-dashboard

-- 通过代理的方式访问 dashboard
kubectl proxy
```

这里是使用 API Server 的形式访问的 dashboard ，具体的地址为：<http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/>

我们可以使用下面的 powershell 命令打印出 token，然后登录到 dashboard 中查看整个集群的信息

```
Copy$TOKEN=((kubectl -n kube-system describe secret default | Select-String "token:") -split " +")[1]
kubectl config set-credentials docker-for-desktop --token="${TOKEN}"
echo $TOKEN
```

如果执行失败，显示没有默认的secret：那么

```shell
# 列出所有 secret 查找正确的名称
kubectl -n kube-system get secrets

# 使用找到的名称替换 'default'
$TOKEN = kubectl -n kube-system describe secret <your-secret-name> | Select-String "token:" | ForEach-Object { ($_ -split '\s+')[1] }
# 验证 token
Write-Host "Token: $TOKEN"
```

如果还有问题！

```shell
# 使用一行命令获取 token
$TOKEN = kubectl -n kube-system create token default --duration=8760h
# 验证 token
Write-Host "Token: $TOKEN"
```

