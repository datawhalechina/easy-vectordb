## Introduction
为什么要做这个,对于RAG的检索数据和性能,不一定要按照宁可获取的数据多,也不能少的原则,而是要达到一个性能和数据量之间的平衡。通过聚类算法，将相似的数据聚在一起，从而提高检索质量。除此之外,也是想做一个方便的接口,只需要输入参数,即可获取一个高性能(我自己认为的,目前没有安排分布式....)的向量数据库方案。
### 基础功能
可快速接入项目的Milvus向量数据库方案，:done
提供CPU和GPU索引选择，:done
登录加密、:done
> 修改milvus配置文件、docker重启，
```shell
   ...
   common:
   ...
   security:
      authorizationEnabled: true
   ...
```
> 启用身份登录功能时，需要提供令牌（在连接逻辑处提供token），不然会报gRPC错误
```python
   client = MilvusClient(
      uri="http://localhost:19530",
      token="root:Milvus"
   )

   # 2. Create a user
   client.create_user(user_name="user_1", password="P@ssw0rd")
```

> 默认root用户登录后，可创建其他身份的用户
> 重置密码，或者超级用户
> 用户名必须以字母开头，32个字符以内，只能包含下划线、字母或数字
> 密码必须在8-64个字符之间，必须包含以下三种字符：大写字母、小写字母、数字和特殊符号。

数据聚类分块选择，done
内存副本以提高吞吐，:done
多模态数据（文本、文件、图片）自动化处理和存储。:done

## 优化
1. 结合基于k8s的日志监控
2. 对结果重排序，将相似的内容分到一起
3. 分布式？内存分片？
4. 文件处理多线程。done
5. 处理失败后重试机制
6. 数据质量评估并标记，将低质量的数据分为其他的collection中


### 关于重排序
有四种方案
1. 基于距离的重排序
   * 在二维平面中,每个聚类都有聚类中心向量,将每个聚类的结果按照与中心向量的距离进行重排序,优先返回距离比较近的结果
2. 基于聚类大小的重排序
   * 按照聚类的大小(每个聚类中包含的结果数量)从大到小排序,优先返回较大的聚类
3. 基于聚类中心的重排序
   * 计算每个聚类的中心点,按质心与查询向量的距离从小到大排序
4. 基于特定业务的重排序
   * 你自己怎么想的就怎么做

以下是根据您提供的内容整理后的完整指南，采用 Markdown 格式：

---

# 基于 Kubernetes 部署 Grafana + Loki 监控 Milvus 日志

## 1. 前置条件
- 已安装 Docker 并启用 Kubernetes
- 确保 kubectl 可正常使用

## 2. 创建监控命名空间
```bash
kubectl create namespace monitoring
```

## 3. 部署 Grafana

### 3.1 创建 ConfigMap
`grafana-config.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-config
  namespace: monitoring
data:
  grafana.ini: |
    [server]
    domain = localhost
    root_url = http://localhost:3000
    serve_from_sub_path = true

    [security]
    admin_user = admin
    admin_password = admin
```

应用配置：
```bash
kubectl apply -f grafana-config.yaml
```

### 3.2 创建 Deployment
`grafana-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-config
          mountPath: /etc/grafana/grafana.ini
          subPath: grafana.ini
      volumes:
      - name: grafana-storage
        emptyDir: {}
      - name: grafana-config
        configMap:
          name: grafana-config
```

应用 Deployment：
```bash
kubectl apply -f grafana-deployment.yaml
```

### 3.3 创建 Service
`grafana-service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  type: NodePort
  ports:
  - port: 3000
    targetPort: 3000
    nodePort: 30001
  selector:
    app: grafana
```

应用 Service：
```bash
kubectl apply -f grafana-service.yaml
```

### 3.4 验证部署
```bash
kubectl get pods -n monitoring
kubectl get svc -n monitoring
```

访问 Grafana：
```
http://<node-ip>:30001
```
使用 admin/admin 登录

### 3.5 持久化存储（可选）
`grafana-pvc.yaml`:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-pvc
  namespace: monitoring
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

更新 Deployment 的 volumes 部分：
```yaml
volumes:
- name: grafana-storage
  persistentVolumeClaim:
    claimName: grafana-pvc
```

## 4. 部署 Loki

### 4.1 创建 ConfigMap
`loki-config.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: monitoring
data:
  local-config.yaml: |
    auth_enabled: false
    server:
      http_listen_port: 3100
    ingester:
      lifecycler:
        address: 127.0.0.1
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1
        final_sleep: 0s
      chunk_idle_period: 5m
      max_chunk_age: 1h
      chunk_target_size: 1536000
      chunk_retain_period: 30s
    schema_config:
      configs:
      - from: 2020-10-24
        store: boltdb-shipper
        object_store: filesystem
        schema: v11
        index:
          prefix: index_
          period: 24h
    storage_config:
      boltdb_shipper:
        active_index_directory: /tmp/loki/boltdb-shipper-active
        cache_location: /tmp/loki/boltdb-shipper-cache
        cache_ttl: 24h
        shared_store: filesystem
      filesystem:
        directory: /tmp/loki/chunks
    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h
    chunk_store_config:
      max_look_back_period: 0s
    table_manager:
      retention_deletes_enabled: false
      retention_period: 0s
```

### 4.2 创建 Deployment
`loki-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      securityContext:
        runAsUser: 0  # 以 root 用户运行
      containers:
      - name: loki
        image: grafana/loki:latest
        args:
        - "-config.file=/etc/loki/local-config.yaml"
        ports:
        - containerPort: 3100
        volumeMounts:
        - name: loki-config
          mountPath: /etc/loki
        - name: loki-wal
          mountPath: /tmp/wal
      volumes:
      - name: loki-config
        configMap:
          name: loki-config
      - name: loki-wal
        emptyDir: {}
```

### 4.3 创建 Service
`loki-service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: monitoring
spec:
  type: NodePort
  ports:
  - port: 3100
    targetPort: 3100
    nodePort: 31090
  selector:
    app: loki
```

## 5. 部署 Promtail

### 5.1 创建 ConfigMap
`promtail-config.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: monitoring
data:
  promtail-local-config.yaml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0
    positions:
      filename: /tmp/positions.yaml
    clients:
      - url: http://loki:3100/loki/api/v1/push
    scrape_configs:
      - job_name: system
        static_configs:
          - targets:
              - localhost
            labels:
              job: varlogs
              __path__: /var/log/*.log
      - job_name: milvus-logs
        static_configs:
          - targets:
              - localhost
            labels:
              job: milvus
              __path__: /host/path/to/logs/*.log
```

### 5.2 创建 DaemonSet
`promtail-daemonset.yaml`:
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    metadata:
      labels:
        app: promtail
    spec:
      containers:
      - name: promtail
        image: grafana/promtail:latest
        args:
        - "-config.file=/etc/promtail/promtail-local-config.yaml"
        volumeMounts:
        - name: logs
          mountPath: /var/log
        - name: milvus-logs
          mountPath: /host/path/to/logs
        - name: promtail-config
          mountPath: /etc/promtail
      volumes:
      - name: logs
        hostPath:
          path: /var/log
      - name: milvus-logs
        hostPath:
          path: /host/path/to/logs
      - name: promtail-config
        configMap:
          name: promtail-config
```

## 6. 配置 Milvus 日志收集

1. 确保 Milvus 容器日志挂载到宿主机：
```bash
docker run -d \
  --name milvus \
  -v /host/path/to/logs:/var/lib/milvus/logs \
  milvusdb/milvus:latest
```

2. 验证日志文件存在：
```bash
ls /host/path/to/logs
```

## 7. 在 Grafana 中添加 Loki 数据源

1. 访问 Grafana (http://<node-ip>:30001)
2. 导航到 Configuration > Data Sources
3. 添加 Loki 数据源：
   - URL: `http://loki:3100`
4. 保存并测试连接

## 8. 查询日志

在 Grafana Explore 页面：
- 选择 Loki 数据源
- 输入查询：`{job="milvus"}`

## 9. 故障排除

### 9.1 Pod 卡在 ContainerCreating 状态
检查事件日志：
```bash
kubectl describe pod <pod-name> -n monitoring
```

常见问题：
1. 镜像拉取失败：检查镜像名称和网络
2. 挂载卷失败：检查 PVC/PV 配置
3. 资源不足：检查节点资源

### 9.2 Loki 权限问题
解决方案：
1. 以 root 用户运行：
```yaml
securityContext:
  runAsUser: 0
```
2. 修改日志目录权限：
```bash
sudo chmod -R 777 /var/lib/loki/wal
```
3. 使用 emptyDir 临时存储

---
