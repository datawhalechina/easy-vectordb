### mysql部署

```dockerfile
docker run  -d 		//后台运行
-v mysql data:/var/lib/mysql   //数据卷挂载，持久化存储，防止重启导致的数据丢失
-p 3306:3306		//开放3306端口号
-e MYSQL_ROOT_PASSWORD=root		//设置root的密码
--restart always	//停止后自动重启
--name mysql	//将该容器命名为blog_mysql
mysql:5.7		//使用的镜像是mysql5.7
```

```
docker ps //查看容器
```



### redis部署

AOF是redis数据持久化：redis是将自己的数据库状态存储到内存里面，一旦服务器退出，那redis数据也没有了。可以设置AOF将redis在内存中的数据库状态保存到磁盘里面。

```dockerfile
//首先要开启AOF
redis-server --appendonly yes
```

```dockerfile
docker run -d  -v redis_data:/data -p 6379:6379 --restart always --name redis redis:7.0 redis-server --appendonly yes
```

### redis的测试

进入容器内部去执行命令`docker exec`

使用交互式的方式`docker exec -it`

指定去哪一个容器,可以用容器id或者容器名称`docker exec -it blog_redis bash`



输入`redis-cli`进入容器内使用命令行

输入`keys *`看看容器里面有哪些数据

`set a 1`手动添加一些数据

`get a` 取出这个key对应的值

`docker volume inspect redis_data`查看容器详情

### 后端应用部署

项目配置文件里面，把redis和mysql的地址和端口修改为服务器的地址和docker容器的端口，

```dockerfile
docker run -p 7777:7777 -d -v /usr/blog:/usr/blog --restart always --name blog_back java:openjdk-8u111 java -jar /usr/blog/xxxx.jar
```

### docker 的网络

首先创建网络`docker network blog_net`

把容器加入对应的网络`docker network connect blog_net blog_mysql`

`docker network connect blog_net blog_redis`

这样就把redis和mysql都加入到了这个网络

可以用`docker inspect blog_net`来查看是否加入进去了

在创建容器mysql和reids的时候，可以使用`--network blog_net`将该容器直接加入到此网络中，这样后端去尝试访问redis或者MySQL的时候，可以在后端配置文件的mysql、redis的地址除直接写mysql、redis。这俩单词就直接代表了他们的地址。

### 前端部署

`npm run build`先打包~捏~(￣▽￣)~*

打包后的资源放到宿主机的目录下面

```dockerfile
docker run -p 80:80 -v /usr/blog/blog-vue/dist:/usr/share/nginx/html --restart always
-d --name blog_vue
 nginx:1.21.5
```

ps:前端应该访问的后端地址应该是宿主机的地址！在app.js里面修改需要请求的地址















