# Docker 环境使用指南

## 📋 问题说明

后端在 Docker 容器中运行时，**无法直接访问宿主机的其他驱动器（如 G 盘）**。

## ✅ 解决方案

### 方案一：使用项目内的 data/import 目录（推荐）

#### 步骤 1：复制文件到项目目录

将您要导入的文件从 `G:\zxz\20150404` 复制到项目的 `data/import` 目录：

```
Knowledge/
├── data/
│   └── import/          # 将文件放在这里
│       ├── IMG_20150404_164927.jpg
│       ├── IMG_20150404_164934.jpg
│       └── ...
```

#### 步骤 2：重启 Docker 容器

如果容器正在运行，需要重启以加载新的 volume 配置：

```bash
docker-compose down
docker-compose up -d
```

或者如果是首次启动：

```bash
docker-compose up -d
```

#### 步骤 3：批量导入

在前端批量导入页面，输入路径：

```
/data/import
```

（注意：这是容器内的路径，不是宿主机路径！）

---

### 方案二：直接挂载 G 盘（仅限 Windows Docker Desktop）

如果您使用的是 Windows Docker Desktop，可以直接挂载 G 盘：

#### 修改 docker-compose.yml：

```yaml
services:
  knowledge-backend:
    volumes:
      - ./backend:/app
      # ... 其他挂载
      - G:/zxz:/data/gzxz  # 挂载 G 盘
```

然后重启容器，在前端输入路径：`/data/gzxz/20150404`

---

## 📂 目录结构说明

项目现在的目录结构：

```
Knowledge/
├── backend/              # 后端代码
├── frontend/             # 前端代码
├── data/                 # 数据目录（新增）
│   ├── import/          # 待导入文件放这里
│   ├── uploads/         # 上传的文件
│   ├── temp/            # 临时文件
│   └── vector_db/       # 向量数据库
├── docker-compose.yml   # 已更新
└── ...
```

---

## 🔧 操作步骤完整流程

### 1. 准备文件

将您的 56 张图片复制到：
```
f:\oaooanew\Knowledge\data\import\
```

### 2. 重启 Docker 服务

```bash
# 停止并删除旧容器
docker-compose down

# 启动新容器
docker-compose up -d

# 查看日志确认启动成功
docker-compose logs -f knowledge-backend
```

### 3. 访问前端

打开浏览器访问：http://localhost:3000

### 4. 批量导入

- 进入"批量导入"页面
- 输入目录路径：`/data/import`
- 点击"开始导入"

### 5. 测试功能

导入完成后，就可以测试：
- 语义搜索
- 以图搜图
- 人脸搜索

---

## 📊 验证步骤

### 检查容器是否正确挂载

```bash
docker-compose exec knowledge-backend ls -la /data/import
```

应该能看到您的图片文件。

### 查看后端日志

```bash
docker-compose logs -f knowledge-backend
```

可以看到批量导入时的详细日志。

---

## ⚠️ 注意事项

1. **文件路径**：在前端输入的是**容器内的路径**，不是宿主机路径
   - ✅ 正确：`/data/import`
   - ❌ 错误：`G:\zxz\20150404`

2. **权限问题**：确保 `data` 目录有读写权限

3. **重启容器**：修改 `docker-compose.yml` 后必须重启容器

4. **数据持久化**：`data` 目录下的内容会持久化，删除容器不会丢失

---

## 🚀 快速开始命令

```bash
# 1. 将文件复制到 data/import
# 手动复制 G:\zxz\20150404\*.jpg 到 data\import\

# 2. 重启容器
cd f:\oaooanew\Knowledge
docker-compose down
docker-compose up -d

# 3. 查看后端日志
docker-compose logs -f knowledge-backend

# 4. 打开浏览器
# 访问 http://localhost:3000
# 批量导入路径: /data/import
```
