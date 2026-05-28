# 本地知识库系统

本地知识库服务核心是**在私有环境（本地电脑/服务器）完成文件的存储、解析、检索、问答**，全程数据不上云，兼顾隐私性与实用性，核心适配图片、视频、文档三类非结构化数据。

## 系统架构

整体为**分层解耦架构**，各模块独立部署、可灵活替换，本地部署优先选择**开源工具+轻量框架**，避免重量级中间件，架构分层如下：

```
前端交互层（本地Web/桌面客户端）→ 应用服务层（检索引擎/问答接口/任务调度）→ 数据处理层（文件解析/特征提取/向量生成）→ 存储层（原始文件/结构化数据/向量库）
```

## 核心功能

1. **文件上传与管理**：支持文档、图片、视频的上传和管理，包括分类标签、重命名、删除等功能。

2. **文件解析**：
   - 文档：PDF、Word、Excel、Markdown、TXT
   - 图片：JPG、PNG、WEBP、BMP
   - 视频：MP4、AVI、MOV、FLV

3. **特征/向量生成**：
   - 文本：使用 CLIP 模型生成文本嵌入向量
   - 图片：提取视觉特征并生成多模态向量（使用 CLIP）
   - 视频：抽帧处理、音频转写（使用 Whisper），融合特征生成向量

4. **向量存储**：
   - 使用 FAISS 进行向量存储和检索（默认）
   - 支持 Qdrant 向量数据库（可选）
   - 支持快速相似度搜索

5. **文件存储**：
   - 本地存储（默认）
   - MinIO 对象存储（可选），支持自动降级到本地存储

6. **检索引擎**：
   - 关键词检索：基于结构化数据库的全文索引
   - 语义检索：基于向量相似度的检索，支持配置相似度阈值
   - 多模态检索：支持以图搜图、以图搜视频
   - 人脸检索：支持人脸检测和人脸搜索
   - 混合检索：融合关键词和语义检索结果

7. **问答接口**：基于 RAG 架构，使用本地 LLM 生成答案，提供答案溯源。

8. **前端界面**：使用 Vue3 + Element Plus 开发，提供友好的用户界面，包括文件管理、检索、问答等功能。

## 技术栈

- **后端**：FastAPI、SQLAlchemy、FAISS、Qdrant、Pillow、FFmpeg、Whisper、CLIP
- **前端**：Vue3、Element Plus、Axios
- **数据库**：SQLite（默认）、MySQL（可选）
- **对象存储**：本地文件系统、MinIO（可选）
- **部署**：Docker、Docker Compose

## 安装与部署

### 方法一：Docker Compose 部署（推荐）

1. **克隆项目**：
   ```bash
   git clone <repository-url>
   cd knowledge-base
   ```

2. **配置环境**：
   - 修改 `backend/config/config.py` 文件，根据需要配置数据库和其他参数
   - 配置 MinIO 连接信息（可选）
   - 配置 Qdrant 连接信息（可选）

3. **启动服务**：
   ```bash
   docker-compose up -d
   ```

4. **访问界面**：
   - 前端界面：http://localhost:3000
   - 后端 API 文档：http://localhost:8000/docs

### 方法二：手动部署

#### 后端部署

1. **安装依赖**：
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **配置数据库**：
   - 默认使用 SQLite，无需额外配置
   - 如需使用 MySQL,postgresql，修改 `backend/config/config.py` 中的数据库配置

3. **配置向量存储**（可选）：
   - 如需使用 Qdrant，设置 `USE_QDRANT: bool = True`

4. **配置对象存储**（可选）：
   - 如需使用 MinIO，配置 MinIO 连接信息

5. **启动后端服务**：
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

#### 前端部署

1. **安装依赖**：
   ```bash
   cd frontend
   npm install
   ```

2. **启动前端服务**：
   ```bash
   npm run dev
   ```

3. **访问界面**：
   前端界面：http://localhost:3000

## 配置选项

### 数据库配置

在 `backend/config/config.py` 中配置：

```python
# 数据库配置
DATABASE_TYPE: str = "sqlite"  # sqlite or mysql,postgresql
DATABASE_URL: str = "sqlite:///./knowledge_base.db"
# MySQL配置
MYSQL_HOST: str = "localhost"
MYSQL_PORT: int = 3306
MYSQL_USER: str = "root"
MYSQL_PASSWORD: str = "password"
MYSQL_DATABASE: str = "knowledge_base"
# PostgreSQL配置
POSTGRES_HOST: str = "localhost"
POSTGRES_PORT: int = 5432
POSTGRES_USER: str = "knowledge"
POSTGRES_PASSWORD: str = "knowledge"
POSTGRES_DATABASE: str = "knowledge_db"
```

### 文件处理配置

```python
# 文件处理配置
UPLOAD_DIR: str = "uploads"
TEMP_DIR: str = "temp"
MAX_FILE_SIZE: int = 10 * 1024 * 1024 * 1024  # 10GB

# 处理配置
MAX_WORKERS: int = 2
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# 视频处理配置
VIDEO_FRAME_INTERVAL: int = 2  # 每2秒抽帧
MAX_FRAMES_PER_VIDEO: int = 300
```

### 模型配置

```python
# 模型配置
OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_MODEL: str = "llama3"
EMBEDDING_DIM: int = 512  # CLIP ViT-B/32 模型输出维度
```

### 向量存储配置

```python
# 向量存储配置
USE_QDRANT: bool = False  # 是否使用 Qdrant
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
```

### 对象存储配置

```python
# 对象存储配置（MinIO）
USE_MINIO: bool = False
MINIO_ENDPOINT: str = "localhost:9000"
MINIO_ACCESS_KEY: str = "minioadmin"
MINIO_SECRET_KEY: str = "minioadmin"
MINIO_BUCKET: str = "knowledge-files"
```

### 搜索配置

```python
# 搜索配置
SEARCH_MIN_SCORE: float = 0.5  # 默认最小相似度阈值（语义搜索）
FACE_SEARCH_THRESHOLD: float = 0.6  # 人脸搜索相似度阈值
```

## 系统要求

### 硬件要求
- **CPU**：4核以上
- **内存**：8GB以上（推荐16GB+）
- **存储**：SSD，至少200GB（原始文件+向量存储）
- **GPU**：可选，加速向量生成和检索

### 软件要求
- **Python**：3.9+
- **Node.js**：18+
- **FFmpeg**：用于视频处理
- **Tesseract OCR**：用于图文识别
- **Ollama**：用于本地 LLM 部署
- **Qdrant**：用于向量数据库（可选）
- **MinIO**：用于对象存储（可选）

## 功能使用

### 1. 文件上传

在"文件管理"页面上传文档、图片、视频文件。系统会自动处理文件，生成向量并存储。

### 2. 文件管理

查看、分类、删除已上传的文件。删除文件时，系统会联动删除相关的结构化数据和向量。

### 3. 检索

在"检索"页面：
- 输入关键词进行文本检索
- 上传图片进行以图搜图/以图搜视频
- 上传图片进行人脸搜索
- 选择文件类型进行筛选
- 调整相似度阈值（通过配置文件或接口参数）

### 4. 问答

在"问答"页面输入问题，系统会基于知识库内容生成答案，并提供答案溯源。



## 注意事项

1. **视频处理**：视频处理可能需要较高的计算资源，建议根据硬件配置调整视频抽帧频率
2. **首次启动**：首次启动时会初始化数据库和向量库，可能需要一些时间
3. **本地 LLM**：本地 LLM 的性能和效果取决于模型大小和硬件配置
4. **数据备份**：定期备份向量库和数据库，防止数据丢失
5. **MySQL 配置**：使用 MySQL 时，需要先创建数据库并确保用户有足够的权限
6. **MinIO 配置**：使用 MinIO 时，确保 MinIO 服务正常运行且 bucket 已创建
7. **Qdrant 配置**：使用 Qdrant 时，确保 Qdrant 服务正常运行



## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request，共同改进系统功能。
