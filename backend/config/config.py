import os
from pydantic_settings import BaseSettings
from typing import Optional, Dict

class Settings(BaseSettings):
    """系统配置类"""
    # 基础配置
    PROJECT_NAME: str = "Local Knowledge Base"
    API_V1_STR: str = "/api"
    # API认证配置
    API_KEY: str = "123456"
    ENABLE_AUTH: bool = True
    # 数据库配置
    DATABASE_TYPE: str = "postgresql"  # sqlite/mysql/postgresql
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
    # 连接池配置
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    # Qdrant 配置
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    # 向量数据库选择
    USE_QDRANT: bool = True
    # MinIO 配置
    MINIO_HOST: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    USE_MINIO: bool = True
    # GPU 配置
    USE_GPU: bool = True  # 是否启用 GPU 加速
    # 文件处理配置
    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "uploads"
    TEMP_DIR: str = "temp"
    # 人脸模型配置
    FACE_MODEL_PATH: str = "models/insightface"  # 模型根目录，FaceAnalysis会在其中查找models子目录
    FACE_MODEL_NAME: str = "buffalo_l"
    # 通用模型目录
    MODELS_DIR: str = "models"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024 * 1024  # 10GB
    # 向量库
    VECTOR_DB_PATH: str = "vector_db"
    EMBEDDING_DIM: int = 512  # 默认，由 _resolve_embedding_dim() 自动覆盖

    # 模型输出维度映射（仅多模态模型）
    MODEL_DIMENSIONS: Dict[str, int] = {
        # OpenAI CLIP
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
        # Chinese-CLIP
        "OFA-Sys/chinese-clip-vit-base-patch16": 512,
        "OFA-Sys/chinese-clip-vit-large-patch14": 768,
        "OFA-Sys/chinese-clip-vit-huge-patch14": 1024,
        # Jina-CLIP-v2（推荐：多语言多模态最强）
        "jinaai/jina-clip-v2": 1024,
        # Qwen3-VL-Embedding
        "Qwen/Qwen3-VL-Embedding-2B": 2048,
        "Qwen/Qwen3-VL-Embedding-8B": 4096,
        # WeCLIP
        "alibaba-nlp/weclip-base": 512,
        "alibaba-nlp/weclip-large": 768,
    }

    # LLM 配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"

    # 嵌入模型（所有模型均为多模态，文本和图像在同一向量空间，天然支持以文搜图+以图搜图）
    # 推荐: jinaai/jina-clip-v2（1024维，中英文多模态最强）
    # 轻量: OFA-Sys/chinese-clip-vit-base-patch16（512维，中文多模态）
    # 极致: Qwen/Qwen3-VL-Embedding-8B（4096维，需GPU 12GB+）
    EMBEDDING_MODEL: str = "jinaai/jina-clip-v2"  # 改为 jinaai/jina-clip-v2 即可升级

    # 处理配置
    MAX_WORKERS: int = 2
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    # 视频处理配置
    VIDEO_FRAME_INTERVAL: int = 2  # 每2秒抽帧
    MAX_FRAMES_PER_VIDEO: int = 300
    # 文件处理开关（性能优化）
    ENABLE_OCR: bool = True  # 图片OCR文字识别
    ENABLE_FACE_DETECTION: bool = True  # 人脸检测
    ENABLE_VIDEO_AUDIO: bool = True  # 视频音频转写
    # 文件存储配置
    KEEP_FILES: bool = True  # 是否保留原始文件（用于预览）
    PERMANENT_STORAGE_DIR: str = "storage"  # 永久存储目录
    # 搜索配置
    SEARCH_MIN_SCORE: float = 0.5  # 默认最小相似度阈值（语义搜索）
    FACE_SEARCH_THRESHOLD: float = 0.6  # 人脸搜索相似度阈值
    class Config:
        env_file = ".env"
        case_sensitive = True

    def _resolve_embedding_dim(self) -> int:
        """自动解析当前模型的输出维度"""
        return self.MODEL_DIMENSIONS.get(self.EMBEDDING_MODEL, 512)


settings = Settings()