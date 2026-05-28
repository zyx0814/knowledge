import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from config.config import settings
from app.models.db import Base
# 根据配置创建数据库引擎
if settings.DATABASE_TYPE == "mysql":
    # MySQL连接URL
    mysql_url = f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
    engine = create_engine(
        mysql_url,
        poolclass=QueuePool,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_recycle=settings.DB_POOL_RECYCLE,
        echo=False
    )
elif settings.DATABASE_TYPE == "postgresql":
    # PostgreSQL连接URL
    postgres_url = f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DATABASE}"
    engine = create_engine(
        postgres_url,
        poolclass=QueuePool,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_recycle=settings.DB_POOL_RECYCLE,
        echo=False
    )
else:
    # SQLite连接URL
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def _wait_for_database(max_retries: int = 30, retry_interval: int = 2):
    """等待数据库连接就绪"""
    if settings.DATABASE_TYPE == "sqlite":
        return True
    for attempt in range(max_retries):
        try:
            # 尝试连接数据库
            with engine.connect():
                print(f"数据库连接成功 (尝试 {attempt + 1}/{max_retries})")
                return True
        except Exception as e:
            print(f"数据库连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
    
    print("数据库连接超时，无法启动服务")
    return False
def init_db():
    """初始化数据库表结构（带重试机制）"""
    print("等待数据库连接...")
    
    if not _wait_for_database():
        raise RuntimeError("无法连接到数据库")
    
    print("初始化数据库表结构...")
    Base.metadata.create_all(bind=engine)
    print("数据库初始化完成")