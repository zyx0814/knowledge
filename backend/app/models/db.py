from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
Base = declarative_base()
class File(Base):
    """文件信息表"""
    __tablename__ = "files"
    __table_args__ = (
        Index('idx_files_gid_type_status', 'gid', 'type', 'status'),
        Index('idx_files_gid_imported', 'gid', 'imported_at'),
    )
    id = Column(String, primary_key=True, index=True)
    rid = Column(String, index=True)  # 外部系统对接依据
    gid = Column(String, index=True, nullable=True)  # 区域ID，用于区分不同区域（改为字符串类型）
    name = Column(String, index=True)
    type = Column(String)  # 文档/图片/视频
    subformat = Column(String)  # PDF/MP4/PNG等
    size = Column(Integer)
    storage_path = Column(String)
    imported_at = Column(DateTime(timezone=True), server_default=func.now())
    modified_at = Column(DateTime(timezone=True), onupdate=func.now())
    category = Column(String, index=True)
    is_parsed = Column(Boolean, default=False)
    is_vectorized = Column(Boolean, default=False)
    status = Column(String, default="pending")  # pending/processing/completed/failed
    # 关系
    faces = relationship("Face", backref="file", cascade="all, delete-orphan")
class TextChunk(Base):
    """文本分块表"""
    __tablename__ = "text_chunks"
    __table_args__ = (
        Index('idx_chunks_file_id_index', 'file_id', 'chunk_index'),
    )
    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"), index=True)
    content = Column(Text)
    chunk_index = Column(Integer)
    start_pos = Column(Integer)
    end_pos = Column(Integer)
    vector_id = Column(String, index=True)
class ImageFrame(Base):
    """图片帧信息表（用于视频抽帧）"""
    __tablename__ = "image_frames"
    __table_args__ = (
        Index('idx_frames_file_id', 'file_id'),
    )
    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"), index=True)
    frame_path = Column(String)
    timestamp = Column(Float)  # 视频时间戳（秒）
    vector_id = Column(String, index=True)
class AudioTranscript(Base):
    """音频转写表"""
    __tablename__ = "audio_transcripts"
    __table_args__ = (
        Index('idx_transcripts_file_id', 'file_id'),
    )
    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"), index=True)
    content = Column(Text)
    start_time = Column(Float)
    end_time = Column(Float)
    vector_id = Column(String, index=True)
class Vector(Base):
    """向量关联表"""
    __tablename__ = "vectors"
    __table_args__ = (
        Index('idx_vectors_file_id', 'file_id'),
    )
    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"), index=True)
    chunk_id = Column(String, index=True)
    vector_path = Column(String)
    vector_type = Column(String)  # text/image/audio
class SearchLog(Base):
    """检索日志表"""
    __tablename__ = "search_logs"
    __table_args__ = (
        Index('idx_search_logs_time', 'search_time'),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text)
    query_type = Column(String)  # text/image
    search_time = Column(DateTime(timezone=True), server_default=func.now())
    result_count = Column(Integer)
    execution_time = Column(Float)
class Face(Base):
    """人脸信息表（包含特征）"""
    __tablename__ = "faces"
    __table_args__ = (
        Index('idx_faces_file_id', 'file_id'),
        Index('idx_faces_group_id', 'group_id'),
    )
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True, index=True)  # 人脸名称，可由用户命名
    group_id = Column(String, nullable=True, index=True)  # 人脸组ID，用于合并相似人脸
    file_id = Column(String, ForeignKey("files.id"), index=True)  # 关联的文件ID
    image_path = Column(String)  # 人脸裁剪图路径
    embedding = Column(Text)  # 人脸特征向量，JSON格式存储
    confidence = Column(Float)  # 人脸检测置信度
    bbox = Column(Text)  # 人脸边界框，JSON格式存储
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
