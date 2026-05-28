import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.models.db import File
from config.config import settings
# 导入 MinIO 存储服务
try:
    from app.services.storage_service import storage_service
    STORAGE_SERVICE_AVAILABLE = True
except ImportError:
    STORAGE_SERVICE_AVAILABLE = False
class FileService:
    """文件服务类"""
    @staticmethod
    def get_permanent_storage_path(file_id: str, file_name: str) -> str:
        """获取永久存储路径（使用两级哈希目录分散文件）"""
        ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
        level1 = file_id[:2] if len(file_id) >= 2 else '00'
        level2 = file_id[2:4] if len(file_id) >= 4 else '00'
        storage_dir = os.path.join(settings.PERMANENT_STORAGE_DIR, level1, level2)
        os.makedirs(storage_dir, exist_ok=True)
        safe_name = f"{file_id}.{ext}" if ext else file_id
        return os.path.join(storage_dir, safe_name)
    @staticmethod
    def save_to_permanent_storage(temp_file_path: str, file_id: str, file_name: str) -> str:
        """将临时文件保存到永久存储"""
        if not settings.KEEP_FILES:
            return ""
        # 优先使用 MinIO 存储（如果可用）
        if STORAGE_SERVICE_AVAILABLE and storage_service.is_available():
            try:
                object_name = storage_service.upload_file(temp_file_path, file_id, file_name)
                if object_name:
                    # 返回 MinIO 对象路径
                    return f"minio://{storage_service.bucket}/{object_name}"
            except Exception:
                pass
        # 降级到本地存储
        permanent_path = FileService.get_permanent_storage_path(file_id, file_name)
        try:
            shutil.copy2(temp_file_path, permanent_path)
            return permanent_path
        except Exception:
            return ""
    @staticmethod
    def create_file(db: Session, rid: str, gid: str, file_name: str, file_type: str, subformat: str, file_size: int) -> File:
        """创建文件记录"""
        file_id = str(uuid.uuid4())
        file = File(
            id=file_id,
            rid=rid,
            gid=gid,
            name=file_name,
            type=file_type,
            subformat=subformat,
            size=file_size,
            storage_path="",
            category="",
            is_parsed=False,
            is_vectorized=False,
            status="pending"
        )
        db.add(file)
        db.commit()
        db.refresh(file)
        return file
    @staticmethod
    def get_file_by_id(db: Session, file_id: str) -> Optional[File]:
        """根据ID获取文件"""
        return db.query(File).filter(File.id == file_id).first()
    @staticmethod
    def get_file_by_rid(db: Session, rid: str) -> Optional[File]:
        """根据RID获取文件"""
        return db.query(File).filter(File.rid == rid).first()
    @staticmethod
    def get_files(db: Session, skip: int = 0, limit: int = 100) -> List[File]:
        """获取文件列表"""
        return db.query(File).offset(skip).limit(limit).all()
    @staticmethod
    def update_file_status(db: Session, file_id: str, status: str) -> Optional[File]:
        """更新文件状态"""
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            file.status = status
            if status == "completed":
                file.is_parsed = True
                file.is_vectorized = True
            db.commit()
            db.refresh(file)
        return file
    @staticmethod
    def update_file_category(db: Session, file_id: str, category: str) -> Optional[File]:
        """更新文件分类"""
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            file.category = category
            db.commit()
            db.refresh(file)
        return file
    @staticmethod
    def delete_file(db: Session, file_id: str) -> bool:
        """删除文件"""
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            from app.models.db import TextChunk, ImageFrame, AudioTranscript, Face
            # 删除关联的文本分块
            db.query(TextChunk).filter(TextChunk.file_id == file_id).delete()
            # 删除关联的图片帧
            image_frames = db.query(ImageFrame).filter(ImageFrame.file_id == file_id).all()
            for frame in image_frames:
                if frame.frame_path:
                    try:
                        if frame.frame_path.startswith("minio://"):
                            # 从 MinIO 删除
                            object_name = frame.frame_path.replace("minio://", "").split("/", 1)[-1]
                            storage_service.delete_file(object_name)
                        elif os.path.exists(frame.frame_path):
                            os.remove(frame.frame_path)
                    except Exception:
                        pass
            db.query(ImageFrame).filter(ImageFrame.file_id == file_id).delete()
            # 删除关联的音频转写
            db.query(AudioTranscript).filter(AudioTranscript.file_id == file_id).delete()
            # 删除关联的人脸
            faces = db.query(Face).filter(Face.file_id == file_id).all()
            for face in faces:
                if face.image_path:
                    try:
                        if face.image_path.startswith("minio://"):
                            # 从 MinIO 删除
                            object_name = face.image_path.replace("minio://", "").split("/", 1)[-1]
                            storage_service.delete_file(object_name)
                        elif os.path.exists(face.image_path):
                            os.remove(face.image_path)
                    except Exception:
                        pass
            db.query(Face).filter(Face.file_id == file_id).delete()
            # 删除向量
            from app.services.vector_service import VectorService
            vector_service = VectorService()
            vector_service.backend.delete_by_file_id(file_id)
            # 删除文件记录
            db.delete(file)
            db.commit()
            return True
        return False
    @staticmethod
    def get_file_details(db: Session, file_id: str) -> Dict[str, Any]:
        """获取文件详细信息（人脸和特征合并后）"""
        from app.models.db import TextChunk, ImageFrame, AudioTranscript, Face
        import json
        details = {
            "text_chunks": [],
            "image_frames": [],
            "audio_transcripts": [],
            "faces": []
        }
        try:
            # 获取文本分块
            text_chunks = db.query(TextChunk).filter(TextChunk.file_id == file_id).all()
            for chunk in text_chunks:
                details["text_chunks"].append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos
                })
            # 获取图片帧
            image_frames = db.query(ImageFrame).filter(ImageFrame.file_id == file_id).all()
            for frame in image_frames:
                details["image_frames"].append({
                    "id": frame.id,
                    "frame_path": frame.frame_path,
                    "timestamp": frame.timestamp
                })
            # 获取音频转写
            audio_transcripts = db.query(AudioTranscript).filter(AudioTranscript.file_id == file_id).all()
            for transcript in audio_transcripts:
                details["audio_transcripts"].append({
                    "id": transcript.id,
                    "content": transcript.content,
                    "start_time": transcript.start_time,
                    "end_time": transcript.end_time
                })
            # 获取人脸信息（人脸和特征合并后）
            faces = db.query(Face).filter(Face.file_id == file_id).all()
            for face in faces:
                # 解析bbox（JSON格式）
                bbox_data = []
                if face.bbox:
                    try:
                        bbox_data = json.loads(face.bbox)
                    except Exception:
                        bbox_data = face.bbox
                details["faces"].append({
                    "id": face.id,
                    "name": face.name,
                    "group_id": face.group_id,
                    "image_path": face.image_path,
                    "confidence": face.confidence,
                    "bbox": bbox_data,
                    "created_at": face.created_at.isoformat() if face.created_at else None
                })
        except Exception:
            pass
        return details
    @staticmethod
    def get_temp_file_path(file_name: str) -> str:
        """获取临时文件路径"""
        temp_dir = os.path.join(settings.TEMP_DIR)
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_name = f"{str(uuid.uuid4())}_{file_name}"
        return os.path.join(temp_dir, temp_file_name)
    @staticmethod
    def clean_temp_file(file_path: str):
        """清理临时文件"""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
