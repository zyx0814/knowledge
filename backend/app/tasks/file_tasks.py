import os
import uuid
from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.services.file_service import FileService
from app.services.file_processor import FileProcessor
from app.services.vector_service import VectorService
from app.models.db import TextChunk, ImageFrame, AudioTranscript
from config.config import settings
file_service = FileService()
file_processor = FileProcessor()
vector_service = VectorService()
@celery_app.task(bind=True, max_retries=3)
def process_file_task(self, file_id: str, temp_file_path: str, file_type: str, subformat: str):
    """异步文件处理任务"""
    db = SessionLocal()
    try:
        file = file_service.get_file_by_id(db, file_id)
        if not file:
            raise ValueError(f"文件不存在: {file_id}")
        file_service.update_file_status(db, file_id, "processing")
        if file_type == "document":
            chunks = file_processor.process_document(temp_file_path, subformat)
            for i, chunk in enumerate(chunks):
                embedding = vector_service.generate_text_embedding(chunk)
                vector_id = vector_service.add_vector(embedding, f"chunk_{file_id}_{i}")
                text_chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    file_id=file_id,
                    content=chunk,
                    chunk_index=i,
                    start_pos=0,
                    end_pos=len(chunk),
                    vector_id=f"chunk_{file_id}_{i}"
                )
                db.add(text_chunk)
        elif file_type == "image":
            result = file_processor.process_image(temp_file_path)
            # 使用CLIP模型从图片路径生成嵌入（与单文件上传一致，支持跨模态检索）
            item_id = f"image_{file_id}"
            embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
            vector_id = vector_service.add_vector(embedding, item_id)
            image_frame = ImageFrame(
                id=str(uuid.uuid4()),
                file_id=file_id,
                frame_path=temp_file_path,
                timestamp=0,
                vector_id=item_id  # 保存item_id而不是索引位置
            )
            db.add(image_frame)
            if settings.ENABLE_OCR and result.get("text"):
                text_embedding = vector_service.generate_text_embedding(result["text"])
                text_vector_id = vector_service.add_vector(text_embedding, f"image_text_{file_id}")
                text_chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    file_id=file_id,
                    content=result["text"],
                    chunk_index=0,
                    start_pos=0,
                    end_pos=len(result["text"]),
                    vector_id=f"image_text_{file_id}"
                )
                db.add(text_chunk)
            if settings.ENABLE_FACE_DETECTION and result.get("faces") and len(result["faces"]) > 0:
                from app.services.face_service import FaceService
                face_service = FaceService()
                face_service.save_face_features(db, file_id, result["faces"])
        elif file_type == "video":
            result = file_processor.process_video(temp_file_path)
            if not result.get("frames"):
                result["frames"] = []
            for i, frame in enumerate(result["frames"]):
                frame_path = frame.get("path", "")
                if not frame_path:
                    continue
                item_id = f"video_frame_{file_id}_{i}"
                embedding = vector_service.generate_image_embedding_from_path(frame_path)
                vector_service.add_vector(embedding, item_id)
                image_frame = ImageFrame(
                    id=str(uuid.uuid4()),
                    file_id=file_id,
                    frame_path=frame_path,
                    timestamp=frame.get("timestamp", 0),
                    vector_id=item_id
                )
                db.add(image_frame)
                if settings.ENABLE_OCR and frame.get("text"):
                    text_embedding = vector_service.generate_text_embedding(frame["text"])
                    vector_service.add_vector(text_embedding, f"video_frame_text_{file_id}_{i}")
                    text_chunk = TextChunk(
                        id=str(uuid.uuid4()),
                        file_id=file_id,
                        content=frame["text"],
                        chunk_index=i,
                        start_pos=0,
                        end_pos=len(frame["text"]),
                        vector_id=f"video_frame_text_{file_id}_{i}"
                    )
                    db.add(text_chunk)
                if settings.ENABLE_FACE_DETECTION and frame.get("faces") and len(frame["faces"]) > 0:
                    from app.services.face_service import FaceService
                    face_service = FaceService()
                    face_service.save_face_features(db, file_id, frame["faces"])
            if settings.ENABLE_VIDEO_AUDIO and result.get("audio_text"):
                audio_text = result["audio_text"]
                chunks = file_processor._split_text(audio_text)
                for j, chunk in enumerate(chunks):
                    embedding = vector_service.generate_text_embedding(chunk)
                    vector_service.add_vector(embedding, f"video_audio_{file_id}_{j}")
                    text_chunk = TextChunk(
                        id=str(uuid.uuid4()),
                        file_id=file_id,
                        content=chunk,
                        chunk_index=j,
                        start_pos=0,
                        end_pos=len(chunk),
                        vector_id=f"video_audio_{file_id}_{j}"
                    )
                    db.add(text_chunk)
                audio_transcript = AudioTranscript(
                    id=str(uuid.uuid4()),
                    file_id=file_id,
                    content=audio_text,
                    start_time=0,
                    end_time=0,
                    vector_id=f"video_audio_{file_id}_0"
                )
                db.add(audio_transcript)
        db.commit()
        vector_service.save_index()
        file_service.update_file_status(db, file_id, "completed")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                pass
        return {"success": True, "file_id": file_id}
    except Exception as exc:
        db.rollback()
        file_service.update_file_status(db, file_id, "failed")
        raise self.retry(exc=exc, countdown=60)
    finally:
        db.close()
