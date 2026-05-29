import os
import asyncio
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from app.core.database import get_db, SessionLocal
from app.services.file_service import FileService
from app.services.file_processor import FileProcessor
from app.services.vector_service import VectorService
from app.services.search_service import SearchService
from app.services.qa_service import QAService
from app.services.task_service import TaskService, TaskStatus
from config.config import settings
router = APIRouter()
file_service = FileService()
file_processor = FileProcessor()
vector_service = VectorService()
search_service = SearchService()
qa_service = QAService()
task_service = TaskService()
# 可修改的配置项白名单（与 config.py 中的 Settings 类字段匹配）
ALLOWED_CONFIG_KEYS = {
    'api_key',
    'enable_auth',
    'data_dir',
    'upload_dir',
    'temp_dir',
    'face_model_path',
    'face_model_name',
    'max_file_size',
    'vector_db_path',
    'embedding_dim',
    'ollama_base_url',
    'llm_model',
    'embedding_model',
    'max_workers',
    'chunk_size',
    'chunk_overlap',
    'video_frame_interval',
    'max_frames_per_video',
    'enable_ocr',
    'enable_face_detection',
    'enable_video_audio',
    'keep_files',
    'permanent_storage_dir'
}
@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """获取当前配置（安全过滤敏感信息）"""
    config_dict = {}
    # 安全地暴露配置项
    for key in ALLOWED_CONFIG_KEYS:
        if hasattr(settings, key.upper()):
            value = getattr(settings, key.upper())
            # 对敏感信息进行脱敏
            if key == 'api_key' and value:
                value = "******" if len(value) > 6 else value
            config_dict[key] = value
    return {"config": config_dict}
@router.put("/config")
async def update_config(
    config_updates: Dict[str, Any]
) -> Dict[str, Any]:
    """修改配置项"""
    updated_keys = []
    skipped_keys = []
    for key, value in config_updates.items():
        # 检查是否在白名单中
        if key not in ALLOWED_CONFIG_KEYS:
            skipped_keys.append(key)
            continue
        # 获取配置项名称（转为大写）
        config_key = key.upper()
        # 验证类型
        if hasattr(settings, config_key):
            current_value = getattr(settings, config_key)
            # 类型检查
            if isinstance(current_value, int):
                try:
                    value = int(value)
                except ValueError:
                    skipped_keys.append(f"{key} (类型错误，期望int)")
                    continue
            elif isinstance(current_value, float):
                try:
                    value = float(value)
                except ValueError:
                    skipped_keys.append(f"{key} (类型错误，期望float)")
                    continue
            elif isinstance(current_value, bool):
                if isinstance(value, str):
                    value = value.lower() == 'true'
                else:
                    value = bool(value)
            # 更新配置（通过修改settings对象的属性）
            setattr(settings, config_key, value)
            updated_keys.append(key)
    result = {
        "success": True,
        "updated": updated_keys,
        "skipped": skipped_keys
    }
    if skipped_keys:
        result["message"] = f"部分配置项未更新：{', '.join(skipped_keys)}"
    return result
def _enrich_search_results(db: Session, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为搜索结果添加文件信息"""
    from app.models.db import File, ImageFrame
    enriched_results = []
    for result in results:
        file_id = result.get("file_id")
        if not file_id:
            continue
        file = db.query(File).filter(File.id == file_id).first()
        if not file:
            continue
        enriched_result = {
            "file_id": str(file_id),
            "score": float(result.get("score", 0.0)),
            "search_type": result.get("search_type", "unknown"),
            "file_info": {
                "id": str(file.id),
                "rid": str(file.rid) if file.rid else None,
                "name": str(file.name),
                "type": str(file.type),
                "subformat": str(file.subformat),
                "size": file.size,
                "imported_at": file.imported_at.isoformat() if file.imported_at else None
            }
        }
        if file.type == "image":
            if file.storage_path:
                enriched_result["storage_path"] = str(file.storage_path)
            else:
                image_frame = db.query(ImageFrame).filter(ImageFrame.file_id == file_id).first()
                if image_frame and image_frame.frame_path:
                    enriched_result["frame_path"] = str(image_frame.frame_path)
        elif file.type == "video":
            if result.get("timestamp") is not None:
                enriched_result["timestamp"] = float(result["timestamp"])
                frame_info = db.query(ImageFrame).filter(
                    ImageFrame.file_id == file_id,
                    ImageFrame.timestamp == result["timestamp"]
                ).first()
                if frame_info and frame_info.frame_path:
                    enriched_result["frame_path"] = str(frame_info.frame_path)
        enriched_results.append(enriched_result)
    enriched_results.sort(key=lambda x: x["score"], reverse=True)
    return enriched_results
@router.post("/file/upload")
async def upload_file(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    rid: str = Form(None),
    gid: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """文件上传接口"""
    try:
        import uuid
        if file:
            temp_file_path = file_service.get_temp_file_path(file.filename)
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            file_name = file.filename
            file_ext = file.filename.split(".")[-1].lower()
        elif base64_data:
            import base64
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            file_name = f"{str(uuid.uuid4())}.png"
            temp_file_path = file_service.get_temp_file_path(file_name)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
            file_ext = "png"
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或base64数据")
        # 文档类支持的文件格式
        document_extensions = ["pdf", "doc", "docx", "rtf", "odt", "txt", "ppt", "pptx", "pps", "odp", "xls", "xlsx", "ods", "csv"]
        # 视频类支持的文件格式
        video_extensions = ["avi", "rm", "rmvb", "mkv", "mov", "wmv", "asf", "mpg", "mpe", "mpeg", "mp4", "m4v", "f4v", "vob", "ogv", "mts", "m2ts", "3gp", "webm", "flv", "wav", "vqf", "ra", "mxf"]
        # 图片类支持的文件格式
        image_extensions = ["jpg", "jpeg", "png", "webp", "bmp", "gif"]
        
        if file_ext in document_extensions:
            file_type = "document"
            subformat = file_ext
        elif file_ext in image_extensions:
            file_type = "image"
            subformat = file_ext
        elif file_ext in video_extensions:
            file_type = "video"
            subformat = file_ext
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        file_size = os.path.getsize(temp_file_path)
        db_file = file_service.create_file(db, rid, gid, file_name, file_type, subformat, file_size)
        if settings.KEEP_FILES:
            permanent_path = file_service.save_to_permanent_storage(temp_file_path, db_file.id, file_name)
            if permanent_path:
                db_file.storage_path = permanent_path
                db.commit()
        from app.models.db import TextChunk, ImageFrame, AudioTranscript
        import uuid
        if file_type == "document":
            chunks = file_processor.process_document(temp_file_path, subformat)
            for i, chunk in enumerate(chunks):
                embedding = vector_service.generate_text_embedding(chunk)
                vector_id = vector_service.add_vector(embedding, f"chunk_{db_file.id}_{i}")
                text_chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    file_id=db_file.id,
                    content=chunk,
                    chunk_index=i,
                    start_pos=0,
                    end_pos=len(chunk),
                    vector_id=f"chunk_{db_file.id}_{i}"
                )
                db.add(text_chunk)
        elif file_type == "image":
            result = file_processor.process_image(temp_file_path)
            # 使用CLIP模型从图片路径生成嵌入（跨模态检索）
            item_id = f"image_{db_file.id}"
            embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
            vector_service.add_vector(embedding, item_id)
            image_frame = ImageFrame(
                id=str(uuid.uuid4()),
                file_id=db_file.id,
                frame_path=temp_file_path,
                timestamp=0,
                vector_id=item_id  # 保存item_id而不是索引位置
            )
            db.add(image_frame)
            if settings.ENABLE_OCR and result.get("text"):
                text_embedding = vector_service.generate_text_embedding(result["text"])
                vector_service.add_vector(text_embedding, f"image_text_{db_file.id}")
                text_chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    file_id=db_file.id,
                    content=result["text"],
                    chunk_index=0,
                    start_pos=0,
                    end_pos=len(result["text"]),
                    vector_id=f"image_text_{db_file.id}"
                )
                db.add(text_chunk)
            if settings.ENABLE_FACE_DETECTION and result.get("faces") and len(result["faces"]) > 0:
                from app.services.face_service import FaceService
                face_service = FaceService()
                face_service.save_face_features(db, db_file.id, result["faces"])
        elif file_type == "video":
            try:
                result = file_processor.process_video(temp_file_path)
                if not result.get("frames"):
                    result["frames"] = []
                for i, frame in enumerate(result["frames"]):
                    frame_path = frame.get("path", "")
                    if not frame_path:
                        continue
                    item_id = f"video_frame_{db_file.id}_{i}"
                    embedding = vector_service.generate_image_embedding_from_path(frame_path)
                    vector_service.add_vector(embedding, item_id)
                    image_frame = ImageFrame(
                        id=str(uuid.uuid4()),
                        file_id=db_file.id,
                        frame_path=frame_path,
                        timestamp=frame.get("timestamp", 0),
                        vector_id=item_id
                    )
                    db.add(image_frame)
                    if settings.ENABLE_OCR and frame.get("text"):
                        text_embedding = vector_service.generate_text_embedding(frame["text"])
                        vector_service.add_vector(text_embedding, f"video_frame_text_{db_file.id}_{i}")
                        text_chunk = TextChunk(
                            id=str(uuid.uuid4()),
                            file_id=db_file.id,
                            content=frame["text"],
                            chunk_index=i,
                            start_pos=0,
                            end_pos=len(frame["text"]),
                            vector_id=f"video_frame_text_{db_file.id}_{i}"
                        )
                        db.add(text_chunk)
                    if settings.ENABLE_FACE_DETECTION and frame.get("faces") and len(frame["faces"]) > 0:
                        from app.services.face_service import FaceService
                        face_service = FaceService()
                        face_service.save_face_features(db, db_file.id, frame["faces"])
                if settings.ENABLE_VIDEO_AUDIO and result.get("audio_text"):
                    audio_text = result["audio_text"]
                    chunks = file_processor._split_text(audio_text)
                    for j, chunk in enumerate(chunks):
                        embedding = vector_service.generate_text_embedding(chunk)
                        vector_id = vector_service.add_vector(embedding, f"video_audio_{db_file.id}_{j}")
                        text_chunk = TextChunk(
                            id=str(uuid.uuid4()),
                            file_id=db_file.id,
                            content=chunk,
                            chunk_index=j,
                            start_pos=0,
                            end_pos=len(chunk),
                            vector_id=f"video_audio_{db_file.id}_{j}"
                        )
                        db.add(text_chunk)
                    audio_transcript = AudioTranscript(
                        id=str(uuid.uuid4()),
                        file_id=db_file.id,
                        content=audio_text,
                        start_time=0,
                        end_time=0,
                        vector_id=f"video_audio_{db_file.id}_0"
                    )
                    db.add(audio_transcript)
            except Exception as e:
                print(f"视频处理失败: {e}")
        
        db.commit()
        file_service.update_file_status(db, db_file.id, "completed")
        file_service.clean_temp_file(temp_file_path)
        # 保存向量索引到磁盘
        vector_service.save_index()
        return {
            "success": True,
            "file_id": db_file.id,
            "rid": db_file.rid,
            "message": "文件上传成功"
        }
    except Exception as e:
        if 'temp_file_path' in locals():
            file_service.clean_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
@router.get("/file/list")
async def get_file_list(
    skip: int = 0,
    limit: int = 100,
    include_details: bool = False,
    rid: str = None,
    gid: str = None,
    keyword: str = None,
    cursor: str = None,
    use_estimate: bool = True,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """获取文件列表（支持游标分页和估算总数）"""
    from app.models.db import File
    from sqlalchemy import text
    from datetime import datetime
    query = db.query(File)
    if rid:
        query = query.filter(File.rid == rid)
    if gid is not None:
        query = query.filter(File.gid == gid)
    if keyword:
        query = query.filter(
            File.name.contains(keyword) |
            File.category.contains(keyword)
        )
    if cursor:
        try:
            cursor_dt = datetime.fromisoformat(cursor)
            query = query.filter(File.imported_at < cursor_dt)
        except (ValueError, TypeError):
            pass
    files = query.order_by(File.imported_at.desc()).offset(skip).limit(limit).all()
    if use_estimate and settings.DATABASE_TYPE == "postgresql":
        result = db.execute(
            text("SELECT reltuples::BIGINT FROM pg_class WHERE relname = 'files'")
        ).scalar()
        total = int(result or 0)
    else:
        total = query.count()
    result = []
    next_cursor = None
    if files:
        next_cursor = files[-1].imported_at.isoformat() if files[-1].imported_at else None
    for file in files:
        file_info = {
            "id": file.id,
            "rid": file.rid,
            "gid": file.gid,
            "name": file.name,
            "type": file.type,
            "subformat": file.subformat,
            "size": file.size,
            "imported_at": file.imported_at,
            "status": file.status,
            "category": file.category
        }
        if include_details:
            details = file_service.get_file_details(db, file.id)
            file_info["details"] = details
        result.append(file_info)
    return {
        "items": result,
        "total": total,
        "page": skip // limit + 1,
        "page_size": limit,
        "next_cursor": next_cursor,
        "has_more": len(files) == limit
    }
@router.get("/file/{file_id}")
async def get_file_detail(
    file_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """获取单个文件详情"""
    from app.models.db import File
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    details = file_service.get_file_details(db, file.id)
    return {
        "id": file.id,
        "rid": file.rid,
        "gid": file.gid,
        "name": file.name,
        "type": file.type,
        "subformat": file.subformat,
        "size": file.size,
        "imported_at": file.imported_at,
        "status": file.status,
        "category": file.category,
        "storage_path": file.storage_path,
        "details":details
    }
@router.post("/search")
async def search(
    query: str,
    file_type: str = None,
    gid: str = None,
    limit: int = 50,
    min_score: float = None,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """混合检索接口"""
    # 使用配置默认值（如果未指定）
    if min_score is None:
        min_score = settings.SEARCH_MIN_SCORE
    results = search_service.hybrid_search(db, query, file_type, gid=gid, limit=limit, min_score=min_score)
    return _enrich_search_results(db, results)
@router.post("/qa")
async def qa(
    question: str,
    file_type: str = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """知识库问答接口"""
    result = qa_service.generate_answer(db, question, file_type)
    return result
@router.post("/search/image")
async def search_image(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    gid: str = Form(None),
    limit: int = 50,
    min_score: float = None,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """以图搜图接口"""
    # 使用配置默认值（如果未指定）
    if min_score is None:
        min_score = settings.SEARCH_MIN_SCORE
    try:
        import uuid
        temp_file_path = None
        if file:
            temp_file_path = f"{settings.TEMP_DIR}/{file.filename}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        elif base64_data:
            import base64
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            file_name = f"{str(uuid.uuid4())}.png"
            temp_file_path = f"{settings.TEMP_DIR}/{file_name}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或base64数据")
        # 使用CLIP模型从图片路径生成嵌入（与上传时使用相同的方法）
        clip_embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
        search_results = search_service.multimodal_search(db, clip_embedding, limit=limit, min_score=min_score)
        # 人脸检测（用于人脸搜索）
        from app.services.file_processor import FileProcessor
        file_processor = FileProcessor()
        result = file_processor.process_image(temp_file_path)
        if result.get("faces") and len(result["faces"]) > 0:
            from app.services.face_service import FaceService
            face_service = FaceService()
            for face_feature in result["faces"]:
                face_embedding = face_feature["embedding"]
                face_search_results = search_service.face_search(db, face_embedding, limit=limit, min_score=min_score)
                search_results.extend(face_search_results)
        if temp_file_path:
            file_service.clean_temp_file(temp_file_path)
        return _enrich_search_results(db, search_results)
    except Exception as e:
        if 'temp_file_path' in locals() and temp_file_path:
            file_service.clean_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"以图搜图失败: {str(e)}")
@router.post("/search/face")
async def search_face(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    gid: str = Form(None),
    limit: int = 10,
    min_score: float = None,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """按人脸搜索接口"""
    # 使用配置默认值（如果未指定）
    if min_score is None:
        min_score = settings.FACE_SEARCH_THRESHOLD
    try:
        import uuid
        temp_file_path = None
        if file:
            temp_file_path = f"{settings.TEMP_DIR}/{file.filename}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        elif base64_data:
            import base64
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            file_name = f"{str(uuid.uuid4())}.png"
            temp_file_path = f"{settings.TEMP_DIR}/{file_name}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或base64数据")
        from app.services.face_service import FaceService
        face_service = FaceService()
        face_features = face_service.extract_face_features(temp_file_path)
        if not face_features:
            return []
        all_results = []
        for face_feature in face_features:
            face_embedding = face_feature["embedding"]
            face_results = search_service.face_search(db, face_embedding, limit=limit, min_score=min_score)
            all_results.extend(face_results)
        if temp_file_path:
            file_service.clean_temp_file(temp_file_path)
        return _enrich_search_results(db, all_results)
    except Exception as e:
        if 'temp_file_path' in locals() and temp_file_path:
            file_service.clean_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"按人脸搜索失败: {str(e)}")
@router.get("/faces")
async def get_face_list(
    skip: int = 0,
    limit: int = 100,
    name: str = None,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """获取人脸列表"""
    from app.services.face_service import FaceService
    face_service = FaceService()
    return face_service.get_face_list(db, skip, limit, name)
@router.get("/faces/{face_id}")
async def get_face_details(
    face_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """获取人脸详情"""
    from app.services.face_service import FaceService
    face_service = FaceService()
    details = face_service.get_face_details(db, face_id)
    if not details:
        raise HTTPException(status_code=404, detail="人脸不存在")
    return details
@router.put("/faces/{face_id}/name")
async def update_face_name(
    face_id: str,
    name: str = Form(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """更新人脸名称"""
    from app.services.face_service import FaceService
    face_service = FaceService()
    success = face_service.update_face_name(db, face_id, name)
    if not success:
        raise HTTPException(status_code=404, detail="人脸不存在")
    return {"success": True, "message": "人脸名称更新成功"}
@router.post("/faces/merge")
async def merge_faces(
    face_ids: List[str] = Form(...),
    name: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """合并人脸"""
    from app.services.face_service import FaceService
    face_service = FaceService()
    group_id = face_service.merge_faces(db, face_ids, name)
    return {"success": True, "group_id": group_id, "message": "人脸合并成功"}
@router.delete("/faces/{face_id}")
async def delete_face(
    face_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """删除人脸"""
    from app.services.face_service import FaceService
    face_service = FaceService()
    success = face_service.delete_face(db, face_id)
    if not success:
        raise HTTPException(status_code=404, detail="人脸不存在")
    return {"success": True, "message": "人脸删除成功"}
@router.post("/faces/add")
async def add_face(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    name: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """添加人脸"""
    try:
        import uuid
        temp_file_path = None
        if file:
            temp_file_path = f"{settings.TEMP_DIR}/{file.filename}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        elif base64_data:
            import base64
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            file_name = f"{str(uuid.uuid4())}.png"
            temp_file_path = f"{settings.TEMP_DIR}/{file_name}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或base64数据")
        from app.services.face_service import FaceService
        face_service = FaceService()
        face_features = face_service.extract_face_features(temp_file_path)
        if not face_features:
            if temp_file_path:
                file_service.clean_temp_file(temp_file_path)
            raise HTTPException(status_code=400, detail="未检测到人脸")
        saved_face_ids = face_service.save_face_features(db, "", face_features)
        if name and saved_face_ids:
            for face_id in saved_face_ids:
                face_service.update_face_name(db, face_id, name)
        if temp_file_path:
            file_service.clean_temp_file(temp_file_path)
        return {
            "success": True,
            "face_ids": saved_face_ids,
            "message": f"成功添加 {len(saved_face_ids)} 个人脸"
        }
    except Exception as e:
        if 'temp_file_path' in locals() and temp_file_path:
            file_service.clean_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"添加人脸失败: {str(e)}")
@router.get("/file/{file_id}")
async def get_file(
    file_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """获取文件详情（包含详细信息）"""
    file = file_service.get_file_by_id(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    details = file_service.get_file_details(db, file_id)
    return {
        "id": file.id,
        "rid": file.rid,
        "gid": file.gid,
        "name": file.name,
        "type": file.type,
        "subformat": file.subformat,
        "size": file.size,
        "imported_at": file.imported_at,
        "status": file.status,
        "category": file.category,
        "details": details
    }
@router.delete("/file/{file_id}")
async def delete_file(
    file_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """删除文件"""
    success = file_service.delete_file(db, file_id)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"success": True, "message": "文件删除成功"}
def _process_batch_import_sync(task_id: str, directory: str, rid: str, gid: str):
    """同步处理批量导入（在后台线程中运行）"""
    import time
    try:
        # 使用同步方式更新任务状态
        # 创建新的事件循环用于异步任务更新
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 使用线程安全的方式更新任务状态
        def run_async(coro):
            try:
                return loop.run_until_complete(coro)
            except Exception as e:
                print(f"[批量导入] 异步操作失败: {e}")
                return None
        run_async(task_service.start_task(task_id))
        db = SessionLocal()
        success_count = 0
        failed_count = 0
        failed_files = []
        directory = directory.strip()
        directory = directory.replace('\\', os.sep).replace('/', os.sep)
        allowed_extensions = {
            'document': ['pdf', 'doc', 'docx', 'rtf', 'odt', 'txt', 'ppt', 'pptx', 'pps', 'odp', 'xls', 'xlsx', 'ods', 'csv'],
            'image': ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'],
            'video': ['avi', 'rm', 'rmvb', 'mkv', 'mov', 'wmv', 'asf', 'mpg', 'mpe', 'mpeg', 'mp4', 'm4v', 'f4v', 'vob', 'ogv', 'mts', 'm2ts', '3gp', 'webm', 'flv', 'wav', 'vqf', 'ra', 'mxf']
        }
        all_files = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                ext = filename.split('.')[-1].lower() if '.' in filename else ''
                file_type = None
                for type_name, extensions in allowed_extensions.items():
                    if ext in extensions:
                        file_type = type_name
                        break
                if file_type:
                    file_path = os.path.join(root, filename)
                    all_files.append({
                        'path': file_path,
                        'name': filename,
                        'type': file_type,
                        'ext': ext
                    })
        total_files = len(all_files)
        if total_files == 0:
            loop.run_until_complete(
                task_service.complete_task(task_id, {
                    "total_files": 0,
                    "success_count": 0,
                    "failed_count": 0,
                    "message": "目录中没有找到允许的文件类型"
                })
            )
            db.close()
            return
        for idx, file_info in enumerate(all_files):
            progress = int((idx / total_files) * 100)
            run_async(task_service.update_task(
                task_id,
                progress=progress,
                message=f"正在处理: {file_info['name']} ({idx+1}/{total_files})"
            ))
            temp_file_path = None
            try:
                file_type = file_info['type']
                subformat = file_info['ext']
                temp_file_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}_{file_info['name']}")
                shutil.copy2(file_info['path'], temp_file_path)
                file_size = os.path.getsize(temp_file_path)
                db_file = file_service.create_file(db, rid, gid, file_info['name'], file_type, subformat, file_size)
                if settings.KEEP_FILES:
                    permanent_path = file_service.save_to_permanent_storage(temp_file_path, db_file.id, file_info['name'])
                    if permanent_path:
                        db_file.storage_path = permanent_path
                        db.commit()
                from app.models.db import TextChunk, ImageFrame, AudioTranscript
                if file_type == "document":
                    chunks = file_processor.process_document(temp_file_path, file_info['ext'])
                    for chunk_idx, chunk in enumerate(chunks):
                        embedding = vector_service.generate_text_embedding(chunk)
                        vector_service.add_vector(embedding, f"chunk_{db_file.id}_{chunk_idx}")
                        text_chunk = TextChunk(
                            id=str(uuid.uuid4()),
                            file_id=db_file.id,
                            content=chunk,
                            chunk_index=chunk_idx,
                            start_pos=0,
                            end_pos=len(chunk),
                            vector_id=f"chunk_{db_file.id}_{chunk_idx}"
                        )
                        db.add(text_chunk)
                elif file_type == "image":
                    result = file_processor.process_image(temp_file_path)
                    # 使用CLIP模型从图片路径生成嵌入（跨模态检索）
                    item_id = f"image_{db_file.id}"
                    embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
                    vector_service.add_vector(embedding, item_id)
                    image_frame = ImageFrame(
                        id=str(uuid.uuid4()),
                        file_id=db_file.id,
                        frame_path=temp_file_path,
                        timestamp=0,
                        vector_id=item_id  # 保存item_id而不是索引位置
                    )
                    db.add(image_frame)
                    if settings.ENABLE_OCR and result.get("text"):
                        text_embedding = vector_service.generate_text_embedding(result["text"])
                        vector_service.add_vector(text_embedding, f"image_text_{db_file.id}")
                        text_chunk = TextChunk(
                            id=str(uuid.uuid4()),
                            file_id=db_file.id,
                            content=result["text"],
                            chunk_index=0,
                            start_pos=0,
                            end_pos=len(result["text"]),
                            vector_id=f"image_text_{db_file.id}"
                        )
                        db.add(text_chunk)
                    if settings.ENABLE_FACE_DETECTION and result.get("faces") and len(result["faces"]) > 0:
                        from app.services.face_service import FaceService
                        face_service = FaceService()
                        face_service.save_face_features(db, db_file.id, result["faces"])
                elif file_type == "video":
                    try:
                        result = file_processor.process_video(temp_file_path)
                        if not result.get("frames"):
                            result["frames"] = []
                        for i, frame in enumerate(result["frames"]):
                            frame_path = frame.get("path", "")
                            if not frame_path:
                                continue
                            item_id = f"video_frame_{db_file.id}_{i}"
                            embedding = vector_service.generate_image_embedding_from_path(frame_path)
                            vector_service.add_vector(embedding, item_id)
                            image_frame = ImageFrame(
                                id=str(uuid.uuid4()),
                                file_id=db_file.id,
                                frame_path=frame_path,
                                timestamp=frame.get("timestamp", 0),
                                vector_id=item_id
                            )
                            db.add(image_frame)
                            if settings.ENABLE_OCR and frame.get("text"):
                                text_embedding = vector_service.generate_text_embedding(frame["text"])
                                vector_service.add_vector(text_embedding, f"video_frame_text_{db_file.id}_{i}")
                                text_chunk = TextChunk(
                                    id=str(uuid.uuid4()),
                                    file_id=db_file.id,
                                    content=frame["text"],
                                    chunk_index=i,
                                    start_pos=0,
                                    end_pos=len(frame["text"]),
                                    vector_id=f"video_frame_text_{db_file.id}_{i}"
                                )
                                db.add(text_chunk)
                            if settings.ENABLE_FACE_DETECTION and frame.get("faces") and len(frame["faces"]) > 0:
                                from app.services.face_service import FaceService
                                face_service = FaceService()
                                face_service.save_face_features(db, db_file.id, frame["faces"])
                        if settings.ENABLE_VIDEO_AUDIO and result.get("audio_text"):
                            audio_text = result["audio_text"]
                            chunks = file_processor._split_text(audio_text)
                            for j, chunk in enumerate(chunks):
                                embedding = vector_service.generate_text_embedding(chunk)
                                vector_id = vector_service.add_vector(embedding, f"video_audio_{db_file.id}_{j}")
                                text_chunk = TextChunk(
                                    id=str(uuid.uuid4()),
                                    file_id=db_file.id,
                                    content=chunk,
                                    chunk_index=j,
                                    start_pos=0,
                                    end_pos=len(chunk),
                                    vector_id=f"video_audio_{db_file.id}_{j}"
                                )
                                db.add(text_chunk)
                            audio_transcript = AudioTranscript(
                                id=str(uuid.uuid4()),
                                file_id=db_file.id,
                                content=audio_text,
                                start_time=0,
                                end_time=0,
                                vector_id=f"video_audio_{db_file.id}_0"
                            )
                            db.add(audio_transcript)
                    except Exception as e:
                        pass
                db.commit()
                file_service.update_file_status(db, db_file.id, "completed")
                file_service.clean_temp_file(temp_file_path)
                success_count += 1
            except Exception as e:
                failed_count += 1
                failed_files.append({
                    'name': file_info['name'],
                    'error': str(e)
                })
                if 'temp_file_path' in locals() and temp_file_path:
                    file_service.clean_temp_file(temp_file_path)
        db.close()
        # 批量导入完成后保存向量索引
        vector_service.save_index()
        run_async(task_service.complete_task(task_id, {
            "total_files": total_files,
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_files": failed_files,
            "message": f"批量导入完成，成功 {success_count} 个，失败 {failed_count} 个"
        }))
    except Exception as e:
        try:
            run_async(task_service.fail_task(task_id, str(e)))
        except:
            # 如果异步调用失败，直接打印错误
            print(f"[批量导入] 更新任务失败状态失败: {e}")


@router.post("/file/batch_import_async")
async def batch_import_files_async(
    background_tasks: BackgroundTasks,
    directory: str = Form(...),
    rid: str = Form(None),
    gid: str = Form(None)
) -> Dict[str, Any]:
    """异步批量导入 - 后台处理，不阻塞其他请求"""
    import threading
    if not rid:
        rid = "default"
    task_id = await task_service.create_task("batch_import", {
        "directory": directory,
        "rid": rid,
        "gid": gid
    })
    thread = threading.Thread(
        target=_process_batch_import_sync,
        args=(task_id, directory, rid, gid)
    )
    thread.daemon = True
    thread.start()
    return {
        "success": True,
        "task_id": task_id,
        "message": "批量导入任务已提交，请通过任务状态接口查询进度"
    }
@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """获取任务状态"""
    task = await task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "task": task}
@router.get("/tasks")
async def get_all_tasks() -> Dict[str, Any]:
    """获取所有任务列表"""
    tasks = await task_service.get_all_tasks()
    return {"success": True, "tasks": tasks}
