import os
import asyncio
import shutil
import uuid
import re
import urllib.parse
import base64
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Optional
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
            "rid": str(file.rid) if file.rid else None,
            "gid": str(file.gid) if file.gid else None,
            "name": str(file.name),
            "type": str(file.type),
            "ext": str(file.subformat),
            "size": file.size,
            "imported_at": file.imported_at.isoformat() if file.imported_at else None,
           
        }
        if file.type == "image":
            if file.storage_path:
                enriched_result["storage_path"] = str(file.storage_path)
            else:
                image_frame = db.query(ImageFrame).filter(ImageFrame.file_id == file_id).first()
                if image_frame and image_frame.frame_path:
                    enriched_result["frame_path"] = str(image_frame.frame_path)
        elif file.type == "video":
            # 处理多个时间戳（视频切片）
            timestamps = result.get("timestamps", [])
            if timestamps:
                enriched_result["timestamps"] = []
                for ts_info in timestamps:
                    timestamp = ts_info.get("timestamp")
                    if timestamp is not None:
                        frame_info = db.query(ImageFrame).filter(
                            ImageFrame.file_id == file_id,
                            ImageFrame.timestamp == timestamp
                        ).first()
                        ts_result = {
                            "timestamp": float(timestamp),
                            "score": float(ts_info.get("score", 0.0))
                        }
                        if frame_info and frame_info.frame_path:
                            ts_result["frame_path"] = str(frame_info.frame_path)
                        enriched_result["timestamps"].append(ts_result)
        enriched_results.append(enriched_result)
    enriched_results.sort(key=lambda x: x["score"], reverse=True)
    return enriched_results
    
@router.post("/file/update")
async def update_file(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    url: str = Form(None),
    rid: str = Form(None),
    gid: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """文件更新接口，根据 rid 更新文件（删除旧文件后重新上传）"""
    try:
        if not rid:
            raise HTTPException(status_code=400, detail="必须提供 rid 参数")
        
        # 先删除该 rid 下的所有文件及其相关数据
        from app.models.db import File, TextChunk, ImageFrame, AudioTranscript
        files_to_delete = db.query(File).filter(File.rid == rid).all()
        deleted_count = 0
        for f in files_to_delete:
            file_id = str(f.id)
            # 删除向量
            vector_service.delete_vectors_by_file_id(file_id)
            # 删除数据库记录
            db.query(TextChunk).filter(TextChunk.file_id == file_id).delete()
            db.query(ImageFrame).filter(ImageFrame.file_id == file_id).delete()
            db.query(AudioTranscript).filter(AudioTranscript.file_id == file_id).delete()
            db.delete(f)
            deleted_count += 1
        db.commit()
        
        # 如果没有提供新文件，只删除不添加
        if not file and not base64_data and not url:
            return {"success": True, "deleted_count": deleted_count, "message": f"成功删除 {deleted_count} 个旧文件，未上传新文件"}
        
        # 调用上传逻辑处理新文件
        return await upload_file(file, base64_data, url, rid, gid, db)
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新文件失败: {str(e)}")

@router.post("/file/upload")
async def upload_file(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    url: str = Form(None),
    rid: str = Form(None),
    gid: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """文件上传接口，支持三种方式：1.直接上传文件 2.Base64 编码 3.URL 下载"""
    try:
        if file:
            temp_file_path = file_service.get_temp_file_path(file.filename)
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            file_name = file.filename
            file_ext = file.filename.split(".")[-1].lower()
        elif base64_data:
            
            # MIME 类型到文件扩展名的映射
            mime_to_ext = {
                'image/png': 'png',
                'image/jpeg': 'jpg',
                'image/jpg': 'jpg',
                'image/gif': 'gif',
                'image/webp': 'webp',
                'image/bmp': 'bmp',
                'image/tiff': 'tif',
                'image/svg+xml': 'svg'
            }
            
            # 默认扩展名
            file_ext = 'png'
            
            # 从 data URL 中提取 MIME 类型
            mime_match = re.match(r'data:([^;]+);base64,', base64_data)
            if mime_match:
                mime_type = mime_match.group(1).lower()
                if mime_type in mime_to_ext:
                    file_ext = mime_to_ext[mime_type]
            
            # 提取 base64 数据
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_data = base64.b64decode(base64_data)
            file_name = f"{str(uuid.uuid4())}.{file_ext}"
            temp_file_path = file_service.get_temp_file_path(file_name)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
                
        elif url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                file_name = None
                parsed_url = urllib.parse.urlparse(url)
                
                # 优先级 1：从 Content-Disposition 头获取文件名
                content_disposition = response.headers.get('content-disposition', '')
                if content_disposition:
                    match = re.search(r'filename[^;=\n]*=((["\']).*?\2|[^;\n]*)', content_disposition)
                    if match:
                        file_name = match.group(1).strip('"\'')
                
                # 优先级2：从 URL 参数中获取 filename
                if not file_name:
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    if 'filename' in query_params:
                        file_name = query_params['filename'][0]
                
                # 优先级 3：从 URL 路径中提取文件名
                if not file_name:
                    path = parsed_url.path
                    file_name = os.path.basename(path)
                
                # 如果都获取不到，返回错误
                if not file_name:
                    raise HTTPException(status_code=400, detail="无法从URL获取文件名，请确保服务器返回Content-Disposition头或URL包含filename参数")
                
                temp_file_path = file_service.get_temp_file_path(file_name)
                with open(temp_file_path, "wb") as f:
                    f.write(response.content)
                
                # 获取文件扩展名
                if '.' in file_name:
                    file_ext = file_name.split(".")[-1].lower()
                else:
                    # 如果没有扩展名，尝试从 Content-Type 推断
                    content_type = response.headers.get('content-type', '')
                    content_type_map = {
                        'image/jpeg': 'jpg',
                        'image/png': 'png',
                        'image/gif': 'gif',
                        'image/webp': 'webp',
                        'application/pdf': 'pdf',
                        'text/plain': 'txt',
                    }
                    file_ext = content_type_map.get(content_type.split(';')[0].strip(), 'bin')
        else:
            raise HTTPException(status_code=400, detail="必须提供文件、base64数据或URL")
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
    gid: List[str] = None,
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
    request: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """知识库问答接口"""
    question = request.get("question", "")
    file_type = request.get("file_type", None)
    
    search_results = search_service.hybrid_search(db, question, file_type, limit=10)
    enriched_results = _enrich_search_results(db, search_results)
    
    has_image_results = any(r.get('type') == 'image' for r in enriched_results)
    
    context = "\n".join([
        f"[{r.get('type', '未知')}: {r.get('name', '未知')}]"
        for r in enriched_results
    ])
    
    if not context:
        context = "知识库为空或未找到相关内容。"
    
    if has_image_results:
        prompt = f"""
        你是一个基于本地知识库的问答助手。用户正在搜索图片相关的内容。
        
        知识库中找到的相关文件：
        {context}
        
        用户问题：{question}
        
        请根据以上信息回答：
        1. 如果找到了图片，请说明找到了多少张相关图片
        2. 回答要简洁明了
        3. 如果没有足够信息，请建议用户查看参考来源
        """
    else:
        prompt = f"""
        你是一个基于本地知识库的问答助手，以下是相关的知识库内容：
        {context}
        
        请根据以上内容回答用户的问题：{question}
        
        要求：
        1. 基于知识库内容回答，不要生成知识库外的信息
        2. 回答要准确、简洁
        3. 如果知识库中没有相关内容，请明确说明
        """
    
    answer = ""
    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt.strip(),
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
    except Exception as e:
        pass
    
    if not answer or answer.strip() == "" or "无法生成准确的答案" in answer or "抱歉，无法生成答案" in answer:
        if has_image_results:
            answer = f"在知识库中找到了 {len([r for r in enriched_results if r.get('type') == 'image'])} 张相关图片，请查看参考来源。"
        elif enriched_results:
            answer = f"在知识库中找到了 {len(enriched_results)} 个相关文件，请查看参考来源了解详情。"
        else:
            answer = "在知识库中未找到相关内容。"
    
    return {
        "answer": answer,
        "sources": enriched_results,
        "context": context,
        "has_images": has_image_results
    }
@router.post("/search/image")
async def search_image(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    gid: List[str] = Form(None),
    limit: int = 50,
    min_score: float = None,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """以图搜图接口"""
    # 使用配置默认值（如果未指定）
    if min_score is None:
        min_score = settings.SEARCH_MIN_SCORE
    try:
        temp_file_path = None
        if file:
            safe_filename = os.path.basename(file.filename).replace(os.sep, '_').replace('/', '_')
            temp_file_path = f"{settings.TEMP_DIR}/{safe_filename}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        elif base64_data:
            
            # MIME 类型到文件扩展名的映射
            mime_to_ext = {
                'image/png': 'png',
                'image/jpeg': 'jpg',
                'image/jpg': 'jpg',
                'image/gif': 'gif',
                'image/webp': 'webp',
                'image/bmp': 'bmp',
                'image/tiff': 'tif',
                'image/svg+xml': 'svg'
            }
            
            # 默认扩展名
            file_ext = 'png'
            
            # 从 data URL 中提取 MIME 类型
            mime_match = re.match(r'data:([^;]+);base64,', base64_data)
            if mime_match:
                mime_type = mime_match.group(1).lower()
                if mime_type in mime_to_ext:
                    file_ext = mime_to_ext[mime_type]
            
            # 提取 base64 数据
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_data = base64.b64decode(base64_data)
            file_name = f"{str(uuid.uuid4())}.{file_ext}"
            temp_file_path = f"{settings.TEMP_DIR}/{file_name}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或base64数据")
        # 使用CLIP模型从图片路径生成嵌入（与上传时使用相同的方法）
        clip_embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
        search_results = search_service.multimodal_search(db, clip_embedding, gid=gid, limit=limit, min_score=min_score)
        # 人脸检测（用于人脸搜索）
        from app.services.file_processor import FileProcessor
        file_processor = FileProcessor()
        result = file_processor.process_image(temp_file_path)
        if result.get("faces") and len(result["faces"]) > 0:
            from app.services.face_service import FaceService
            face_service = FaceService()
            for face_feature in result["faces"]:
                face_embedding = np.array(face_feature['embedding'])
                face_results = search_service.face_search(db, face_embedding, gid=gid, limit=limit, min_score=min_score)
                search_results.extend(face_results)
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return _enrich_search_results(db, search_results)
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"图片搜索失败: {str(e)}")

@router.post("/search/face")
async def search_face(
    file: UploadFile = File(None),
    base64_data: str = Form(None),
    gid: List[str] = Form(None),
    limit: int = 10,
    min_score: float = None,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """按人脸搜索接口"""
    # 使用配置默认值（如果未指定）
    if min_score is None:
        min_score = settings.FACE_SEARCH_THRESHOLD
    try:
        temp_file_path = None
        if file:
            safe_filename = os.path.basename(file.filename).replace(os.sep, '_').replace('/', '_')
            temp_file_path = f"{settings.TEMP_DIR}/{safe_filename}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        elif base64_data:
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            file_name = f"{uuid.uuid4()}.png"
            temp_file_path = f"{settings.TEMP_DIR}/{file_name}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或 base64 数据")
        from app.services.face_service import FaceService
        face_service = FaceService()
        face_features = face_service.extract_face_features(temp_file_path)
        if not face_features:
            return []
        all_results = []
        for face_feature in face_features:
            face_embedding = face_feature["embedding"]
            face_results = search_service.face_search(db, face_embedding, gid=gid, limit=limit, min_score=min_score)
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
        temp_file_path = None
        if file:
            safe_filename = os.path.basename(file.filename).replace(os.sep, '_').replace('/', '_')
            temp_file_path = f"{settings.TEMP_DIR}/{safe_filename}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        elif base64_data:
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            file_name = f"{uuid.uuid4()}.png"
            temp_file_path = f"{settings.TEMP_DIR}/{file_name}"
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供文件或 base64 数据")
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


@router.get("/stats")
async def get_stats(
    rid: str = None,
    gid: str = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """获取统计信息"""
    from app.models.db import File, TextChunk, ImageFrame, AudioTranscript
    from sqlalchemy import func
    query = db.query(File)
    if rid:
        query = query.filter(File.rid == rid)
    if gid is not None:
        query = query.filter(File.gid == gid)
    total_files = query.count()
    document_count = query.filter(File.type == "document").count()
    image_count = query.filter(File.type == "image").count()
    video_count = query.filter(File.type == "video").count()
    text_chunk_count = db.query(func.count(TextChunk.id))
    image_frame_count = db.query(func.count(ImageFrame.id))
    audio_transcript_count = db.query(func.count(AudioTranscript.id))
    if rid:
        text_chunk_count = text_chunk_count.join(TextChunk.file).filter(File.rid == rid)
        image_frame_count = image_frame_count.join(ImageFrame.file).filter(File.rid == rid)
        audio_transcript_count = audio_transcript_count.join(AudioTranscript.file).filter(File.rid == rid)
    if gid is not None:
        text_chunk_count = text_chunk_count.join(TextChunk.file).filter(File.gid == gid)
        image_frame_count = image_frame_count.join(ImageFrame.file).filter(File.gid == gid)
        audio_transcript_count = audio_transcript_count.join(AudioTranscript.file).filter(File.gid == gid)
    return {
        "total_files": total_files,
        "document_count": document_count,
        "image_count": image_count,
        "video_count": video_count,
        "text_chunk_count": text_chunk_count.scalar(),
        "image_frame_count": image_frame_count.scalar(),
        "audio_transcript_count": audio_transcript_count.scalar()
    }
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查接口"""
    health_status = {
        "status": "healthy",
        "components": {}
    }
    try:
        db.execute("SELECT 1")
        health_status["components"]["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = f"error: {str(e)}"
    try:
        vector_service = VectorService()
        if vector_service.client:
            health_status["components"]["qdrant"] = "connected"
        else:
            health_status["components"]["qdrant"] = "not configured"
    except Exception as e:
        health_status["components"]["qdrant"] = f"error: {str(e)}"
    return health_status
@router.delete("/file/{file_id}")
async def delete_file(
    file_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """删除文件及其相关数据"""
    try:
        from app.models.db import File, TextChunk, ImageFrame, AudioTranscript, Face
        file = db.query(File).filter(File.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")
        # 清理向量索引
        vector_service.delete_vectors_by_file_id(file_id)
        # 清理人脸向量索引
        try:
            from app.services.face_service import FaceService
            face_service = FaceService()
            faces = db.query(Face).filter(Face.file_id == file_id).all()
            for face in faces:
                face_service.face_index.remove(str(face.id))
            if faces:
                face_service.face_index.save()
        except Exception as e:
            pass  # 人脸索引清理失败不阻塞删除
        # 删除数据库记录
        db.query(TextChunk).filter(TextChunk.file_id == file_id).delete()
        db.query(ImageFrame).filter(ImageFrame.file_id == file_id).delete()
        db.query(AudioTranscript).filter(AudioTranscript.file_id == file_id).delete()
        db.delete(file)
        db.commit()
        # 保存向量索引
        vector_service.save_index()
        return {"success": True, "message": "文件删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")
@router.delete("/file/batch")
async def batch_delete_files(
    file_ids: List[str],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """批量删除文件"""
    try:
        from app.models.db import File, TextChunk, ImageFrame, AudioTranscript, Face
        from app.services.face_service import FaceService
        face_service = FaceService()
        deleted_count = 0
        any_face_deleted = False
        for file_id in file_ids:
            file = db.query(File).filter(File.id == file_id).first()
            if file:
                # 清理向量索引
                vector_service.delete_vectors_by_file_id(file_id)
                # 清理人脸向量索引
                try:
                    faces = db.query(Face).filter(Face.file_id == file_id).all()
                    for face in faces:
                        face_service.face_index.remove(str(face.id))
                    if faces:
                        any_face_deleted = True
                except Exception:
                    pass
                db.query(TextChunk).filter(TextChunk.file_id == file_id).delete()
                db.query(ImageFrame).filter(ImageFrame.file_id == file_id).delete()
                db.query(AudioTranscript).filter(AudioTranscript.file_id == file_id).delete()
                db.delete(file)
                deleted_count += 1
        db.commit()
        vector_service.save_index()
        if any_face_deleted:
            face_service.face_index.save()
        return {"success": True, "deleted_count": deleted_count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")
@router.delete("/file/by_rid")
async def delete_files_by_rid(
    rid: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """通过 rid 删除所有相关文件"""
    try:
        from app.models.db import File, TextChunk, ImageFrame, AudioTranscript, Face
        from app.services.face_service import FaceService
        face_service = FaceService()
        # 获取该 rid 下的所有文件
        files = db.query(File).filter(File.rid == rid).all()
        if not files:
            return {"success": True, "deleted_count": 0, "message": "没有找到该 rid 下的文件"}
        deleted_count = 0
        any_face_deleted = False
        for file in files:
            file_id = str(file.id)
            # 清理向量索引
            vector_service.delete_vectors_by_file_id(file_id)
            # 清理人脸向量索引
            try:
                faces = db.query(Face).filter(Face.file_id == file_id).all()
                for face in faces:
                    face_service.face_index.remove(str(face.id))
                if faces:
                    any_face_deleted = True
            except Exception:
                pass
            db.query(TextChunk).filter(TextChunk.file_id == file_id).delete()
            db.query(ImageFrame).filter(ImageFrame.file_id == file_id).delete()
            db.query(AudioTranscript).filter(AudioTranscript.file_id == file_id).delete()
            db.delete(file)
            deleted_count += 1
        db.commit()
        vector_service.save_index()
        if any_face_deleted:
            face_service.face_index.save()
        return {"success": True, "deleted_count": deleted_count, "message": f"成功删除 {deleted_count} 个文件"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
@router.post("/task/start")
async def start_task(
    task_type: str,
    params: Dict[str, Any] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """启动后台任务"""
    from datetime import datetime
    task_id = str(uuid.uuid4())
    task_id = await task_service.create_task(task_type, params or {})
    return {
        "task_id": task_id,
        "status": "pending",
        "created_at": None
    }
@router.get("/task/{task_id}")
async def get_task(
    task_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """获取任务状态"""
    task = await task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {
        "task": {
            "id": task.get("id", ""),
            "type": task.get("type", ""),
            "status": task.get("status", ""),
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
            "result": task.get("result"),
            "error": task.get("error"),
            "created_at": task.get("created_at"),
            "completed_at": task.get("completed_at")
        }
    }
@router.post("/task/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """取消任务"""
    success = await task_service.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或无法取消")
    return {"success": True, "message": "任务已取消"}

@router.get("/tasks")
async def get_tasks(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """????????"""
    tasks = await task_service.get_all_tasks()
    return {
        "tasks": [
            {
                "id": t.get("id", ""),
                "type": t.get("type", ""),
                "status": t.get("status", ""),
                "progress": t.get("progress", 0),
                "message": t.get("message", ""),
                "result": t.get("result"),
                "error": t.get("error"),
                "created_at": t.get("created_at"),
                "completed_at": t.get("completed_at")
            }
            for t in tasks
        ]
    }


@router.post("/import/folder")
async def import_folder(
    directory: str,
    rid: str = Form(None),
    gid: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """批量导入文件夹中的文件"""
    task_id = await task_service.create_task("folder_import", {"directory": directory, "rid": rid, "gid": gid})
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_in_executor(None, process_folder_import, task_id, directory, rid, gid)
    return {
        "task_id": task_id,
        "message": "文件夹导入任务已启动"
    }
def process_folder_import(task_id: str, directory: str, rid: str = None, gid: str = None):
    """在后台线程中处理文件夹导入"""
    from app.core.database import SessionLocal
    from app.services.task_service import TaskService
    from app.services.file_service import FileService
    from app.services.file_processor import FileProcessor
    from app.services.vector_service import VectorService
    from app.models.db import File, TextChunk, ImageFrame, AudioTranscript
    from datetime import datetime
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
            vector_service = VectorService()
            file_processor = FileProcessor()
            if file_type == "document":
                chunks = file_processor.process_document(temp_file_path, subformat)
                for i, chunk in enumerate(chunks):
                    embedding = vector_service.generate_text_embedding(chunk)
                    vector_service.add_vector(embedding, f"chunk_{db_file.id}_{i}")
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
                item_id = f"image_{db_file.id}"
                embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
                vector_service.add_vector(embedding, item_id)
                image_frame = ImageFrame(
                    id=str(uuid.uuid4()),
                    file_id=db_file.id,
                    frame_path=temp_file_path,
                    timestamp=0,
                    vector_id=item_id
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
                            vector_service.add_vector(embedding, f"video_audio_{db_file.id}_{j}")
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
            if temp_file_path:
                file_service.clean_temp_file(temp_file_path)
            vector_service.save_index()
            success_count += 1
        except Exception as e:
            print(f"处理文件失败: {file_info['name']}, 错误: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                file_service.clean_temp_file(temp_file_path)
            failed_count += 1
            failed_files.append({"name": file_info['name'], "error": str(e)})
            continue
    loop.run_until_complete(
        task_service.complete_task(task_id, {
            "total_files": total_files,
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_files": failed_files,
            "message": f"导入完成: 成功 {success_count}, 失败 {failed_count}"
        })
    )
    db.close()

@router.post("/file/batch_import_async")
async def batch_import_async(
    directory: str = Form(...),
    rid: str = Form(None),
    gid: Optional[str] = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """????????????"""
    task_id = await task_service.create_task("folder_import", {"directory": directory, "rid": rid, "gid": gid})
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_in_executor(None, process_folder_import, task_id, directory, rid, gid)
    return {
        "task_id": task_id,
        "message": "??????????"
    }


@router.post("/file/batch_import")
async def batch_import(
    directory: str = Form(...),
    rid: str = Form(None),
    gid: Optional[str] = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """????????????"""
    result_holder = {}

    def run_and_wait():
        import asyncio as _asyncio
        db2 = SessionLocal()
        try:
            _loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(_loop)
            task_id_local = _loop.run_until_complete(
                task_service.create_task("folder_import", {"directory": directory, "rid": rid, "gid": gid}))
            process_folder_import(task_id_local, directory, rid, gid)
            task = _loop.run_until_complete(task_service.get_task(task_id_local))
            if task and task.get("result"):
                result_holder["result"] = task["result"]
            else:
                result_holder["result"] = {"success_count": 0, "failed_count": 0, "failed_files": []}
        except Exception as e:
            result_holder["result"] = {"success_count": 0, "failed_count": 1, "failed_files": [{"error": str(e)}]}
        finally:
            db2.close()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_and_wait)
    return result_holder.get("result", {"success_count": 0, "failed_count": 0, "failed_files": []})


@router.post("/import/url")
async def import_from_url(
    url: str = Form(...),
    rid: str = Form(None),
    gid: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """从URL批量导入文件（支持动态URL）"""
    task_id = str(uuid.uuid4())
    task_service.create_task(db, task_id, "url_import", {"url": url, "rid": rid, "gid": gid})
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_in_executor(None, process_url_import, task_id, url, rid, gid)
    return {
        "task_id": task_id,
        "message": "URL导入任务已启动"
    }
def process_url_import(task_id: str, url: str, rid: str = None, gid: str = None):
    """在后台线程中处理URL导入"""
    from app.core.database import SessionLocal
    from app.services.task_service import TaskService
    from app.services.file_service import FileService
    from app.services.file_processor import FileProcessor
    from app.services.vector_service import VectorService
    from app.models.db import File, TextChunk, ImageFrame, AudioTranscript
    from datetime import datetime
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def run_async(coro):
        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            print(f"[URL导入] 异步操作失败: {e}")
            return None
    run_async(task_service.start_task(task_id))
    db = SessionLocal()
    success_count = 0
    failed_count = 0
    failed_files = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        allowed_extensions = {
            'document': ['pdf', 'doc', 'docx', 'rtf', 'odt', 'txt', 'ppt', 'pptx', 'pps', 'odp', 'xls', 'xlsx', 'ods', 'csv'],
            'image': ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'],
            'video': ['avi', 'rm', 'rmvb', 'mkv', 'mov', 'wmv', 'asf', 'mpg', 'mpe', 'mpeg', 'mp4', 'm4v', 'f4v', 'vob', 'ogv', 'mts', 'm2ts', '3gp', 'webm', 'flv', 'wav', 'vqf', 'ra', 'mxf']
        }
        run_async(task_service.update_task(task_id, progress=10, message="正在获取URL内容..."))
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            content_disposition = response.headers.get('content-disposition', '')
            if 'attachment' in content_disposition or 'filename' in content_disposition:
                match = re.search(r'filename[^;=\n]*=((["\']).*?\2|[^;\n]*)', content_disposition)
                if match:
                    file_name = match.group(1).strip('"\'')
                else:
                    file_name = f"downloaded_file"
            else:
                file_name = f"downloaded_file"
            file_ext = file_name.split(".")[-1].lower() if "." in file_name else ""
            file_type = None
            for type_name, extensions in allowed_extensions.items():
                if file_ext in extensions:
                    file_type = type_name
                    break
            if not file_type:
                if 'image' in content_type:
                    file_type = 'image'
                    file_ext = content_type.split('/')[-1].split(';')[0].lower()
                    if file_ext in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'webp']:
                        file_name = f"{file_name}.{file_ext}"
                elif 'video' in content_type:
                    file_type = 'video'
                    file_ext = content_type.split('/')[-1].split(';')[0].lower()
                    file_name = f"{file_name}.{file_ext}"
                elif 'application/pdf' in content_type:
                    file_type = 'document'
                    file_ext = 'pdf'
                    file_name = f"{file_name}.pdf"
                else:
                    file_type = 'document'
            if not file_name or "." not in file_name:
                file_name = f"downloaded_file.{file_ext}"
            temp_file_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}_{file_name}")
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(response.content)
            run_async(task_service.update_task(task_id, progress=30, message=f"正在处理文件: {file_name}"))
            file_size = os.path.getsize(temp_file_path)
            db_file = file_service.create_file(db, rid, gid, file_name, file_type, file_ext, file_size)
            if settings.KEEP_FILES:
                permanent_path = file_service.save_to_permanent_storage(temp_file_path, db_file.id, file_name)
                if permanent_path:
                    db_file.storage_path = permanent_path
                    db.commit()
            vector_service = VectorService()
            file_processor = FileProcessor()
            if file_type == "document":
                chunks = file_processor.process_document(temp_file_path, file_ext)
                for i, chunk in enumerate(chunks):
                    embedding = vector_service.generate_text_embedding(chunk)
                    vector_service.add_vector(embedding, f"chunk_{db_file.id}_{i}")
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
                item_id = f"image_{db_file.id}"
                embedding = vector_service.generate_image_embedding_from_path(temp_file_path)
                vector_service.add_vector(embedding, item_id)
                image_frame = ImageFrame(
                    id=str(uuid.uuid4()),
                    file_id=db_file.id,
                    frame_path=temp_file_path,
                    timestamp=0,
                    vector_id=item_id
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
                            vector_service.add_vector(embedding, f"video_audio_{db_file.id}_{j}")
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
            if temp_file_path:
                file_service.clean_temp_file(temp_file_path)
            vector_service.save_index()
            success_count = 1
            run_async(task_service.update_task(task_id, progress=100, message="导入完成"))
    except Exception as e:
        print(f"URL导入失败: {e}")
        failed_count = 1
        failed_files.append({"url": url, "error": str(e)})
    loop.run_until_complete(
        task_service.complete_task(task_id, {
            "total_files": 1,
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_files": failed_files,
            "message": f"URL导入完成: 成功 {success_count}, 失败 {failed_count}"
        })
    )
    db.close()
@router.post("/export")
async def export_data(
    export_type: str,
    rid: str = Form(None),
    gid: str = Form(None),
    file_type: str = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """导出数据"""
    import json
    from datetime import datetime
    try:
        from app.models.db import File, TextChunk, ImageFrame, AudioTranscript
        query = db.query(File)
        if rid:
            query = query.filter(File.rid == rid)
        if gid is not None:
            query = query.filter(File.gid == gid)
        if file_type:
            query = query.filter(File.type == file_type)
        files = query.all()
        export_data = []
        for file in files:
            file_data = {
                "id": str(file.id),
                "name": file.name,
                "type": file.type,
                "subformat": file.subformat,
                "size": file.size,
                "imported_at": file.imported_at.isoformat() if file.imported_at else None,
                "status": file.status
            }
            if export_type in ["all", "chunks"]:
                text_chunks = db.query(TextChunk).filter(TextChunk.file_id == file.id).all()
                file_data["chunks"] = [
                    {
                        "id": str(chunk.id),
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index
                    }
                    for chunk in text_chunks
                ]
            if export_type in ["all", "frames"]:
                image_frames = db.query(ImageFrame).filter(ImageFrame.file_id == file.id).all()
                file_data["frames"] = [
                    {
                        "id": str(frame.id),
                        "timestamp": frame.timestamp,
                        "frame_path": frame.frame_path
                    }
                    for frame in image_frames
                ]
            if export_type in ["all", "audio"]:
                audio_transcripts = db.query(AudioTranscript).filter(AudioTranscript.file_id == file.id).all()
                file_data["audio_transcripts"] = [
                    {
                        "id": str(transcript.id),
                        "content": transcript.content,
                        "start_time": transcript.start_time,
                        "end_time": transcript.end_time
                    }
                    for transcript in audio_transcripts
                ]
            export_data.append(file_data)
        export_dir = os.path.join(settings.DATA_DIR, "exports")
        os.makedirs(export_dir, exist_ok=True)
        export_file = os.path.join(export_dir, f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        return {
            "success": True,
            "export_file": export_file,
            "file_count": len(export_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")
