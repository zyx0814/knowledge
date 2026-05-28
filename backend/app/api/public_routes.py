import io
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.db import Face, File, ImageFrame
from app.services.storage_service import storage_service
from config.config import settings
public_router = APIRouter()
@public_router.get("/faces/{face_id}/image")
async def get_face_image(
    face_id: str,
    db: Session = Depends(get_db)
):
    """获取人脸图片（公开访问，不受认证限制，支持 MinIO 存储）"""
    face = db.query(Face).filter(Face.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="人脸不存在")
    if not face.image_path:
        raise HTTPException(status_code=404, detail="人脸图片不存在")
    # 检查是否是 MinIO 存储路径
    if face.image_path.startswith("minio://"):
        # 解析 MinIO 对象路径
        object_name = face.image_path.replace("minio://", "").split("/", 1)[-1]
        try:
            response = storage_service.client.get_object(
                bucket_name=storage_service.bucket,
                object_name=object_name
            )
            file_content = response.read()
            return Response(content=file_content, media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=404, detail="人脸图片不存在")
    # 本地路径处理
    if not os.path.exists(face.image_path):
        raise HTTPException(status_code=404, detail="人脸图片不存在")
    return FileResponse(face.image_path, media_type="image/jpeg")
@public_router.get("/file")
async def get_file_by_path(
    path: str = Query(..., description="文件路径，支持本地路径和 MinIO 路径（minio://bucket/object_name）")
):
    """通过路径获取文件（公开访问，支持 MinIO 存储）"""
    # 检查是否是 MinIO 存储路径
    if path.startswith("minio://"):
        # 解析 MinIO 对象路径
        # 格式: minio://bucket/object_name
        try:
            path_parts = path.replace("minio://", "").split("/", 1)
            if len(path_parts) != 2:
                raise HTTPException(status_code=400, detail="无效的 MinIO 路径格式")
            bucket_name = path_parts[0]
            object_name = path_parts[1]
            # 从 MinIO 获取文件内容
            if storage_service.client:
                response = storage_service.client.get_object(
                    bucket_name=bucket_name,
                    object_name=object_name
                )
                file_content = response.read()
                ext = os.path.splitext(object_name)[1].lower()
                media_type = "application/octet-stream"
                if ext in [".jpg", ".jpeg"]:
                    media_type = "image/jpeg"
                elif ext == ".png":
                    media_type = "image/png"
                elif ext == ".webp":
                    media_type = "image/webp"
                elif ext == ".bmp":
                    media_type = "image/bmp"
                return Response(content=file_content, media_type=media_type)
            else:
                raise HTTPException(status_code=500, detail="MinIO 服务不可用")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail="文件不存在")
    # 本地路径处理
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    ext = os.path.splitext(path)[1].lower()
    media_type = "application/octet-stream"
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"
    elif ext == ".bmp":
        media_type = "image/bmp"
    return FileResponse(path, media_type=media_type)
@public_router.get("/preview/{file_id}")
async def preview_file(
    file_id: str,
    db: Session = Depends(get_db)
):
    """预览文件（优先显示图片，支持 MinIO 存储）"""
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    if file.type == "image":
        if file.storage_path:
            # 检查是否是 MinIO 存储路径
            if file.storage_path.startswith("minio://"):
                # 解析 MinIO 对象路径
                # 格式: minio://bucket/object_name
                object_name = file.storage_path.replace("minio://", "").split("/", 1)[-1]
                # 从 MinIO 获取文件内容
                try:
                    response = storage_service.client.get_object(
                        bucket_name=storage_service.bucket,
                        object_name=object_name
                    )
                    file_content = response.read()
                    ext = os.path.splitext(object_name)[1].lower()
                    media_type = "image/jpeg"
                    if ext == ".png":
                        media_type = "image/png"
                    elif ext == ".webp":
                        media_type = "image/webp"
                    return Response(content=file_content, media_type=media_type)
                except Exception as e:
                    raise HTTPException(status_code=500, detail="无法获取 MinIO 文件")
            elif os.path.exists(file.storage_path):
                ext = os.path.splitext(file.storage_path)[1].lower()
                media_type = "image/jpeg"
                if ext == ".png":
                    media_type = "image/png"
                elif ext == ".webp":
                    media_type = "image/webp"
                return FileResponse(file.storage_path, media_type=media_type)
        image_frame = db.query(ImageFrame).filter(ImageFrame.file_id == file_id).first()
        if image_frame and image_frame.frame_path:
            if image_frame.frame_path.startswith("minio://"):
                object_name = image_frame.frame_path.replace("minio://", "").split("/", 1)[-1]
                try:
                    response = storage_service.client.get_object(
                        bucket_name=storage_service.bucket,
                        object_name=object_name
                    )
                    file_content = response.read()
                    ext = os.path.splitext(object_name)[1].lower()
                    media_type = "image/jpeg"
                    if ext == ".png":
                        media_type = "image/png"
                    elif ext == ".webp":
                        media_type = "image/webp"
                    return Response(content=file_content, media_type=media_type)
                except Exception as e:
                    raise HTTPException(status_code=500, detail="无法获取 MinIO 文件")
            elif os.path.exists(image_frame.frame_path):
                ext = os.path.splitext(image_frame.frame_path)[1].lower()
                media_type = "image/jpeg"
                if ext == ".png":
                    media_type = "image/png"
                elif ext == ".webp":
                    media_type = "image/webp"
                return FileResponse(image_frame.frame_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="无法预览此文件")