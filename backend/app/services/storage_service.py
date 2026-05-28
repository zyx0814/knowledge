import os
from typing import Optional
from config.config import settings
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
class StorageService:
    """MinIO 对象存储服务"""
    def __init__(self):
        """初始化 MinIO 客户端"""
        self.client = None
        self.bucket = "knowledge-files"
        self.use_minio = getattr(settings, 'USE_MINIO', False)
        if MINIO_AVAILABLE and self.use_minio:
            try:
                minio_host = getattr(settings, 'MINIO_HOST', 'localhost')
                minio_port = getattr(settings, 'MINIO_PORT', 9000)
                minio_access_key = getattr(settings, 'MINIO_ACCESS_KEY', 'minioadmin')
                minio_secret_key = getattr(settings, 'MINIO_SECRET_KEY', 'minioadmin')
                minio_secure = getattr(settings, 'MINIO_SECURE', False)
                self.client = Minio(
                    f"{minio_host}:{minio_port}",
                    access_key=minio_access_key,
                    secret_key=minio_secret_key,
                    secure=minio_secure
                )
                self._ensure_bucket()
            except Exception as e:
                self.client = None

    def _ensure_bucket(self):
        """确保 bucket 存在"""
        if not self.client:
            return
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except Exception:
            pass

    def is_available(self) -> bool:
        """检查 MinIO 是否可用"""
        return self.client is not None

    def upload_file(self, local_path: str, file_id: str, file_name: str) -> Optional[str]:
        """上传文件到 MinIO，返回对象路径"""
        if not self.client:
            return None
        try:
            ext = file_name.rsplit('.', 1)[-1].lower() if '.' in file_name else ''
            object_name = f"{file_id[:2]}/{file_id[2:4]}/{file_id}.{ext}" if ext else f"{file_id[:2]}/{file_id[2:4]}/{file_id}"
            self.client.fput_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=local_path
            )
            return object_name
        except Exception as e:
            return None

    def download_file(self, object_name: str, local_path: str) -> bool:
        """从 MinIO 下载文件到本地"""
        if not self.client:
            return False
        try:
            self.client.fget_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=local_path
            )
            return True
        except Exception as e:
            return False

    def get_presigned_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """获取预签名下载链接"""
        if not self.client:
            return None
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name=self.bucket,
                object_name=object_name,
                expires=timedelta(seconds=expires)
            )
            return url
        except Exception as e:
            return None

    def delete_file(self, object_name: str) -> bool:
        """删除 MinIO 中的文件"""
        if not self.client:
            return False
        try:
            self.client.remove_object(
                bucket_name=self.bucket,
                object_name=object_name
            )
            return True
        except Exception as e:
            return False

    def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在"""
        if not self.client:
            return False
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False
        except Exception as e:
            return False
            
storage_service = StorageService()
