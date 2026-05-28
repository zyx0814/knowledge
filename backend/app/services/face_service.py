import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.db import Face
from config.config import settings
from app.core.gpu_utils import gpu_manager
# 导入 MinIO 存储服务
try:
    from app.services.storage_service import storage_service
    STORAGE_SERVICE_AVAILABLE = True
except ImportError:
    STORAGE_SERVICE_AVAILABLE = False
class FaceService:
    """人脸服务类"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        """初始化人脸服务"""
        if self._initialized:
            return
        self.face_db_path = os.path.join(settings.DATA_DIR, "faces")
        os.makedirs(self.face_db_path, exist_ok=True)
        self.face_embedding_dim = 512  # InsightFace默认特征维度
        self.insightface_available = False
        # 延迟导入InsightFace，避免启动时依赖问题
        self.face_analyzer = None
        self._init_face_analyzer()
        self._initialized = True

    def _init_face_analyzer(self):
        """初始化人脸分析器 - 自动选择 GPU/CPU"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # 获取 ONNX Runtime providers
            use_gpu = getattr(settings, 'USE_GPU', False)
            providers = gpu_manager.get_onnx_providers(prefer_gpu=use_gpu)
            # InsightFace会在 root/models/name 路径查找模型
            root_path = settings.FACE_MODEL_PATH
            model_path = os.path.join(root_path, "models", settings.FACE_MODEL_NAME)
            # 检查模型是否已存在（需要检查实际模型文件）
            model_exists = False
            if os.path.exists(model_path):
                # 检查是否有模型文件
                model_files = ['2d106det.onnx', 'det_10g.onnx', 'genderage.onnx', '1k3d68.onnx', 'w600k_r50.onnx']
                model_exists = all(os.path.exists(os.path.join(model_path, f)) for f in model_files)
            if model_exists:
                # 设置环境变量，强制使用本地模型路径
                os.environ['INSIGHTFACE_HOME'] = root_path
                self.face_analyzer = FaceAnalysis(
                    name=settings.FACE_MODEL_NAME,
                    root=root_path,
                    providers=providers
                )
            else:
                self.face_analyzer = FaceAnalysis(
                    name=settings.FACE_MODEL_NAME,
                    providers=providers
                )
            # ctx_id: -1=CPU, 0=GPU0
            ctx_id = 0 if use_gpu and gpu_manager.cuda_available else -1
            # 初始化模型（如果模型不存在会抛出异常）
            self.face_analyzer.prepare(
                ctx_id=ctx_id, 
                det_size=(640, 640)
            )
            self.insightface_available = True
        except ImportError as e:
            self.face_analyzer = None
            self.insightface_available = False
        except RuntimeError as e:
            self.face_analyzer = None
            self.insightface_available = False
        except Exception as e:
            self.face_analyzer = None
            self.insightface_available = False

    def _read_image(self, image_path: str):
        """读取图片，支持本地路径和 MinIO 路径"""
        import cv2
        import numpy as np
        if image_path.startswith("minio://") and STORAGE_SERVICE_AVAILABLE:
            # 从 MinIO 读取图片
            try:
                object_name = image_path.replace("minio://", "").split("/", 1)[-1]
                response = storage_service.client.get_object(
                    bucket_name=storage_service.bucket,
                    object_name=object_name
                )
                image_data = response.read()
                # 将二进制数据转换为 OpenCV 图片
                img_array = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
            except Exception as e:
                return None
        else:
            # 从本地读取图片
            return cv2.imread(image_path)

    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """检测图片中的人脸"""
        if not self.insightface_available or not self.face_analyzer:
            return []
        try:
            import cv2
            # 读取图片（支持 MinIO）
            img = self._read_image(image_path)
            if img is None:
                return []
            # 检测人脸
            faces = self.face_analyzer.get(img)
        
            # 处理检测结果
            result = []
            for i, face in enumerate(faces):
                if face is None:
                    continue
                # 检查 bbox
                if not hasattr(face, 'bbox') or face.bbox is None:
                    continue
                # 获取置信度
                if not hasattr(face, 'det_score') or face.det_score is None:
                    confidence = 0.0
                else:
                    confidence = float(face.det_score)
                # 获取 embedding（必需）
                if not hasattr(face, 'embedding') or face.embedding is None:
                    continue
                embedding = face.embedding.tolist()
                # 获取 landmark（可选）
                if not hasattr(face, 'landmark') or face.landmark is None:
                    landmark = []
                else:
                    landmark = face.landmark.tolist()
                result.append({
                    "bbox": face.bbox.tolist(),
                    "confidence": confidence,
                    "embedding": embedding,
                    "landmark": landmark
                })
            return result
        except Exception as e:
           
            return []
    def extract_face_features(self, image_path: str) -> List[Dict[str, Any]]:
        """提取图片中的人脸特征"""
        # 检测人脸
        faces = self.detect_faces(image_path)
        # 对每个检测到的人脸提取特征
        result = []
        for face in faces:
            # 保存人脸裁剪图
            face_crop_path = self._save_face_crop(image_path, face["bbox"])
            result.append({
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "embedding": face["embedding"],
                "image_path": face_crop_path
            })
        return result
    def _save_face_crop(self, image_path: str, bbox: List[float]) -> str:
        """保存人脸裁剪图（支持 MinIO 存储）"""
        try:
            import cv2
            import tempfile
            # 读取图片（支持 MinIO）
            img = self._read_image(image_path)
            if img is None:
                return ""
            # 裁剪人脸
            x1, y1, x2, y2 = map(int, bbox)
            # 扩展边界，确保包含完整人脸
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img.shape[1], x2 + margin)
            y2 = min(img.shape[0], y2 + margin)
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                return ""
            # 生成人脸ID
            face_id = str(uuid.uuid4())
            # 先保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            cv2.imwrite(temp_path, face_crop)
            # 优先上传到 MinIO（如果可用）
            if STORAGE_SERVICE_AVAILABLE and storage_service.is_available():
                try:
                    object_name = f"faces/{face_id[:2]}/{face_id[2:4]}/{face_id}.jpg"
                    storage_service.client.fput_object(
                        bucket_name=storage_service.bucket,
                        object_name=object_name,
                        file_path=temp_path
                    )
                    # 删除临时文件
                    os.remove(temp_path)
                    return f"minio://{storage_service.bucket}/{object_name}"
                except Exception as e:
                    pass
            # 降级到本地存储
            face_crop_path = os.path.join(self.face_db_path, f"{face_id}.jpg")
            # 移动临时文件到永久位置
            import shutil
            shutil.move(temp_path, face_crop_path)
            return face_crop_path
        except Exception as e:
            return ""
    def find_similar_faces(self, embedding: List[float], threshold: float = 0.6, limit: int = 10) -> List[Dict[str, Any]]:
        """查找相似人脸（人脸和特征合并后）"""
        from sqlalchemy.orm import Session
        from app.core.database import get_db
        db = next(get_db())
        try:
            # 获取所有人脸（现在Face表直接包含特征）
            all_faces = db.query(Face).all()
            # 计算相似度
            import numpy as np
            query_embedding = np.array(embedding)
            similar_faces = []
            for face in all_faces:
                # 解析嵌入向量
                try:
                    # 检查embedding属性是否存在且不为空
                    if not hasattr(face, 'embedding') or not face.embedding:
                        continue
                    face_embedding = np.array(json.loads(face.embedding))
                    # 计算余弦相似度
                    similarity = np.dot(query_embedding, face_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(face_embedding)
                    )
                    # 过滤相似度高于阈值的人脸
                    if similarity >= threshold:
                        similar_faces.append({
                            "face_id": face.id,
                            "feature_id": face.id,  # 人脸和特征合并后，ID相同
                            "file_id": face.file_id,
                            "image_path": face.image_path,
                            "similarity": float(similarity),
                            "confidence": face.confidence
                        })
                except Exception as e:
                    continue
            # 按相似度排序并限制结果数量
            similar_faces.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_faces[:limit]
        finally:
            db.close()
    def save_face_features(self, db: Session, file_id: str, face_features: List[Dict[str, Any]]) -> List[str]:
        """保存人脸特征到数据库（人脸和特征合并）"""
        saved_face_ids = []
        similarity_threshold = 0.6  # 相似度阈值
        for feature in face_features:
            # 检查是否存在相似人脸
            existing_face_id = None
            max_similarity = 0.0
            # 获取所有人脸（现在Face表直接包含特征）
            all_faces = db.query(Face).all()
            import numpy as np
            current_embedding = np.array(feature["embedding"])
            for existing_face in all_faces:
                try:
                    # 检查embedding属性是否存在且不为空
                    if not hasattr(existing_face, 'embedding') or not existing_face.embedding:
                        continue
                    existing_embedding = np.array(json.loads(existing_face.embedding))
                    # 计算余弦相似度
                    similarity = np.dot(current_embedding, existing_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(existing_embedding)
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity
                        existing_face_id = existing_face.id
                except Exception as e:
                    continue
            # 如果找到相似度高于阈值的人脸，跳过添加
            if max_similarity >= similarity_threshold and existing_face_id:
                saved_face_ids.append(existing_face_id)
                # 删除已生成的人脸图片
                if feature.get("image_path"):
                    try:
                        image_path = feature["image_path"]
                        if image_path.startswith("minio://") and STORAGE_SERVICE_AVAILABLE:
                            # 从 MinIO 删除
                            object_name = image_path.replace("minio://", "").split("/", 1)[-1]
                            storage_service.delete_file(object_name)
                        elif os.path.exists(image_path):
                            os.remove(image_path)
                    except Exception as e:
                        pass
                continue
            # 生成新人脸ID并创建记录（人脸和特征合并）
            face_id = str(uuid.uuid4())
            face = Face(
                id=face_id,
                name=None,  # 初始无名称
                group_id=None,  # 初始无分组
                file_id=file_id,
                image_path=feature["image_path"],
                embedding=json.dumps(feature["embedding"]),
                confidence=feature["confidence"],
                bbox=json.dumps(feature["bbox"])
            )
            db.add(face)
            saved_face_ids.append(face_id)
        db.commit()
        return saved_face_ids
    def merge_faces(self, db: Session, face_ids: List[str], new_name: str = None) -> str:
        """合并相似人脸"""
        # 处理逗号分隔的ID字符串
        processed_ids = []
        for face_id in face_ids:
            if ',' in face_id:
                processed_ids.extend([id.strip() for id in face_id.split(',')])
            else:
                processed_ids.append(face_id)
        # 生成新的人脸组ID
        group_id = str(uuid.uuid4())
        # 更新人脸记录，设置相同的group_id
        for face_id in processed_ids:
            face = db.query(Face).filter(Face.id == face_id).first()
            if face:
                face.group_id = group_id
                if new_name:
                    face.name = new_name
        db.commit()
        return group_id
    def get_face_list(self, db: Session, skip: int = 0, limit: int = 100, name: str = None) -> List[Dict[str, Any]]:
        """获取人脸列表（人脸和特征合并后）"""
        # 构建查询
        query = db.query(Face)
        # 按人名模糊查询
        if name:
            query = query.filter(Face.name.contains(name))
        faces = query.offset(skip).limit(limit).all()
        result = []
        for face in faces:
            result.append({
                "id": face.id,
                "name": face.name,
                "group_id": face.group_id,
                "file_id": face.file_id,
                "image_path": face.image_path,
                "confidence": face.confidence,
                "bbox": json.loads(face.bbox) if face.bbox else [],
                "created_at": face.created_at,
                "updated_at": face.updated_at
            })
        return result
    def get_face_details(self, db: Session, face_id: str) -> Dict[str, Any]:
        """获取人脸详情（人脸和特征合并后）"""
        face = db.query(Face).filter(Face.id == face_id).first()
        if not face:
            return {}
        return {
            "id": face.id,
            "name": face.name,
            "group_id": face.group_id,
            "file_id": face.file_id,
            "image_path": face.image_path,
            "confidence": face.confidence,
            "bbox": json.loads(face.bbox) if face.bbox else [],
            "created_at": face.created_at,
            "updated_at": face.updated_at
        }
    def update_face_name(self, db: Session, face_id: str, name: str) -> bool:
        """更新人脸名称"""
        face = db.query(Face).filter(Face.id == face_id).first()
        if not face:
            return False
        face.name = name
        db.commit()
        return True
    def delete_face(self, db: Session, face_id: str) -> bool:
        """删除人脸（人脸和特征合并后）"""
        # 删除人脸记录
        face = db.query(Face).filter(Face.id == face_id).first()
        if not face:
            return False
        # 删除人脸裁剪图
        if face.image_path:
            try:
                image_path = face.image_path
                if image_path.startswith("minio://") and STORAGE_SERVICE_AVAILABLE:
                    # 从 MinIO 删除
                    object_name = image_path.replace("minio://", "").split("/", 1)[-1]
                    storage_service.delete_file(object_name)
                elif os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                pass
        # 删除人脸记录
        db.delete(face)
        db.commit()
        return True
