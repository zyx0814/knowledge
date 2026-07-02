import os
import json
import uuid
import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple
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


class FaceVectorIndex:
    """人脸向量 FAISS 索引（单例）

    使用 IndexIDMap(IndexFlatIP) 实现基于余弦相似度的人脸搜索。
    IndexFlatIP: 归一化向量的内积 = 余弦相似度，精确搜索。
    IndexIDMap: 支持 add_with_ids / remove_ids，通过外部 ID 直接删除。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceVectorIndex, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        import faiss
        self.dim = 512  # InsightFace 默认特征维度
        self.faiss = faiss
        self.face_index_path = os.path.join(settings.FACE_INDEX_PATH)
        os.makedirs(self.face_index_path, exist_ok=True)
        self.index_file = os.path.join(self.face_index_path, "face_index.faiss")
        self.id_map_file = os.path.join(self.face_index_path, "face_id_map.npy")
        self.id_map: Dict[int, str] = {}  # faiss_int_id → face_db_uuid
        self.index: Optional[faiss.Index] = None
        self._dirty = False
        self._lock = threading.Lock()
        self._load_or_create()
        self._initialized = True

    def _load_or_create(self):
        """从磁盘加载索引，失败则创建空索引（首次运行时从 DB 重建）"""
        if os.path.exists(self.index_file) and os.path.exists(self.id_map_file):
            try:
                self.index = self.faiss.read_index(self.index_file)
                loaded_map = np.load(self.id_map_file, allow_pickle=True).item()
                self.id_map = {int(k): str(v) for k, v in loaded_map.items()}
                print(f"[INFO] Face index loaded: {self.index.ntotal} faces")
                return
            except Exception as e:
                print(f"[WARN] Failed to load face index from disk: {e}")
        # 创建空索引
        self.index = self.faiss.IndexIDMap(self.faiss.IndexFlatIP(self.dim))
        self.id_map = {}
        self._dirty = True
        print("[INFO] Created new empty face index")

    def _faiss_id(self, face_db_id: str) -> int:
        """将 face DB UUID 字符串映射为 FAISS int64 ID"""
        return hash(face_db_id) % (2**63)

    def add(self, face_db_id: str, embedding: List[float]) -> bool:
        """添加一个人脸向量到索引（自动 L2 归一化）"""
        with self._lock:
            try:
                emb = np.array([embedding], dtype=np.float32)
                self.faiss.normalize_L2(emb)
                faiss_id = self._faiss_id(face_db_id)

                if faiss_id in self.id_map:
                    self.remove(face_db_id)

                self.index.add_with_ids(emb, np.array([faiss_id], dtype=np.int64))
                self.id_map[faiss_id] = face_db_id
                self._dirty = True
                return True
            except Exception as e:
                print(f"[ERROR] FaceVectorIndex.add failed: {e}")
                return False

    def add_batch(self, face_db_ids: List[str], embeddings: List[List[float]]) -> int:
        """批量添加人脸向量"""
        if not face_db_ids or not embeddings:
            return 0
        with self._lock:
            try:
                emb_array = np.array(embeddings, dtype=np.float32)
                self.faiss.normalize_L2(emb_array)
                faiss_ids = np.array([self._faiss_id(fid) for fid in face_db_ids], dtype=np.int64)

                for fid in face_db_ids:
                    faiss_id = self._faiss_id(fid)
                    if faiss_id in self.id_map:
                        self.remove(fid)

                self.index.add_with_ids(emb_array, faiss_ids)
                for i, fid in enumerate(face_db_ids):
                    self.id_map[int(faiss_ids[i])] = fid
                self._dirty = True
                return len(face_db_ids)
            except Exception as e:
                print(f"[ERROR] FaceVectorIndex.add_batch failed: {e}")
                return 0

    def search(self, embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """搜索最相似的 k 个人脸，返回 [(face_db_id, similarity), ...]"""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            try:
                emb = np.array([embedding], dtype=np.float32)
                self.faiss.normalize_L2(emb)
                actual_k = min(k, self.index.ntotal)
                distances, indices = self.index.search(emb, actual_k)

                results = []
                for i in range(len(indices[0])):
                    faiss_id = indices[0][i]
                    similarity = float(distances[0][i])
                    if faiss_id != -1 and faiss_id in self.id_map:
                        results.append((self.id_map[faiss_id], similarity))
                return results
            except Exception as e:
                print(f"[ERROR] FaceVectorIndex.search failed: {e}")
                return []

    def remove(self, face_db_id: str) -> bool:
        """从索引中删除一个人脸"""
        with self._lock:
            try:
                faiss_id = self._faiss_id(face_db_id)
                if faiss_id not in self.id_map:
                    return False
                selector = self.faiss.IDSelectorArray([faiss_id])
                self.index.remove_ids(selector)
                self.id_map.pop(faiss_id, None)
                self._dirty = True
                return True
            except Exception as e:
                print(f"[ERROR] FaceVectorIndex.remove failed: {e}")
                return False

    def remove_batch(self, face_db_ids: List[str]) -> int:
        """批量删除人脸"""
        with self._lock:
            count = 0
            try:
                faiss_ids = []
                for fid in face_db_ids:
                    faiss_id = self._faiss_id(fid)
                    if faiss_id in self.id_map:
                        faiss_ids.append(faiss_id)
                        self.id_map.pop(faiss_id, None)
                if faiss_ids:
                    selector = self.faiss.IDSelectorArray(faiss_ids)
                    self.index.remove_ids(selector)
                    count = len(faiss_ids)
                    self._dirty = True
            except Exception as e:
                print(f"[ERROR] FaceVectorIndex.remove_batch failed: {e}")
            return count

    def rebuild_from_db(self, db: Session):
        """从 Face 表全量重建索引（用于恢复/迁移）"""
        try:
            print("[INFO] Rebuilding face index from database...")
            all_faces = db.query(Face).all()

            # 创建新索引
            new_index = self.faiss.IndexIDMap(self.faiss.IndexFlatIP(self.dim))
            new_id_map: Dict[int, str] = {}

            if all_faces:
                embeddings = []
                faiss_ids = []
                for face in all_faces:
                    if not face.embedding:
                        continue
                    try:
                        emb = np.array(json.loads(face.embedding), dtype=np.float32)
                        embeddings.append(emb)
                        faiss_id = self._faiss_id(str(face.id))
                        faiss_ids.append(faiss_id)
                        new_id_map[faiss_id] = str(face.id)
                    except Exception:
                        continue

                if embeddings:
                    emb_array = np.stack(embeddings)
                    self.faiss.normalize_L2(emb_array)
                    faiss_id_array = np.array(faiss_ids, dtype=np.int64)
                    new_index.add_with_ids(emb_array, faiss_id_array)

            self.index = new_index
            self.id_map = new_id_map
            self._dirty = True
            self.save()
            print(f"[INFO] Face index rebuilt: {self.index.ntotal} faces")
        except Exception as e:
            print(f"[ERROR] Face index rebuild failed: {e}")
            import traceback
            traceback.print_exc()

    def save(self):
        """持久化索引到磁盘"""
        with self._lock:
            if not self._dirty:
                return
            try:
                self.faiss.write_index(self.index, self.index_file)
                clean_map = {int(k): str(v) for k, v in self.id_map.items()}
                np.save(self.id_map_file, clean_map)
                self._dirty = False
            except Exception as e:
                print(f"[ERROR] FaceVectorIndex.save failed: {e}")

    def count(self) -> int:
        """返回索引入脸数量"""
        if self.index is None:
            return 0
        return self.index.ntotal

    def ensure_loaded(self, db: Optional[Session] = None):
        """确保索引已加载（若为空则尝试从 DB 重建）"""
        if self.index is None or self.index.ntotal == 0:
            if db is not None:
                self.rebuild_from_db(db)


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
        # 人脸向量 FAISS 索引
        self.face_index = FaceVectorIndex()
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
        """查找相似人脸（使用 FAISS 索引，O(log n)）"""
        from sqlalchemy.orm import Session
        from app.core.database import get_db

        # 确保索引已加载
        self.face_index.ensure_loaded()

        # FAISS 搜索
        search_results = self.face_index.search(embedding, k=limit * 2)

        if not search_results:
            return []

        # 收集匹配的 face_db_id 和相似度
        face_scores: Dict[str, float] = {}
        matched_ids = []
        for face_db_id, similarity in search_results:
            if similarity >= threshold:
                matched_ids.append(face_db_id)
                face_scores[face_db_id] = similarity

        if not matched_ids:
            return []

        # 批量查询 DB 获取完整信息
        db = next(get_db())
        try:
            faces = db.query(Face).filter(Face.id.in_(matched_ids)).all()
            similar_faces = []
            for face in faces:
                similar_faces.append({
                    "face_id": str(face.id),
                    "feature_id": str(face.id),
                    "file_id": str(face.file_id) if face.file_id else "",
                    "image_path": face.image_path,
                    "similarity": face_scores.get(str(face.id), 0.0),
                    "confidence": face.confidence
                })
            # 按相似度排序
            similar_faces.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_faces[:limit]
        finally:
            db.close()
    def save_face_features(self, db: Session, file_id: str, face_features: List[Dict[str, Any]]) -> List[str]:
        """保存人脸特征到数据库（使用 FAISS 索引查找重复，O(log n)）"""
        saved_face_ids = []
        similarity_threshold = 0.6  # 相似度阈值

        # 确保索引已加载
        self.face_index.ensure_loaded(db)

        for feature in face_features:
            embedding = feature["embedding"]

            # 使用 FAISS 索引查找最近邻（O(log n) vs 原先的 O(n)）
            existing_match = None
            if self.face_index.count() > 0:
                search_results = self.face_index.search(embedding, k=1)
                if search_results and search_results[0][1] >= similarity_threshold:
                    existing_match = search_results[0][0]  # face_db_id

            # 如果找到相似度高于阈值的人脸，跳过添加
            if existing_match:
                saved_face_ids.append(existing_match)
                # 删除已生成的人脸图片
                if feature.get("image_path"):
                    try:
                        image_path = feature["image_path"]
                        if image_path.startswith("minio://") and STORAGE_SERVICE_AVAILABLE:
                            object_name = image_path.replace("minio://", "").split("/", 1)[-1]
                            storage_service.delete_file(object_name)
                        elif os.path.exists(image_path):
                            os.remove(image_path)
                    except Exception:
                        pass
                continue

            # 生成新人脸ID并创建记录
            face_id = str(uuid.uuid4())
            face = Face(
                id=face_id,
                name=None,
                group_id=None,
                file_id=file_id,
                image_path=feature["image_path"],
                embedding=json.dumps(embedding),
                confidence=feature["confidence"],
                bbox=json.dumps(feature["bbox"])
            )
            db.add(face)
            saved_face_ids.append(face_id)

            # 添加到 FAISS 索引
            self.face_index.add(face_id, embedding)

        db.commit()
        # 保存索引到磁盘
        self.face_index.save()
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
        """删除人脸（同时清理 FAISS 索引）"""
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
            except Exception:
                pass
        # 从 FAISS 索引中移除
        self.face_index.remove(face_id)
        # 删除人脸记录
        db.delete(face)
        db.commit()
        self.face_index.save()
        return True
