import io
import os
import uuid
import hashlib
import numpy as np
import faiss
import threading
from typing import List, Dict, Any, Optional
from config.config import settings
from app.core.gpu_utils import gpu_manager
from app.services.embedding_registry import (
    generate_text_embedding as _reg_text_embed,
    generate_image_embedding_from_path as _reg_image_embed_path,
    generate_image_embedding as _reg_image_embed,
    generate_video_embedding as _reg_video_embed,
)

def stable_hash(item_id: str) -> int:
    """生成稳定的哈希值，用于Qdrant的point id"""
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (2**63)
# 导入 MinIO 存储服务
try:
    from app.services.storage_service import storage_service
    STORAGE_SERVICE_AVAILABLE = True
except ImportError:
    STORAGE_SERVICE_AVAILABLE = False
# 全局CLIP模型（用于跨模态检索）
_clip_model = None
_clip_preprocess = None
_clip_model_lock = threading.Lock()
def _read_image_from_path(image_path: str):
    """从路径读取图片，支持本地路径和 MinIO 路径"""
    from PIL import Image
    if image_path.startswith("minio://") and STORAGE_SERVICE_AVAILABLE:
        # 从 MinIO 读取图片
        try:
            object_name = image_path.replace("minio://", "").split("/", 1)[-1]
            response = storage_service.client.get_object(
                bucket_name=storage_service.bucket,
                object_name=object_name
            )
            image_data = response.read()
            # 将二进制数据转换为 PIL 图片
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            return None
    else:
        # 从本地读取图片
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            return None

def _init_clip_model():
    """初始化CLIP模型（支持本地模型和离线模式）"""
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    with _clip_model_lock:
        if _clip_model is not None:
            return _clip_model, _clip_preprocess
        try:
            import torch
            from clip import clip
            device = "cuda" if gpu_manager.cuda_available else "cpu"
            model_name = "ViT-B/32"
            local_model_path = os.path.join(settings.MODELS_DIR, 'clip', model_name)
            local_clip_path = os.path.join(local_model_path, 'clip.pt')
            if os.path.exists(local_clip_path):
                try:
                    model_state_dict = torch.load(local_clip_path, map_location=device)
                    from clip.model import CLIP, build_model
                    _clip_model = build_model(model_state_dict['model_state_dict']).to(device)
                    _clip_model.eval()
                    from clip.simple_tokenizer import SimpleTokenizer
                    _clip_preprocess = clip._transform(model_state_dict.get('input_resolution', 224))
                    return _clip_model, _clip_preprocess
                except Exception:
                    pass
            _clip_model, _clip_preprocess = clip.load(model_name, device=device)
            save_path = os.path.join(settings.MODELS_DIR, 'clip', model_name)
            os.makedirs(save_path, exist_ok=True)
            save_clip_path = os.path.join(save_path, 'clip.pt')
            try:
                torch.save({
                    'model_state_dict': _clip_model.state_dict(),
                    'input_resolution': _clip_model.visual.input_resolution,
                    'context_length': _clip_model.context_length,
                    'vocab_size': _clip_model.vocab_size
                }, save_clip_path)
            except Exception:
                pass
            return _clip_model, _clip_preprocess
        except ImportError:
            return None, None
        except RuntimeError:
            raise
        except Exception:
            return None, None

class FAISSVectorService:
    """FAISS 向量服务实现"""
    def __init__(self):
        """初始化FAISS向量库"""
        self.vector_db_path = os.path.join(settings.VECTOR_DB_PATH)
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.index_path = os.path.join(self.vector_db_path, "faiss_index.bin")
        self.id_map_path = os.path.join(self.vector_db_path, "id_map.npy")
        self._pending_vectors = []
        self._pending_ids = []
        self._dirty = False
        self._add_count = 0
        # 从配置中读取保存间隔和批量大小
        self._save_interval = getattr(settings, 'VECTOR_INDEX_BATCH_SIZE', 100)
        self._auto_save_interval = getattr(settings, 'VECTOR_INDEX_SAVE_INTERVAL', 30)
        # Tombstone 删除机制：删除时只标记，搜索时过滤，避免全量重建
        self._deleted_indices: set = set()
        self._tombstone_threshold = 0.1  # 删除比例超过 10% 时触发物理压缩
        self.use_gpu = getattr(settings, 'USE_GPU', False) and gpu_manager.faiss_gpu_available
        self.gpu_resources = None
        self.gpu_index = None
        self.cpu_index = None
        self._lock = threading.Lock()
        # 定时保存相关
        self._last_save_time = 0
        self._save_thread = None
        self._shutdown_event = threading.Event()
        if os.path.exists(self.index_path):
            self.cpu_index = faiss.read_index(self.index_path)
            self.id_map = np.load(self.id_map_path, allow_pickle=True).item()
            if self.use_gpu:
                self._move_to_gpu()
            self.index = self.gpu_index if self.gpu_index else self.cpu_index
        else:
            self.cpu_index = self._create_index(0)
            self.id_map = {}
            if self.use_gpu:
                self._move_to_gpu()
            self.index = self.gpu_index if self.gpu_index else self.cpu_index
        # 启动自动保存线程
        self._start_auto_save()

    def _move_to_gpu(self):
        """将索引移动到 GPU"""
        if not gpu_manager.faiss_gpu_available:
            return
        try:
            self.gpu_resources = gpu_manager.get_faiss_gpu_resources()
            if self.gpu_resources and self.cpu_index:
                self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.cpu_index)
        except Exception:
            self.gpu_index = None
            self.gpu_resources = None

    def _save_to_cpu(self):
        """确保有 CPU 索引用于保存"""
        if self.gpu_index and self.gpu_resources:
            try:
                self.cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
            except Exception as e:
                pass

    def _create_index(self, n_vectors: int = 0) -> faiss.Index:
        """根据数据量创建最优索引"""
        dim = settings._resolve_embedding_dim()
        if n_vectors < 10000:
            return faiss.IndexFlatL2(dim)
        elif n_vectors < 500000:
            nlist = min(int(4 * np.sqrt(n_vectors)), 4096)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            return index
        else:
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
            return index

    def _maybe_rebuild_index(self):
        """检查是否需要重建索引以适应数据规模"""
        n_vectors = self.index.ntotal
        current_type = type(self.index).__name__
        needs_rebuild = False
        if n_vectors >= 500000 and current_type != 'IndexHNSWFlat':
            needs_rebuild = True
        elif n_vectors >= 10000 and n_vectors < 500000 and current_type != 'IndexIVFFlat':
            needs_rebuild = True
        if needs_rebuild:
            new_index = self._create_index(n_vectors)
            if self.index.ntotal > 0:
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
                if hasattr(new_index, 'train') and not new_index.is_trained:
                    new_index.train(vectors)
                new_index.add(vectors)
            self.index = new_index
            self._dirty = True
    def generate_text_embedding(self, text: str) -> List[float]:
        """??????????????????"""
        return _reg_text_embed(text)

    def generate_image_embedding_from_path(self, image_path: str) -> List[float]:
        """????????????????????????? MinIO?"""
        return _reg_image_embed_path(image_path)

    def generate_image_embedding(self, features: List[float]) -> List[float]:
        """???????????????"""
        return _reg_image_embed(features)

    def generate_video_embedding(self, frame_features: List[List[float]], audio_text: str) -> List[float]:
        """????????"""
        return _reg_video_embed(frame_features, audio_text)



    def add_vector(self, vector: List[float], item_id: str, auto_save: bool = False, 
                   file_id: str = None, vector_type: str = None, gid: str = None) -> int:
        """添加单个向量到向量库"""
        with self._lock:
            vector_np = np.array([vector], dtype=np.float32)
            idx = self.index.ntotal
            if hasattr(self.index, 'train') and not self.index.is_trained and idx > 0:
                self.index.train(vector_np)
            self.index.add(vector_np)
            self.id_map[idx] = item_id
            self._dirty = True
            self._add_count += 1
            # 移除自动保存逻辑，改用定时保存
            return idx

    def add_vectors_batch(self, vectors: List[List[float]], item_ids: List[str],
                          file_ids: List[str] = None, vector_types: List[str] = None) -> List[int]:
        """批量添加向量（性能优化）"""
        if not vectors or not item_ids or len(vectors) != len(item_ids):
            return []
        with self._lock:
            vectors_np = np.array(vectors, dtype=np.float32)
            start_idx = self.index.ntotal
            if hasattr(self.index, 'train') and not self.index.is_trained and start_idx == 0:
                self.index.train(vectors_np)
            self.index.add(vectors_np)
            indices = []
            for i, item_id in enumerate(item_ids):
                self.id_map[start_idx + i] = item_id
                indices.append(start_idx + i)
            self._dirty = True
            self._add_count += len(vectors)
            # 移除自动保存逻辑，改用定时保存
            return indices

    def _start_auto_save(self):
        """启动自动保存线程"""
        import time
        def auto_save_worker():
            while not self._shutdown_event.is_set():
                time.sleep(self._auto_save_interval)
                if self._dirty:
                    self._save_index()
        self._save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self._save_thread.start()

    def _save_index(self, force: bool = False):
        """保存索引到磁盘（保存前自动压缩 tombstone）"""
        with self._lock:
            if not self._dirty and not force:
                return
            try:
                # 保存前先压缩，确保磁盘数据干净
                self._maybe_compact()
                self._save_to_cpu()
                faiss.write_index(self.cpu_index, self.index_path)
                np.save(self.id_map_path, self.id_map)
                self._dirty = False
                import time
                self._last_save_time = time.time()
            except Exception as e:
                pass

    def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似向量（自动过滤 tombstone 标记的已删除项）"""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            query_np = np.array([query_vector], dtype=np.float32)
            try:
                expanded_k = min(top_k + len(self._deleted_indices) + 10, self.index.ntotal)
                distances, indices = self.index.search(query_np, expanded_k)
            except Exception as e:
                return []
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                distance = distances[0][i]
                if idx == -1 or idx in self._deleted_indices:
                    continue
                if idx in self.id_map:
                    results.append({
                        "id": self.id_map[idx],
                        "distance": float(distance),
                        "score": float(1.0 / (1.0 + distance))
                    })
                if len(results) >= top_k:
                    break
            return results

    def delete_vector(self, item_id: str) -> bool:
        """删除指定向量（tombstone 标记，O(k)，k=匹配数）"""
        with self._lock:
            indices_to_remove = [idx for idx, id_val in self.id_map.items() if id_val == item_id]
            if not indices_to_remove:
                return False
            for idx in indices_to_remove:
                del self.id_map[idx]
                self._deleted_indices.add(idx)
            self._dirty = True
            self._maybe_compact()
            return True

    def delete_vectors_by_file_id(self, file_id: str) -> int:
        """根据 file_id 删除所有相关向量（tombstone 标记，避免全量重建）"""
        with self._lock:
            indices_to_remove = []
            for idx, id_val in self.id_map.items():
                if id_val.startswith(f"chunk_{file_id}_") or \
                   id_val.startswith(f"image_{file_id}") or \
                   id_val.startswith(f"video_frame_{file_id}_") or \
                   id_val.startswith(f"video_audio_{file_id}_"):
                    indices_to_remove.append(idx)
            if not indices_to_remove:
                return 0
            for idx in indices_to_remove:
                del self.id_map[idx]
                self._deleted_indices.add(idx)
            self._dirty = True
            self._maybe_compact()
            return len(indices_to_remove)

    def _maybe_compact(self):
        """检查 tombstone 比例，超过阈值时触发物理重建"""
        total = self.index.ntotal
        if total == 0:
            return
        deleted_count = len(self._deleted_indices)
        ratio = deleted_count / total
        if ratio >= self._tombstone_threshold:
            print(f"[INFO] Tombstone ratio {ratio:.2%} >= {self._tombstone_threshold:.0%}, compacting index...")
            self._rebuild_index()
            self._deleted_indices.clear()
            print(f"[INFO] Index compacted: {self.index.ntotal} vectors remaining")

    def _rebuild_index(self):
        """物理重建索引（排除 tombstone 项）"""
        if self.index.ntotal == 0:
            return
        try:
            # 读取所有向量
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            # 只保留未被删除的索引
            valid_indices = sorted([idx for idx in self.id_map.keys() if idx not in self._deleted_indices])
            if not valid_indices:
                # 全部被删除，创建新空索引
                self.cpu_index = self._create_index(0)
                self.id_map = {}
                self._deleted_indices.clear()
                self.gpu_index = None
                if self.use_gpu:
                    self._move_to_gpu()
                self.index = self.gpu_index if self.gpu_index else self.cpu_index
                self._dirty = True
                return

            valid_vectors = vectors[valid_indices]
            new_index = self._create_index(len(valid_indices))
            if hasattr(new_index, 'train') and not new_index.is_trained:
                new_index.train(valid_vectors)
            new_index.add(valid_vectors)

            # 更新 id_map（重映射内部索引）
            new_id_map = {}
            for new_idx, old_idx in enumerate(valid_indices):
                new_id_map[new_idx] = self.id_map[old_idx]

            self.index = new_index
            self.cpu_index = new_index
            self.id_map = new_id_map
            self._deleted_indices.clear()
            self.gpu_index = None
            if self.use_gpu:
                self._move_to_gpu()
                self.index = self.gpu_index if self.gpu_index else self.cpu_index
            self._dirty = True
        except Exception as e:
            print(f"[ERROR] Index rebuild failed: {e}")
            import traceback
            traceback.print_exc()

    def get_vector_count(self) -> int:
        """获取向量数量"""
        return self.index.ntotal

    def begin_batch(self):
        """开始批量添加模式"""
        pass

    def add_to_batch(self, vector: List[float], item_id: str):
        """添加到批量队列"""
        self._pending_vectors.append(vector)
        self._pending_ids.append(item_id)

    def commit_batch(self) -> List[int]:
        """提交批量添加"""
        if not self._pending_vectors:
            return []
        result = self.add_vectors_batch(self._pending_vectors, self._pending_ids)
        self._pending_vectors = []
        self._pending_ids = []
        return result

    def close(self):
        """关闭服务，保存索引"""
        # 停止自动保存线程
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=5.0)
        # 强制保存一次
        self._save_index(force=True)

class QdrantVectorService:
    """Qdrant 向量服务实现"""
    def __init__(self):
        """初始化Qdrant客户端"""
        self.client = None
        self.collection_name = "knowledge_vectors"
        self.PointStruct = None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, VectorParams, Distance
            host = getattr(settings, 'QDRANT_HOST', 'localhost')
            port = getattr(settings, 'QDRANT_PORT', 6333)
            self.client = QdrantClient(host=host, port=port)
            self.PointStruct = PointStruct
            # 检查并创建集合（处理并发创建的竞态条件）
            try:
                if not self.client.collection_exists(self.collection_name):
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=settings._resolve_embedding_dim(),
                            distance=Distance.COSINE
                        )
                    )
            except Exception as create_e:
                # 如果集合已存在（并发创建时可能发生），忽略此错误
                if "already exists" in str(create_e):
                    pass
                else:
                    raise
            
        except ImportError:
           print("qdrant-client not installed")
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def _extract_file_id(self, item_id: str) -> str:
        """从item_id中提取file_id"""
        parts = item_id.split('_')
        if len(parts) >= 3 and parts[0] in ['chunk', 'image', 'video']:
            return parts[1]
        return ""

    def _extract_vector_type(self, item_id: str) -> str:
        """从item_id中提取向量类型"""
        if item_id.startswith('chunk_'):
            return 'text'
        elif item_id.startswith('image_'):
            return 'image'
        elif item_id.startswith('video_'):
            return 'video'
        return 'unknown'

    def generate_text_embedding(self, text: str) -> List[float]:
        """??????????????????"""
        return _reg_text_embed(text)

    def generate_image_embedding_from_path(self, image_path: str) -> List[float]:
        """????????????????????????? MinIO?"""
        return _reg_image_embed_path(image_path)

    def generate_image_embedding(self, features: List[float]) -> List[float]:
        """???????????????"""
        return _reg_image_embed(features)

    def generate_video_embedding(self, frame_features: List[List[float]], audio_text: str) -> List[float]:
        """????????"""
        return _reg_video_embed(frame_features, audio_text)



    def add_vector(self, vector: List[float], item_id: str, auto_save: bool = False,
                   file_id: str = None, vector_type: str = None, gid: str = None) -> str:
        """??????????"""
        if not self.client:
            return item_id
        try:
            point = self.PointStruct(
                id=stable_hash(item_id),
                vector=vector,
                payload={
                    "item_id": item_id,
                    "file_id": file_id if file_id else self._extract_file_id(item_id),
                    "vector_type": vector_type if vector_type else self._extract_vector_type(item_id),
                    "gid": gid
                }
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            return item_id
        except Exception as e:
            return item_id

    def add_vectors_batch(self, vectors: List[List[float]], item_ids: List[str],
                          file_ids: List[str] = None, vector_types: List[str] = None) -> List[str]:
        """批量添加（推荐用于导入）"""
        if not self.client:
            return item_ids
        try:
            points = []
            for i, (item_id, vector) in enumerate(zip(item_ids, vectors)):
                fid = file_ids[i] if file_ids else self._extract_file_id(item_id)
                vtype = vector_types[i] if vector_types else self._extract_vector_type(item_id)
                point = self.PointStruct(
                    id=stable_hash(item_id),
                    vector=vector,
                    payload={"item_id": item_id, "file_id": fid, "vector_type": vtype}
                )
                points.append(point)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return item_ids
        except Exception as e:
            return item_ids

    def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似向量（兼容不同版本的 Qdrant 客户端）"""
        if not self.client:
            return []
        try:
            # 尝试使用新版本的 query_points 方法
            if hasattr(self.client, 'query_points'):
                from qdrant_client.http.models import PointIdsList, Filter, SearchRequest
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k
                )
                return [
                    {
                        "id": hit.payload.get("item_id", str(hit.id)),
                        "file_id": hit.payload.get("file_id", ""),
                        "score": float(hit.score),
                        "distance": float(1.0 / (1.0 + float(hit.score)))
                    }
                    for hit in search_result.points
                ]
            # 尝试使用旧版本的 search 方法
            elif hasattr(self.client, 'search'):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
                return [
                    {
                        "id": hit.payload.get("item_id", str(hit.id)),
                        "file_id": hit.payload.get("file_id", ""),
                        "score": hit.score,
                        "distance": float(1.0 / (1.0 + hit.score))
                    }
                    for hit in results
                ]
            else:
                return []
        except Exception as e:
            return []

    def delete_vector(self, item_id: str) -> bool:
        """删除指定向量"""
        if not self.client:
            return False
        try:
            point_id = abs(hash(item_id)) % (2**63)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            return True
        except Exception as e:
            return False
    
    def delete_vectors_by_file_id(self, file_id: str) -> int:
        """根据 file_id 删除所有相关向量"""
        if not self.client:
            return 0
        try:
            # 使用条件过滤删除所有匹配的向量
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="file_id",
                        match=MatchValue(value=file_id)
                    )
                ]
            )
            
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            # 返回删除的数量
            if hasattr(result, 'deleted_count'):
                return result.deleted_count
            return 0
        except Exception as e:
            return 0

    def get_vector_count(self) -> int:
        """获取向量数量（兼容不同版本的 Qdrant 客户端）"""
        if not self.client:
            return 0
        try:
            info = self.client.get_collection(self.collection_name)
            # 兼容不同版本的 Qdrant 客户端
            if hasattr(info, 'vectors_count'):
                return info.vectors_count
            elif hasattr(info, 'points_count'):
                return info.points_count
            elif hasattr(info, 'status') and hasattr(info.status, 'count'):
                return info.status.count
            else:
                return 0
        except Exception as e:
            return 0

    def begin_batch(self):
        """开始批量添加模式"""
        pass

    def add_to_batch(self, vector: List[float], item_id: str):
        """添加到批量队列"""
        self.add_vector(vector, item_id)

    def commit_batch(self) -> List[int]:
        """提交批量添加"""
        return []

    def close(self):
        """关闭服务"""
        pass
class VectorService:
    """统一向量服务接口（单例模式）"""
    _instance = None
    def __new__(cls):
        """单例模式：确保整个应用只有一个VectorService实例"""
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
            # 只在第一次创建时初始化
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化向量服务（只执行一次）"""
        if self._initialized:
            return
        use_qdrant = getattr(settings, 'USE_QDRANT', False)
        if use_qdrant:
            self.backend = QdrantVectorService()
        else:
            self.backend = FAISSVectorService()
        self._initialized = True

    def generate_text_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        return self.backend.generate_text_embedding(text)
        
    def generate_image_embedding(self, features: List[float]) -> List[float]:
        """生成图片嵌入向量"""
        return self.backend.generate_image_embedding(features)
    def generate_image_embedding_from_path(self, image_path: str) -> List[float]:
        """从图片路径生成图片嵌入向量"""
        if hasattr(self.backend, 'generate_image_embedding_from_path'):
            return self.backend.generate_image_embedding_from_path(image_path)
        return self.backend.generate_image_embedding([])
    def generate_video_embedding(self, frame_features: List[List[float]], audio_text: str) -> List[float]:
        """生成视频嵌入向量"""
        return self.backend.generate_video_embedding(frame_features, audio_text)
    def add_vector(self, vector: List[float], item_id: str, auto_save: bool = False) -> int:
        """添加单个向量到向量库"""
        return self.backend.add_vector(vector, item_id, auto_save)
    def add_vectors_batch(self, vectors: List[List[float]], item_ids: List[str]) -> List[int]:
        """批量添加向量"""
        return self.backend.add_vectors_batch(vectors, item_ids)
    def begin_batch(self):
        """开始批量添加模式"""
        self.backend.begin_batch()
    def add_to_batch(self, vector: List[float], item_id: str):
        """添加到批量队列"""
        self.backend.add_to_batch(vector, item_id)
    def commit_batch(self) -> List[int]:
        """提交批量添加"""
        return self.backend.commit_batch()
    def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        return self.backend.search_vectors(query_vector, top_k)
    def save_index(self):
        """保存索引到磁盘"""
        if hasattr(self.backend, '_save_index'):
            self.backend._save_index()
        elif hasattr(self.backend, 'save_index'):
            self.backend.save_index()
    def delete_vector(self, item_id: str) -> bool:
        """删除指定向量"""
        return self.backend.delete_vector(item_id)
    
    def delete_vectors_by_file_id(self, file_id: str) -> int:
        """根据 file_id 删除所有相关向量"""
        return self.backend.delete_vectors_by_file_id(file_id)
    
    def get_vector_count(self) -> int:
        """获取向量数量"""
        return self.backend.get_vector_count()
    def close(self):
        """关闭服务"""
        self.backend.close()
