import io
import os
import uuid
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from config.config import settings
from app.core.gpu_utils import gpu_manager
# 导入 MinIO 存储服务
try:
    from app.services.storage_service import storage_service
    STORAGE_SERVICE_AVAILABLE = True
except ImportError:
    STORAGE_SERVICE_AVAILABLE = False
# 全局CLIP模型（用于跨模态检索）
_clip_model = None
_clip_preprocess = None
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
    try:
        import torch
        from clip import clip
        device = "cuda" if gpu_manager.cuda_available else "cpu"
        model_name = "ViT-B/32"
        # 首先尝试从本地目录加载
        local_model_path = os.path.join(settings.MODELS_DIR, 'clip', model_name)
        local_clip_path = os.path.join(local_model_path, 'clip.pt')
        if os.path.exists(local_clip_path):
            try:
                # 加载本地模型
                model_state_dict = torch.load(local_clip_path, map_location=device)
                # 创建模型
                from clip.model import CLIP, build_model
                _clip_model = build_model(model_state_dict['model_state_dict']).to(device)
                _clip_model.eval()
                # 创建预处理
                from clip.simple_tokenizer import SimpleTokenizer
                _clip_preprocess = clip._transform(model_state_dict.get('input_resolution', 224))
                return _clip_model, _clip_preprocess
            except Exception:
                pass
        # 默认方式加载（自动下载）
        _clip_model, _clip_preprocess = clip.load(model_name, device=device)
        # 保存到本地目录以便下次使用
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
        self._save_interval = 1000
        self._add_count = 0
        self.use_gpu = getattr(settings, 'USE_GPU', False) and gpu_manager.faiss_gpu_available
        self.gpu_resources = None
        self.gpu_index = None
        self.cpu_index = None
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
        dim = settings.EMBEDDING_DIM
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
        """生成文本嵌入向量（使用CLIP）"""
        global _clip_model
        clip_model, _ = _init_clip_model()
        if clip_model is not None:
            try:
                import torch
                from clip import clip
                device = "cuda" if gpu_manager.cuda_available else "cpu"
                text_tokens = clip.tokenize([text]).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_tokens)
                # 归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().flatten().tolist()
            except Exception as e:
                pass
        # 降级到随机向量
        return np.random.rand(settings.EMBEDDING_DIM).tolist()

    def generate_image_embedding_from_path(self, image_path: str) -> List[float]:
        """从图片路径生成图片嵌入向量（使用CLIP，支持 MinIO）"""
        global _clip_model, _clip_preprocess
        clip_model, clip_preprocess = _init_clip_model()
        if clip_model is not None and clip_preprocess is not None:
            try:
                import torch
                device = "cuda" if gpu_manager.cuda_available else "cpu"
                # 使用统一的图片读取函数（支持 MinIO）
                image = _read_image_from_path(image_path)
                if image is None:
                    return np.random.rand(settings.EMBEDDING_DIM).tolist()
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy().flatten().tolist()
            except Exception as e:
                pass
        # 降级到简单特征（像素均值）- 这会导致语义搜索失败！
        try:
            # 使用统一的图片读取函数（支持 MinIO）
            img = _read_image_from_path(image_path)
            if img is None:
                return np.random.rand(settings.EMBEDDING_DIM).tolist()
            img = img.resize((224, 224))
            img_array = np.array(img)
            features = img_array.mean(axis=(0, 1)).tolist()
            embedding = np.zeros(settings.EMBEDDING_DIM)
            embedding[:len(features)] = features
            return embedding.tolist()
        except Exception as e:
            return np.random.rand(settings.EMBEDDING_DIM).tolist()

    def generate_image_embedding(self, features: List[float]) -> List[float]:
        """生成图片嵌入向量（兼容旧接口）"""
        embedding = np.zeros(settings.EMBEDDING_DIM)
        embedding[:len(features)] = features
        return embedding.tolist()

    def generate_video_embedding(self, frame_features: List[List[float]], audio_text: str) -> List[float]:
        """生成视频嵌入向量"""
        if not frame_features:
            if audio_text:
                return self.generate_text_embedding(audio_text)
            else:
                return np.random.rand(settings.EMBEDDING_DIM).tolist()
        frame_mean = np.mean(frame_features, axis=0)
        if audio_text:
            audio_embedding = self.generate_text_embedding(audio_text)
            embedding = (frame_mean + np.array(audio_embedding)) / 2
        else:
            embedding = frame_mean
        return embedding.tolist()

    def add_vector(self, vector: List[float], item_id: str, auto_save: bool = False, 
                   file_id: str = None, vector_type: str = None, gid: str = None) -> int:
        """添加单个向量到向量库"""
        vector_np = np.array([vector], dtype=np.float32)
        idx = self.index.ntotal
        if hasattr(self.index, 'train') and not self.index.is_trained and idx > 0:
            self.index.train(vector_np)
        self.index.add(vector_np)
        self.id_map[idx] = item_id
        self._dirty = True
        self._add_count += 1
        if auto_save and self._add_count % self._save_interval == 0:
            self._save_index()
        return idx

    def add_vectors_batch(self, vectors: List[List[float]], item_ids: List[str],
                          file_ids: List[str] = None, vector_types: List[str] = None) -> List[int]:
        """批量添加向量（性能优化）"""
        if not vectors or not item_ids or len(vectors) != len(item_ids):
            return []
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
        if self._add_count % self._save_interval >= len(vectors):
            self._save_index()
        return indices

    def _save_index(self):
        """保存索引到磁盘"""
        if not self._dirty:
            return
        try:
            self._save_to_cpu()
            faiss.write_index(self.cpu_index, self.index_path)
            np.save(self.id_map_path, self.id_map)
            self._dirty = False
        except Exception as e:
            pass

    def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if self.index.ntotal == 0:
            return []
        query_np = np.array([query_vector], dtype=np.float32)
        try:
            distances, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
        except Exception as e:
            return []
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            if idx != -1 and idx in self.id_map:
                results.append({
                    "id": self.id_map[idx],
                    "distance": float(distance),
                    "score": float(1.0 / (1.0 + distance))
                })
        return results

    def delete_vector(self, item_id: str) -> bool:
        """删除指定向量"""
        indices_to_remove = [idx for idx, id_val in self.id_map.items() if id_val == item_id]
        if not indices_to_remove:
            return False
        for idx in sorted(indices_to_remove, reverse=True):
            del self.id_map[idx]
        # 需要重建索引
        self._rebuild_index()
        return True

    def _rebuild_index(self):
        """重建索引"""
        if self.index.ntotal == 0:
            return
        try:
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            new_index = self._create_index(len(self.id_map))
            if hasattr(new_index, 'train') and not new_index.is_trained:
                new_index.train(vectors)
            valid_indices = sorted(self.id_map.keys())
            valid_vectors = vectors[valid_indices]
            new_index.add(valid_vectors)
            self.index = new_index
            self.cpu_index = new_index
            self.gpu_index = None
            if self.use_gpu:
                self._move_to_gpu()
            self._dirty = True
        except Exception as e:
            pass

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
        self._save_index()

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
            # 检查并创建集合
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
            
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
        """生成文本嵌入向量（使用CLIP）"""
        global _clip_model
        clip_model, _ = _init_clip_model()
        if clip_model is not None:
            try:
                import torch
                from clip import clip
                device = "cuda" if gpu_manager.cuda_available else "cpu"
                text_tokens = clip.tokenize([text]).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().flatten().tolist()
            except Exception as e:
                pass
        return np.random.rand(settings.EMBEDDING_DIM).tolist()

    def generate_image_embedding_from_path(self, image_path: str) -> List[float]:
        """从图片路径生成图片嵌入向量（使用CLIP，支持 MinIO）"""
        global _clip_model, _clip_preprocess
        clip_model, clip_preprocess = _init_clip_model()
        if clip_model is not None and clip_preprocess is not None:
            try:
                import torch
                device = "cuda" if gpu_manager.cuda_available else "cpu"
                # 使用统一的图片读取函数（支持 MinIO）
                image = _read_image_from_path(image_path)
                if image is None:
                    return np.random.rand(settings.EMBEDDING_DIM).tolist()
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy().flatten().tolist()
            except Exception as e:
                pass
        # 降级到简单特征（像素均值）- 这会导致语义搜索失败！
        try:
            # 使用统一的图片读取函数（支持 MinIO）
            img = _read_image_from_path(image_path)
            if img is None:
                return np.random.rand(settings.EMBEDDING_DIM).tolist()
            img = img.resize((224, 224))
            img_array = np.array(img)
            features = img_array.mean(axis=(0, 1)).tolist()
            embedding = np.zeros(settings.EMBEDDING_DIM)
            embedding[:len(features)] = features
            return embedding.tolist()
        except Exception as e:
            return np.random.rand(settings.EMBEDDING_DIM).tolist()

    def generate_image_embedding(self, features: List[float]) -> List[float]:
        """生成图片嵌入向量（兼容旧接口）"""
        embedding = np.zeros(settings.EMBEDDING_DIM)
        embedding[:len(features)] = features
        return embedding.tolist()

    def generate_video_embedding(self, frame_features: List[List[float]], audio_text: str) -> List[float]:
        """生成视频嵌入向量"""
        if not frame_features:
            if audio_text:
                return self.generate_text_embedding(audio_text)
            else:
                return np.random.rand(settings.EMBEDDING_DIM).tolist()
        frame_mean = np.mean(frame_features, axis=0)
        if audio_text:
            audio_embedding = self.generate_text_embedding(audio_text)
            embedding = (frame_mean + np.array(audio_embedding)) / 2
        else:
            embedding = frame_mean
        return embedding.tolist()

    def add_vector(self, vector: List[float], item_id: str, auto_save: bool = False,
                   file_id: str = None, vector_type: str = None, gid: str = None) -> str:
        """添加向量（带元数据）"""
        if not self.client:
            return item_id
        try:
            point = self.PointStruct(
                id=abs(hash(item_id)) % (2**63),
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
                    id=abs(hash(item_id)) % (2**63),
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
    def get_vector_count(self) -> int:
        """获取向量数量"""
        return self.backend.get_vector_count()
    def close(self):
        """关闭服务"""
        self.backend.close()
