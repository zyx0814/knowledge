from typing import List, Dict, Any, Union
from sqlalchemy.orm import Session
from app.models.db import File, TextChunk, ImageFrame, AudioTranscript
from app.services.vector_service import VectorService
from config.config import settings
class SearchService:
    """搜索服务类"""
    def __init__(self):
        """初始化搜索服务"""
        self.vector_service = VectorService()
        # 从配置读取默认相似度阈值
        self.default_min_score = settings.SEARCH_MIN_SCORE
        self.default_face_threshold = settings.FACE_SEARCH_THRESHOLD
    def _normalize_gid(self, gid: Union[str, List[Any], None]) -> List[str]:
        """标准化 gid 参数，支持单个字符串、逗号分隔字符串和数组"""
        if gid is None:
            return []
        if isinstance(gid, str):
            return [g.strip() for g in gid.split(',') if g.strip()]
        if isinstance(gid, list):
            normalized = []
            for g in gid:
                str_g = str(g).strip()
                if ',' in str_g:
                    normalized.extend([item.strip() for item in str_g.split(',') if item.strip()])
                else:
                    normalized.append(str_g)
            return normalized
        return []
    def _extract_file_id_from_vector_id(self, vector_id: str) -> str:
        """从向量ID中提取file_id"""
        vector_id = str(vector_id)
        # 更具体的模式要放在前面
        if vector_id.startswith("image_text_"):
            # image_text_fileid -> fileid (fileid可能包含下划线)
            return vector_id[len("image_text_"):]
        elif vector_id.startswith("video_frame_text_"):
            # video_frame_text_fileid_idx -> fileid (fileid可能包含下划线)
            parts = vector_id.split("_")
            return "_".join(parts[3:-1]) if len(parts) > 4 else "_".join(parts[3:])
        elif vector_id.startswith("video_frame_"):
            # video_frame_fileid_idx -> fileid (fileid可能包含下划线)
            parts = vector_id.split("_")
            return "_".join(parts[2:-1]) if len(parts) > 3 else "_".join(parts[2:])
        elif vector_id.startswith("video_audio_"):
            # video_audio_fileid_idx -> fileid (fileid可能包含下划线)
            parts = vector_id.split("_")
            return "_".join(parts[2:-1]) if len(parts) > 3 else "_".join(parts[2:])
        elif vector_id.startswith("chunk_"):
            # chunk_fileid_idx -> fileid (fileid可能包含下划线)
            parts = vector_id.split("_")
            return "_".join(parts[1:-1]) if len(parts) > 2 else "_".join(parts[1:])
        elif vector_id.startswith("doc_"):
            # doc_fileid_chunkidx -> fileid (fileid可能包含下划线)
            parts = vector_id.split("_")
            return "_".join(parts[1:-1]) if len(parts) > 2 else "_".join(parts[1:])
        elif vector_id.startswith("image_"):
            # image_fileid -> fileid (fileid可能包含下划线)
            return vector_id[len("image_"):]
        elif vector_id.startswith("video_"):
            # video_fileid -> fileid (fileid可能包含下划线)
            return vector_id[len("video_"):]
        return ""
        
    def _dedup_and_limit_by_file_id(self, results: List[Dict[str, Any]], limit: int, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """按file_id去重、过滤最小分数并限制数量"""
        file_results = {}
        for result in results:
            file_id = result.get("file_id", "")
            score = float(result.get("score", 0.0))
            search_type = result.get("search_type", "unknown")
            timestamp = result.get("timestamp")
            if not file_id:
                continue
            if score < min_score:
                continue
            if file_id not in file_results:
                file_results[file_id] = {
                    "file_id": file_id,
                    "score": score,
                    "search_type": search_type,
                    "timestamps": []  # 存储多个时间戳和对应的分数
                }
            else:
                if score > file_results[file_id]["score"]:
                    file_results[file_id]["score"] = score
                    file_results[file_id]["search_type"] = search_type
            # 收集时间戳信息（支持多个视频切片）
            if timestamp is not None:
                file_results[file_id]["timestamps"].append({
                    "timestamp": float(timestamp),
                    "score": float(score)
                })
        # 对每个文件的时间戳按分数排序
        for file_id in file_results:
            if file_results[file_id]["timestamps"]:
                file_results[file_id]["timestamps"].sort(key=lambda x: x["score"], reverse=True)
        sorted_list = sorted(file_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_list[:limit]

    def keyword_search(self, db: Session, query: str, file_type: str = None, gid: Union[str, List[str], None] = None, limit: int = 10, min_score: float = None) -> List[Dict[str, Any]]:
        """关键词检索"""
        results = []
        gid_list = self._normalize_gid(gid)
        file_query = db.query(File)
        if file_type:
            file_query = file_query.filter(File.type == file_type)
        if gid_list:
            file_query = file_query.filter(File.gid.in_(gid_list))
        files = file_query.filter(
            File.name.contains(query) |
            File.category.contains(query)
        ).limit(limit * 2).all()
        for file in files:
            results.append({
                "file_id": str(file.id),
                "score": 1.0,
                "search_type": "keyword"
            })
        text_query = db.query(TextChunk).join(File)
        if file_type:
            text_query = text_query.filter(File.type == file_type)
        if gid_list:
            text_query = text_query.filter(File.gid.in_(gid_list))
        chunks = text_query.filter(TextChunk.content.contains(query)).limit(limit * 2).all()
        for chunk in chunks:
            results.append({
                "file_id": str(chunk.file_id),
                "score": 0.9,
                "search_type": "keyword"
            })
        # 使用配置默认值（如果未指定）
        if min_score is None:
            min_score = self.default_min_score
        return self._dedup_and_limit_by_file_id(results, limit, min_score)

    def semantic_search(self, db: Session, query: str, file_type: str = None, gid: Union[str, List[str], None] = None, limit: int = 10, min_score: float = None) -> List[Dict[str, Any]]:
        """语义检索"""
        query_vector = self.vector_service.generate_text_embedding(query)
        vector_count = self.vector_service.get_vector_count()
        if vector_count == 0:
            return []
        gid_list = self._normalize_gid(gid)
        if gid_list:
            gid_file_count = db.query(File).filter(File.gid.in_(gid_list)).count()
            top_k = max(limit * 5, gid_file_count + limit)
        else:
            top_k = limit * 3
        vector_results = self.vector_service.search_vectors(query_vector, top_k=top_k)
        results = []
        for result in vector_results:
            vector_id = str(result["id"])
            distance = result.get("distance", 0.0)
            score = result.get("score", 0.0)
            file_id = self._extract_file_id_from_vector_id(vector_id)
            if not file_id:
                continue
            file_type_from_id = None
            is_video_frame = False
            if vector_id.startswith("chunk_"):
                file_type_from_id = "document"
            elif vector_id.startswith("image_") or vector_id.startswith("image_text_"):
                file_type_from_id = "image"
            elif vector_id.startswith("video_") or vector_id.startswith("video_frame_") or vector_id.startswith("video_audio_"):
                file_type_from_id = "video"
                is_video_frame = True
            if file_type and file_type_from_id != file_type:
                continue
            if gid_list:
                file = db.query(File).filter(File.id == file_id).first()
                if not file or file.gid not in gid_list:
                    continue
            score = result.get("score")
            if score is not None:
                score = float(score)
            else:
                distance = float(result.get("distance", 0.0))
                score = float(1.0 / (1.0 + distance))
            result_item = {
                "file_id": file_id,
                "score": score,
                "search_type": "semantic"
            }
            # 如果是视频切片，获取时间戳
            if is_video_frame and (vector_id.startswith("video_frame_") or vector_id.startswith("video_frame_text_")):
                frame_info = db.query(ImageFrame).filter(
                    ImageFrame.file_id == file_id,
                    ImageFrame.vector_id == vector_id
                ).first()
                if frame_info and frame_info.timestamp is not None:
                    result_item["timestamp"] = float(frame_info.timestamp)
            results.append(result_item)
        # 使用配置默认值（如果未指定）
        if min_score is None:
            min_score = 0.0
        final_results = self._dedup_and_limit_by_file_id(results, limit, min_score)
      
        return final_results

    def multimodal_search(self, db: Session, image_embedding: List[float], gid: Union[str, List[str], None] = None, limit: int = 10, min_score: float = None) -> List[Dict[str, Any]]:
        """多模态检索（使用 CLIP 模型进行跨模态匹配）"""
        image_vector = image_embedding
      
        vector_count = self.vector_service.get_vector_count()
        if vector_count == 0:
            return []
        gid_list = self._normalize_gid(gid)

        if gid_list:
            gid_file_count = db.query(File).filter(File.gid.in_(gid_list)).count()
            top_k = max(limit * 5, gid_file_count + limit)
        else:
            top_k = limit * 3
        vector_results = self.vector_service.search_vectors(image_vector, top_k=top_k)
        results = []
        for result in vector_results:
            vector_id = str(result["id"])
            file_id = self._extract_file_id_from_vector_id(vector_id)
            if not file_id:
                continue
            # gid 过滤
            if gid_list:
                file = db.query(File).filter(File.id == file_id).first()
                if not file or file.gid not in gid_list:
                    continue
            score = result.get("score")
            if score is not None:
                score = float(score)
            else:
                distance = float(result.get("distance", 0.0))
                score = float(1.0 / (1.0 + distance))
            result_item = {
                "file_id": file_id,
                "score": score,
                "search_type": "multimodal"
            }
            # 如果是视频帧，获取时间戳
            if vector_id.startswith("video_frame_"):
                frame_info = db.query(ImageFrame).filter(
                    ImageFrame.file_id == file_id,
                    ImageFrame.vector_id == vector_id
                ).first()
                if frame_info and frame_info.timestamp is not None:
                    result_item["timestamp"] = float(frame_info.timestamp)
            results.append(result_item)
        # 使用配置默认值（如果未指定）
        if min_score is None:
            min_score = self.default_min_score
        return self._dedup_and_limit_by_file_id(results, limit, min_score)

    def face_search(self, db: Session, face_embedding: List[float], gid: Union[str, List[str], None] = None, limit: int = 10, min_score: float = None) -> List[Dict[str, Any]]:
        """按人脸搜索"""
        from app.models.db import File
        from app.services.face_service import FaceService
        face_service = FaceService()
        gid_list = self._normalize_gid(gid)
        if gid_list:
            gid_file_count = db.query(File).filter(File.gid.in_(gid_list)).count()
            search_limit = max(limit * 5, gid_file_count + limit)
        else:
            search_limit = limit * 2
        similar_faces = face_service.find_similar_faces(face_embedding, limit=search_limit)
        results = []
        for face in similar_faces:
            file_id = str(face.get("file_id", ""))
            if not file_id:
                continue
            # gid 过滤
            if gid_list:
                file = db.query(File).filter(File.id == file_id).first()
                if not file or file.gid not in gid_list:
                    continue
            similarity = float(face.get("similarity", 0.0))
            results.append({
                "file_id": file_id,
                "score": similarity,
                "search_type": "face"
            })
        # 使用配置默认值（如果未指定）
        if min_score is None:
            min_score = self.default_min_score
        return self._dedup_and_limit_by_file_id(results, limit, min_score)

    def hybrid_search(self, db: Session, query: str, file_type: str = None, gid: Union[str, List[str], None] = None, limit: int = 10, min_score: float = None) -> List[Dict[str, Any]]:
        """混合检索"""
        # 使用配置默认值（如果未指定）
        if min_score is None:
            min_score = self.default_min_score
        keyword_results = self.keyword_search(db, query, file_type, gid, limit, min_score=min_score)
        semantic_results = self.semantic_search(db, query, file_type, gid, limit, min_score=min_score)
        combined_results = {}
        for result in keyword_results:
            file_id = result["file_id"]
            combined_results[file_id] = result
        for result in semantic_results:
            file_id = result["file_id"]
            if file_id in combined_results:
                if result["score"] > combined_results[file_id]["score"]:
                    combined_results[file_id]["score"] = result["score"]
                    combined_results[file_id]["search_type"] = "semantic"
            else:
                combined_results[file_id] = result
        filtered_results = []
        for result in combined_results.values():
            score = float(result.get("score", 0.0))
            if score >= min_score:
                result["score"] = score
                filtered_results.append(result)
        sorted_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)[:limit]
        return sorted_results
