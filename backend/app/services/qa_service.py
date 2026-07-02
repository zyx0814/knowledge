import requests
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.services.search_service import SearchService
from config.config import settings

class QAService:
    """问答服务类"""
    
    def __init__(self):
        """初始化问答服务"""
        self.search_service = SearchService()
    
    def generate_answer(self, db: Session, question: str, file_type: str = None) -> Dict[str, Any]:
        """生成答案"""
        search_results = self.search_service.hybrid_search(db, question, file_type, limit=10)
        
        enriched_results = self._enrich_search_results(db, search_results)
        
        has_image_results = any(r.get('type') == 'image' for r in enriched_results)
        
        context = self._build_context(enriched_results)
        
        prompt = self._build_prompt(question, context, has_image_results)
        
        answer = self._call_llm(prompt)
        
        if not answer or answer.strip() == "" or "无法生成准确的答案" in answer or "抱歉，无法生成答案" in answer:
            if has_image_results:
                answer = f"在知识库中找到了 {len([r for r in enriched_results if r.get('type') == 'image'])} 张相关图片，请查看参考来源。"
            elif enriched_results:
                answer = f"在知识库中找到了 {len(enriched_results)} 个相关文件，请查看参考来源了解详情。"
            else:
                answer = "在知识库中未找到相关内容。"
        
        sources = []
        for result in enriched_results:
            sources.append({
                "id": result.get("file_id"),
                "name": result.get("name"),
                "type": result.get("type"),
                "score": result.get("score"),
                "storage_path": result.get("storage_path"),
                "frame_path": result.get("frame_path")
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "has_images": has_image_results
        }
    
    def _enrich_search_results(self, db: Session, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为搜索结果添加文件信息"""
        from app.models.db import File, ImageFrame
        
        enriched_results = []
        for result in results:
            file_id = result.get("file_id") or result.get("id")
            if not file_id:
                continue
            
            file = db.query(File).filter(File.id == file_id).first()
            if not file:
                enriched_result = {
                    "file_id": str(file_id),
                    "score": float(result.get("score", 0.0)),
                    "search_type": result.get("search_type", "unknown"),
                    "name": f"文件 {file_id}",
                    "type": "unknown"
                }
                enriched_results.append(enriched_result)
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
            
            enriched_results.append(enriched_result)
        
        enriched_results.sort(key=lambda x: x["score"], reverse=True)
        return enriched_results
    
    def _build_context(self, enriched_results: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        context_parts = []
        
        for result in enriched_results:
            file_name = result.get('name', '未知文件')
            file_type = result.get('type', '未知类型')
            
            if file_type == 'image':
                context_parts.append(f"[图片: {file_name}] 图片文件，可能包含相关视觉内容")
            elif file_type == 'video':
                context_parts.append(f"[视频: {file_name}] 视频文件，可能包含相关内容")
            elif file_type == 'document':
                context_parts.append(f"[文档: {file_name}]")
            else:
                context_parts.append(f"[{file_type}: {file_name}]")
        
        if not context_parts:
            return "知识库为空或未找到相关内容。"
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, has_image_results: bool) -> str:
        """构建prompt"""
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
        
        return prompt.strip()
    
    def _call_llm(self, prompt: str) -> str:
        """调用本地LLM"""
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                return ""
        except Exception as e:
            return ""