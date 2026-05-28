import requests
from typing import Dict, Any
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
        # 1. 检索相关内容
        search_results = self.search_service.hybrid_search(db, question, file_type, limit=5)
        # 2. 构建上下文
        context = "\n".join([
            f"[来源: {result.get('name', '未知')}] {result.get('content', '')}"
            for result in search_results
        ])
        # 3. 构建prompt
        prompt = f"""
        你是一个基于本地知识库的问答助手，以下是相关的知识库内容：
        {context}
        请根据以上内容回答用户的问题：{question}
        要求：
        1. 基于知识库内容回答，不要生成知识库外的信息
        2. 回答要准确、简洁
        3. 引用来源信息
        """
        # 4. 调用本地LLM生成答案
        answer = self._call_llm(prompt)
        # 5. 构建溯源信息
        sources = []
        for result in search_results:
            sources.append({
                "id": result.get("id"),
                "name": result.get("name"),
                "type": result.get("type"),
                "score": result.get("score")
            })
        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }
    def _call_llm(self, prompt: str) -> str:
        """调用本地LLM"""
        try:
            # 调用Ollama API
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                return "抱歉，无法生成答案，请稍后重试。"
        except Exception as e:
            # 返回一个默认答案
            return "基于知识库内容，我无法生成准确的答案。请尝试更具体的问题。"
