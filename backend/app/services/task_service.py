import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
class TaskService:
    """任务服务 - 管理后台任务"""
    _tasks: Dict[str, Dict[str, Any]] = {}
    _lock = asyncio.Lock()
    @classmethod
    async def create_task(cls, task_type: str, task_data: Dict[str, Any] = None) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        async with cls._lock:
            cls._tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "status": TaskStatus.PENDING,
                "progress": 0,
                "message": "等待处理",
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "data": task_data or {}
            }
        return task_id
    @classmethod
    async def update_task(cls, task_id: str, **kwargs) -> bool:
        """更新任务状态"""
        async with cls._lock:
            if task_id not in cls._tasks:
                return False
            cls._tasks[task_id].update(kwargs)
            return True
    @classmethod
    async def get_task(cls, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        async with cls._lock:
            return cls._tasks.get(task_id)
    @classmethod
    async def get_all_tasks(cls) -> List[Dict[str, Any]]:
        """获取所有任务"""
        async with cls._lock:
            return list(cls._tasks.values())
    @classmethod
    async def start_task(cls, task_id: str) -> bool:
        """标记任务开始"""
        return await cls.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            message="处理中..."
        )
    @classmethod
    async def complete_task(cls, task_id: str, result: Any = None) -> bool:
        """标记任务完成"""
        return await cls.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            progress=100,
            message="处理完成",
            result=result
        )
    @classmethod
    async def fail_task(cls, task_id: str, error: str) -> bool:
        """标记任务失败"""
        return await cls.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            message=f"失败: {error}",
            error=error
        )
