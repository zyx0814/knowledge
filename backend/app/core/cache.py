import json
import functools
from typing import Optional, Any, Callable
from config.config import settings
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
class CacheService:
    """Redis 缓存服务"""
    def __init__(self):
        self.client = None
        if REDIS_AVAILABLE:
            try:
                redis_host = getattr(settings, 'REDIS_HOST', 'localhost')
                redis_port = getattr(settings, 'REDIS_PORT', 6379)
                self.client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=2,
                    decode_responses=True
                )
                self.client.ping()
            except Exception as e:
                self.client = None
                self._memory_cache = {}
    def is_available(self) -> bool:
        """检查 Redis 是否可用"""
        return self.client is not None
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if self.client:
            try:
                data = self.client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
        elif hasattr(self, '_memory_cache'):
            return self._memory_cache.get(key)
        return None
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """设置缓存"""
        try:
            serialized = json.dumps(value, default=str)
            if self.client:
                return self.client.setex(key, ttl, serialized)
            elif hasattr(self, '_memory_cache'):
                self._memory_cache[key] = value
                return True
        except Exception as e:
        return False
    def delete(self, key: str) -> bool:
        """删除缓存"""
        if self.client:
            try:
                return bool(self.client.delete(key))
            except Exception as e:
        elif hasattr(self, '_memory_cache') and key in self._memory_cache:
            del self._memory_cache[key]
            return True
        return False
    def delete_pattern(self, pattern: str) -> int:
        """按模式删除缓存"""
        count = 0
        if self.client:
            try:
                keys = self.client.keys(pattern)
                if keys:
                    count = self.client.delete(*keys)
            except Exception as e:
        elif hasattr(self, '_memory_cache'):
            keys_to_delete = [k for k in self._memory_cache.keys() if k.startswith(pattern.replace('*', ''))]
            for k in keys_to_delete:
                del self._memory_cache[k]
            count = len(keys_to_delete)
        return count
    def clear_all(self) -> bool:
        """清空所有缓存"""
        if self.client:
            try:
                return self.client.flushdb()
            except Exception as e:
        elif hasattr(self, '_memory_cache'):
            self._memory_cache.clear()
            return True
        return False
cache_service = CacheService()
def cache(ttl: int = 300, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{hash(str(args) + str(kwargs))}"
            cached = cache_service.get(cache_key)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            return result
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{hash(str(args) + str(kwargs))}"
            cached = cache_service.get(cache_key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            return result
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
