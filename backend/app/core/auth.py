from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from config.config import settings
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def verify_api_key(api_key: str = Security(api_key_header)):
    if not settings.ENABLE_AUTH:
        return True
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )
    return True