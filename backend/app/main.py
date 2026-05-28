from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from config.config import settings
from app.api.routes import router
from app.core.database import init_db
from app.core.auth import verify_api_key
# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="本地知识库系统API"
)
# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 注册路由（需要认证）
app.include_router(router, prefix=settings.API_V1_STR, dependencies=[Depends(verify_api_key)])
# 注册公开访问的路由（不需要认证）
from app.api.public_routes import public_router
app.include_router(public_router, prefix="/public")
# 初始化数据库
@app.on_event("startup")
async def startup_event():
    init_db()
# 根路径
@app.get("/")
async def root():
    return {
        "message": "Welcome to Local Knowledge Base API",
        "version": "1.0.0",
        "docs": "/docs"
    }
# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
