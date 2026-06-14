# 后端Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置HuggingFace国内镜像（加速模型下载）
ENV HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc g++ \
    cmake \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY backend/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
