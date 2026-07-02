# 后端Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 设置HuggingFace国内镜像（加速模型下载）
ENV HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户（安全最佳实践）
RUN useradd -m -u 1001 appuser \
    && mkdir -p /app/data /app/vector_db /app/uploads /app/temp /app/storage /opt/venv \
    && chown -R appuser:appuser /app /opt/venv

# 切换到非root用户
USER appuser

# 设置Python虚拟环境（放在/app之外，避免被挂载覆盖）
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 在虚拟环境中升级pip
RUN pip install --upgrade pip setuptools wheel

# 复制依赖文件
COPY --chown=appuser:appuser backend/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY --chown=appuser:appuser backend/ .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
