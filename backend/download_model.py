#!/usr/bin/env python3
"""
模型下载脚本 - 用于在有网络的环境中下载文本嵌入模型
使用方法:
    python download_model.py
    python download_model.py --model all-MiniLM-L6-v2
    python download_model.py --help
"""
import os
import argparse
import sys
def download_model(model_name: str = "all-MiniLM-L6-v2", save_dir: str = None):
    """
    下载SentenceTransformer模型并保存到本地
    Args:
        model_name: 模型名称，默认为 all-MiniLM-L6-v2
        save_dir: 保存目录，默认为 models/sentence_transformers/{model_name}
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(1)
    # 设置保存目录
    if save_dir is None:
        save_dir = os.path.join("models", "sentence_transformers", model_name)
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    }")
    try:
        # 检查是否已经存在
        if os.path.exists(os.path.join(save_dir, "config.json")):
            overwrite = input("是否覆盖? (y/N): ").strip().lower()
            if overwrite != 'y':
                return
        # 下载模型
        ")
        # 设置镜像（如果环境变量已配置）
        original_endpoint = os.environ.get('HF_ENDPOINT')
        hf_endpoint = os.environ.get('HF_HUB_ENDPOINT', '')
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
        model = SentenceTransformer(model_name)
        # 恢复原始环境变量
        if original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_endpoint
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        # 保存模型
        model.save(save_dir)
        # 测试模型
        test_embedding = model.encode("测试文本")
        }")
        }")
        return True
    except Exception as e:
        }")
        }")
        return False
def main():
    parser = argparse.ArgumentParser(description="下载SentenceTransformer文本嵌入模型")
    parser.add_argument(
        "--model", 
        default="all-MiniLM-L6-v2", 
        help="模型名称，默认为 all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--save-dir", 
        default=None, 
        help="保存目录，默认自动创建"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出推荐的模型"
    )
    args = parser.parse_args()
    if args.list_models:
        return
    download_model(args.model, args.save_dir)
if __name__ == "__main__":
    main()
