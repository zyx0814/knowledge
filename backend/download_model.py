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
        print("错误: 请先安装 sentence-transformers")
        print("命令: pip install sentence-transformers")
        sys.exit(1)
    
    # 设置保存目录
    if save_dir is None:
        save_dir = os.path.join("models", "sentence_transformers", model_name)
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"="*60)
    print(f"开始下载模型: {model_name}")
    print(f"保存目录: {os.path.abspath(save_dir)}")
    print("="*60)
    
    try:
        # 检查是否已经存在
        if os.path.exists(os.path.join(save_dir, "config.json")):
            print(f"警告: 模型已存在于 {save_dir}")
            overwrite = input("是否覆盖? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("取消下载")
                return
        
        # 下载模型
        print("\n正在下载模型... (这可能需要几分钟)")
        print("提示: 如果下载速度慢，可以设置环境变量使用镜像")
        print("      export HF_HUB_ENDPOINT=https://hf-mirror.com")
        
        # 设置镜像（如果环境变量已配置）
        original_endpoint = os.environ.get('HF_ENDPOINT')
        hf_endpoint = os.environ.get('HF_HUB_ENDPOINT', '')
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
            print(f"使用镜像: {hf_endpoint}")
        
        model = SentenceTransformer(model_name)
        
        # 恢复原始环境变量
        if original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_endpoint
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        
        # 保存模型
        print(f"\n保存模型到 {save_dir}...")
        model.save(save_dir)
        
        # 测试模型
        print("测试模型...")
        test_embedding = model.encode("测试文本")
        print(f"模型测试成功! 嵌入维度: {len(test_embedding)}")
        
        print("\n" + "="*60)
        print(f"模型下载完成!")
        print(f"模型路径: {os.path.abspath(save_dir)}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n错误: 下载失败 - {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 设置环境变量使用镜像:")
        print("   export HF_HUB_ENDPOINT=https://hf-mirror.com")
        print("3. 手动下载模型文件:")
        print(f"   https://huggingface.co/sentence-transformers/{model_name}")
        print(f"   然后解压到: {os.path.abspath(save_dir)}")
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
        print("推荐的文本嵌入模型:")
        print("-" * 50)
        print("all-MiniLM-L6-v2      - 轻量级，384维（推荐）")
        print("all-mpnet-base-v2     - 高质量，768维")
        print("all-distilroberta-v1  - DistilBERT基础，768维")
        print("paraphrase-MiniLM-L6-v2 - 短句匹配专用")
        print("multi-qa-MiniLM-L6-cos-v1 - 问答专用")
        print("-" * 50)
        print("更多模型请访问: https://huggingface.co/sentence-transformers")
        return
    
    download_model(args.model, args.save_dir)

if __name__ == "__main__":
    main()
