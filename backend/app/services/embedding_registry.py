# -*- coding: utf-8 -*-
"""
嵌入模型注册表 —— 仅多模态模型
所有模型文本和图像编码在同一向量空间，天然支持以文搜图 + 以图搜图。

支持的模型：
- OpenAI CLIP: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
- Chinese-CLIP: OFA-Sys/chinese-clip-vit-*
- Qwen3-VL-Embedding: Qwen/Qwen3-VL-Embedding-*
- Jina-CLIP-v2: jinaai/jina-clip-v2
- WeCLIP: alibaba-nlp/weclip-*
"""

import io
import os
import threading
import numpy as np
from typing import List, Optional

from config.config import settings
from app.core.gpu_utils import gpu_manager

# GPU 推理锁：防止多个 Celery worker / 协程同时调用全局模型导致 CUDA 竞争
_inference_lock = threading.Lock()


# --- 辅助函数 ---

def _download_clip_model(model_name: str = "ViT-B/32", save_dir: str = None):
    """下载OpenAI CLIP模型"""
    try:
        import torch
        from clip import clip
        
        if save_dir is None:
            save_dir = os.path.join(settings.MODELS_DIR, "clip", model_name.replace("/", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[INFO] 正在下载 CLIP {model_name} 模型...")
        
        device = "cpu"
        model, preprocess = clip.load(model_name, device=device)
        
        save_path = os.path.join(save_dir, "clip.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_resolution': model.visual.input_resolution,
            'context_length': model.context_length,
            'vocab_size': model.vocab_size
        }, save_path)
        
        print(f"[INFO] CLIP模型已保存到: {save_dir}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载CLIP模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _download_transformers_model(model_name: str, save_dir: str = None, trust_remote_code: bool = False):
    """下载Transformers模型"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        if save_dir is None:
            save_dir = os.path.join(settings.MODELS_DIR, model_name.replace("/", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[INFO] 正在下载 {model_name} 模型...")
        
        original_endpoint = os.environ.get('HF_ENDPOINT')
        hf_endpoint = os.environ.get('HF_HUB_ENDPOINT', '')
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
            print(f"[INFO] 使用HF端点: {hf_endpoint}")
        
        # Jina-CLIP-v2 使用 Mistral tokenizer，需要修复 regex 模式
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if "jina" in model_name.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        print(f"[INFO] Tokenizer 加载成功")
        
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        print(f"[INFO] 模型加载成功")
        
        if original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_endpoint
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        print(f"[INFO] 模型已保存到: {save_dir}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载Transformers模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _download_qwen_vl_model(model_name: str = "Qwen/Qwen3-VL-Embedding-2B", save_dir: str = None):
    """下载Qwen3-VL嵌入模型"""
    try:
        from transformers import AutoTokenizer, AutoModel, Qwen2VLProcessor
        
        if save_dir is None:
            save_dir = os.path.join(settings.MODELS_DIR, "qwen_vl", model_name.replace("/", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[INFO] 正在下载 {model_name} 模型...")
        
        original_endpoint = os.environ.get('HF_ENDPOINT')
        hf_endpoint = os.environ.get('HF_HUB_ENDPOINT', '')
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(model_name)
        
        if original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_endpoint
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        
        print(f"[INFO] Qwen3-VL模型已保存到: {save_dir}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载Qwen3-VL模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _download_weclip_model(model_name: str, save_dir: str = None):
    """下载WeCLIP模型"""
    try:
        from transformers import AutoModel, AutoProcessor
        
        if save_dir is None:
            save_dir = os.path.join(settings.MODELS_DIR, "weclip", model_name.replace("/", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[INFO] 正在下载 {model_name} 模型...")
        
        original_endpoint = os.environ.get('HF_ENDPOINT')
        hf_endpoint = os.environ.get('HF_HUB_ENDPOINT', '')
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        if original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = original_endpoint
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        
        processor.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        print(f"[INFO] WeCLIP模型已保存到: {save_dir}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载WeCLIP模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _read_image_from_path(image_path: str):
    """从路径读取图片，支持本地路径和 MinIO"""
    from PIL import Image
    try:
        from app.services.storage_service import storage_service
        STORAGE_AVAILABLE = True
    except ImportError:
        STORAGE_AVAILABLE = False

    if image_path.startswith("minio://") and STORAGE_AVAILABLE:
        try:
            object_name = image_path.replace("minio://", "").split("/", 1)[-1]
            response = storage_service.client.get_object(
                bucket_name=storage_service.bucket, object_name=object_name)
            return Image.open(io.BytesIO(response.read())).convert("RGB")
        except Exception:
            return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def _get_device():
    import torch
    return "cuda" if gpu_manager.cuda_available else "cpu"


# --- 模型缓存 ---
_model = None
_model_type = None
_preprocess = None
_tokenizer = None


# --- 各模型初始化 ---

def _init_clip(model_name="ViT-B/32"):
    global _model, _model_type, _preprocess
    cache_key = f"clip_{model_name}"
    if _model is not None and _model_type == cache_key:
        return _model, _preprocess
    try:
        import torch
        from clip import clip
        device = _get_device()
        local_path = os.path.join(settings.MODELS_DIR, "clip", model_name.replace("/", "_"))
        local_pt = os.path.join(local_path, "clip.pt")
        
        # 尝试从本地加载
        if os.path.exists(local_pt):
            try:
                sd = torch.load(local_pt, map_location=device)
                from clip.model import build_model
                _model = build_model(sd["model_state_dict"]).to(device)
                _model.eval()
                _preprocess = clip._transform(sd.get("input_resolution", 224))
                _model_type = cache_key
                print(f"[INFO] CLIP model loaded from local file: {local_pt}")
                return _model, _preprocess
            except Exception as e:
                print(f"[WARN] Failed to load CLIP from local file {local_pt}: {e}")
                import traceback
                traceback.print_exc()
        
        # 本地不存在，尝试在线下载
        print(f"[INFO] CLIP model not found locally at {local_pt}, attempting to download...")
        if _download_clip_model(model_name, local_path):
            # 下载成功后重新加载
            try:
                sd = torch.load(local_pt, map_location=device)
                from clip.model import build_model
                _model = build_model(sd["model_state_dict"]).to(device)
                _model.eval()
                _preprocess = clip._transform(sd.get("input_resolution", 224))
                _model_type = cache_key
                print(f"[INFO] CLIP model downloaded and loaded successfully")
                return _model, _preprocess
            except Exception as e:
                print(f"[ERROR] Failed to load CLIP after download: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[ERROR] CLIP model not found locally and download failed")
        return None, None
    except Exception as e:
        print(f"[ERROR] _init_clip failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _init_chinese_clip(model_name):
    global _model, _model_type, _preprocess, _tokenizer
    cache_key = f"chinese_clip_{model_name}"
    if _model is not None and _model_type == cache_key:
        return _model, _preprocess, _tokenizer
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
        device = _get_device()
        local_path = os.path.join(settings.MODELS_DIR, "chinese_clip", model_name.replace("/", "_"))
        
        # 尝试从本地加载
        if os.path.exists(local_path):
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                _model = AutoModel.from_pretrained(local_path, local_files_only=True).to(device)
                _model.eval()
                _preprocess = CLIPImageProcessor.from_pretrained(local_path, local_files_only=True)
                _model_type = cache_key
                print(f"[INFO] Chinese-CLIP model loaded successfully from {local_path}")
                return _model, _preprocess, _tokenizer
            except Exception as e:
                print(f"[WARN] Chinese-CLIP local loading failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 本地不存在，尝试在线下载
        print(f"[INFO] Chinese-CLIP model not found locally at {local_path}, attempting to download...")
        # 清除离线模式环境变量
        if 'HF_HUB_OFFLINE' in os.environ:
            del os.environ['HF_HUB_OFFLINE']
        if 'TRANSFORMERS_OFFLINE' in os.environ:
            del os.environ['TRANSFORMERS_OFFLINE']
        
        if _download_transformers_model(model_name, local_path):
            # 下载成功后重新加载
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                _model = AutoModel.from_pretrained(local_path, local_files_only=True).to(device)
                _model.eval()
                _preprocess = CLIPImageProcessor.from_pretrained(local_path, local_files_only=True)
                _model_type = cache_key
                print(f"[INFO] Chinese-CLIP model downloaded and loaded successfully")
                return _model, _preprocess, _tokenizer
            except Exception as e:
                print(f"[ERROR] Failed to load Chinese-CLIP after download: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[ERROR] Chinese-CLIP model not found locally and download failed")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] _init_chinese_clip failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _init_jina_clip_v2(model_name="jinaai/jina-clip-v2"):
    global _model, _model_type, _preprocess
    cache_key = f"jina_clip_{model_name}"
    if _model is not None and _model_type == cache_key:
        return _model, _preprocess
    try:
        import torch, shutil, glob, traceback
        from transformers import AutoModel
        device = _get_device()
        local_path = os.path.join(settings.MODELS_DIR, "jina_clip", model_name.replace("/", "_"))
        
        # 检查本地是否存在模型文件
        model_exists = os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "model.safetensors"))
        
        # 如果本地不存在，尝试在线下载
        if not model_exists:
            print(f"[INFO] Jina-CLIP-v2 model not found locally at {local_path}, attempting to download...")
            # 清除离线模式环境变量
            if 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']
            if 'TRANSFORMERS_OFFLINE' in os.environ:
                del os.environ['TRANSFORMERS_OFFLINE']
            
            if not _download_transformers_model(model_name, local_path, trust_remote_code=True):
                print(f"[ERROR] Jina-CLIP-v2 download failed")
                return None, None
        
        # Pre-populate the HuggingFace cache with implementation files
        hf_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
        cache_repo = model_name.replace("/", "_").replace("-", "_hyphen_")
        cache_dir = os.path.join(hf_cache, cache_repo)
        os.makedirs(cache_dir, exist_ok=True)
        for py_file in glob.glob(os.path.join(local_path, "*.py")):
            dest = os.path.join(cache_dir, os.path.basename(py_file))
            if not os.path.exists(dest):
                shutil.copy2(py_file, dest)
        print(f"[INFO] Copied {len(list(glob.glob(os.path.join(local_path, '*.py'))))} Python files to HF cache: {cache_dir}")
        
        # Also pre-populate the text encoder's implementation files
        emb_model_dir = os.path.join(settings.MODELS_DIR, "jina_embeddings", "jinaai_jina-embeddings-v3")
        emb_cache_repo = "jinaai/xlm-roberta-flash-implementation"
        emb_cache_dir = os.path.join(hf_cache, emb_cache_repo)
        os.makedirs(emb_cache_dir, exist_ok=True)
        for py_file in glob.glob(os.path.join(emb_model_dir, "*.py")):
            dest = os.path.join(emb_cache_dir, os.path.basename(py_file))
            if not os.path.exists(dest):
                shutil.copy2(py_file, dest)
        if os.path.exists(emb_model_dir):
            print(f"[INFO] Copied text encoder Python files to HF cache: {emb_cache_dir}")
        
        # Set offline mode 
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        print(f"[INFO] Loading Jina-CLIP-v2 model from {local_path}...")
        _model = AutoModel.from_pretrained(local_path, trust_remote_code=True, local_files_only=True).to(device)
        _model.eval()
        _preprocess = True
        _model_type = cache_key
        print(f"[INFO] Jina-CLIP-v2 model loaded successfully on {device}")
        return _model, _preprocess
    except Exception as e:
        print(f"[ERROR] Failed to load Jina-CLIP-v2 model: {e}")
        traceback.print_exc()
        return None, None


def _init_qwen_vl(model_name):
    global _model, _model_type, _preprocess, _tokenizer
    cache_key = f"qwen_vl_{model_name}"
    if _model is not None and _model_type == cache_key:
        return _model, _preprocess, _tokenizer
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, Qwen2VLProcessor
        device = _get_device()
        local_path = os.path.join(settings.MODELS_DIR, "qwen_vl", model_name.replace("/", "_"))
        
        # 尝试从本地加载
        if os.path.exists(local_path):
            try:
                _tokenizer = AutoTokenizer.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = Qwen2VLProcessor.from_pretrained(local_path)
                _model_type = cache_key
                print(f"[INFO] Qwen3-VL model loaded successfully from {local_path}")
                return _model, _preprocess, _tokenizer
            except Exception as e:
                print(f"[WARN] Qwen3-VL local loading failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 本地不存在，尝试在线下载
        print(f"[INFO] Qwen3-VL model not found locally at {local_path}, attempting to download...")
        if _download_qwen_vl_model(model_name, local_path):
            # 下载成功后重新加载
            try:
                _tokenizer = AutoTokenizer.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = Qwen2VLProcessor.from_pretrained(local_path)
                _model_type = cache_key
                print(f"[INFO] Qwen3-VL model downloaded and loaded successfully")
                return _model, _preprocess, _tokenizer
            except Exception as e:
                print(f"[ERROR] Failed to load Qwen3-VL after download: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[ERROR] Qwen3-VL model not found locally and download failed")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] _init_qwen_vl failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _init_weclip(model_name):
    global _model, _model_type, _preprocess, _tokenizer
    cache_key = f"weclip_{model_name}"
    if _model is not None and _model_type == cache_key:
        return _model, _preprocess, _tokenizer
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        device = _get_device()
        local_path = os.path.join(settings.MODELS_DIR, "weclip", model_name.replace("/", "_"))
        
        # 尝试从本地加载
        if os.path.exists(local_path):
            try:
                _tokenizer = AutoProcessor.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = _tokenizer
                _model_type = cache_key
                print(f"[INFO] WeCLIP model loaded successfully from {local_path}")
                return _model, _preprocess, _tokenizer
            except Exception as e:
                print(f"[WARN] WeCLIP local loading failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 本地不存在，尝试在线下载
        print(f"[INFO] WeCLIP model not found locally at {local_path}, attempting to download...")
        if _download_weclip_model(model_name, local_path):
            # 下载成功后重新加载
            try:
                _tokenizer = AutoProcessor.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = _tokenizer
                _model_type = cache_key
                print(f"[INFO] WeCLIP model downloaded and loaded successfully")
                return _model, _preprocess, _tokenizer
            except Exception as e:
                print(f"[ERROR] Failed to load WeCLIP after download: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[ERROR] WeCLIP model not found locally and download failed")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] _init_weclip failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# --- 统一嵌入生成接口 ---

def generate_text_embedding(text: str, model_name: str = None) -> List[float]:
    import numpy as np
    """生成文本嵌入向量。文本和图像在同一向量空间，可直接搜图。"""
    import torch
    if model_name is None:
        model_name = settings.EMBEDDING_MODEL
    target_dim = settings.MODEL_DIMENSIONS.get(model_name, 512)
    device = _get_device()
    emb = None

    # CLIP
    if model_name in ("ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"):
        model, _ = _init_clip(model_name)
        if model:
            try:
                from clip import clip
                tokens = clip.tokenize([text]).to(device)
                with _inference_lock:
                    with torch.no_grad():
                        f = model.encode_text(tokens)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate text embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # Chinese-CLIP
    elif model_name.startswith("OFA-Sys/chinese-clip-"):
        model, _, tokenizer = _init_chinese_clip(model_name)
        if model and tokenizer:
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                with _inference_lock:
                    with torch.no_grad():
                        f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # Qwen3-VL
    elif model_name.startswith("Qwen/Qwen3-VL-Embedding-"):
        model, _, tokenizer = _init_qwen_vl(model_name)
        if model and tokenizer:
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                with _inference_lock:
                    with torch.no_grad():
                        f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # Jina-CLIP-v2
    elif model_name == "jinaai/jina-clip-v2":
        model, _ = _init_jina_clip_v2(model_name)
        if model:
            try:
                with _inference_lock:
                    with torch.no_grad():
                        f = model.encode_text(text, task="retrieval.query", truncate_dim=None)
                if isinstance(f, torch.Tensor):
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
                else:
                    arr = np.array(f)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    emb = arr.flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # WeCLIP
    elif model_name.startswith("alibaba-nlp/weclip-"):
        model, _, tokenizer = _init_weclip(model_name)
        if model and tokenizer:
            try:
                inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                with _inference_lock:
                    with torch.no_grad():
                        f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    if emb is None:
        print(f"[ERROR] Failed to generate embedding for model {model_name}")
        return []
    # 维度对齐
    actual = len(emb)
    if actual < target_dim:
        emb = emb + [0.0] * (target_dim - actual)
    elif actual > target_dim:
        emb = emb[:target_dim]
    return emb


def generate_image_embedding_from_path(image_path: str, model_name: str = None) -> List[float]:
    """从图片路径生成图像嵌入向量。与文本向量在同一空间，天然支持以文搜图。"""
    import numpy as np
    import torch
    if model_name is None:
        model_name = settings.EMBEDDING_MODEL
    target_dim = settings.MODEL_DIMENSIONS.get(model_name, 512)
    device = _get_device()
    emb = None

    # CLIP
    if model_name in ("ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"):
        model, preprocess = _init_clip(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(img).unsqueeze(0).to(device)
                    with _inference_lock:
                        with torch.no_grad():
                            f = model.encode_image(inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # Chinese-CLIP
    elif model_name.startswith("OFA-Sys/chinese-clip-"):
        model, preprocess, _ = _init_chinese_clip(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(images=img, return_tensors="pt").to(device)
                    with _inference_lock:
                        with torch.no_grad():
                            f = model.get_image_features(**inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # Qwen3-VL
    elif model_name.startswith("Qwen/Qwen3-VL-Embedding-"):
        model, preprocess, _ = _init_qwen_vl(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(images=img, return_tensors="pt").to(device)
                    with _inference_lock:
                        with torch.no_grad():
                            f = model.get_image_features(**inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # Jina-CLIP-v2
    elif model_name == "jinaai/jina-clip-v2":
        model, _ = _init_jina_clip_v2(model_name)
        if model:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    # Get image size from config
                    image_size = model.config.vision_config.image_size

                    # Create preprocessing using torchvision transforms (CLIP-style)
                    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
                    preprocess = Compose([
                        Resize(image_size, interpolation=2),  # BICUBIC
                        CenterCrop(image_size),
                        ToTensor(),
                        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
                    ])

                    # Apply preprocessing
                    inp = preprocess(img).unsqueeze(0).to(device)

                    # Ensure model is on the same device as input
                    model.to(device)
                    model.eval()

                    # Get image features (serialized via GPU inference lock)
                    with _inference_lock:
                        with torch.no_grad():
                            f = model.get_image_features(pixel_values=inp)

                    # Normalize and convert to list
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    # WeCLIP
    elif model_name.startswith("alibaba-nlp/weclip-"):
        model, preprocess, _ = _init_weclip(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(images=img, return_tensors="pt").to(device)
                    with _inference_lock:
                        with torch.no_grad():
                            f = model.get_image_features(**inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                print(f"[ERROR] Failed to generate embedding for model {model_name}")
                import traceback
                traceback.print_exc()

    if emb is None:
        print(f"[ERROR] Failed to generate embedding for model {model_name}")
        return []
    actual = len(emb)
    if actual < target_dim:
        emb = emb + [0.0] * (target_dim - actual)
    elif actual > target_dim:
        emb = emb[:target_dim]
    return emb


def generate_image_embedding(features: List[float]) -> List[float]:
    """兼容旧接口：从特征列表生成图像嵌入"""
    dim = settings._resolve_embedding_dim()
    out = np.zeros(dim)
    out[:len(features)] = features
    return out.tolist()


def generate_video_embedding(frame_features, audio_text: str) -> List[float]:
    """生成视频嵌入向量"""
    dim = settings._resolve_embedding_dim()
    if not frame_features:
        return generate_text_embedding(audio_text) if audio_text else np.random.rand(dim).tolist()
    frame_mean = np.mean(frame_features, axis=0)
    if audio_text:
        audio_emb = generate_text_embedding(audio_text)
        emb = (frame_mean + np.array(audio_emb)) / 2
    else:
        emb = frame_mean
    result = emb.tolist()
    if len(result) < dim:
        result = result + [0.0] * (dim - len(result))
    elif len(result) > dim:
        result = result[:dim]
    return result