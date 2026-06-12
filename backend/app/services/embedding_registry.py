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
import numpy as np
from typing import List, Optional

from config.config import settings
from app.core.gpu_utils import gpu_manager


# --- 辅助函数 ---

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
        if os.path.exists(local_pt):
            try:
                sd = torch.load(local_pt, map_location=device)
                from clip.model import build_model
                _model = build_model(sd["model_state_dict"]).to(device)
                _model.eval()
                _preprocess = clip._transform(sd.get("input_resolution", 224))
                _model_type = cache_key
                return _model, _preprocess
            except Exception:
                pass
        _model, _preprocess = clip.load(model_name, device=device)
        _model_type = cache_key
        os.makedirs(local_path, exist_ok=True)
        try:
            torch.save({"model_state_dict": _model.state_dict(),
                         "input_resolution": _model.visual.input_resolution,
                         "context_length": _model.context_length,
                         "vocab_size": _model.vocab_size}, local_pt)
        except Exception:
            pass
        return _model, _preprocess
    except Exception:
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
        if os.path.exists(local_path):
            try:
                _tokenizer = AutoTokenizer.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = CLIPImageProcessor.from_pretrained(local_path)
                _model_type = cache_key
                return _model, _preprocess, _tokenizer
            except Exception:
                pass
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(device)
        _model.eval()
        _preprocess = CLIPImageProcessor.from_pretrained(model_name)
        _model_type = cache_key
        os.makedirs(local_path, exist_ok=True)
        try:
            _tokenizer.save_pretrained(local_path)
            _model.save_pretrained(local_path)
        except Exception:
            pass
        return _model, _preprocess, _tokenizer
    except Exception:
        return None, None, None


def _init_jina_clip_v2(model_name="jinaai/jina-clip-v2"):
    global _model, _model_type, _preprocess
    cache_key = f"jina_clip_{model_name}"
    if _model is not None and _model_type == cache_key:
        return _model, _preprocess
    try:
        import torch, shutil, glob
        from transformers import AutoModel
        device = _get_device()
        local_path = os.path.join(settings.MODELS_DIR, "jina_clip", model_name.replace("/", "_"))
        if not os.path.exists(local_path) or not os.path.exists(os.path.join(local_path, "model.safetensors")):
            return None, None
        
        # Pre-populate the HuggingFace cache with implementation files
        # so transformers can find them offline
        hf_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
        cache_repo = model_name.replace("/", "_").replace("-", "_hyphen_")
        cache_dir = os.path.join(hf_cache, cache_repo)
        os.makedirs(cache_dir, exist_ok=True)
        for py_file in glob.glob(os.path.join(local_path, "*.py")):
            dest = os.path.join(cache_dir, os.path.basename(py_file))
            if not os.path.exists(dest):
                shutil.copy2(py_file, dest)
        
        # Set offline mode to prevent any network access
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        
        _model = AutoModel.from_pretrained(local_path, trust_remote_code=True).to(device)
        _model.eval()
        _preprocess = True  # Jina-CLIP-v2: encode_text()/encode_image() handle preprocessing internally
        _model_type = cache_key
        return _model, _preprocess
    except Exception:
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
        if os.path.exists(local_path):
            try:
                _tokenizer = AutoTokenizer.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = Qwen2VLProcessor.from_pretrained(local_path)
                _model_type = cache_key
                return _model, _preprocess, _tokenizer
            except Exception:
                pass
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(device)
        _model.eval()
        _preprocess = Qwen2VLProcessor.from_pretrained(model_name)
        _model_type = cache_key
        os.makedirs(local_path, exist_ok=True)
        try:
            _tokenizer.save_pretrained(local_path)
            _model.save_pretrained(local_path)
            _preprocess.save_pretrained(local_path)
        except Exception:
            pass
        return _model, _preprocess, _tokenizer
    except Exception:
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
        if os.path.exists(local_path):
            try:
                _tokenizer = AutoProcessor.from_pretrained(local_path)
                _model = AutoModel.from_pretrained(local_path).to(device)
                _model.eval()
                _preprocess = _tokenizer
                _model_type = cache_key
                return _model, _preprocess, _tokenizer
            except Exception:
                pass
        _tokenizer = AutoProcessor.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(device)
        _model.eval()
        _preprocess = _tokenizer
        _model_type = cache_key
        os.makedirs(local_path, exist_ok=True)
        try:
            _model.save_pretrained(local_path)
            _tokenizer.save_pretrained(local_path)
        except Exception:
            pass
        return _model, _preprocess, _tokenizer
    except Exception:
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
                with torch.no_grad():
                    f = model.encode_text(tokens)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    # Chinese-CLIP
    elif model_name.startswith("OFA-Sys/chinese-clip-"):
        model, _, tokenizer = _init_chinese_clip(model_name)
        if model and tokenizer:
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    # Qwen3-VL
    elif model_name.startswith("Qwen/Qwen3-VL-Embedding-"):
        model, _, tokenizer = _init_qwen_vl(model_name)
        if model and tokenizer:
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    # Jina-CLIP-v2
    elif model_name == "jinaai/jina-clip-v2":
        model, _ = _init_jina_clip_v2(model_name)
        if model:
            try:
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
                pass

    # WeCLIP
    elif model_name.startswith("alibaba-nlp/weclip-"):
        model, _, tokenizer = _init_weclip(model_name)
        if model and tokenizer:
            try:
                inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    if emb is None:
        return np.random.rand(target_dim).tolist()
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
                    with torch.no_grad():
                        f = model.encode_image(inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    # Chinese-CLIP
    elif model_name.startswith("OFA-Sys/chinese-clip-"):
        model, preprocess, _ = _init_chinese_clip(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        f = model.get_image_features(**inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    # Qwen3-VL
    elif model_name.startswith("Qwen/Qwen3-VL-Embedding-"):
        model, preprocess, _ = _init_qwen_vl(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        f = model.get_image_features(**inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    # Jina-CLIP-v2
    elif model_name == "jinaai/jina-clip-v2":
        model, _ = _init_jina_clip_v2(model_name)
        if model:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    with torch.no_grad():
                        f = model.encode_image(img, truncate_dim=None)
                    if isinstance(f, torch.Tensor):
                        f = f / f.norm(dim=-1, keepdim=True)
                        emb = f.cpu().numpy().flatten().tolist()
                    else:
                        import numpy as np
                        arr = np.array(f)
                        norm = np.linalg.norm(arr)
                        if norm > 0:
                            arr = arr / norm
                        emb = arr.flatten().tolist()
            except Exception:
                pass

    # WeCLIP
    elif model_name.startswith("alibaba-nlp/weclip-"):
        model, preprocess, _ = _init_weclip(model_name)
        if model and preprocess:
            try:
                img = _read_image_from_path(image_path)
                if img:
                    inp = preprocess(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        f = model.get_image_features(**inp)
                    f = f / f.norm(dim=-1, keepdim=True)
                    emb = f.cpu().numpy().flatten().tolist()
            except Exception:
                pass

    if emb is None:
        return np.random.rand(target_dim).tolist()
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