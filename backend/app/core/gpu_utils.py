import os
from typing import Tuple, List, Optional
class GPUManager:
    """GPU 管理器 - 自动检测和配置 GPU 加速"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        if self._initialized:
            return
        self.cuda_available = False
        self.faiss_gpu_available = False
        self.onnxruntime_gpu_available = False
        self.gpu_count = 0
        self.gpu_info = []
        self._detect_gpu()
        self._initialized = True
    def _detect_gpu(self):
        """检测 GPU 可用性"""
        self._check_cuda()
        self._check_faiss_gpu()
        self._check_onnxruntime_gpu()
    def _check_cuda(self):
        """检查 CUDA 是否可用"""
        try:
            import torch
            if torch.cuda.is_available():
                self.cuda_available = True
                self.gpu_count = torch.cuda.device_count()
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_info.append({
                        "id": i,
                        "name": props.name,
                        "memory_total": props.total_memory,
                        "memory_available": props.total_memory
                    })
                print(f"检测到 {self.gpu_count} 个 CUDA 设备")
            else:
                print("CUDA 不可用，将使用 CPU")
        except ImportError:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.cuda_available = True
                    self.gpu_count = int(result.stdout.strip())
                    print(f"检测到 {self.gpu_count} 个 NVIDIA GPU (via nvidia-smi)")
            except Exception:
                print("未检测到 NVIDIA GPU，将使用 CPU")
    
    def _check_faiss_gpu(self):
        """检查 FAISS GPU 是否可用"""
        try:
            import faiss
            if hasattr(faiss, 'StandardGpuResources'):
                self.faiss_gpu_available = True
                print("FAISS GPU 可用")
            else:
                print("FAISS GPU 不可用，将使用 CPU 版本")
        except ImportError:
            print("FAISS 未安装")
    
    def _check_onnxruntime_gpu(self):
        """检查 ONNX Runtime GPU 是否可用"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                self.onnxruntime_gpu_available = True
                print("ONNX Runtime GPU 可用")
            else:
                print("ONNX Runtime GPU 不可用，可用的 providers:", providers)
        except ImportError:
            print("ONNX Runtime 未安装")
    
    def get_faiss_gpu_resources(self, device_id: int = 0):
        """获取 FAISS GPU 资源"""
        if not self.faiss_gpu_available:
            return None
        try:
            import faiss
            res = faiss.StandardGpuResources()
            return res
        except Exception as e:
            print(f"获取 FAISS GPU 资源失败: {e}")
            return None
    def get_onnx_providers(self, prefer_gpu: bool = True) -> List[str]:
        """获取 ONNX Runtime providers"""
        try:
            import onnxruntime as ort
            providers = []
            if prefer_gpu and self.onnxruntime_gpu_available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            return providers
        except ImportError:
            return ['CPUExecutionProvider']
    def is_gpu_available(self) -> bool:
        """检查是否有任何 GPU 可用"""
        return self.cuda_available
    def get_gpu_info(self) -> List[dict]:
        """获取 GPU 信息列表"""
        return self.gpu_info
    def get_best_device(self) -> str:
        """获取最佳设备"""
        if self.cuda_available:
            return "cuda"
        return "cpu"
gpu_manager = GPUManager()
