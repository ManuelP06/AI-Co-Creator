"""
GPU Management Service
Handles GPU memory management and resource allocation
"""

import os
import gc
import torch
import psutil
from typing import Dict, Optional, Any
from contextlib import contextmanager

from app.config import settings
from app.core.logging_config import get_logger
from app.core.exceptions import InsufficientResourcesException

logger = get_logger("gpu_manager")


class GPUManager:
    """Centralized GPU resource management."""

    def __init__(self):
        self._models = {}
        self._setup_gpu_config()

    def _setup_gpu_config(self):
        """Setup GPU configuration."""
        if torch.cuda.is_available():
            # Apply CUDA memory configuration
            if settings.pytorch_cuda_alloc_conf:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.pytorch_cuda_alloc_conf

            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            logger.info("No GPU available, using CPU")

    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """Get detailed GPU memory information."""
        if not torch.cuda.is_available():
            return None

        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': free,
                'utilization': (reserved / total) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return None

    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'used_gb': memory.used / 1024**3,
            'percent': memory.percent
        }

    def check_resources(self, min_gpu_memory_gb: float = 8.0, min_ram_gb: float = 4.0) -> Dict[str, Any]:
        """Check if system has sufficient resources."""
        gpu_info = self.get_gpu_memory_info()
        ram_info = self.get_system_memory_info()

        result = {
            'gpu_available': gpu_info is not None,
            'gpu_sufficient': False,
            'ram_sufficient': ram_info['available_gb'] >= min_ram_gb,
            'gpu_info': gpu_info,
            'ram_info': ram_info,
            'recommendations': []
        }

        if gpu_info:
            result['gpu_sufficient'] = gpu_info['free_gb'] >= min_gpu_memory_gb
            if not result['gpu_sufficient']:
                result['recommendations'].append(
                    f"GPU memory insufficient: {gpu_info['free_gb']:.1f}GB available, "
                    f"{min_gpu_memory_gb}GB required"
                )

        if not result['ram_sufficient']:
            result['recommendations'].append(
                f"RAM insufficient: {ram_info['available_gb']:.1f}GB available, "
                f"{min_ram_gb}GB required"
            )

        return result

    @contextmanager
    def memory_context(self):
        """Context manager for memory cleanup."""
        try:
            yield
        finally:
            self.cleanup_memory()

    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        try:
            # Python garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")

            # CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("CUDA memory cache cleared")

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def should_use_gpu(self, min_memory_gb: float = 6.0) -> bool:
        """Determine if GPU should be used based on available memory."""
        gpu_info = self.get_gpu_memory_info()

        if not gpu_info:
            return False

        if gpu_info['free_gb'] < min_memory_gb:
            logger.info(f"GPU memory insufficient ({gpu_info['free_gb']:.1f}GB), using CPU")
            return False

        if gpu_info['utilization'] > 90:
            logger.info(f"GPU highly utilized ({gpu_info['utilization']:.1f}%), using CPU")
            return False

        return True

    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """Get optimal batch size based on available memory."""
        gpu_info = self.get_gpu_memory_info()

        if not gpu_info:
            return 1

        # Conservative scaling based on available memory
        memory_factor = min(gpu_info['free_gb'] / 8.0, 2.0)  # Max 2x scaling
        optimal_size = max(1, int(base_batch_size * memory_factor))

        logger.debug(f"Optimal batch size: {optimal_size} (base: {base_batch_size})")
        return optimal_size

    def monitor_resources(self) -> Dict[str, Any]:
        """Get comprehensive resource monitoring data."""
        return {
            'gpu': self.get_gpu_memory_info(),
            'ram': self.get_system_memory_info(),
            'recommendations': self.check_resources()['recommendations'],
            'timestamp': torch.utils.data.get_worker_info() if torch.utils.data.get_worker_info() else None
        }


# Global GPU manager instance
gpu_manager = GPUManager()