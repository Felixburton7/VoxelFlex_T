"""
System related utilities (CPU, GPU, Memory).
"""

import os
import gc
import logging
import psutil
import torch
from typing import Optional, Dict

logger = logging.getLogger("voxelflex.utils.system") # Use submodule logger

def get_cpu_cores() -> int:
    """Get the number of logical CPU cores."""
    try:
        cores = os.cpu_count()
        logger.debug(f"Detected {cores} logical CPU cores.")
        return cores
    except NotImplementedError:
        logger.warning("Could not detect number of CPU cores. Defaulting to 1.")
        return 1

def get_gpu_details() -> dict:
    """Get details about available NVIDIA GPUs."""
    details = {"count": 0, "names": [], "memory_gb": [], "cuda_available": False}
    if torch.cuda.is_available():
        details["cuda_available"] = True
        details["count"] = torch.cuda.device_count()
        for i in range(details["count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                details["names"].append(props.name)
                details["memory_gb"].append(props.total_memory / (1024**3))
            except Exception as e:
                 logger.error(f"Could not get properties for GPU {i}: {e}")
                 details["names"].append("Error")
                 details["memory_gb"].append(0)

    return details

def log_gpu_details(logger_instance: logging.Logger = logger):
    """Logs GPU details."""
    gpu_info = get_gpu_details()
    if gpu_info["cuda_available"]:
        logger_instance.info(f"CUDA Available: Yes. Found {gpu_info['count']} GPU(s).")
        for i in range(gpu_info['count']):
            logger_instance.info(f"  GPU {i}: {gpu_info['names'][i]} - Memory: {gpu_info['memory_gb'][i]:.2f} GB")
    else:
        logger_instance.info("CUDA Available: No. Running on CPU.")

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Gets the recommended device (GPU if available and preferred, else CPU)."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        log_gpu_details(logger) # Log details when selecting GPU
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    logger.info(f"Selected device: {device}")
    return device

def check_memory_usage() -> dict:
    """Returns system memory usage statistics."""
    mem_info = psutil.virtual_memory()
    return {
        "total_gb": mem_info.total / (1024**3),
        "available_gb": mem_info.available / (1024**3),
        "percent_used": mem_info.percent,
        "used_gb": mem_info.used / (1024**3)
    }

def clear_memory(force_gc: bool = True, clear_cuda: bool = True):
    """Attempts to clear memory by running GC and emptying CUDA cache."""
    logger.debug("Attempting to clear memory...")
    if force_gc:
        gc.collect()
        logger.debug("Ran garbage collector.")
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Emptied CUDA cache.")

def set_num_threads(num_threads: Optional[int] = None):
    """Sets the number of threads used by PyTorch."""
    if num_threads is not None and num_threads > 0:
        logger.info(f"Setting PyTorch CPU threads to: {num_threads}")
        torch.set_num_threads(num_threads)
    else:
        # Use default setting (often uses all cores, which is usually fine)
        cores = get_cpu_cores()
        logger.debug(f"Using default PyTorch CPU thread settings (likely based on {cores} cores).")
        # torch.set_num_threads(1) # Option: Force single thread if needed

def adjust_workers_for_memory(requested_workers: int, memory_threshold: float = 85.0) -> int:
    """
    Adjusts the number of DataLoader workers based on available memory.
    Simple heuristic: reduces workers if memory usage is high.
    """
    if requested_workers <= 0:
        return 0

    try:
        mem_percent = check_memory_usage().get("percent_used", 0)
        if mem_percent > memory_threshold:
            reduced_workers = max(0, requested_workers // 2) # Halve workers
            logger.warning(f"High memory usage ({mem_percent:.1f}%) detected. Reducing DataLoader workers from {requested_workers} to {reduced_workers}.")
            return reduced_workers
        else:
            return requested_workers
    except Exception as e:
        logger.warning(f"Could not check memory to adjust workers: {e}. Using requested value: {requested_workers}")
        return requested_workers
