"""
Logging utilities for VoxelFlex.
"""

import logging
import sys
import time
import psutil
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any
import os

import torch
from tqdm import tqdm

# --- Global Logger Setup ---

def setup_logging(
    log_file: Optional[str] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    name: str = "voxelflex" # Root logger name
) -> logging.Logger:
    """
    Configures logging for the entire application.

    Args:
        log_file: Path to the log file. If None, only console logging is enabled.
        console_level: Logging level for console output (e.g., "DEBUG", "INFO", "WARNING").
        file_level: Logging level for file output (e.g., "DEBUG", "INFO").
        name: Name of the root logger for the application.

    Returns:
        The configured root logger instance.
    """
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Set root logger level to lowest (DEBUG)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    try:
        # Convert string level names to logging constants
        console_log_level_int = getattr(logging, console_level.upper(), logging.INFO)
    except AttributeError:
        print(f"Warning: Invalid console log level '{console_level}'. Defaulting to INFO.")
        console_log_level_int = logging.INFO
    console_handler.setLevel(console_log_level_int)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    if log_file:
        try:
            # Convert string level names to logging constants
            file_log_level_int = getattr(logging, file_level.upper(), logging.DEBUG)
        except AttributeError:
            print(f"Warning: Invalid file log level '{file_level}'. Defaulting to DEBUG.")
            file_log_level_int = logging.DEBUG

        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_handler.setLevel(file_log_level_int)
        file_formatter = logging.Formatter(
             '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file} (Level: {file_level.upper()})")

    logger.info(f"Console logging level set to: {console_level.upper()}")

    # --- Handle external libraries ---
    # Reduce verbosity of common libraries if desired
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance, inheriting configuration from the root."""
    return logging.getLogger(f"voxelflex.{name}") # Use hierarchical naming


# --- Progress Bar ---
class EnhancedProgressBar(tqdm):
    """Custom tqdm progress bar."""
    def __init__(self, *args, stage_info: Optional[str] = None, **kwargs):
        # Note: The main change is ensuring callers use 'desc' instead of 'prefix'.
        # We don't need to explicitly handle 'prefix' here if callers use 'desc'.
        # self.stage_info = stage_info # Keep if you plan to use stage_info later

        # Set some sensible defaults if not provided by the caller
        kwargs.setdefault('ncols', 100) # Wider default?
        kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        # Ensure description (prefix) is handled by tqdm via 'desc' kwarg passed by caller
        super().__init__(*args, **kwargs)

    # update and finish methods remain the same as generated previously
    def update(self, n=1):
        super().update(n)
        # Example: Add dynamic postfix (optional)
        # mem_info = psutil.virtual_memory()
        # self.set_postfix_str(f"Mem: {mem_info.percent:.1f}%", refresh=False)

    def finish(self):
        """Close the progress bar."""
        # Ensure the bar is refreshed before closing to show final state
        self.refresh()
        self.close()

# --- Context Managers and Helpers ---

@contextmanager
def log_stage(stage_name: str, description: Optional[str] = None):
    """Logs the start and end of a processing stage."""
    logger = get_logger("pipeline") # Use a dedicated pipeline logger
    logger.info(f"--- Starting Stage: {stage_name} ---")
    if description:
        logger.info(f"  {description}")
    start_time = time.time()
    log_memory_usage(logger, level=logging.DEBUG) # Log memory at start
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"--- Finished Stage: {stage_name} (Duration: {duration:.2f}s) ---")
        log_memory_usage(logger, level=logging.DEBUG) # Log memory at end
        gc.collect() # Optional: Force GC after stage


def log_section_header(logger_instance: logging.Logger, title: str):
    """Logs a formatted section header."""
    logger_instance.info("")
    logger_instance.info("=" * 80)
    logger_instance.info(f"===== {title.upper()} =====")
    logger_instance.info("=" * 80)


def log_memory_usage(logger_instance: logging.Logger, level: int = logging.INFO):
    """Logs current system and GPU memory usage."""
    try:
        # System RAM
        mem_info = psutil.virtual_memory()
        total_gb = mem_info.total / (1024**3)
        available_gb = mem_info.available / (1024**3)
        used_gb = mem_info.used / (1024**3)
        percent_used = mem_info.percent
        logger_instance.log(level, f"Sys Memory: {used_gb:.2f}/{total_gb:.2f} GB Used ({percent_used:.1f}%) | Available: {available_gb:.2f} GB")

        # Process Memory (Current Python process)
        process = psutil.Process(os.getpid())
        process_mem_mb = process.memory_info().rss / (1024**2) # Resident Set Size
        logger_instance.log(level, f"Process Memory: {process_mem_mb:.2f} MB")

        # GPU Memory (if CUDA available)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
                allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
                free_mem = total_mem - reserved_mem # More accurate available estimate
                logger_instance.log(level, f"  GPU {i} Mem: Allocated={allocated_mem:.2f} GB | Reserved={reserved_mem:.2f} GB | Total={total_mem:.2f} GB")
    except Exception as e:
        logger_instance.warning(f"Could not retrieve memory usage details: {e}", exc_info=False) # Avoid full traceback for this warning


# Example usage at the end of a script or major section
def log_final_memory_state(logger_instance: logging.Logger):
    logger_instance.info("--- Final Memory State ---")
    log_memory_usage(logger_instance, level=logging.INFO)


# Simple pipeline tracker (can be expanded)
pipeline_tracker: Dict[str, Any] = {}

def track_metric(key: str, value: Any):
    """Stores a metric in the global tracker."""
    pipeline_tracker[key] = value

def log_tracked_metrics(logger_instance: logging.Logger):
    """Logs all tracked metrics."""
    if pipeline_tracker:
        logger_instance.info("--- Tracked Pipeline Metrics ---")
        for key, value in pipeline_tracker.items():
            logger_instance.info(f"  {key}: {value}")
        pipeline_tracker.clear() # Clear after logging
