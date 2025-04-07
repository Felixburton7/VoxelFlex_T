# src/voxelflex/cli/commands/train.py (Optimized)
"""
Training command for VoxelFlex (Temperature-Aware).

Loads sample metadata from the preprocessed file, reads corresponding
HDF5 voxel data on demand via DataLoader workers, and trains the model.
"""

import os
import time
import json
import logging
import math
from typing import Dict, Any, Tuple, List, Optional, Callable
import shutil
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import BatchSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from scipy.stats import pearsonr

logger = logging.getLogger("voxelflex.cli.train")

# Project imports
from voxelflex.data.data_loader import VoxelDataset, worker_init_fn, simple_collate_fn
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import (
    log_stage, EnhancedProgressBar, log_memory_usage, log_section_header, get_logger
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, resolve_path
from voxelflex.utils.system_utils import (
    get_device, clear_memory, check_memory_usage,
    set_num_threads, adjust_workers_for_memory
)

# --- Timer for Performance Monitoring ---
class Timing:
    """Simple class to track execution time of sections of code."""
    def __init__(self):
        self.timings = {}
        self.starts = {}
        
    def start(self, section):
        """Start timing a section."""
        self.starts[section] = time.time()
        
    def end(self, section):
        """End timing a section and record elapsed time."""
        if section in self.starts:
            elapsed = time.time() - self.starts.pop(section)
            self.timings.setdefault(section, []).append(elapsed)
            return elapsed
        return 0
        
    def get_stats(self):
        """Get timing statistics."""
        stats = {}
        for section, times in self.timings.items():
            stats[section] = {
                'avg': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
                'total': sum(times),
                'count': len(times)
            }
        return stats

# --- Train/Validate Epoch Functions ---

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    timing: Optional[Timing] = None
) -> Tuple[float, float]:
    """
    Performs one training epoch with performance tracking.
    """
    model.train() # Set model to training mode
    running_loss_sum = 0.0
    batch_preds_all = []
    batch_targets_all = []
    num_samples_processed = 0

    train_cfg = config['training']
    gradient_clip_cfg = train_cfg.get('gradient_clipping', {})
    gradient_clip_norm = gradient_clip_cfg.get('max_norm') if gradient_clip_cfg.get('enabled') else None
    show_progress = config['logging'].get('show_progress_bars', True)
    autocast_enabled = scaler is not None and scaler.is_enabled()
    
    # Gradient accumulation steps
    gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    effective_batch_multiplier = gradient_accumulation_steps
    
    # Set up progress bar
    loader_len = len(train_loader) if hasattr(train_loader, '__len__') else None
    if show_progress:
        progress = EnhancedProgressBar(
            total=loader_len, 
            desc=f"Epoch {epoch+1} Train"
        )
    else:
        progress = None
    
    if loader_len == 0:
        logger.warning("Training loader has zero length. Skipping epoch.")
        if progress: progress.close()
        return 0.0, 0.0

    # For tracking total batch processing time
    if timing:
        timing.start('train_epoch_total')
        
    optimizer.zero_grad(set_to_none=True)  # Initial grad clear
    running_grad_norm = 0.0
    
    # --- Batch Loop ---
    for i, batch_dict in enumerate(train_loader):
        if timing:
            timing.start('batch_process')
            
        # Handle potential None batches from collate_fn if all samples failed
        if batch_dict is None:
            logger.debug(f"Skipping None batch at index {i}")
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue

        try:
            if timing: timing.start('data_transfer')
            # Move data to device
            voxel_inputs = batch_dict['voxels'].to(device, non_blocking=True)
            scaled_temps = batch_dict['scaled_temps'].to(device, non_blocking=True)
            targets = batch_dict['targets'].to(device, non_blocking=True)
            if timing: timing.end('data_transfer')
            
            current_batch_size = voxel_inputs.size(0)
        except Exception as load_e:
            logger.debug(f"Error preparing batch {i}: {load_e}")
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue
            
        # --- Forward Pass ---
        try:
            if timing: timing.start('forward')
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(voxel_input=voxel_inputs, scaled_temp=scaled_temps)
                loss = criterion(outputs, targets) / gradient_accumulation_steps  # Scale loss for accumulation
            if timing: timing.end('forward')
                
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss in batch {i}. Skipping.")
                if progress: progress.update(1)
                if timing: timing.end('batch_process')
                continue
        except Exception as forward_e:
            logger.debug(f"Forward pass error in batch {i}: {forward_e}")
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue

        # --- Backward Pass ---
        try:
            if timing: timing.start('backward')
            if scaler is not None:  # AMP enabled
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if timing: timing.end('backward')
                
            # Accumulate unscaled gradients for average norm calculation
            if gradient_clip_norm:
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.detach().data.norm(2)
                            running_grad_norm += param_norm.item() ** 2
        except Exception as backward_e:
            logger.debug(f"Backward pass error in batch {i}: {backward_e}")
            optimizer.zero_grad(set_to_none=True)
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue

        # --- Step Optimizer if at gradient accumulation boundary ---
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            if timing: timing.start('optimizer_step')
            
            # Calculate gradient norm (before clipping)
            total_norm = 0.0
            if gradient_clip_norm:
                total_norm = running_grad_norm ** 0.5
                running_grad_norm = 0.0
            
            try:
                if scaler is not None:  # AMP enabled
                    if gradient_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:  # No AMP
                    if gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                
                if gradient_clip_norm and total_norm > gradient_clip_norm * 10:
                    logger.debug(f"Epoch {epoch+1}, batch {i}: Large gradient norm {total_norm:.1f} clipped to {gradient_clip_norm}")
                    
            except Exception as optim_e:
                logger.debug(f"Optimizer step error in batch {i}: {optim_e}")
                optimizer.zero_grad(set_to_none=True)
                
            if timing: timing.end('optimizer_step')

        # --- Accumulate Results ---
        running_loss_sum += (loss.item() * gradient_accumulation_steps) * current_batch_size  # Rescale loss
        batch_preds_all.append(outputs.detach().cpu().numpy())
        batch_targets_all.append(targets.detach().cpu().numpy())
        num_samples_processed += current_batch_size

        # --- Cleanup and Progress ---
        if progress: progress.update(1)
        del voxel_inputs, scaled_temps, targets, outputs, loss
        if i % 20 == 0 and device.type == 'cuda': 
            torch.cuda.empty_cache()
            
        if timing: timing.end('batch_process')
    # --- End Batch Loop ---

    if progress: progress.close()
    
    # Log cache statistics if available
    try:
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            logger.debug(f"Train cache stats: {cache_stats}")
    except Exception:
        pass

    # --- Calculate Epoch Metrics ---
    if timing: timing.start('metrics_calculation')
    avg_loss = running_loss_sum / num_samples_processed if num_samples_processed > 0 else 0.0
    epoch_pearson = 0.0
    if num_samples_processed > 1 and batch_preds_all and batch_targets_all:
        try:
            all_preds_flat = np.concatenate([b.flatten() for b in batch_preds_all])
            all_targets_flat = np.concatenate([b.flatten() for b in batch_targets_all])
            valid_mask = np.isfinite(all_preds_flat) & np.isfinite(all_targets_flat)
            preds_valid = all_preds_flat[valid_mask]
            targets_valid = all_targets_flat[valid_mask]
            if len(preds_valid) > 1 and np.std(preds_valid) > 1e-6 and np.std(targets_valid) > 1e-6:
                corr, _ = pearsonr(preds_valid, targets_valid)
                epoch_pearson = corr if not np.isnan(corr) else 0.0
        except Exception as e:
            logger.debug(f"Correlation calculation error: {e}")
    if timing: timing.end('metrics_calculation')
    
    if timing:
        timing.end('train_epoch_total')
        timing_stats = timing.get_stats()
        logger.debug(f"Train epoch timing: total={timing_stats.get('train_epoch_total', {}).get('total', 0):.2f}s")

    return avg_loss, epoch_pearson


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    timing: Optional[Timing] = None
) -> Tuple[float, float]:
    """
    Performs validation with performance tracking.
    """
    model.eval() # Set model to evaluation mode
    running_loss_sum = 0.0
    batch_preds_all = []
    batch_targets_all = []
    num_samples_processed = 0
    autocast_enabled = config['training'].get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda'
    show_progress = config['logging'].get('show_progress_bars', True)

    # Progress bar setup
    loader_len = len(val_loader) if hasattr(val_loader, '__len__') else None
    if show_progress:
        progress = EnhancedProgressBar(total=loader_len, desc="Validation")
    else:
        progress = None

    if loader_len == 0:
        logger.warning("Validation loader has zero length. Skipping validation.")
        if progress: progress.close()
        return float('inf'), 0.0

    # Start timing validation epoch
    if timing:
        timing.start('val_epoch_total')

    # --- Batch Loop ---
    with torch.no_grad(): # Disable gradient calculations for validation
        for i, batch_dict in enumerate(val_loader):
            if timing:
                timing.start('val_batch_process')
                
            if batch_dict is None:
                if progress: progress.update(1)
                if timing: timing.end('val_batch_process')
                continue
                
            try:
                if timing: timing.start('val_data_transfer')
                voxel_inputs = batch_dict['voxels'].to(device, non_blocking=True)
                scaled_temps = batch_dict['scaled_temps'].to(device, non_blocking=True)
                targets = batch_dict['targets'].to(device, non_blocking=True)
                if timing: timing.end('val_data_transfer')
                
                current_batch_size = voxel_inputs.size(0)
            except Exception as e:
                logger.debug(f"Val data transfer error in batch {i}: {e}")
                if progress: progress.update(1)
                if timing: timing.end('val_batch_process')
                continue

            try:
                if timing: timing.start('val_forward')
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                    outputs = model(voxel_input=voxel_inputs, scaled_temp=scaled_temps)
                    loss = criterion(outputs, targets)
                if timing: timing.end('val_forward')
                    
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.debug(f"Val NaN/Inf loss in batch {i}. Skipping.")
                    if progress: progress.update(1)
                    if timing: timing.end('val_batch_process')
                    continue

                # Accumulate results
                running_loss_sum += loss.item() * current_batch_size
                batch_preds_all.append(outputs.cpu().numpy())
                batch_targets_all.append(targets.cpu().numpy())
                num_samples_processed += current_batch_size

            except Exception as e:
                logger.debug(f"Val forward error in batch {i}: {e}")
                if progress: progress.update(1)
                if timing: timing.end('val_batch_process')
                continue

            # --- Cleanup and Progress ---
            if progress: progress.update(1)
            del voxel_inputs, scaled_temps, targets, outputs, loss
            if i % 20 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
            if timing: timing.end('val_batch_process')
        # --- End Batch Loop ---

    if progress: progress.close()
    
    # Log cache statistics if available
    try:
        if hasattr(val_loader.dataset, 'get_cache_stats'):
            cache_stats = val_loader.dataset.get_cache_stats()
            logger.debug(f"Val cache stats: {cache_stats}")
    except Exception:
        pass

    # --- Calculate Metrics ---
    if timing: timing.start('val_metrics_calculation')
    avg_loss = running_loss_sum / num_samples_processed if num_samples_processed > 0 else float('inf')
    epoch_pearson = 0.0
    if num_samples_processed > 1 and batch_preds_all and batch_targets_all:
        try:
            all_preds_flat = np.concatenate([b.flatten() for b in batch_preds_all])
            all_targets_flat = np.concatenate([b.flatten() for b in batch_targets_all])
            valid_mask = np.isfinite(all_preds_flat) & np.isfinite(all_targets_flat)
            preds_valid = all_preds_flat[valid_mask]
            targets_valid = all_targets_flat[valid_mask]
            if len(preds_valid) > 1 and np.std(preds_valid) > 1e-6 and np.std(targets_valid) > 1e-6:
                corr, _ = pearsonr(preds_valid, targets_valid)
                epoch_pearson = corr if not np.isnan(corr) else 0.0
        except Exception as e:
            logger.debug(f"Val correlation calculation error: {e}")
    if timing: timing.end('val_metrics_calculation')
    
    if timing:
        timing.end('val_epoch_total')
        timing_stats = timing.get_stats()
        logger.debug(f"Val epoch timing: total={timing_stats.get('val_epoch_total', {}).get('total', 0):.2f}s")

    return avg_loss, epoch_pearson

# --- Helper Functions: Optimizer and Scheduler ---
def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Creates an optimizer based on the configuration."""
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer created: AdamW (LR={lr:.2e}, WeightDecay={weight_decay:.2e})")
    return optimizer

def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], num_epochs: int):
    """Creates a learning rate scheduler based on the configuration."""
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'reduce_on_plateau').lower()
    monitor_metric = scheduler_config.get('monitor_metric', 'val_pearson')
    if monitor_metric not in ['val_loss', 'val_pearson']:
        logger.warning(f"Invalid scheduler metric '{monitor_metric}'. Defaulting to 'val_pearson'.")
        monitor_metric = 'val_pearson'
    mode = scheduler_config.get('mode')
    if mode not in ['min', 'max']:
        mode = 'max' if 'pearson' in monitor_metric else 'min'
        logger.debug(f"Scheduler mode auto-set to '{mode}'.")
    scheduler = None
    if scheduler_type == 'reduce_on_plateau':
        patience=int(scheduler_config.get('patience', 5))
        factor=float(scheduler_config.get('factor', 0.5))
        min_lr=float(scheduler_config.get('min_lr', 1e-7))
        threshold=float(scheduler_config.get('threshold', 0.001))
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr, verbose=True)
        logger.info(f"Using ReduceLROnPlateau scheduler (Metric: {monitor_metric}, Mode: {mode})")
    elif scheduler_type == 'cosine_annealing':
        T_max = int(scheduler_config.get('T_max', num_epochs))
        eta_min = float(scheduler_config.get('eta_min', 1e-7))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        logger.info(f"Using CosineAnnealingLR scheduler (T_max={T_max}, eta_min={eta_min:.1e})")
    elif scheduler_type == 'step':
        step_size = int(scheduler_config.get('step_size', 10))
        gamma = float(scheduler_config.get('gamma', 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Using StepLR scheduler (Step: {step_size}, Gamma: {gamma})")
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. None used.")
    return scheduler, monitor_metric, mode

# --- Main Training Function ---

def train_model(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, List[float]]]]:
    """
    Main function to train the model using VoxelDataset with on-demand loading.
    Assumes preprocessing has run and the master sample file exists.
    """
    # --- Setup directories and device ---
    run_output_dir = config["output"]["run_dir"]
    models_dir = config["output"]["models_dir"]
    ensure_dir(models_dir)
    
    log_section_header(logger, f"STARTING TRAINING RUN (Optimized HDF5): {config['output']['run_name']}")
    log_memory_usage(logger)
    
    # Set up device and seed
    device = get_device(config["system_utilization"].get("adjust_for_gpu", True))
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Initialize timer if enabled
    log_timing = config.get('logging', {}).get('log_timing', False)
    timing = Timing() if log_timing else None

    # --- Load Datasets & DataLoaders ---
    with log_stage("DATA_PREPARATION", "Creating Datasets & DataLoaders"):
        try:
            master_samples_path = os.path.join(config["data"]["processed_dir"], config["data"]["master_samples_file"])
            if not os.path.exists(master_samples_path):
                raise FileNotFoundError(f"Master sample file not found: {master_samples_path}.")
            
            voxel_hdf5_path = config['input']['voxel_file']
            if not voxel_hdf5_path or not os.path.exists(voxel_hdf5_path):
                raise FileNotFoundError(f"Voxel HDF5 path invalid/missing: {voxel_hdf5_path}")

            voxel_c = config['model']['input_channels']
            voxel_d = config['model'].get('voxel_depth', 21)
            voxel_h = config['model'].get('voxel_height', 21)
            voxel_w = config['model'].get('voxel_width', 21)
            voxel_shape = (voxel_c, voxel_d, voxel_h, voxel_w)
            expected_channels = voxel_c

            logger.info("Creating training dataset...")
            train_dataset = VoxelDataset(
                config["data"]["processed_dir"], 
                'train', 
                voxel_hdf5_path, 
                config,  # Pass full config now
                expected_channels=expected_channels, 
                target_shape_chw=voxel_shape
            )
            
            logger.info("Creating validation dataset...")
            val_dataset = VoxelDataset(
                config["data"]["processed_dir"], 
                'val', 
                voxel_hdf5_path, 
                config,  # Pass full config
                expected_channels=expected_channels, 
                target_shape_chw=voxel_shape
            )

            if len(train_dataset) == 0:
                raise ValueError("Training dataset empty.")
            if len(val_dataset) == 0:
                raise ValueError("Validation dataset empty.")

            train_batch_size = config['training']['batch_size']
            val_batch_size = train_batch_size * 2
            num_workers = config['training'].get('num_workers', 0)
            # num_workers = adjust_workers_for_memory(num_workers)  # No longer needed with better cache
            
            pin_memory = config['training'].get('pin_memory', True) and (device.type == 'cuda')
            persistent_workers = config['training'].get('persistent_workers', True)  # Default to True now
            prefetch_factor = config['training'].get('prefetch_factor', 6)  # Increased from default
            
            logger.info(f"Train DataLoader: {len(train_dataset)} samples, BatchSize={train_batch_size}, Workers={num_workers}")
            train_loader = DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                shuffle=True,
                num_workers=num_workers, 
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=simple_collate_fn,  # Use optimized collate
                drop_last=True, 
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                worker_init_fn=worker_init_fn  # Use persistent HDF5 connections
            )
            
            logger.info(f"Val DataLoader: {len(val_dataset)} samples, BatchSize={val_batch_size}, Workers={num_workers}")
            val_loader = DataLoader(
                val_dataset, 
                batch_size=val_batch_size, 
                shuffle=False,
                num_workers=num_workers, 
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=simple_collate_fn,  # Use optimized collate
                drop_last=False, 
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                worker_init_fn=worker_init_fn  # Use persistent HDF5 connections
            )

        except Exception as e:
            logger.exception(f"Fatal error loading sample data/creating DataLoaders: {e}")
            return None, None

    # --- Model & Optimizer ---
    with log_stage("MODEL_CREATION", "Creating model and optimizer"):
        try:
            input_shape = voxel_shape
            logger.info(f"Using model input shape: {input_shape}")
            
            model = get_model(config['model'], input_shape=input_shape)
            model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model '{config['model']['architecture']}' created. " 
                       f"Params: Total={total_params:,}, Trainable={trainable_params:,}")
            
            optimizer = get_optimizer(model, config)
            num_epochs = config['training']['num_epochs']
            scheduler, monitor_metric, scheduler_mode = get_scheduler(optimizer, config, num_epochs)
            
        except Exception as e:
            logger.exception(f"Fatal error creating model/optimizer: {e}")
            return None, None

    # --- Resume / Training Prep ---
    start_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_pearson': [], 'val_pearson': [], 
        'lr': [],
        'epoch_time': []  # Track epoch duration
    }
    best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
    best_epoch = -1
    resume_path = config['training'].get('resume_checkpoint')
    
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Attempting resume from: {resume_path}")
        try: 
            checkpoint = torch.load(resume_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing_keys or unexpected_keys:
                logger.warning(f"Resume state dict - Missing: {missing_keys or 'None'}, "
                              f"Unexpected: {unexpected_keys or 'None'}")
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            history = checkpoint.get('history', history)
            best_metric_value = checkpoint.get(f'best_{monitor_metric}', best_metric_value)
            best_epoch = checkpoint.get('best_epoch', -1)
            
            if scheduler and 'scheduler_state_dict' in checkpoint and type(scheduler).__name__ == checkpoint.get('scheduler_type'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Resuming scheduler state.")
            elif scheduler and 'scheduler_state_dict' in checkpoint:
                logger.warning("Scheduler type mismatch. Cannot resume state.")
                
            logger.info(f"Resumed from epoch {start_epoch}. "
                       f"Best '{monitor_metric}': {best_metric_value:.6f} @ epoch {best_epoch}")
            
            del checkpoint
            gc.collect()
            
        except Exception as e:
            logger.exception(f"Failed resume: {e}. Starting fresh.")
            start_epoch = 0
            history = {
                'train_loss': [], 'val_loss': [], 
                'train_pearson': [], 'val_pearson': [], 
                'lr': [],
                'epoch_time': []
            }
            best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
            best_epoch = -1
    else:
        logger.info("No resume checkpoint. Starting fresh.")

    # --- Training Loop Setup ---
    criterion = nn.MSELoss()
    
    # Ensure mixed precision is properly configured
    mixed_precision_enabled = config['training']['mixed_precision'].get('enabled', False)
    use_amp = mixed_precision_enabled and device.type == 'cuda'
    
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP)")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        logger.info("Mixed precision disabled")
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    # Early stopping setup
    early_stopping_cfg = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_cfg.get('enabled', True)
    early_stopping_patience = early_stopping_cfg.get('patience', 10)
    early_stopping_delta = early_stopping_cfg.get('min_delta', 0.001)
    early_stopping_counter = 0
    early_stopping_metric = early_stopping_cfg.get('monitor_metric', monitor_metric)
    early_stopping_mode = early_stopping_cfg.get('mode', scheduler_mode)
    
    if early_stopping_metric != monitor_metric or early_stopping_mode != scheduler_mode:
        logger.warning(f"Early stopping metric/mode differs. Using scheduler's.")
        early_stopping_metric = monitor_metric
        early_stopping_mode = scheduler_mode
        
    save_best_metric = config['training'].get('save_best_metric', monitor_metric)
    save_best_mode = config['training'].get('save_best_mode', scheduler_mode)
    
    # Gradient accumulation setup
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        logger.info(f"Effective batch size: {train_batch_size * gradient_accumulation_steps}")

    # --- TRAINING LOOP ---
    start_train_loop = time.time()
    log_section_header(logger, f"STARTING TRAINING LOOP (Epochs {start_epoch+1} to {num_epochs})")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # --- Run Train Epoch ---
        avg_epoch_train_loss, avg_epoch_train_corr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler, timing
        )
        
        # Handle potential NaN/Inf values
        if np.isnan(avg_epoch_train_loss) or np.isinf(avg_epoch_train_loss):
            avg_epoch_train_loss = history['train_loss'][-1] if history['train_loss'] else 1e9
            logger.error(f"Ep {epoch+1}: Invalid train loss. Fallback: {avg_epoch_train_loss:.4f}")
            
        if np.isnan(avg_epoch_train_corr) or np.isinf(avg_epoch_train_corr):
            avg_epoch_train_corr = history['train_pearson'][-1] if history['train_pearson'] else 0.0
            logger.error(f"Ep {epoch+1}: Invalid train corr. Fallback: {avg_epoch_train_corr:.4f}")
            
        history['train_loss'].append(avg_epoch_train_loss)
        history['train_pearson'].append(avg_epoch_train_corr)

        # --- Run Validation Epoch ---
        avg_epoch_val_loss, avg_epoch_val_corr = validate(
            model, val_loader, criterion, device, config, timing
        )
        
        # Handle potential NaN/Inf values
        if np.isnan(avg_epoch_val_loss) or np.isinf(avg_epoch_val_loss):
            avg_epoch_val_loss = history['val_loss'][-1] if history['val_loss'] else 1e9
            logger.error(f"Ep {epoch+1}: Invalid val loss. Fallback: {avg_epoch_val_loss:.4f}")
            
        if np.isnan(avg_epoch_val_corr) or np.isinf(avg_epoch_val_corr):
            avg_epoch_val_corr = history['val_pearson'][-1] if history['val_pearson'] else 0.0
            logger.error(f"Ep {epoch+1}: Invalid val corr. Fallback: {avg_epoch_val_corr:.4f}")
            
        history['val_loss'].append(avg_epoch_val_loss)
        history['val_pearson'].append(avg_epoch_val_corr)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Track epoch time
        epoch_duration = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_duration)

        # --- Log Epoch Results ---
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} --- | Time: {epoch_duration:.1f}s | LR: {history['lr'][-1]:.2e}")
        logger.info(f"  Train -> Loss: {avg_epoch_train_loss:.6f} | Pearson: {avg_epoch_train_corr:.6f}")
        logger.info(f"  Valid -> Loss: {avg_epoch_val_loss:.6f} | Pearson: {avg_epoch_val_corr:.6f}")

        # --- Update Scheduler ---
        current_metric_val = avg_epoch_val_corr if monitor_metric == 'val_pearson' else avg_epoch_val_loss
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_metric_val)
            else:
                scheduler.step()

        # --- Check for Best Model ---
        is_best = False
        if not np.isnan(current_metric_val):
            if save_best_mode == 'max' and current_metric_val > best_metric_value + early_stopping_delta:
                is_best = True
                improvement = current_metric_val - best_metric_value
            elif save_best_mode == 'min' and current_metric_val < best_metric_value - early_stopping_delta:
                is_best = True
                improvement = best_metric_value - current_metric_val
        else:
            logger.warning(f"Ep {epoch+1}: Monitored metric '{save_best_metric}' NaN.")

        # --- Save Checkpoints ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'input_shape': input_shape,
            # Save hard-coded temperature scaling constants (no file reference)
            'temp_scaling_params': {
                'temp_min': train_dataset.temp_min,
                'temp_max': train_dataset.temp_max
            },
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scheduler_type': type(scheduler).__name__ if scheduler else None
        }
        
        if is_best:
            logger.info(f"  >>> New best {save_best_metric}: {current_metric_val:.6f}. Saving best model...")
            best_metric_value = current_metric_val
            best_epoch = epoch + 1
            early_stopping_counter = 0
            
            best_model_path = os.path.join(models_dir, "best_model.pt")
            best_checkpoint_data = checkpoint_data.copy()
            best_checkpoint_data[f'best_{save_best_metric}'] = best_metric_value
            best_checkpoint_data['best_epoch'] = best_epoch
            
            try:
                torch.save(best_checkpoint_data, best_model_path)
            except Exception as save_e:
                logger.error(f"Failed save best model: {save_e}")
        else:
            early_stopping_counter += 1
            logger.info(f"  {save_best_metric} did not improve. "
                      f"Best: {best_metric_value:.6f} @ Ep {best_epoch}. "
                      f"Early stopping: {early_stopping_counter}/{early_stopping_patience}")

        # Periodic checkpoint saving
        chkpt_interval = config['training'].get('checkpoint_interval', 0)
        if chkpt_interval > 0 and (epoch + 1) % chkpt_interval == 0:
            chkpt_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1}.pt")
            logger.info(f"Saving periodic checkpoint: {chkpt_path}")
            
            periodic_checkpoint_data = checkpoint_data.copy()
            periodic_checkpoint_data['history'] = history
            periodic_checkpoint_data[f'best_{save_best_metric}'] = best_metric_value
            periodic_checkpoint_data['best_epoch'] = best_epoch
            
            try:
                torch.save(periodic_checkpoint_data, chkpt_path)
            except Exception as save_e:
                logger.error(f"Failed save periodic checkpoint: {save_e}")

        # Always save latest model
        latest_model_path = os.path.join(models_dir, "latest_model.pt")
        latest_checkpoint_data = checkpoint_data.copy()
        latest_checkpoint_data['history'] = history
        
        try:
            torch.save(latest_checkpoint_data, latest_model_path)
        except Exception as save_e:
            logger.error(f"Failed save latest model: {save_e}")

        # Early stopping check
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
        # Clean up memory
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
        
        # Log memory usage and cache statistics if enabled
        if config.get('logging', {}).get('log_memory_usage', True):
            log_memory_usage(logger)

        # Log performance timing if enabled
        if timing:
            timing_stats = timing.get_stats()
            logger.info("Timing statistics:")
            for section, stats in timing_stats.items():
                logger.info(f"  {section}: avg={stats['avg']:.4f}s, total={stats['total']:.2f}s, count={stats['count']}")

    # --- Finalization ---
    training_duration = time.time() - start_train_loop
    log_section_header(logger, "TRAINING FINISHED")
    logger.info(f"Total Training Time: {training_duration:.2f}s ({training_duration/60:.1f} mins)")
    
    final_model_path = None
    if best_epoch != -1:
        logger.info(f"Best {save_best_metric}: {best_metric_value:.6f} @ epoch {best_epoch}")
        final_model_path = os.path.join(models_dir, "best_model.pt")
    else:
        logger.warning("No improvement observed.")
        final_model_path = os.path.join(models_dir, "latest_model.pt")
        logger.info(f"Using latest model from epoch {epoch+1}: {final_model_path}")
        
    if not final_model_path or not os.path.exists(final_model_path):
        logger.error("Could not find final model checkpoint.")
        final_model_path = None
        
    # Save complete training history
    history_path = os.path.join(run_output_dir, "training_history.json")
    try:
        save_json(history, history_path)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e:
        logger.error(f"Failed save history: {e}")
        
    # Generate training plots
    try:
        from voxelflex.cli.commands.visualize import create_loss_curve, create_correlation_curve
        viz_dir = config["output"]["visualizations_dir"]
        logger.info("Generating final training plots...")
        
        if history.get('train_loss') and history.get('val_loss'):
            create_loss_curve(
                history, viz_dir, 
                save_format=config['visualization']['save_format'], 
                dpi=config['visualization']['dpi']
            )
            
        if history.get('train_pearson') and history.get('val_pearson'):
            create_correlation_curve(
                history, viz_dir, 
                save_format=config['visualization']['save_format'], 
                dpi=config['visualization']['dpi']
            )
    except ImportError:
        logger.warning("Could not import visualize functions.")
    except Exception as plot_e:
        logger.error(f"Failed generate final plots: {plot_e}")
        
    # Final cleanup
    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    
    return final_model_path, history