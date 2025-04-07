#!/bin/bash

# ==========================================================
# Create VoxelFlex (Temperature-Aware) Package Script
# ==========================================================
# This script generates the directory structure and Python files
# for the VoxelFlex package based on the preprocessing workflow.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Creating VoxelFlex package structure..."

# --- Create Directories ---
# Package source
mkdir -p src/voxelflex/config
mkdir -p src/voxelflex/data
mkdir -p src/voxelflex/models
mkdir -p src/voxelflex/utils
mkdir -p src/voxelflex/cli/commands
# Tests
mkdir -p tests
# Example Input Data Structure
mkdir -p input_data/voxel
mkdir -p input_data/rmsf
mkdir -p input_data/processed/train
mkdir -p input_data/processed/val
mkdir -p input_data/processed/test
# Default Output Base Directory
mkdir -p outputs

echo "Creating Python package files (__init__.py)..."
# --- Create __init__.py files ---
touch src/voxelflex/__init__.py
touch src/voxelflex/config/__init__.py
touch src/voxelflex/data/__init__.py
touch src/voxelflex/models/__init__.py
touch src/voxelflex/utils/__init__.py
touch src/voxelflex/cli/__init__.py
touch src/voxelflex/cli/commands/__init__.py
touch tests/__init__.py

echo "Generating Python source files..."

# --- src/voxelflex/config/default_config.yaml ---
cat << 'EOF' > src/voxelflex/config/default_config.yaml
# src/voxelflex/config/default_config.yaml (Preprocessing Workflow)

input:
  # --- Raw Data ---
  voxel_file: input_data/voxel/mdcath_voxelized.hdf5
  aggregated_rmsf_file: input_data/rmsf/aggregated_rmsf_all_temps.csv
  # --- Domain Split Files (REQUIRED for preprocessing) ---
  train_split_file: input_data/train_domains.txt
  val_split_file: input_data/val_domains.txt
  test_split_file: input_data/test_domains.txt # Optional if no testing needed
  # --- Optional Filtering ---
  # domain_ids: [] # Filter specific domains BEFORE preprocessing (applied to split lists)
  max_domains: null # Limit total domains considered during preprocessing (applied to split lists)

data:
  # --- Processed Data Output ---
  # Base directory where preprocessed batches (.pt) and metadata (.meta) will be stored
  processed_dir: input_data/processed/ # Suggest storing processed data near input
  # Filename (relative to output.models_dir) for storing/loading temp scaling params
  # This is generated during preprocessing based on training split
  temp_scaling_params_file: "temp_scaling_params.json" # Base name, full path constructed later

  # --- Preprocessing Settings ---
  # Batch size used when SAVING preprocessed .pt files (can differ from training batch size)
  # Controls how many samples are grouped into each saved .pt file.
  preprocessing_batch_size: 256 # Larger batch size for saving might be efficient
  # Max number of domains to hold in the in-memory voxel cache during preprocessing a split.
  # Adjust based on available RAM and typical domain size. ~500-1000 often feasible.
  preprocessing_cache_limit: 750

output:
  base_dir: outputs/
  run_name: "voxelflex_run_{timestamp}" # Timestamp will be filled in
  log_file: voxelflex.log # Relative to run_dir/logs/

model:
  # Preferred architecture based on dissertation context
  architecture: multipath_rmsf_net # Options: densenet3d_regression, dilated_resnet3d, multipath_rmsf_net
  input_channels: 5 # Should match the first dimension of transposed voxel data
  # --- DenseNet Specific ---
  densenet: {
    growth_rate: 16,
    block_config: [4, 4, 4], # Layers per block (e.g., 3 blocks)
    num_init_features: 32, # Features after initial conv
    bn_size: 4              # Bottleneck factor
  }
  # --- DilatedResNet / Multipath Specific ---
  channel_growth_rate: 1.5 # Factor to increase channels in blocks
  num_residual_blocks: 3   # Number of residual blocks (per path for multipath)
  base_filters: 32         # Initial filter count for ResNet/Multipath
  # --- General ---
  dropout_rate: 0.3

training:
  # Batch size for TRAINING DataLoader (loading preprocessed .pt files).
  # Should generally match or be related to `preprocessing_batch_size`.
  batch_size: 256
  num_epochs: 50
  learning_rate: 0.0005
  weight_decay: 1e-4
  seed: 42
  # --- Data Loading during Training ---
  num_workers: 8 # Adjust based on available CPU cores
  pin_memory: true # Set to true if using GPU for potentially faster transfer
  persistent_workers: true # Keep workers alive between epochs if num_workers > 0
  # --- Checkpointing & Saving ---
  resume_checkpoint: null # Path to a checkpoint file to resume training
  save_best_metric: "val_pearson" # Metric to monitor ['val_loss', 'val_pearson']
  save_best_mode: "max"           # 'min' for loss, 'max' for pearson
  checkpoint_interval: 5          # Save a checkpoint every N epochs (0 to disable)
  # --- Optimization & Scheduling ---
  gradient_clipping: { enabled: true, max_norm: 1.0 }
  mixed_precision: { enabled: true } # Enable Automatic Mixed Precision (AMP) if using GPU
  scheduler:
    type: reduce_on_plateau # Options: reduce_on_plateau, cosine_annealing, step
    monitor_metric: "val_pearson" # Metric to monitor ['val_loss', 'val_pearson']
    # --- ReduceLROnPlateau ---
    mode: "max"       # 'min' for loss, 'max' for pearson
    patience: 5       # Epochs to wait for improvement
    factor: 0.5       # Factor to reduce LR by
    min_lr: 1e-7      # Minimum learning rate
    threshold: 0.001  # Min change to qualify as improvement
    # --- CosineAnnealingLR ---
    T_max: 50         # Max iterations (usually num_epochs)
    eta_min: 1e-7     # Minimum learning rate
    # --- StepLR ---
    step_size: 10     # Decay LR every N epochs
    gamma: 0.1        # Factor to decay LR by
  early_stopping:
    enabled: true
    patience: 10 # Epochs to wait after last best epoch
    monitor_metric: "val_pearson" # Metric to monitor ['val_loss', 'val_pearson']
    mode: "max"           # 'min' for loss, 'max' for pearson
    min_delta: 0.001      # Minimum change to qualify as improvement

prediction:
  batch_size: 512 # Batch size for running inference (predict/evaluate)

evaluation:
  calculate_stratified_metrics: true
  calculate_permutation_importance: true
  sasa_bins: [0.0, 0.1, 0.4, 1.01] # Bins for relative accessibility stratification
  permutation_n_repeats: 5         # Number of repeats for permutation importance

logging:
  level: INFO         # Overall level for logger setup
  console_level: INFO # Level for console output (can be overridden by -v)
  file_level: DEBUG   # Level for file output
  show_progress_bars: true

visualization:
  # Plots to generate during the 'visualize' command
  plot_loss: true
  plot_correlation: true
  plot_predictions: true         # Standard scatter plot
  plot_density_scatter: true     # Scatter plot with density coloring
  plot_error_distribution: true  # Histogram of errors
  plot_residue_type_analysis: true # Boxplot of errors by residue type
  plot_sasa_error_analysis: true   # Boxplot of errors by SASA bin
  plot_ss_error_analysis: true     # Boxplot of errors by secondary structure
  plot_amino_acid_performance: false # Bar plots of metrics per AA type
  # Settings
  save_format: png
  dpi: 150
  max_scatter_points: 1000 # Max points to plot in scatter (samples if more)
  save_plot_data: true     # Save data used for plots as CSV

system_utilization:
  detect_cores: true      # Automatically detect CPU cores for num_workers default
  adjust_for_gpu: true    # Automatically select GPU if available
EOF

# --- src/voxelflex/config/config.py ---
cat << 'EOF' > src/voxelflex/config/config.py
# src/voxelflex/config/config.py (Preprocessing Workflow Version)
"""
Configuration module for VoxelFlex (Temperature-Aware).

Handles loading, validation, merging with defaults, and path expansion
for YAML configuration files.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

# Use centralized logger
logger = logging.getLogger("voxelflex.config")

from voxelflex.utils.file_utils import resolve_path, ensure_dir

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file, merge with defaults, validate,
    expand paths, and create necessary output directories.

    Args:
        config_path: Path to the user-provided configuration file.

    Returns:
        A dictionary containing the fully processed configuration.
    """
    config_path_resolved = resolve_path(config_path)
    logger.info(f"Loading user configuration from: {config_path_resolved}")

    if not os.path.exists(config_path_resolved):
        raise FileNotFoundError(f"Configuration file not found: {config_path_resolved}")

    try:
        with open(config_path_resolved, 'r') as f:
            user_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {config_path_resolved}") from e

    if user_config is None:
        # Allow empty user config, defaults will be used
        logger.warning(f"User configuration file is empty: {config_path_resolved}. Using defaults.")
        user_config = {}

    # Load default config to merge missing keys
    default_config = get_default_config()
    if not default_config:
         raise RuntimeError("Failed to load default configuration.")

    config = merge_configs(default_config, user_config)

    # Validate the merged configuration before path expansion
    validate_config(config)

    # Expand paths that refer to existing files or directories
    config = expand_paths(config)

    # Add timestamped run name if needed
    if "run_name" not in config["output"] or "{timestamp}" in config["output"].get("run_name", ""):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name_template = config["output"].get("run_name", "voxelflex_run_{timestamp}")
        config["output"]["run_name"] = run_name_template.format(timestamp=timestamp)
        logger.info(f"Generated run name: {config['output']['run_name']}")

    # Construct full output paths relative to base_dir and run_name
    base_output_dir = config["output"]["base_dir"] # Already expanded
    run_output_dir = os.path.join(base_output_dir, config["output"]["run_name"])
    config["output"]["run_dir"] = run_output_dir
    config["output"]["log_dir"] = os.path.join(run_output_dir, "logs")
    config["output"]["models_dir"] = os.path.join(run_output_dir, "models")
    config["output"]["metrics_dir"] = os.path.join(run_output_dir, "metrics")
    config["output"]["visualizations_dir"] = os.path.join(run_output_dir, "visualizations")

    # Create base run directories needed early (e.g., for logging)
    # These are created here because logging setup might happen before command execution
    ensure_dir(config["output"]["log_dir"])
    ensure_dir(config["output"]["models_dir"]) # Needed for temp scaler path construction
    ensure_dir(config["output"]["metrics_dir"])
    ensure_dir(config["output"]["visualizations_dir"])

    # Construct full path for temp scaling params file RELATIVE TO MODELS_DIR
    scaling_file_name = os.path.basename(config["data"]["temp_scaling_params_file"])
    if not scaling_file_name: # Ensure basename is not empty
        scaling_file_name = "temp_scaling_params.json"
    config["data"]["temp_scaling_params_file"] = os.path.join(
        config["output"]["models_dir"],
        scaling_file_name
    )
    logger.debug(f"Temperature scaling parameter file path set to: {config['data']['temp_scaling_params_file']}")


    # Construct full paths for processed data relative to processed_dir
    processed_base = config["data"]["processed_dir"] # Already expanded
    config["data"]["processed_train_dir"] = os.path.join(processed_base, "train")
    config["data"]["processed_val_dir"] = os.path.join(processed_base, "val")
    config["data"]["processed_test_dir"] = os.path.join(processed_base, "test")
    config["data"]["processed_train_meta"] = os.path.join(processed_base, "train_batches.meta")
    config["data"]["processed_val_meta"] = os.path.join(processed_base, "val_batches.meta")
    config["data"]["processed_test_meta"] = os.path.join(processed_base, "test_batches.meta")

    logger.debug("Configuration loaded and processed successfully.")
    return config


def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config into default config."""
    merged = default.copy()
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the structure and types of the configuration dictionary."""
    logger.debug("Validating configuration structure...")
    required_sections = ['input', 'output', 'model', 'training', 'data', 'logging', 'evaluation', 'visualization', 'system_utilization']
    for section in required_sections:
        if section not in config: raise ValueError(f"Missing required section in config: '{section}'")
        if not isinstance(config[section], dict): raise ValueError(f"Section '{section}' must be a dictionary.")

    # Input section validation
    input_cfg = config['input']
    req_input = ['voxel_file', 'aggregated_rmsf_file', 'train_split_file', 'val_split_file']
    for key in req_input:
        if key not in input_cfg or not input_cfg[key]:
            raise ValueError(f"Missing or empty required input parameter: 'input.{key}'")
    if 'test_split_file' not in input_cfg or not input_cfg['test_split_file']:
         logger.warning(f"Optional input parameter 'input.test_split_file' is missing or empty. Test set processing/evaluation will be skipped if 'test' command is used.")

    # Data section validation
    data_cfg = config['data']
    req_data = ['processed_dir', 'temp_scaling_params_file', 'preprocessing_batch_size', 'preprocessing_cache_limit']
    for key in req_data:
        # Check presence and non-null value for required keys
        if key not in data_cfg or data_cfg[key] is None:
             raise ValueError(f"Missing or unset required data parameter: 'data.{key}'")
    if not isinstance(data_cfg['preprocessing_batch_size'], int) or data_cfg['preprocessing_batch_size'] <= 0:
         raise ValueError("'data.preprocessing_batch_size' must be a positive integer.")
    if not isinstance(data_cfg['preprocessing_cache_limit'], int) or data_cfg['preprocessing_cache_limit'] <= 0:
         raise ValueError("'data.preprocessing_cache_limit' must be a positive integer.")

    # Output section validation
    if 'base_dir' not in config['output'] or not config['output']['base_dir']:
        raise ValueError("Missing or empty required output parameter: 'output.base_dir'")

    # Model section validation
    model_cfg = config.get('model', {})
    if 'architecture' not in model_cfg: raise ValueError("Missing 'model.architecture'")
    valid_arch = ['densenet3d_regression', 'dilated_resnet3d', 'multipath_rmsf_net']
    if model_cfg['architecture'] not in valid_arch: raise ValueError(f"Invalid 'model.architecture'. Must be one of: {valid_arch}")
    if model_cfg['architecture'] == 'densenet3d_regression':
        if 'densenet' not in model_cfg or not isinstance(model_cfg['densenet'], dict): raise ValueError("Missing/invalid 'model.densenet' section for DenseNet.")
        req_densenet = ['growth_rate', 'block_config', 'num_init_features', 'bn_size']
        for key in req_densenet:
             if key not in model_cfg['densenet']: raise ValueError(f"Missing 'model.densenet.{key}'")
        if not isinstance(model_cfg['densenet']['block_config'], list): raise ValueError("'model.densenet.block_config' must be a list (e.g., [4, 4, 4])")

    # Training section validation
    train_cfg = config.get('training', {})
    req_train = ['batch_size', 'num_epochs', 'learning_rate', 'weight_decay', 'seed']
    for key in req_train:
        if key not in train_cfg: raise ValueError(f"Missing 'training.{key}'")
    if not isinstance(train_cfg.get('batch_size'), int) or train_cfg.get('batch_size', 0) <= 0: raise ValueError("'training.batch_size' must be positive.")
    if not isinstance(train_cfg.get('num_epochs'), int) or train_cfg.get('num_epochs', 0) <= 0: raise ValueError("'training.num_epochs' must be positive.")
    if not isinstance(train_cfg.get('num_workers', 0), int) or train_cfg.get('num_workers', 0) < 0: raise ValueError("'training.num_workers' must be non-negative.")

    # Validate metric choices
    valid_metrics = ['val_loss', 'val_pearson']
    def validate_monitor_metric(cfg_section: dict, section_key: str, section_name: str):
        metric = cfg_section.get(section_key, {}).get('monitor_metric')
        if metric and metric not in valid_metrics:
             raise ValueError(f"'training.{section_name}.monitor_metric' ('{metric}') must be one of {valid_metrics}")

    if 'save_best_metric' in train_cfg and train_cfg['save_best_metric'] not in valid_metrics:
         raise ValueError(f"'training.save_best_metric' ('{train_cfg['save_best_metric']}') must be one of {valid_metrics}")
    validate_monitor_metric(train_cfg, 'scheduler', 'scheduler')
    validate_monitor_metric(train_cfg, 'early_stopping', 'early_stopping')

    # Validate scheduler type
    sched_cfg = train_cfg.get('scheduler', {})
    if 'type' in sched_cfg and sched_cfg['type'] not in ['reduce_on_plateau', 'cosine_annealing', 'step']:
        raise ValueError(f"Invalid scheduler type: {sched_cfg['type']}")

    # System Utilization validation
    sys_cfg = config.get('system_utilization', {})
    if not isinstance(sys_cfg.get('detect_cores'), bool): raise ValueError("'system_utilization.detect_cores' must be boolean.")
    if not isinstance(sys_cfg.get('adjust_for_gpu'), bool): raise ValueError("'system_utilization.adjust_for_gpu' must be boolean.")

    logger.debug("Configuration validation passed.")


def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand path strings in the configuration relative to CWD or user home."""
    logger.debug("Expanding paths in configuration (relative to CWD)...")
    paths_to_expand = {
        ('input', 'voxel_file'),
        ('input', 'aggregated_rmsf_file'),
        ('input', 'train_split_file'),
        ('input', 'val_split_file'),
        ('input', 'test_split_file'),
        ('output', 'base_dir'),
        ('data', 'processed_dir'),
        ('training', 'resume_checkpoint'),
    }
    for section, key in paths_to_expand:
        # Check if section exists and key exists within the section
        if config.get(section) is not None and isinstance(config[section], dict) and config[section].get(key):
            original_path = config[section][key]
            # Only resolve if it's a non-empty string
            if isinstance(original_path, str) and original_path:
                resolved = resolve_path(original_path)
                # Log only if the path actually changed during resolution
                # if resolved != original_path:
                #     logger.debug(f"Path '{original_path}' resolved to '{resolved}'")
                config[section][key] = resolved
            # Handle cases where path might be None or empty, leave as is
            elif not original_path:
                 config[section][key] = None # Explicitly set empty paths to None if desired, or keep original
                 # logger.debug(f"Path for {section}.{key} is empty, setting to None.")

    return config


def get_default_config() -> Dict[str, Any]:
    """Load the default configuration from the embedded default_config.yaml file."""
    default_config_path = os.path.join(os.path.dirname(__file__),'default_config.yaml')
    # Use a local logger for this function to avoid potential setup issues if called early
    local_logger = logging.getLogger("voxelflex.config.default")
    local_logger.debug(f"Attempting to load default configuration from: {default_config_path}")
    if not os.path.exists(default_config_path):
        local_logger.error(f"Default configuration file NOT FOUND at: {default_config_path}")
        return {}
    try:
        with open(default_config_path, 'r') as f:
             default_config = yaml.safe_load(f)
        if default_config is None:
             local_logger.error("Default configuration file is empty!")
             return {}
        local_logger.debug("Default configuration loaded successfully.")
        return default_config
    except Exception as e:
         local_logger.error(f"Failed to load default configuration from {default_config_path}: {e}")
         return {}
EOF

# --- src/voxelflex/data/validators.py ---
cat << 'EOF' > src/voxelflex/data/validators.py
# src/voxelflex/data/validators.py
"""
Data validation module for VoxelFlex (Temperature-Aware).

Validates aggregated RMSF data. Voxel validation happens during loading.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Use the centralized logger
logger = logging.getLogger("voxelflex.data") # Use parent logger name


def validate_aggregated_rmsf_data(rmsf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate aggregated RMSF DataFrame for required columns, types, and potential issues.

    Args:
        rmsf_df: DataFrame loaded from the aggregated RMSF CSV.

    Returns:
        Validated (potentially filtered) DataFrame.

    Raises:
        ValueError: If input is empty or essential columns are missing/invalid.
    """
    logger.info("Validating aggregated RMSF data...")
    if not isinstance(rmsf_df, pd.DataFrame) or rmsf_df.empty:
        raise ValueError("Input RMSF data is not a non-empty DataFrame.")

    df_validated = rmsf_df.copy()

    # Define required columns and their expected types (approximate)
    required_cols = {
        'domain_id': str,
        'resid': int,
        'resname': str,
        'temperature_feature': float,
        'target_rmsf': float
    }
    optional_cols_for_eval = { # Needed for evaluation/visualization
        'relative_accessibility': float,
        'dssp': str, # Or secondary_structure_encoded: int
        'secondary_structure_encoded': int # Allow either format
    }

    # Check for required columns
    missing_req = [col for col in required_cols if col not in df_validated.columns]
    if missing_req:
        raise ValueError(f"Aggregated RMSF data missing required columns: {missing_req}. Found: {list(df_validated.columns)}")

    # Check for optional columns needed later (warn if missing)
    missing_opt = []
    for col in optional_cols_for_eval:
         # Skip check if it's one of the secondary structure alternatives
         if col in ['dssp', 'secondary_structure_encoded']: continue
         if col not in df_validated.columns: missing_opt.append(col)

    # Special check for secondary structure (allow either 'dssp' or 'secondary_structure_encoded')
    ss_col_found = 'dssp' in df_validated.columns or 'secondary_structure_encoded' in df_validated.columns
    if not ss_col_found:
         missing_opt.append("dssp OR secondary_structure_encoded")

    if missing_opt:
        logger.warning(f"Optional columns for evaluation/visualization missing: {missing_opt}. Stratified analysis may be limited.")


    initial_rows = len(df_validated)
    logger.info(f"Initial RMSF rows: {initial_rows}")

    # --- Type Conversion and NaN Handling ---
    issues = {"nan": 0, "type": 0, "negative_rmsf": 0, "duplicates": 0}

    # Convert required columns first
    for col, expected_type in required_cols.items():
        try:
            if expected_type == int:
                # Use Int64 to handle potential NaNs before dropping
                df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce').astype('Int64')
            elif expected_type == float:
                df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
            elif expected_type == str:
                 # Ensure conversion to string, handle potential NaNs as 'nan' string if needed, but dropna later
                 df_validated[col] = df_validated[col].astype(str)

            # Check for NaNs *after* conversion attempts
            nan_count = df_validated[col].isnull().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}': Found {nan_count} NaN/invalid values after conversion. Affected rows will be dropped.")
                issues["nan"] += nan_count # Count NaNs identified before dropping

        except Exception as e:
            logger.error(f"Error converting column '{col}' to {expected_type}: {e}. Skipping further checks on this column.")
            issues["type"] += df_validated.shape[0] # Mark all rows as potentially problematic for this column

    # Drop rows with NaNs in ANY required column AFTER attempting conversion for all
    orig_len = len(df_validated)
    df_validated.dropna(subset=list(required_cols.keys()), inplace=True)
    rows_dropped_nan = orig_len - len(df_validated)
    if rows_dropped_nan > 0:
        logger.info(f"Dropped {rows_dropped_nan} rows due to NaN/invalid values in required columns.")

    logger.info(f"Rows after NaN removal in required columns: {len(df_validated)}")

    # Attempt conversion for optional eval columns if they exist, but don't drop rows
    for col, expected_type in optional_cols_for_eval.items():
         if col in df_validated.columns:
              try:
                   if expected_type == float:
                        df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
                   elif expected_type == int: # e.g., secondary_structure_encoded
                        df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce').astype('Int64')
                   # Add warning for NaNs but don't drop rows based on optional columns
                   nan_count = df_validated[col].isnull().sum()
                   if nan_count > 0:
                        logger.warning(f"Optional column '{col}' contains {nan_count} NaN/invalid values.")
              except Exception as e:
                   logger.warning(f"Could not process optional column '{col}': {e}")


    # --- Specific Value Checks ---
    # Check for negative RMSF (only if column still exists after potential drop)
    if 'target_rmsf' in df_validated.columns:
         neg_rmsf_mask = df_validated['target_rmsf'] < 0
         neg_count = neg_rmsf_mask.sum()
         if neg_count > 0:
              logger.warning(f"Found {neg_count} negative 'target_rmsf' values. Setting them to 0.")
              df_validated.loc[neg_rmsf_mask, 'target_rmsf'] = 0.0
              issues["negative_rmsf"] += neg_count

    # Check for duplicate (domain_id, resid, temperature_feature) entries
    key_cols = ['domain_id', 'resid', 'temperature_feature']
    if all(c in df_validated.columns for c in key_cols):
        # Ensure types are consistent before checking duplicates
        # resid should be int by now if dropna worked
        try:
            df_validated['resid'] = df_validated['resid'].astype(int)
        except TypeError as e:
             logger.error(f"Cannot convert 'resid' column to int after NaN drop: {e}. Duplicates check might be affected.")
             # Proceed cautiously, or raise error? Let's proceed but warn.

        df_validated['temperature_feature'] = df_validated['temperature_feature'].astype(float)

        duplicates_mask = df_validated.duplicated(subset=key_cols, keep='first')
        dup_count = duplicates_mask.sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate entries based on {key_cols}. Keeping first occurrence.")
            df_validated = df_validated[~duplicates_mask].copy() # Use copy to avoid SettingWithCopyWarning
            issues["duplicates"] += dup_count

    final_rows = len(df_validated)
    if initial_rows > 0:
        percentage_str = f"({final_rows / initial_rows:.1%})"
    else:
        percentage_str = "(0.0%)"
    logger.info(f"RMSF validation finished. Valid rows: {final_rows} / {initial_rows} {percentage_str}.")
    if sum(issues.values()) > 0:
         # Refined reporting of dropped rows vs corrected rows
         logger.warning(f"Issues handled: NaN/Type rows dropped={rows_dropped_nan}, Negative RMSF corrected={issues['negative_rmsf']}, Duplicates Removed={issues['duplicates']}")


    if final_rows == 0:
        raise ValueError("No valid RMSF data remaining after validation.")

    # --- Log Summary Statistics ---
    logger.info("Summary statistics of validated RMSF data:")
    try:
        logger.info(f"  Unique Domains: {df_validated['domain_id'].nunique()}")
        logger.info(f"  Unique Temperatures: {sorted(df_validated['temperature_feature'].unique())}")
        if 'target_rmsf' in df_validated.columns and not df_validated['target_rmsf'].empty:
             logger.info(f"  Target RMSF Range: [{df_validated['target_rmsf'].min():.4f}, {df_validated['target_rmsf'].max():.4f}]")
             logger.info(f"  Target RMSF Mean: {df_validated['target_rmsf'].mean():.4f}, Median: {df_validated['target_rmsf'].median():.4f}")
        else: logger.warning("  Target RMSF column missing or empty for summary.")

        if 'resname' in df_validated and not df_validated['resname'].empty:
            logger.info(f"  Residue Types Found: {sorted(df_validated['resname'].unique())}")
        if 'relative_accessibility' in df_validated.columns:
             sasa_col_valid = df_validated['relative_accessibility'].dropna()
             if not sasa_col_valid.empty:
                 logger.info(f"  Rel. Accessibility Range: [{sasa_col_valid.min():.2f}, {sasa_col_valid.max():.2f}]")
             else: logger.warning("  Relative accessibility column contains only NaN/invalid values.")

        ss_col = next((c for c in ['dssp', 'secondary_structure_encoded'] if c in df_validated.columns), None)
        if ss_col:
             ss_col_valid = df_validated[ss_col].dropna()
             if not ss_col_valid.empty:
                 logger.info(f"  Secondary Structure Types ({ss_col}): {sorted(ss_col_valid.unique())}")
             else: logger.warning(f"  Secondary structure column ('{ss_col}') contains only NaN/invalid values.")

    except Exception as e:
        logger.warning(f"Could not generate full summary statistics: {e}")

    return df_validated
EOF

# --- src/voxelflex/data/data_loader.py ---
cat << 'EOF' > src/voxelflex/data/data_loader.py
"""
Data loading module for VoxelFlex (Temperature-Aware).

Defines:
- PreprocessedVoxelFlexDataset: Loads pre-computed batches (.pt files) for training/validation.
- PredictionDataset: Minimal dataset for prediction, loading raw voxels on demand.
- Helper functions for loading raw RMSF data and creating lookups/mappings.
- Robust function for loading raw voxel data from HDF5.
"""

import os
import logging
import time
import gc
import h5py
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("voxelflex.data") # Use parent logger name

# Internal imports (avoid circular dependency with preprocess)
from voxelflex.data.validators import validate_aggregated_rmsf_data
from voxelflex.utils.file_utils import resolve_path, ensure_dir
from voxelflex.utils.logging_utils import EnhancedProgressBar

# --- Datasets ---

class PreprocessedVoxelFlexDataset(Dataset):
    """
    Dataset for loading batches from preprocessed .pt files listed in a metadata file.
    Designed for use during the training and validation phases. Loads full batches.
    """
    def __init__(self, metadata_file: str, processed_dir: str):
        """
        Initializes the dataset by reading the list of .pt batch files from the meta file.

        Args:
            metadata_file: Path to the .meta file listing relative paths of .pt files.
            processed_dir: Base directory where the .pt files are stored.
        """
        self.metadata_file = resolve_path(metadata_file)
        self.processed_dir = resolve_path(processed_dir)
        self.batch_files: List[str] = []

        if not os.path.exists(self.metadata_file):
            # Log an error but allow initialization with empty list, caller should handle
            logger.error(f"Metadata file not found: {self.metadata_file}. Dataset will be empty.")
            return
            # raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        if not os.path.isdir(self.processed_dir):
             logger.error(f"Processed data directory not found: {self.processed_dir}. Cannot load batches.")
             return
            # raise NotADirectoryError(f"Processed data directory not found: {self.processed_dir}")

        try:
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    relative_path = line.strip()
                    if relative_path:
                        full_path = os.path.join(self.processed_dir, relative_path)
                        # Check existence *here* to avoid issues later
                        if os.path.exists(full_path):
                            self.batch_files.append(full_path)
                        else:
                            logger.warning(f"Batch file listed in meta file not found, skipping: {full_path}")
            logger.info(f"Initialized PreprocessedVoxelFlexDataset with {len(self.batch_files)} batch files from {self.metadata_file}")
            if not self.batch_files and os.path.exists(self.metadata_file):
                logger.warning(f"Meta file {self.metadata_file} exists but contains no valid/existing batch files.")
        except Exception as e:
            logger.exception(f"Error loading metadata file {self.metadata_file}: {e}")
            # Allow initialization with empty list
            self.batch_files = []

    def __len__(self) -> int:
        """Return the number of preprocessed batch files."""
        return len(self.batch_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a preprocessed batch dictionary from a .pt file.

        Args:
            idx: The index of the batch file to load.

        Returns:
            A dictionary {'voxels': Tensor, 'scaled_temps': Tensor, 'targets': Tensor}.
            Returns an empty dictionary if loading fails.
        """
        if idx >= len(self.batch_files):
            # This should ideally not happen if DataLoader uses __len__ correctly
            logger.error(f"Index {idx} out of range for PreprocessedVoxelFlexDataset (size {len(self.batch_files)}).")
            return self._get_empty_batch()

        batch_file_path = self.batch_files[idx]
        try:
            # Load directly to CPU first, DataLoader handles moving to device if needed
            batch_data = torch.load(batch_file_path, map_location='cpu')

            # Validate the loaded batch structure
            if not isinstance(batch_data, dict) or \
               'voxels' not in batch_data or \
               'scaled_temps' not in batch_data or \
               'targets' not in batch_data:
                logger.error(f"Invalid data format in batch file: {batch_file_path}. Missing required keys.")
                return self._get_empty_batch()

            # Optional: Validate tensor types/shapes if needed, but can impact performance
            # if not isinstance(batch_data['voxels'], torch.Tensor) or ...

            return batch_data
        except FileNotFoundError:
             logger.error(f"Batch file listed in meta not found during getitem: {batch_file_path}")
             return self._get_empty_batch()
        except Exception as e:
            logger.exception(f"Error loading or processing batch file {batch_file_path}: {e}")
            return self._get_empty_batch()

    def _get_empty_batch(self) -> Dict[str, torch.Tensor]:
        """Returns an empty dictionary to signal a failed batch load."""
        # Returning empty tensors can cause downstream issues if not handled
        # Returning an empty dict is clearer for the DataLoader collation to skip
        return {}


class PredictionDataset(Dataset):
    """
    Minimal Dataset for prediction/evaluation, yielding raw voxels and identifiers.
    Loads voxel data on demand using `load_process_voxels_from_hdf5`.
    Initializes only with samples for which voxel data can be successfully loaded.
    """
    def __init__(
        self,
        samples_to_load: List[Tuple[str, str]], # List of (voxel_domain_id, resid_str) to attempt loading
        voxel_hdf5_path: str,
        expected_channels: int = 5,
        target_shape_chw: Optional[Tuple[int, int, int, int]] = (5, 21, 21, 21) # C,D,H,W - For validation
    ):
        """
        Initializes the dataset by attempting to load voxels for the requested samples.

        Args:
            samples_to_load: List of (voxel_domain_id, resid_str) tuples to try loading.
            voxel_hdf5_path: Path to the HDF5 file containing voxel data.
            expected_channels: Expected number of channels for validation.
            target_shape_chw: Expected final shape (Channels, D, H, W) for validation.
        """
        self.voxel_hdf5_path = resolve_path(voxel_hdf5_path)
        self.expected_channels = expected_channels
        self.target_shape_chw = target_shape_chw
        self.samples: List[Tuple[str, str]] = [] # List of successfully loaded samples
        self.voxel_cache: Dict[Tuple[str, str], np.ndarray] = {} # Cache loaded numpy arrays
        self._dummy_shape: Optional[Tuple[int,...]] = None

        logger.info(f"Initializing PredictionDataset: Attempting to load voxels for {len(samples_to_load)} requested samples...")

        # Group samples by domain to optimize HDF5 reading
        domains_needed = defaultdict(list)
        for domain_id, resid_str in samples_to_load:
            domains_needed[domain_id].append(resid_str)

        loaded_count = 0
        failed_load_samples = 0
        progress = EnhancedProgressBar(len(domains_needed), prefix="Prefetching Voxels (Prediction)")

        # Iterate through domains needed
        for i, (domain_id, resid_list) in enumerate(domains_needed.items()):
            try:
                # Load data for only this domain
                domain_voxel_data = load_process_voxels_from_hdf5(
                    self.voxel_hdf5_path,
                    domain_ids=[domain_id],
                    expected_channels=self.expected_channels,
                    target_shape_chw=self.target_shape_chw
                )

                # Check if the domain itself was loaded and process its residues
                if domain_id in domain_voxel_data:
                    for resid_str in resid_list:
                        if resid_str in domain_voxel_data[domain_id]:
                            voxel_array = domain_voxel_data[domain_id][resid_str]
                            # Store successful sample and cache the data
                            self.samples.append((domain_id, resid_str))
                            self.voxel_cache[(domain_id, resid_str)] = voxel_array
                            loaded_count += 1
                            if self._dummy_shape is None: # Store shape from first success
                                self._dummy_shape = voxel_array.shape
                        else:
                            # Residue within the domain failed loading inside the function
                            failed_load_samples += 1
                            # logger.debug(f"Residue {domain_id}:{resid_str} not found in successfully loaded domain data.")
                else:
                    # Domain itself failed to load or had no valid residues
                    failed_load_samples += len(resid_list)
                    # logger.debug(f"Domain {domain_id} could not be loaded or had no valid residues.")

            except Exception as e:
                 logger.warning(f"Error pre-fetching voxels for domain {domain_id}: {e}")
                 failed_load_samples += len(resid_list) # Mark all residues for this domain as failed

            progress.update(i + 1)

        progress.finish()
        logger.info(f"PredictionDataset initialized. Successfully loaded and cached voxels for {loaded_count} samples.")
        if failed_load_samples > 0:
             logger.warning(f"Failed to load voxels for {failed_load_samples} requested samples. They will be excluded.")

        if not self.samples:
            logger.error("PredictionDataset initialization failed: No valid voxel data could be loaded for any requested samples.")
            # Set dummy shape based on config/defaults if nothing loaded
            if self.target_shape_chw:
                 self._dummy_shape = self.target_shape_chw
            else:
                 self._dummy_shape = (self.expected_channels or 5, 21, 21, 21)


    def __len__(self) -> int:
        """Return the number of samples with successfully loaded voxel data."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[str, str, torch.Tensor]:
        """
        Returns (voxel_domain_id, resid_str, voxel_tensor) for a given index.
        Retrieves the pre-cached numpy array and converts it to a tensor.
        """
        if idx >= len(self.samples):
             logger.error(f"Prediction Dataset: Index {idx} out of bounds ({len(self.samples)}).")
             # This should ideally not happen if DataLoader uses __len__ correctly
             return "ERROR", "0", self._get_dummy_voxel()

        voxel_domain_id, resid_str = self.samples[idx]
        key = (voxel_domain_id, resid_str)

        try:
            # Retrieve from cache (should exist because initialization preloaded it)
            voxel_np = self.voxel_cache.get(key)
            if voxel_np is None:
                # This indicates an internal inconsistency
                logger.error(f"Internal Error: Voxel data missing from cache for sample {key}. This should not happen.")
                return voxel_domain_id, resid_str, self._get_dummy_voxel()

            # Convert numpy array (already float32) to tensor
            voxel_tensor = torch.from_numpy(voxel_np)
            return voxel_domain_id, resid_str, voxel_tensor

        except Exception as e:
            logger.exception(f"Error retrieving prediction item {idx} ({key}) from cache: {e}")
            return voxel_domain_id, resid_str, self._get_dummy_voxel()

    def _get_dummy_voxel(self) -> torch.Tensor:
        """Creates a dummy zero tensor based on the expected shape."""
        if self._dummy_shape is None:
            # Fallback if initialization failed to set shape
            shape = (self.expected_channels or 5, 21, 21, 21)
            logger.warning(f"Could not infer dummy shape, using default: {shape}")
            self._dummy_shape = shape
        return torch.zeros(self._dummy_shape, dtype=torch.float32)


# --- Robust Voxel Loading Function ---

def load_process_voxels_from_hdf5(
    voxel_hdf5_path: str,
    domain_ids: List[str],
    expected_channels: int = 5,
    target_shape_chw: Optional[Tuple[int, int, int, int]] = (5, 21, 21, 21) # C,D,H,W
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads voxel data for specified domains from an HDF5 file.

    Handles:
    - Finding the correct chain/residue group within each domain.
    - Reading individual residue datasets.
    - Checking dtype: Casts bool to float32.
    - Checking shape: Transposes (D,H,W,C) to (C,D,H,W) if needed.
    - Validating final shape and numeric types.
    - Skipping and logging faulty residues individually.

    Args:
        voxel_hdf5_path: Path to the HDF5 file.
        domain_ids: List of HDF5 domain keys to load.
        expected_channels: Expected number of channels (first dim after transpose).
        target_shape_chw: Expected final shape (C, D, H, W) for validation.

    Returns:
        A dictionary: {domain_id: {resid_str: voxel_numpy_array}}
        Includes only successfully loaded and processed voxels.
    """
    voxel_hdf5_path = resolve_path(voxel_hdf5_path)
    logger.debug(f"Loading voxels for {len(domain_ids)} domains from {voxel_hdf5_path}")
    processed_voxels: Dict[str, Dict[str, np.ndarray]] = {}
    domains_not_found = 0
    residues_processed_count = 0
    residues_failed_count = 0

    if not os.path.exists(voxel_hdf5_path):
        logger.error(f"Voxel HDF5 file not found: {voxel_hdf5_path}")
        return processed_voxels

    try:
        with h5py.File(voxel_hdf5_path, 'r') as f_h5:
            for domain_id in domain_ids:
                if domain_id not in f_h5:
                    logger.debug(f"Domain '{domain_id}' not found in HDF5 file.")
                    domains_not_found += 1
                    continue

                domain_group = f_h5[domain_id]
                residue_group = None

                # Find the first group that looks like a chain with residues
                potential_chain_keys = sorted([k for k in domain_group.keys() if isinstance(domain_group[k], h5py.Group)])
                if not potential_chain_keys:
                     logger.warning(f"No groups (potential chains) found under domain '{domain_id}'.")
                     continue

                for chain_key in potential_chain_keys:
                    try:
                        potential_residue_group = domain_group[chain_key]
                        # Check if it contains keys that are all digits
                        if any(key.isdigit() for key in potential_residue_group.keys()):
                            residue_group = potential_residue_group
                            if len(potential_chain_keys) > 1:
                                 logger.debug(f"Multiple potential chain groups found in '{domain_id}'. Using first valid one found: '{chain_key}'.")
                            break # Use the first one found
                    except Exception as e:
                         logger.debug(f"Error accessing subgroup {chain_key} in {domain_id}: {e}")

                if residue_group is None:
                    logger.warning(f"Could not find a valid residue group (e.g., Chain 'A') within domain '{domain_id}'. Searched keys: {potential_chain_keys}")
                    continue

                domain_data_dict: Dict[str, np.ndarray] = {}
                for resid_str in residue_group.keys():
                    # Ensure the key is actually a residue ID (all digits)
                    if not resid_str.isdigit():
                        continue

                    residues_processed_count += 1
                    voxel_dataset = None
                    voxel_array = None
                    try:
                        voxel_dataset = residue_group[resid_str]
                        if not isinstance(voxel_dataset, h5py.Dataset):
                            raise TypeError(f"Expected HDF5 Dataset for {domain_id}:{resid_str}, got {type(voxel_dataset)}.")

                        # --- Read Data ---
                        voxel_raw = voxel_dataset[:]
                        original_dtype = voxel_raw.dtype
                        original_shape = voxel_raw.shape

                        # --- Process Dtype (Bool -> Float32) ---
                        if original_dtype == bool:
                            voxel_array = voxel_raw.astype(np.float32)
                            logger.debug(f"Casted {domain_id}:{resid_str} from bool to float32.")
                        elif np.issubdtype(original_dtype, np.floating):
                            # Ensure it's float32, copy if necessary
                            voxel_array = voxel_raw.astype(np.float32, copy=False)
                        elif np.issubdtype(original_dtype, np.integer):
                             logger.warning(f"Voxel {domain_id}:{resid_str} has integer dtype ({original_dtype}). Casting to float32.")
                             voxel_array = voxel_raw.astype(np.float32)
                        else:
                             raise TypeError(f"Unsupported voxel dtype {original_dtype} for {domain_id}:{resid_str}.")

                        # --- Process Shape (Transpose if needed) ---
                        if voxel_array.ndim == 4 and voxel_array.shape[-1] == expected_channels:
                            # Assume (D, H, W, C) -> Transpose to (C, D, H, W)
                            voxel_array = np.transpose(voxel_array, (3, 0, 1, 2))
                            logger.debug(f"Transposed {domain_id}:{resid_str} from {original_shape} to {voxel_array.shape}.")
                        elif voxel_array.ndim == 4 and voxel_array.shape[0] == expected_channels:
                             # Already channels-first, do nothing
                             pass
                        elif voxel_array.ndim != 4:
                             raise ValueError(f"Unexpected voxel dimensions {voxel_array.ndim} (expected 4) for {domain_id}:{resid_str}.")
                        else:
                             # 4D but wrong channel dimension
                             raise ValueError(f"Unexpected shape {voxel_array.shape}, expected {expected_channels} channels (found {voxel_array.shape[0]} or {voxel_array.shape[-1]}).")

                        # --- Validate Final Shape and Content ---
                        if target_shape_chw and voxel_array.shape != target_shape_chw:
                            # Only warn if shape mismatch, allow processing if channels are right
                            logger.warning(f"Voxel {domain_id}:{resid_str} final shape {voxel_array.shape} differs from target {target_shape_chw}.")
                            # Option to raise error here if exact shape match is mandatory:
                            # raise ValueError(f"Final shape mismatch: {voxel_array.shape} vs {target_shape_chw}")

                        if np.isnan(voxel_array).any() or np.isinf(voxel_array).any():
                            raise ValueError(f"Voxel {domain_id}:{resid_str} contains NaN or Inf values after processing.")

                        # Success - add to dictionary for this domain
                        domain_data_dict[resid_str] = voxel_array

                    except Exception as e:
                        logger.warning(f"Failed to load/process voxel for {domain_id}:{resid_str}. Error: {e}. Skipping residue.")
                        residues_failed_count += 1
                    finally:
                         # Explicitly delete intermediate arrays if created
                         if 'voxel_raw' in locals(): del voxel_raw
                         # voxel_array is kept if successful

                # Add successfully processed residues for this domain
                if domain_data_dict:
                    processed_voxels[domain_id] = domain_data_dict

    except FileNotFoundError:
        # Already logged error above
        pass
    except Exception as e:
        logger.exception(f"An error occurred while processing HDF5 file {voxel_hdf5_path}: {e}")

    logger.debug(f"Finished loading domains. Found {len(processed_voxels)} domains with some valid voxels.")
    logger.debug(f"  Residues attempted: {residues_processed_count}, Failed: {residues_failed_count}")
    if domains_not_found > 0:
        logger.debug(f"  Domains specified but not found in HDF5: {domains_not_found}")

    return processed_voxels


# --- Helper Functions for Preprocessing ---

def load_aggregated_rmsf_data(aggregated_rmsf_file: str) -> pd.DataFrame:
    """Loads and validates the aggregated RMSF data from a CSV file."""
    rmsf_file = resolve_path(aggregated_rmsf_file)
    logger.info(f"Loading aggregated RMSF data from: {rmsf_file}")
    if not os.path.exists(rmsf_file):
        raise FileNotFoundError(f"Aggregated RMSF file not found: {rmsf_file}")
    try:
        # Use low_memory=False for potentially mixed type columns
        rmsf_df = pd.read_csv(rmsf_file, low_memory=False)
        logger.info(f"Successfully loaded {len(rmsf_df)} rows from aggregated RMSF file.")
    except Exception as e:
        logger.exception(f"Failed to read aggregated RMSF CSV file {rmsf_file}: {e}")
        raise
    # Validate the loaded data
    try:
        validated_df = validate_aggregated_rmsf_data(rmsf_df)
        logger.info(f"Aggregated RMSF data validated. {len(validated_df)} valid rows remaining.")
        return validated_df
    except ValueError as ve:
        logger.error(f"Validation of aggregated RMSF data failed: {ve}")
        raise


def create_master_rmsf_lookup(
    rmsf_df: pd.DataFrame
) -> Dict[Tuple[str, int], List[Tuple[float, float]]]:
    """
    Creates lookup: (domain_id, resid_int) -> [(temp, rmsf), ...].
    Handles base name matching (e.g., '1abcA00_pdb' -> '1abcA00').

    Args:
        rmsf_df: Validated DataFrame from load_aggregated_rmsf_data.

    Returns:
        The lookup dictionary.
    """
    logger.info("Creating master RMSF/Temperature lookup from aggregated data...")
    start_time = time.time()
    # Ensure required columns are present after validation
    required_cols = ['domain_id', 'resid', 'temperature_feature', 'target_rmsf']
    if not all(col in rmsf_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in rmsf_df.columns]
        raise ValueError(f"Validated RMSF DataFrame missing required columns for lookup: {missing}")

    # Ensure types are correct (should be handled by validator, but double check)
    try:
        rmsf_df['resid_int'] = rmsf_df['resid'].astype(int)
        rmsf_df['temp_float'] = rmsf_df['temperature_feature'].astype(float)
        rmsf_df['rmsf_float'] = rmsf_df['target_rmsf'].astype(float)
    except Exception as e:
        logger.error(f"Error ensuring correct types for RMSF lookup creation: {e}")
        raise

    # Main lookup creation using groupby for efficiency
    lookup: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
    grouped = rmsf_df.groupby(['domain_id', 'resid_int'])

    for name, group in grouped:
        domain_id, resid_int = name
        # Extract valid temperature and RMSF pairs
        temp_rmsf_pairs = list(zip(group['temp_float'], group['rmsf_float']))
        lookup[(str(domain_id), resid_int)] = temp_rmsf_pairs

    # Optional: Add base name matching if needed (e.g., '1abcA00_pdb' should match '1abcA00')
    # This assumes suffixes like '_pdb' should be ignored if exact match fails.
    base_name_lookup: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    for (domain_id, resid_int), pairs in lookup.items():
        # Simple base name logic: split by '_' and take the first part
        base_name = str(domain_id).split('_')[0]
        if base_name != domain_id:
            base_key = (base_name, resid_int)
            # Add base name key ONLY if it doesn't already exist (prioritize specific keys)
            if base_key not in lookup:
                # If multiple suffixed versions exist, which one should the base map to?
                # For simplicity, let the first one encountered populate the base key.
                # Or, if base_key already in base_name_lookup, don't overwrite.
                if base_key not in base_name_lookup:
                    base_name_lookup[base_key] = pairs

    # Update the main lookup with non-conflicting base names
    original_keys = len(lookup)
    lookup.update(base_name_lookup)
    base_added = len(lookup) - original_keys

    duration = time.time() - start_time
    logger.info(f"Created master RMSF lookup with {len(lookup)} unique (domain, residue) keys ({base_added} base names added) in {duration:.2f}s.")

    # Clean up temporary columns if they were added
    rmsf_df.drop(columns=['resid_int', 'temp_float', 'rmsf_float'], inplace=True, errors='ignore')
    return dict(lookup) # Convert back to standard dict


def create_domain_mapping(
    voxel_domain_keys: List[str],
    rmsf_domain_ids: List[str]
) -> Dict[str, str]:
    """
    Create mapping from HDF5 voxel domain keys to RMSF CSV domain IDs.
    Prioritizes exact matches, then matches by removing trailing suffixes
    like '_pdb' from the HDF5 key.

    Args:
        voxel_domain_keys: List of unique keys found in the HDF5 file.
        rmsf_domain_ids: List of unique domain_id values from the RMSF CSV.

    Returns:
        Dictionary mapping {hdf5_key: rmsf_domain_id}.
    """
    logger.info("Creating domain mapping (HDF5 key -> RMSF domain ID)...")
    if not voxel_domain_keys:
        logger.warning("Voxel domain key list is empty. Returning empty mapping.")
        return {}
    if not rmsf_domain_ids:
        logger.warning("RMSF domain ID list is empty. Returning empty mapping.")
        return {}

    voxel_keys_set = set(voxel_domain_keys)
    rmsf_ids_set = set(rmsf_domain_ids)
    mapping: Dict[str, str] = {}
    matches = {'exact': 0, 'base_match': 0}

    # Build a lookup for base RMSF IDs to the "best" full RMSF ID (e.g., prefer shorter or specific pattern if multiple exist)
    # For now, just map base -> first encountered full ID
    rmsf_base_to_full: Dict[str, str] = {}
    for rmsf_id in rmsf_domain_ids:
        base = str(rmsf_id).split('_')[0]
        if base not in rmsf_base_to_full:
            rmsf_base_to_full[base] = rmsf_id

    # Attempt mapping for each HDF5 key
    for hdf5_key in voxel_keys_set:
        # 1. Exact Match
        if hdf5_key in rmsf_ids_set:
            mapping[hdf5_key] = hdf5_key
            matches['exact'] += 1
            continue

        # 2. Base Name Match (HDF5 base -> RMSF exact OR HDF5 base -> RMSF base)
        base_hdf5 = str(hdf5_key).split('_')[0]
        if base_hdf5 in rmsf_ids_set: # e.g., hdf5='1abcA00_pdb', base='1abcA00', rmsf has '1abcA00'
            mapping[hdf5_key] = base_hdf5
            matches['base_match'] += 1
        elif base_hdf5 in rmsf_base_to_full: # e.g., hdf5='1abcA00_pdb', base='1abcA00', rmsf has '1abcA00_xxx'
            mapping[hdf5_key] = rmsf_base_to_full[base_hdf5]
            matches['base_match'] += 1
            logger.debug(f"Mapped HDF5 key '{hdf5_key}' to RMSF ID '{mapping[hdf5_key]}' via base name '{base_hdf5}'.")


    total_mapped = len(mapping)
    coverage = (total_mapped / len(voxel_domain_keys)) * 100 if voxel_domain_keys else 0
    logger.info(f"Domain mapping: {total_mapped}/{len(voxel_domain_keys)} HDF5 keys mapped ({coverage:.1f}% coverage).")
    logger.info(f"  Match types: Exact={matches['exact']}, Base Name={matches['base_match']}")

    if total_mapped < len(voxel_domain_keys):
        unmapped_count = len(voxel_domain_keys) - total_mapped
        unmapped_keys_example = [k for k in voxel_domain_keys if k not in mapping][:5]
        logger.warning(f"{unmapped_count} HDF5 keys could not be mapped to any RMSF domain ID.")
        logger.warning(f"  Examples: {unmapped_keys_example}{'...' if unmapped_count > 5 else ''}")
        logger.warning("  Check naming conventions (e.g., suffixes) between HDF5 keys and CSV 'domain_id'.")

    return mapping
EOF

# --- src/voxelflex/models/cnn_models.py ---
cat << 'EOF' > src/voxelflex/models/cnn_models.py
"""
CNN models for VoxelFlex (Temperature-Aware).

Contains 3D CNN architectures adapted for RMSF prediction, including a
temperature feature input. Includes DenseNet3D, DilatedResNet3D, MultipathRMSFNet.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Use centralized logger
logger = logging.getLogger("voxelflex.models") # Use parent logger name

# --- DenseNet Building Blocks ---

class _DenseLayer(nn.Module):
    """Single layer within a DenseBlock."""
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ):
        super().__init__()
        self.norm1: nn.BatchNorm3d
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv3d
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.norm2: nn.BatchNorm3d
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv3d
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
        # Checkpointing not implemented here for simplicity, but could be added via torch.utils.checkpoint

    def bn_function(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Correctly handle list of tensors or single tensor input
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input_features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Ensure input is treated as a list for concatenation in bn_function
        if isinstance(input_features, torch.Tensor):
            prev_features = [input_features]
        else:
            prev_features = input_features

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    """Dense Convolutional Block."""
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        # Correctly iterate over layers in the ModuleDict
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """Transition layer between DenseBlocks."""
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


# --- Temperature-Aware Models ---

class DenseNet3DRegression(nn.Module):
    """
    3D DenseNet architecture adapted for temperature-aware RMSF regression.
    """
    def __init__(
        self,
        input_channels: int = 5,
        growth_rate: int = 16,
        block_config: Tuple[int, ...] = (4, 4, 4), # Use Tuple[int, ...] for flexibility
        num_init_features: int = 32,
        bn_size: int = 4,
        dropout_rate: float = 0.3,
        memory_efficient: bool = False # Add memory efficient option if needed later
    ):
        """
        Initialize DenseNet3DRegression.

        Args:
            input_channels: Number of input voxel channels.
            growth_rate: How many features to add per layer (k).
            block_config: Tuple containing number of layers in each dense block.
            num_init_features: Number of features after initial convolution.
            bn_size: Multiplicative factor for bottleneck layers.
            dropout_rate: Dropout rate for dense layers and final FC layer.
            memory_efficient: Use checkpointing to save memory (slower).
        """
        super().__init__()
        logger.info("Initializing DenseNet3DRegression model...")

        # --- Initial Convolution ---
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        logger.info(f"  Initial Conv: {input_channels} -> {num_init_features} features")

        # --- Dense Blocks ---
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            logger.info(f"  Dense Block {i+1}: {num_layers} layers, Output features: {num_features}")

            # Add transition layer if not the last block
            if i != len(block_config) - 1:
                num_output_features = num_features // 2
                trans = _Transition(num_input_features=num_features, num_output_features=num_output_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_output_features
                logger.info(f"  Transition {i+1}: Output features: {num_features}")

        # --- Final Batch Norm ---
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # --- Global Pooling ---
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # --- Temperature-Aware Regression Head ---
        # Input size = features from DenseNet + 1 (scaled temperature)
        self.classifier_input_features = num_features + 1
        # Define regression head layers
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Final RMSF prediction
        )

        logger.info(f"  Regression Head Input Features: {self.classifier_input_features}")

        # --- Weight Initialization ---
        self._initialize_weights()
        logger.info("DenseNet3DRegression initialized.")


    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Standard initialization for linear layers in regression head
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, voxel_input: torch.Tensor, scaled_temp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass incorporating voxel data and scaled temperature.

        Args:
            voxel_input: Tensor of shape (Batch, Channels, D, H, W).
            scaled_temp: Tensor of shape (Batch, 1) with scaled temperature values [0, 1].

        Returns:
            Tensor of shape (Batch,) with predicted RMSF values.
        """
        # Process voxel data through DenseNet features
        features = self.features(voxel_input)
        out = F.relu(features, inplace=True) # Final ReLU after last BatchNorm
        out = self.global_avg_pool(out)
        voxel_features = torch.flatten(out, 1) # Shape: (Batch, num_features)

        # Ensure scaled_temp has shape (Batch, 1)
        if scaled_temp.ndim == 1:
            scaled_temp = scaled_temp.unsqueeze(1)
        elif scaled_temp.shape[1] != 1:
             raise ValueError(f"scaled_temp input must have shape (Batch, 1), but got {scaled_temp.shape}")

        # Concatenate voxel features and scaled temperature
        combined_features = torch.cat((voxel_features, scaled_temp), dim=1)

        # Pass through regression head
        predictions = self.classifier(combined_features)

        return predictions.squeeze(1) # Return shape (Batch,)


# --- Residual Block (Used by DilatedResNet3D) ---
class ResidualBlock3D(nn.Module):
    """3D Residual block with optional dilation and dropout."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        padding = dilation # Standard padding for dilated convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity() # Use Identity if no dropout

        # Skip connection: Adjust channels if necessary
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out) # Apply dropout after first activation
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual # Add skip connection BEFORE final activation
        out = F.relu(out, inplace=True)
        return out

class DilatedResNet3D(nn.Module):
    """
    Dilated ResNet 3D architecture adapted for temperature-aware RMSF prediction.
    """
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 4,
        dropout_rate: float = 0.3
    ):
        """Initialize DilatedResNet3D."""
        super().__init__()
        logger.info("Initializing DilatedResNet3D model...")

        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        channels = [base_filters]
        for i in range(num_residual_blocks):
            # Ensure channel count increases, minimum of +1 channel
            next_channels = max(channels[-1] + 1, int(channels[-1] * channel_growth_rate))
            channels.append(next_channels)

        self.res_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            dilation = 2**(i % 3) # Dilations: 1, 2, 4, 1...
            block = ResidualBlock3D(channels[i], channels[i+1], dilation=dilation, dropout_rate=dropout_rate)
            self.res_blocks.append(block)
            logger.info(f"  Res Block {i+1}: {channels[i]}->{channels[i+1]} filters, Dilation={dilation}")

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Temperature-Aware Regression Head
        self.cnn_output_features = channels[-1]
        self.classifier_input_features = self.cnn_output_features + 1 # +1 for temp
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Final RMSF prediction
        )

        logger.info(f"  Regression Head Input Features: {self.classifier_input_features}")
        self._initialize_weights()
        logger.info("DilatedResNet3D initialized.")

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, voxel_input: torch.Tensor, scaled_temp: torch.Tensor) -> torch.Tensor:
        """Forward pass incorporating voxel data and scaled temperature."""
        # Initial convolution
        x = self.conv1(voxel_input)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        voxel_features = torch.flatten(x, 1) # Shape: (Batch, cnn_output_features)

        # Prepare and concatenate temperature
        if scaled_temp.ndim == 1: scaled_temp = scaled_temp.unsqueeze(1) # Ensure (Batch, 1)
        elif scaled_temp.shape[1] != 1: raise ValueError(f"scaled_temp input must have shape (Batch, 1), but got {scaled_temp.shape}")
        combined_features = torch.cat((voxel_features, scaled_temp), dim=1)

        # Pass through regression head
        predictions = self.classifier(combined_features)

        return predictions.squeeze(1)


class MultipathRMSFNet(nn.Module):
    """
    Multi-path 3D CNN architecture adapted for temperature-aware RMSF prediction.
    Preferred model based on dissertation context.
    """
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 3, # Number of blocks *per path* after initial pooling
        dropout_rate: float = 0.3
    ):
        """Initialize MultipathRMSFNet."""
        super().__init__()
        logger.info("Initializing MultipathRMSFNet model...")

        c1 = base_filters
        # Channel size within paths (after first conv in path)
        c2 = max(c1 + 1, int(c1 * channel_growth_rate))
        # Channel size after fusion
        c3 = max(c2*3 + 1, int(c2 * 3 * channel_growth_rate / 2)) # Grow channels after fusion, but maybe less aggressively

        # Initial shared convolution
        self.conv1 = nn.Conv3d(input_channels, c1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c1)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Multi-path branches
        # kernel_size defines the main conv kernel for the path
        self.path1 = self._create_path(c1, c2, kernel_size=3, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        self.path2 = self._create_path(c1, c2, kernel_size=5, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        self.path3 = self._create_path(c1, c2, kernel_size=7, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        logger.info(f"  Paths created: {num_residual_blocks} blocks each, output channels={c2}")

        # Fusion layer (adjusts channels after concatenation)
        self.fusion_conv = nn.Conv3d(c2 * 3, c3, kernel_size=1, bias=False) # 1x1 convolution for channel fusion
        self.fusion_bn = nn.BatchNorm3d(c3)
        logger.info(f"  Fusion layer: {c2*3} -> {c3} channels")

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Temperature-Aware Regression Head
        self.cnn_output_features = c3
        self.classifier_input_features = self.cnn_output_features + 1 # +1 for temp
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Final RMSF prediction
        )

        logger.info(f"  Regression Head Input Features: {self.classifier_input_features}")
        self._initialize_weights()
        logger.info("MultipathRMSFNet initialized.")

    def _create_path(
        self, in_channels: int, path_channels: int, kernel_size: int, blocks: int, dropout_rate: float
    ) -> nn.Sequential:
        """Create a single path for the Multipath network."""
        layers = []
        padding = kernel_size // 2

        # First convolution in the path
        layers.append(nn.Conv3d(in_channels, path_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm3d(path_channels))
        layers.append(nn.ReLU(inplace=True))
        # No pooling here, pooling happens before paths diverge

        # Additional blocks within the path (using kernel_size=3 for subsequent blocks for consistency?)
        # Or keep using the path's main kernel size? Let's keep it path-specific for now.
        current_channels = path_channels
        for _ in range(blocks - 1): # If blocks=3, adds 2 more conv layers
            layers.append(nn.Conv3d(current_channels, current_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(current_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout3d(dropout_rate))
            # Note: Could add residual connections within the path if desired

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, voxel_input: torch.Tensor, scaled_temp: torch.Tensor) -> torch.Tensor:
        """Forward pass incorporating voxel data and scaled temperature."""
        # Initial shared convolution & pooling
        x = self.conv1(voxel_input)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x_pooled = self.pool1(x) # Pool once after initial conv

        # Multi-path processing
        out1 = self.path1(x_pooled)
        out2 = self.path2(x_pooled)
        out3 = self.path3(x_pooled)

        # Concatenate path outputs along the channel dimension
        out_cat = torch.cat([out1, out2, out3], dim=1)

        # Fusion layer
        fused = self.fusion_conv(out_cat)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused, inplace=True)

        # Global average pooling
        pooled = self.global_avg_pool(fused)
        voxel_features = torch.flatten(pooled, 1) # Shape: (Batch, cnn_output_features)

        # Prepare and concatenate temperature
        if scaled_temp.ndim == 1: scaled_temp = scaled_temp.unsqueeze(1) # Ensure (Batch, 1)
        elif scaled_temp.shape[1] != 1: raise ValueError(f"scaled_temp input must have shape (Batch, 1), but got {scaled_temp.shape}")
        combined_features = torch.cat((voxel_features, scaled_temp), dim=1)

        # Pass through regression head
        predictions = self.classifier(combined_features)

        return predictions.squeeze(1)


# --- Helper function to get model instance ---

def get_model(config: Dict[str, Any], input_shape: Tuple[int, ...]) -> nn.Module:
    """
    Get a model instance based on the configuration.

    Args:
        config: Model configuration dictionary section (config['model']).
        input_shape: Shape of a single voxel input (Channels, D, H, W). Used for validation.

    Returns:
        Initialized PyTorch model.
    """
    architecture = config.get('architecture')
    input_channels = config.get('input_channels', 5) # Get from config or default
    dropout_rate = config.get('dropout_rate', 0.3)

    # Verify input channels match data shape (first dimension)
    if input_shape[0] != input_channels:
         logger.warning(f"Model 'input_channels' in config ({input_channels}) does not match "
                        f"detected data channels ({input_shape[0]}). Using detected data channels value.")
         input_channels = input_shape[0] # Override config value


    logger.info(f"Creating model architecture: {architecture} with {input_channels} input channels.")

    if architecture == "densenet3d_regression":
        densenet_cfg = config.get('densenet', {})
        if not densenet_cfg: logger.warning("DenseNet config section ('model.densenet') not found or empty in config. Using defaults.")
        return DenseNet3DRegression(
            input_channels=input_channels,
            growth_rate=densenet_cfg.get('growth_rate', 16),
            block_config=tuple(densenet_cfg.get('block_config', [4, 4, 4])),
            num_init_features=densenet_cfg.get('num_init_features', 32),
            bn_size=densenet_cfg.get('bn_size', 4),
            dropout_rate=dropout_rate
        )
    elif architecture == "dilated_resnet3d":
        return DilatedResNet3D(
            input_channels=input_channels,
            base_filters=config.get('base_filters', 32),
            channel_growth_rate=config.get('channel_growth_rate', 1.5),
            num_residual_blocks=config.get('num_residual_blocks', 4),
            dropout_rate=dropout_rate
        )
    elif architecture == "multipath_rmsf_net":
        return MultipathRMSFNet(
            input_channels=input_channels,
            base_filters=config.get('base_filters', 32),
            channel_growth_rate=config.get('channel_growth_rate', 1.5),
            num_residual_blocks=config.get('num_residual_blocks', 3),
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown architecture specified in config: {architecture}. Valid options: 'densenet3d_regression', 'dilated_resnet3d', 'multipath_rmsf_net'")
EOF

# --- src/voxelflex/utils/file_utils.py ---
cat << 'EOF' > src/voxelflex/utils/file_utils.py
"""
File system utilities for VoxelFlex.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

logger = logging.getLogger("voxelflex.utils.file") # Use submodule logger

def resolve_path(path: Union[str, Path]) -> str:
    """Resolves a path relative to CWD or user home."""
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    # Resolve relative paths based on the current working directory
    if not p.is_absolute():
        p = Path.cwd() / p
    # Use resolve() to make the path absolute and resolve symlinks,
    # but don't raise error if path doesn't exist yet.
    # We handle existence checks later where needed.
    try:
        # return str(p.resolve(strict=False)) # strict=False allows non-existent paths
        # Let's just return the absolute path for now, resolve can cause issues
        # if parts of the path don't exist yet (e.g., output dirs)
        return str(p.absolute())
    except Exception as e:
         logger.warning(f"Could not fully resolve path {p}: {e}. Returning absolute path.")
         return str(p.absolute())


def ensure_dir(dir_path: Union[str, Path]) -> None:
    """Ensure a directory exists, creating it if necessary."""
    if dir_path:
        path = Path(dir_path)
        if not path.exists():
            logger.debug(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")

def save_json(data: Union[Dict, List], file_path: Union[str, Path], indent: int = 4) -> None:
    """Save dictionary or list to JSON file."""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.debug(f"Saved JSON data to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise

def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """Load dictionary or list from JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON data from: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise ValueError(f"Invalid JSON format in {file_path}") from e
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise

def load_list_from_file(file_path: Union[str, Path]) -> List[str]:
    """Load a list of strings from a file, one item per line."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found for loading list: {file_path}. Returning empty list.")
        return []
    try:
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            items = [line.strip() for line in f if line.strip()]
        logger.debug(f"Loaded {len(items)} items from list file: {file_path}")
        return items
    except Exception as e:
        logger.error(f"Failed to load list from {file_path}: {e}")
        return [] # Return empty list on error

def save_list_to_file(data: List[str], file_path: Union[str, Path]) -> None:
    """Save a list of strings to a file, one item per line."""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    try:
        with open(file_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")
        logger.debug(f"Saved {len(data)} items to list file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save list to {file_path}: {e}")
        raise
EOF

# --- src/voxelflex/utils/logging_utils.py ---
cat << 'EOF' > src/voxelflex/utils/logging_utils.py
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
    """Custom tqdm progress bar with optional stage info."""
    def __init__(self, *args, stage_info: Optional[str] = None, **kwargs):
        self.stage_info = stage_info
        # Sensible defaults if not provided
        kwargs.setdefault('ncols', 80)
        kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        # Add dynamic info like memory usage if needed
        # mem_info = psutil.virtual_memory()
        # self.set_postfix_str(f"Mem: {mem_info.percent:.1f}%", refresh=False)

    def finish(self):
        """Close the progress bar."""
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
EOF

# --- src/voxelflex/utils/system_utils.py ---
cat << 'EOF' > src/voxelflex/utils/system_utils.py
"""
System related utilities (CPU, GPU, Memory).
"""

import os
import gc
import logging
import psutil
import torch

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
EOF

# --- src/voxelflex/utils/temp_scaling.py ---
cat << 'EOF' > src/voxelflex/utils/temp_scaling.py
"""
Utility functions for temperature scaling.
"""

import json
import os
import logging
from typing import Callable, Tuple, List, Dict, Any

import numpy as np

from voxelflex.utils.file_utils import ensure_dir, load_json, save_json

logger = logging.getLogger("voxelflex.utils.temp_scaling") # Use submodule logger

def calculate_and_save_temp_scaling(
    train_temps: List[float],
    output_path: str
) -> Tuple[float, float]:
    """
    Calculates min/max temperature from the training data and saves them.

    Args:
        train_temps: List of raw temperature values from the training set.
        output_path: Full path where the JSON file with scaling params will be saved.

    Returns:
        Tuple of (temp_min, temp_max).

    Raises:
        ValueError: If no valid temperature values are provided.
    """
    if not train_temps:
        raise ValueError("No training temperatures provided. Cannot calculate scaling parameters.")

    valid_temps = [t for t in train_temps if t is not None and not np.isnan(t)]
    if not valid_temps:
        raise ValueError("No valid (non-NaN) temperatures found in training data.")

    temp_min = float(np.min(valid_temps))
    temp_max = float(np.max(valid_temps))
    temp_range = temp_max - temp_min

    logger.info(f"Calculating temperature scaling based on {len(valid_temps)} training samples.")
    logger.info(f"  Min Temperature: {temp_min:.2f} K")
    logger.info(f"  Max Temperature: {temp_max:.2f} K")
    logger.info(f"  Temperature Range: {temp_range:.2f} K")

    if abs(temp_range) < 1e-6:
        logger.warning("Temperature range is near zero. Scaling will result in a constant value (0.5).")

    scaling_params = {'temp_min': temp_min, 'temp_max': temp_max}

    try:
        # Ensure the directory exists before saving
        ensure_dir(os.path.dirname(output_path))
        save_json(scaling_params, output_path)
        logger.info(f"Saved temperature scaling parameters to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save temperature scaling parameters to {output_path}: {e}")
        raise # Re-raise error as saving params is crucial

    return temp_min, temp_max


def get_temperature_scaler(params_path: Optional[str] = None, params: Optional[Dict[str, float]] = None) -> Callable[[float], float]:
    """
    Loads scaling parameters from a JSON file or dict and returns a scaling function.

    Args:
        params_path: Path to the JSON file containing 'temp_min' and 'temp_max'.
        params: Alternatively, provide the parameters directly as a dictionary.

    Returns:
        A function that takes a raw temperature (float) and returns a scaled value [0, 1].

    Raises:
        ValueError: If neither params_path nor params is provided, or if params are invalid.
        FileNotFoundError: If params_path is provided but the file doesn't exist.
        KeyError: If required keys are missing in the parameters.
    """
    if params is None and params_path:
        logger.info(f"Loading temperature scaling parameters from: {params_path}")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Temperature scaling file not found: {params_path}")
        try:
            params = load_json(params_path)
        except Exception as e:
            logger.error(f"Error loading or parsing temperature scaling file {params_path}: {e}")
            raise
    elif params is None:
         raise ValueError("Must provide either params_path or params dictionary to get_temperature_scaler.")
    # else: use provided params dict

    try:
        temp_min = float(params['temp_min'])
        temp_max = float(params['temp_max'])
    except KeyError as e:
        raise KeyError(f"Missing key '{e}' in temperature scaling parameters: {params}") from e
    except (TypeError, ValueError) as e:
         raise ValueError(f"Invalid values in temperature scaling parameters {params}: {e}") from e

    temp_range = temp_max - temp_min
    logger.info(f"  Loaded Temp Scaler -> Min: {temp_min:.2f}, Max: {temp_max:.2f}, Range: {temp_range:.2f}")

    # Define the scaling function using the loaded parameters
    if abs(temp_range) < 1e-6:
        logger.warning("Temperature range is near zero. Scaling function will return 0.5.")
        # Return a lambda function that outputs a constant 0.5
        scaler_func = lambda t: 0.5
    else:
        # Return the standard Min-Max scaling function
        # Add epsilon for numerical stability if range is very small but non-zero
        epsilon = 1e-8
        # Use closure to capture temp_min and temp_range
        scaler_func = lambda t: (float(t) - temp_min) / (temp_range + epsilon)

    return scaler_func
EOF

# --- src/voxelflex/cli/cli.py ---
cat << 'EOF' > src/voxelflex/cli/cli.py
# src/voxelflex/cli/cli.py
"""
Command Line Interface for VoxelFlex (Temperature-Aware).
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

# Setup logging early, potentially before config is loaded for basic messages
from voxelflex.utils.logging_utils import setup_logging, get_logger, log_section_header, log_final_memory_state
# Logger instance for this module
logger = get_logger("cli")

# Import command functions only when needed to avoid heavy imports at startup
# from voxelflex.config.config import load_config
# from voxelflex.utils.system_utils import log_gpu_details, check_memory_usage
# from voxelflex.cli.commands.preprocess import run_preprocessing
# from voxelflex.cli.commands.train import train_model
# from voxelflex.cli.commands.predict import predict_rmsf
# from voxelflex.cli.commands.evaluate import evaluate_model
# from voxelflex.cli.commands.visualize import create_visualizations

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="voxelflex",
        description="VoxelFlex (Temp-Aware): Preprocess, train, predict, evaluate protein flexibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Common arguments for subparsers
    common_parser_args = argparse.ArgumentParser(add_help=False)
    common_parser_args.add_argument(
         '-v', '--verbose', action='count', default=0,
         help="Increase verbosity level (-v INFO, -vv DEBUG)."
    )
    common_parser_args.add_argument(
         "--config", type=str, required=True, help="Path to YAML config file."
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run", required=True)

    # --- Preprocess Command ---
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess raw data into batches.",
        parents=[common_parser_args], # Inherit common args
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add specific args for preprocess if any needed later

    # --- Train Command ---
    train_parser = subparsers.add_parser(
        "train", help="Train a model using preprocessed data.",
        parents=[common_parser_args], # Inherit common args
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        "--force_preprocess", action="store_true",
        help="Run preprocessing first, even if .meta files exist."
    )

    # --- Predict Command ---
    predict_parser = subparsers.add_parser(
        "predict", help="Predict RMSF at a target temperature.",
        parents=[common_parser_args], # Inherit common args
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint (.pt)."
    )
    predict_parser.add_argument(
        "--temperature", type=float, required=True, help="Target prediction temperature (K)."
    )
    predict_parser.add_argument(
        "--domains", type=str, nargs='*', default=None,
        help="Optional: List specific HDF5 domain keys to predict. If omitted, predicts on test split domains found in HDF5."
    )
    predict_parser.add_argument(
        "--output_csv", type=str, default=None, help="Optional: Specify output CSV filename (relative to metrics dir)."
    )

    # --- Evaluate Command ---
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model predictions against ground truth.",
        parents=[common_parser_args], # Inherit common args
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    evaluate_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint (.pt)."
    )
    evaluate_parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions CSV file generated by 'predict'."
    )
    # Output json is now automatically named based on predictions filename

    # --- Visualize Command ---
    visualize_parser = subparsers.add_parser(
        "visualize", help="Create performance visualizations.",
        parents=[common_parser_args], # Inherit common args
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    visualize_parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions CSV file."
    )
    visualize_parser.add_argument(
        "--history", type=str, default=None, help="Optional: Path to training history JSON file (e.g., from run output)."
    )

    return parser.parse_args(args)


def main(cli_args: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    start_time = time.time()
    args = parse_args(cli_args)

    # Setup initial console logging based on verbosity
    console_log_level = "DEBUG" if args.verbose >= 2 else "INFO" if args.verbose == 1 else "WARNING"
    # Configure root logger initially (file handler added after config load)
    setup_logging(console_level=console_log_level, file_level="DEBUG", log_file=None)

    config = None
    log_file_path = None # Keep track of log file path

    try:
        # --- Load Configuration ---
        # Config is required for all commands currently
        if not hasattr(args, 'config') or not args.config:
             raise ValueError("A configuration file path (--config) is required.")

        # Import config loading function here
        from voxelflex.config.config import load_config
        config = load_config(args.config)

        # --- Re-setup Logging with File Handler ---
        log_file_path = os.path.join(config["output"]["log_dir"], config["output"]["log_file"])
        # Ensure log dir exists (should be created by load_config)
        os.makedirs(config["output"]["log_dir"], exist_ok=True)
        # Reconfigure logging to include the file handler
        setup_logging(log_file=log_file_path,
                      console_level=console_log_level, # Keep console level from args
                      file_level=config["logging"].get("file_level", "DEBUG"))
        logger.info(f"Logging re-initialized. Log file: {log_file_path}")
        logger.info(f"Run output directory: {config['output']['run_dir']}")

        # Log system info
        from voxelflex.utils.system_utils import log_gpu_details
        log_gpu_details(logger)

        # --- Dispatch Command ---
        log_section_header(logger, f"EXECUTING COMMAND: {args.command}")

        if args.command == "preprocess":
            from voxelflex.cli.commands.preprocess import run_preprocessing
            run_preprocessing(config)
        elif args.command == "train":
            from voxelflex.cli.commands.train import train_model
            train_meta = config["data"]["processed_train_meta"]
            val_meta = config["data"]["processed_val_meta"]
            # Check if preprocessed data exists
            preprocessed_exists = os.path.exists(train_meta) and os.path.exists(val_meta)
            if not preprocessed_exists or args.force_preprocess:
                 if args.force_preprocess: logger.info("Preprocessing forced by --force_preprocess.")
                 else: logger.warning("Preprocessed data metadata not found. Running preprocessing first...")
                 # Run preprocessing
                 from voxelflex.cli.commands.preprocess import run_preprocessing
                 run_preprocessing(config)
                 # Verify again after running
                 if not os.path.exists(train_meta) or not os.path.exists(val_meta):
                      raise RuntimeError("Preprocessing ran but required metadata files still missing. Cannot proceed with training.")
                 logger.info("Preprocessing finished. Proceeding with training.")
            else: logger.info("Preprocessed data found. Skipping preprocessing.")
            # Run training
            train_model(config)
        elif args.command == "predict":
            from voxelflex.cli.commands.predict import predict_rmsf
            predict_rmsf(
                config=config,
                model_path=args.model,
                target_temperature=args.temperature,
                domain_ids_to_predict=args.domains,
                output_csv_filename=args.output_csv # Pass optional filename
            )
        elif args.command == "evaluate":
            from voxelflex.cli.commands.evaluate import evaluate_model
            evaluate_model(
                config=config,
                model_path=args.model,
                predictions_path=args.predictions
            )
        elif args.command == "visualize":
             from voxelflex.cli.commands.visualize import create_visualizations
             # Resolve history path relative to CWD if provided and not absolute
             history_file_resolved = None
             if args.history:
                 from voxelflex.utils.file_utils import resolve_path
                 history_file_resolved = resolve_path(args.history)
                 if not os.path.exists(history_file_resolved):
                      logger.warning(f"Specified history file not found: {history_file_resolved}")
                      history_file_resolved = None # Reset if not found

             create_visualizations(
                 config=config,
                 predictions_path=args.predictions,
                 history_path=history_file_resolved
             )
        else:
            # This case should not be reached due to argparse 'required=True'
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

        log_section_header(logger, f"COMMAND '{args.command}' COMPLETED")

    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime Error: {e}")
        sys.exit(1)
    except Exception as e:
        # Log the full exception traceback for unexpected errors
        logger.exception(f"An unexpected error occurred during command '{args.command}': {e}")
        sys.exit(1)
    finally:
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Total execution time: {total_duration:.2f} seconds.")
        log_final_memory_state(logger)
        logging.shutdown() # Ensure all handlers are closed properly

if __name__ == "__main__":
    main()
EOF

# --- src/voxelflex/cli/commands/preprocess.py ---
cat << 'EOF' > src/voxelflex/cli/commands/preprocess.py
# src/voxelflex/cli/commands/preprocess.py
"""
Preprocessing command for VoxelFlex (Temperature-Aware).

Reads raw HDF5 voxel data and aggregated RMSF CSV data, processes voxels
robustly (handling type/shape, skipping faulty residues), scales temperature,
batches samples, and saves optimized tensor files (.pt) for faster training/evaluation.
Uses an in-memory cache for recently loaded domains during batch creation.
"""

import os
import time
import json
import logging
import gc
import math
import h5py
from typing import Dict, Any, Tuple, List, Optional, Callable, Set, DefaultDict
from collections import defaultdict, OrderedDict, deque

import numpy as np
import pandas as pd
import torch

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.preprocess")

# Project Imports
from voxelflex.data.data_loader import (
    load_aggregated_rmsf_data,
    create_master_rmsf_lookup,
    create_domain_mapping,
    load_process_voxels_from_hdf5 # Use the primary robust loader
)
from voxelflex.utils.logging_utils import (
    log_stage, EnhancedProgressBar, log_memory_usage, log_section_header, get_logger
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, load_list_from_file, save_list_to_file, resolve_path
from voxelflex.utils.system_utils import clear_memory
from voxelflex.utils.temp_scaling import calculate_and_save_temp_scaling, get_temperature_scaler

# Define a simple LRU cache using OrderedDict for the voxel data
class SimpleLRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if key not in self.cache:
            self.misses += 1
            return None
        else:
            self.hits += 1
            # Move the accessed key to the end to mark it as recently used
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Dict[str, np.ndarray]):
        if key in self.cache:
             # Move the existing key to the end
             self.cache.move_to_end(key)
        self.cache[key] = value
        # Check if capacity is exceeded
        if len(self.cache) > self.capacity:
            # Pop the first item (least recently used)
            evicted_key, _ = self.cache.popitem(last=False)
            logger.debug(f"Cache limit ({self.capacity}) reached. Evicted domain: {evicted_key}")

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache Stats: Size={len(self.cache)}/{self.capacity}, Hits={self.hits}, Misses={self.misses}, HitRate={hit_rate:.1f}%"


def run_preprocessing(config: Dict[str, Any]):
    """Main function to execute the preprocessing pipeline."""
    log_section_header(logger, "STARTING PREPROCESSING")
    start_time = time.time()

    # --- Configuration & Setup ---
    input_cfg = config['input']
    data_cfg = config['data']
    output_cfg = config['output']
    model_cfg = config['model']

    voxel_file_path = input_cfg['voxel_file']
    rmsf_file_path = input_cfg['aggregated_rmsf_file']
    processed_dir = data_cfg['processed_dir']
    run_output_dir = output_cfg['run_dir']
    cache_limit = data_cfg['preprocessing_cache_limit']
    preprocessing_batch_size = data_cfg['preprocessing_batch_size']
    expected_channels = model_cfg['input_channels']

    # Ensure output directories exist
    ensure_dir(processed_dir)
    ensure_dir(run_output_dir)
    # Explicitly create train/val/test subdirs within processed_dir
    ensure_dir(data_cfg['processed_train_dir'])
    ensure_dir(data_cfg['processed_val_dir'])
    ensure_dir(data_cfg['processed_test_dir'])

    # Initialize tracking for failed domains
    failed_domains: Set[str] = set() # Domains where no samples could be processed
    split_samples_processed_count: DefaultDict[str, int] = defaultdict(int)
    split_samples_skipped_count: DefaultDict[str, int] = defaultdict(int)

    try:
        # --- 1. Load RMSF Data and Create Lookups ---
        with log_stage("PREPROCESS", "Loading RMSF Data & Mappings"):
            rmsf_df = load_aggregated_rmsf_data(rmsf_file_path)
            rmsf_lookup = create_master_rmsf_lookup(rmsf_df)
            # Check if RMSF lookup is empty, which indicates a problem
            if not rmsf_lookup:
                raise ValueError("RMSF lookup dictionary is empty after processing. Check RMSF data file.")

            # Get HDF5 keys safely
            try:
                with h5py.File(voxel_file_path, 'r') as f_h5:
                     hdf5_domain_keys = list(f_h5.keys())
            except Exception as e:
                 raise RuntimeError(f"Failed to read keys from HDF5 file {voxel_file_path}: {e}")

            if not hdf5_domain_keys:
                 raise ValueError(f"No domain keys found in HDF5 file: {voxel_file_path}")

            rmsf_domain_ids = rmsf_df['domain_id'].unique().tolist()
            domain_mapping = create_domain_mapping(hdf5_domain_keys, rmsf_domain_ids)
            available_hdf5_keys = set(domain_mapping.keys()) # Keys that are in HDF5 AND mappable to RMSF

            if not available_hdf5_keys:
                raise ValueError("No HDF5 domain keys could be mapped to RMSF domain IDs.")
            logger.info(f"Found {len(available_hdf5_keys)} mappable HDF5 domain keys.")

            # Free RMSF dataframe memory
            del rmsf_df; gc.collect()


        # --- 2. Load Domain Splits & Filter ---
        with log_stage("PREPROCESS", "Loading and Filtering Domain Splits"):
            split_domains: Dict[str, List[str]] = {}
            all_split_hdf5_keys_in_use: Set[str] = set()
            for split in ["train", "val", "test"]:
                split_file = input_cfg.get(f"{split}_split_file")
                if not split_file or not os.path.exists(split_file):
                    logger.warning(f"Split file for '{split}' not found or not specified ({split_file}). Skipping this split.")
                    split_domains[split] = []
                    continue

                domains_in_file = load_list_from_file(split_file)
                if not domains_in_file:
                     logger.warning(f"Split file for '{split}' ({split_file}) is empty. Skipping this split.")
                     split_domains[split] = []
                     continue

                # Filter domains: must be in HDF5 AND mappable
                valid_split_domains = []
                for d in domains_in_file:
                    if d in available_hdf5_keys:
                         valid_split_domains.append(d)
                    elif d in hdf5_domain_keys:
                         logger.warning(f"Split '{split}': Domain '{d}' exists in HDF5 but could not be mapped to RMSF data. Excluding.")
                    else:
                         logger.warning(f"Split '{split}': Domain '{d}' not found in HDF5 file. Excluding.")

                # Apply max_domains limit if specified
                max_doms = input_cfg.get('max_domains')
                if max_doms is not None and max_doms > 0 and len(valid_split_domains) > max_doms:
                     logger.info(f"Limiting '{split}' split to {max_doms} domains (from {len(valid_split_domains)}) due to 'max_domains' setting.")
                     valid_split_domains = valid_split_domains[:max_doms]

                split_domains[split] = valid_split_domains
                all_split_hdf5_keys_in_use.update(valid_split_domains)
                logger.info(f"Split '{split}': Using {len(valid_split_domains)} valid/mappable domains specified in {split_file}.")

            # Essential checks
            if not split_domains["train"]: raise ValueError("Train split is empty after filtering. Cannot proceed.")
            if not split_domains["val"]: raise ValueError("Validation split is empty after filtering. Cannot proceed.")
            if "test" not in split_domains or not split_domains["test"]: logger.warning("Test split is empty or not specified. Evaluation on test set will not be possible.")

        # --- 3. Generate Master Sample List ---
        with log_stage("PREPROCESS", "Generating Master Sample List"):
            master_samples: List[Tuple[str, str, int, float, float]] = []
            residues_without_rmsf = 0
            domains_with_residues_checked: Set[str] = set()

            logger.info(f"Checking residues for {len(all_split_hdf5_keys_in_use)} domains across all splits...")
            progress_domains = EnhancedProgressBar(len(all_split_hdf5_keys_in_use), prefix="Checking Residues")

            # Iterate only through domains present in the filtered splits
            with h5py.File(voxel_file_path, 'r') as f_h5:
                for i, hdf5_domain_id in enumerate(all_split_hdf5_keys_in_use):
                    if hdf5_domain_id not in f_h5: continue # Should not happen due to earlier check, but safety first

                    domain_group = f_h5[hdf5_domain_id]
                    residue_group = None
                    # Find the first valid residue group (chain) - same logic as data_loader
                    potential_chain_keys = sorted([k for k in domain_group.keys() if isinstance(domain_group[k], h5py.Group)])
                    for chain_key in potential_chain_keys:
                        try:
                             potential_residue_group = domain_group[chain_key]
                             if any(key.isdigit() for key in potential_residue_group.keys()):
                                  residue_group = potential_residue_group; break
                        except Exception: continue # Ignore errors accessing subgroups here

                    if residue_group is None:
                        logger.warning(f"No residue group found for domain '{hdf5_domain_id}' during sample generation.")
                        failed_domains.add(hdf5_domain_id) # Mark domain as failed if no residues found
                        progress_domains.update(i+1)
                        continue

                    domains_with_residues_checked.add(hdf5_domain_id)
                    # Get mapped RMSF domain ID
                    rmsf_domain_id = domain_mapping.get(hdf5_domain_id)
                    if rmsf_domain_id is None: continue # Should not happen if key is in available_hdf5_keys

                    for resid_str in residue_group.keys():
                        if not resid_str.isdigit(): continue
                        try:
                            resid_int = int(resid_str)
                            # Check if this residue exists in the RMSF lookup
                            lookup_key = (rmsf_domain_id, resid_int)
                            temp_rmsf_pairs = rmsf_lookup.get(lookup_key)

                            # Try base name if exact match failed
                            if temp_rmsf_pairs is None:
                                base_rmsf_id = rmsf_domain_id.split('_')[0]
                                if base_rmsf_id != rmsf_domain_id:
                                    lookup_key_base = (base_rmsf_id, resid_int)
                                    temp_rmsf_pairs = rmsf_lookup.get(lookup_key_base)

                            if temp_rmsf_pairs:
                                for raw_temp, target_rmsf in temp_rmsf_pairs:
                                    # Basic validation of temp/rmsf values
                                    if raw_temp is not None and not np.isnan(raw_temp) and \
                                       target_rmsf is not None and not np.isnan(target_rmsf) and target_rmsf >= 0:
                                        master_samples.append((
                                            hdf5_domain_id, # Use HDF5 key for voxel lookup
                                            resid_str,
                                            resid_int,
                                            float(raw_temp),
                                            float(target_rmsf)
                                        ))
                            else:
                                residues_without_rmsf += 1
                                # logger.debug(f"Residue {hdf5_domain_id}:{resid_str} found in HDF5 but not in RMSF lookup for '{rmsf_domain_id}' or its base.")

                        except ValueError:
                             logger.warning(f"Invalid residue format '{resid_str}' in domain '{hdf5_domain_id}'. Skipping.")
                        except Exception as e:
                             logger.warning(f"Error processing residue {hdf5_domain_id}:{resid_str} during sample list generation: {e}")

                    progress_domains.update(i+1)
            progress_domains.finish()

            if not master_samples:
                raise ValueError("Master sample list is empty. No overlap between HDF5 residues and RMSF data.")

            logger.info(f"Generated {len(master_samples)} total samples across all splits.")
            if residues_without_rmsf > 0:
                logger.info(f"  {residues_without_rmsf} HDF5 residues lacked corresponding RMSF data.")
            # Check if any domains that were supposed to be checked ended up having no samples
            for domain_id in all_split_hdf5_keys_in_use:
                 if domain_id not in domains_with_residues_checked:
                      if domain_id not in failed_domains: # Avoid double logging if already failed finding residues
                           logger.warning(f"Domain '{domain_id}' was in splits but yielded no samples (possibly due to RMSF lookup failures for all its residues).")
                           failed_domains.add(domain_id)

            del rmsf_lookup; gc.collect() # Free lookup memory


        # --- 4. Calculate Temperature Scaler ---
        with log_stage("PREPROCESS", "Calculating Temperature Scaler"):
            # Filter master samples to include only those belonging to the training split domains
            train_split_set = set(split_domains["train"])
            train_samples = [s for s in master_samples if s[0] in train_split_set]

            if not train_samples:
                raise ValueError("No samples found belonging to the training split domains. Cannot calculate temperature scaler.")

            train_temps = [s[3] for s in train_samples]
            temp_scaling_params_path = data_cfg["temp_scaling_params_file"]
            try:
                _, _ = calculate_and_save_temp_scaling(train_temps, temp_scaling_params_path)
                temp_scaler_func = get_temperature_scaler(params_path=temp_scaling_params_path)
            except Exception as e:
                logger.error(f"Failed to calculate, save, or load temperature scaler: {e}")
                raise

        # --- 5. Process Each Split (Create Batches) ---
        voxel_cache = SimpleLRUCache(capacity=cache_limit) # Initialize cache

        for split in ["train", "val", "test"]:
            split_hdf5_keys = split_domains.get(split)
            if not split_hdf5_keys:
                logger.info(f"Skipping batch creation for empty or unspecified split: '{split}'")
                # Ensure meta file exists even if empty
                meta_file_path = data_cfg[f"processed_{split}_meta"]
                try: open(meta_file_path, 'w').close()
                except IOError as e: logger.error(f"Could not create empty meta file {meta_file_path}: {e}")
                continue

            log_section_header(logger, f"PROCESSING SPLIT: {split.upper()}")
            split_output_dir = data_cfg[f"processed_{split}_dir"]
            meta_file_path = data_cfg[f"processed_{split}_meta"]
            ensure_dir(split_output_dir) # Ensure directory exists

            # Filter master sample list for the current split
            current_split_set = set(split_hdf5_keys)
            split_samples = [s for s in master_samples if s[0] in current_split_set]
            num_split_samples = len(split_samples)

            if num_split_samples == 0:
                logger.warning(f"No samples available for split '{split}' after filtering master list. Skipping batch creation.")
                try: open(meta_file_path, 'w').close() # Create empty meta file
                except IOError as e: logger.error(f"Could not create empty meta file {meta_file_path}: {e}")
                continue

            logger.info(f"Processing {num_split_samples} samples for '{split}' split.")
            num_batches = math.ceil(num_split_samples / preprocessing_batch_size)
            logger.info(f"Saving into {num_batches} batches (Size: {preprocessing_batch_size}).")

            processed_batch_paths_rel: List[str] = [] # Store relative paths for meta file
            split_domain_failures = defaultdict(int) # Track failures per domain within this split
            split_domain_total_residues = defaultdict(int) # Track total residues attempted per domain

            progress_batches = EnhancedProgressBar(num_batches, prefix=f"Save Batches {split}", stage_info="BATCH_SAVE")

            voxel_cache.clear() # Clear cache before processing a new split
            logger.info(f"Voxel cache cleared for split '{split}'.")

            for i in range(0, num_split_samples, preprocessing_batch_size):
                batch_idx = i // preprocessing_batch_size
                current_batch_samples = split_samples[i : i + preprocessing_batch_size]
                if not current_batch_samples: continue

                # --- Load Voxel Data for Batch (Using Cache) ---
                domains_needed_for_batch: Set[str] = set(s[0] for s in current_batch_samples)
                domains_to_load_from_hdf5: List[str] = []

                # Check cache first
                batch_voxel_data: Dict[str, Dict[str, np.ndarray]] = {}
                for domain_id in domains_needed_for_batch:
                     cached_data = voxel_cache.get(domain_id) # Returns None if miss
                     if cached_data is not None:
                          batch_voxel_data[domain_id] = cached_data
                     else:
                          domains_to_load_from_hdf5.append(domain_id)

                # Load missing domains from HDF5
                if domains_to_load_from_hdf5:
                     logger.debug(f"Batch {batch_idx+1}: Loading {len(domains_to_load_from_hdf5)} domains from HDF5...")
                     try:
                          loaded_domain_data = load_process_voxels_from_hdf5(
                               voxel_file_path,
                               domains_to_load_from_hdf5,
                               expected_channels=expected_channels
                               # target_shape_chw can be passed if strict validation needed
                          )
                          # Add successfully loaded domains to cache and batch data
                          for domain_id, domain_residues in loaded_domain_data.items():
                               if domain_residues: # Only cache if residues were loaded
                                    batch_voxel_data[domain_id] = domain_residues
                                    voxel_cache.put(domain_id, domain_residues)
                               else:
                                    # Domain was attempted but yielded no valid residues
                                    logger.warning(f"Domain '{domain_id}' loaded from HDF5 but contained no processable residues for batch {batch_idx+1}.")
                                    split_domain_failures[domain_id] += 1 # Count as failure if expected
                                    # Do not add to cache
                          # Log cache stats periodically
                          if batch_idx % 50 == 0: logger.debug(voxel_cache.stats())

                     except Exception as load_e:
                          logger.error(f"Critical error loading domain batch for batch {batch_idx+1}: {load_e}")
                          # Mark all requested domains as failed for this batch attempt
                          for domain_id in domains_to_load_from_hdf5:
                               split_domain_failures[domain_id] += 1


                # --- Assemble Batch Tensors ---
                batch_voxels_list: List[torch.Tensor] = []
                batch_temps_list: List[float] = []
                batch_targets_list: List[float] = []
                samples_in_batch_processed = 0
                samples_in_batch_skipped = 0

                for sample_tuple in current_batch_samples:
                    hdf5_domain_id, resid_str, _, raw_temp, target_rmsf = sample_tuple
                    split_domain_total_residues[hdf5_domain_id] += 1 # Count attempt

                    # Try to get voxel data from the potentially cached data
                    voxel_array = batch_voxel_data.get(hdf5_domain_id, {}).get(resid_str)

                    if voxel_array is not None:
                        try:
                            scaled_temp = temp_scaler_func(raw_temp)
                            batch_voxels_list.append(torch.from_numpy(voxel_array)) # Convert to tensor
                            batch_temps_list.append(scaled_temp)
                            batch_targets_list.append(target_rmsf)
                            samples_in_batch_processed += 1
                        except Exception as assemble_e:
                             logger.warning(f"Error assembling sample {hdf5_domain_id}:{resid_str} into batch: {assemble_e}")
                             samples_in_batch_skipped += 1
                             split_domain_failures[hdf5_domain_id] += 1
                    else:
                        # Voxel data was not loaded successfully (either domain failed or residue failed)
                        # logger.debug(f"Skipping sample {hdf5_domain_id}:{resid_str} in batch {batch_idx+1}: Voxel data not available.")
                        samples_in_batch_skipped += 1
                        split_domain_failures[hdf5_domain_id] += 1 # Count as failure for this domain

                # Update overall split counts
                split_samples_processed_count[split] += samples_in_batch_processed
                split_samples_skipped_count[split] += samples_in_batch_skipped

                # --- Save Batch ---
                if not batch_voxels_list:
                    logger.warning(f"Batch {batch_idx + 1}/{num_batches} for '{split}' is empty after assembly. Skipping save.")
                    progress_batches.update(batch_idx + 1)
                    continue # Skip to next batch

                try:
                    # Stack tensors
                    voxel_batch_tensor = torch.stack(batch_voxels_list)
                    scaled_temp_batch_tensor = torch.tensor(batch_temps_list, dtype=torch.float32).unsqueeze(1) # Add channel dim
                    target_rmsf_batch_tensor = torch.tensor(batch_targets_list, dtype=torch.float32)

                    # Define batch filename and path
                    batch_filename = f"batch_{batch_idx:06d}.pt"
                    batch_filepath = os.path.join(split_output_dir, batch_filename)

                    # Save the batch dictionary
                    batch_data_to_save = {
                        'voxels': voxel_batch_tensor,
                        'scaled_temps': scaled_temp_batch_tensor,
                        'targets': target_rmsf_batch_tensor
                    }
                    torch.save(batch_data_to_save, batch_filepath)

                    # Store relative path for meta file
                    relative_batch_path = os.path.join(split, batch_filename)
                    processed_batch_paths_rel.append(relative_batch_path)

                except Exception as save_e:
                    logger.error(f"Error stacking or saving batch file {batch_filepath}: {save_e}")
                    # Attempt to clean up if file was partially created? os.remove? Risky.

                finally:
                     # Clean up tensors for this batch explicitly
                     del batch_voxels_list, batch_temps_list, batch_targets_list
                     if 'voxel_batch_tensor' in locals(): del voxel_batch_tensor
                     if 'scaled_temp_batch_tensor' in locals(): del scaled_temp_batch_tensor
                     if 'target_rmsf_batch_tensor' in locals(): del target_rmsf_batch_tensor
                     if batch_idx % 50 == 0: gc.collect() # Periodic GC

                progress_batches.update(batch_idx + 1) # Update progress bar

            progress_batches.finish()
            logger.info(voxel_cache.stats()) # Log final cache stats for the split

            # Save Metadata File for the split
            try:
                with open(meta_file_path, 'w') as f_meta:
                    for rel_path in processed_batch_paths_rel:
                        f_meta.write(f"{rel_path}\n")
                logger.info(f"Saved metadata for {len(processed_batch_paths_rel)} batches to {meta_file_path}")
            except Exception as meta_e:
                logger.error(f"Error writing metadata file {meta_file_path}: {meta_e}")

            # Check for domains that completely failed within this split
            for domain_id in split_hdf5_keys:
                 attempted = split_domain_total_residues.get(domain_id, 0)
                 failed = split_domain_failures.get(domain_id, 0)
                 if attempted > 0 and failed >= attempted:
                      logger.warning(f"Split '{split}': All {attempted} attempted samples for domain '{domain_id}' failed processing.")
                      failed_domains.add(domain_id) # Add to global failed list

            logger.info(f"Finished processing split '{split}'. Processed {split_samples_processed_count[split]} samples, Skipped {split_samples_skipped_count[split]} samples.")


        # --- 6. Final Summary and Cleanup ---
        log_section_header(logger, "PREPROCESSING FINISHED")
        total_duration = time.time() - start_time
        logger.info(f"Total Preprocessing Time: {total_duration:.2f}s.")

        total_processed = sum(split_samples_processed_count.values())
        total_skipped = sum(split_samples_skipped_count.values())
        logger.info(f"Overall: Processed {total_processed} samples, Skipped {total_skipped} samples.")

        if failed_domains:
            failed_list_path = os.path.join(run_output_dir, "failed_preprocess_domains.txt")
            logger.warning(f"Found {len(failed_domains)} domains where no samples could be successfully processed.")
            try:
                sorted_failures = sorted(list(failed_domains))
                save_list_to_file(sorted_failures, failed_list_path)
                logger.info(f"List of failed domains saved to: {failed_list_path}")
            except Exception as save_err:
                logger.error(f"Could not save list of failed domains: {save_err}")
        else:
            logger.info("No domains completely failed during preprocessing.")

    except Exception as e:
        logger.exception(f"Preprocessing pipeline failed with error: {e}")
        # Optionally save partial failure info if possible
        if failed_domains:
             try:
                  failed_list_path = os.path.join(run_output_dir, "failed_preprocess_domains_partial.txt")
                  save_list_to_file(sorted(list(failed_domains)), failed_list_path)
                  logger.info(f"Saved partial list of failed domains to: {failed_list_path}")
             except: pass # Ignore errors during cleanup save
        raise # Re-raise the exception to signal failure

    finally:
        # Final cleanup
        del master_samples # Ensure large list is cleared
        voxel_cache.clear()
        gc.collect()
        logger.info("End of preprocessing run.")
        log_memory_usage(logger, level=logging.INFO) # Log final memory
EOF

# --- src/voxelflex/cli/commands/train.py ---
cat << 'EOF' > src/voxelflex/cli/commands/train.py
# src/voxelflex/cli/commands/train.py (Simplified for Preprocessed Data)
"""
Training command for VoxelFlex (Temperature-Aware).

Loads preprocessed data batches and trains the model.
"""

import os
import time
import json
import logging
import math # Import math
from typing import Dict, Any, Tuple, List, Optional, Callable
import shutil
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # Import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from scipy.stats import pearsonr

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.train")

# Project imports
from voxelflex.data.data_loader import PreprocessedVoxelFlexDataset # Use the correct dataset name
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import (
    log_stage, EnhancedProgressBar, log_memory_usage, log_section_header, get_logger
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, resolve_path
from voxelflex.utils.system_utils import (
    get_device, clear_memory, check_memory_usage,
    set_num_threads, adjust_workers_for_memory
)
from voxelflex.utils.temp_scaling import get_temperature_scaler # Needed for loading scaler info for checkpoints

# --- Train/Validate Epoch Functions (Simplified Signatures) ---

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]: # Returns avg_loss, avg_pearson
    """Train model for one epoch using preprocessed batches."""
    model.train()
    running_loss_sum = 0.0
    batch_preds_all: List[np.ndarray] = []
    batch_targets_all: List[np.ndarray] = []
    num_samples_processed = 0

    train_cfg = config['training']
    gradient_clip_cfg = train_cfg.get('gradient_clipping', {})
    gradient_clip_norm = gradient_clip_cfg.get('max_norm') if gradient_clip_cfg.get('enabled') else None
    show_progress = config['logging'].get('show_progress_bars', True)
    autocast_enabled = scaler is not None and scaler.is_enabled()

    progress = None
    if show_progress and train_loader is not None and len(train_loader) > 0:
        # Use math.ceil for calculating total steps if needed, though len(loader) is direct
        progress = EnhancedProgressBar(len(train_loader), prefix=f"Epoch {epoch+1} Train")

    if train_loader is None or len(train_loader) == 0:
        logger.warning("Training loader is empty. Skipping training epoch.")
        return 0.0, 0.0

    for i, batch_dict in enumerate(train_loader):
        # Skip potentially empty batches loaded from file system issues
        if not batch_dict:
            logger.warning(f"Skipping empty batch dict at index {i} in train_loader.")
            continue

        try:
            # Load tensors from the preprocessed batch dictionary
            # Ensure keys exist before accessing
            if not all(k in batch_dict for k in ['voxels', 'scaled_temps', 'targets']):
                 logger.error(f"Train batch {i} dict missing required keys: {batch_dict.keys()}. Skipping.")
                 continue

            voxel_inputs = batch_dict['voxels'].to(device, non_blocking=True)
            scaled_temps = batch_dict['scaled_temps'].to(device, non_blocking=True) # Already scaled
            targets = batch_dict['targets'].to(device, non_blocking=True)

            if voxel_inputs.numel() == 0:
                logger.warning(f"Skipping empty tensor batch at index {i} in train_loader.")
                continue # Skip if tensors are empty
            current_batch_size = voxel_inputs.size(0)

        except Exception as load_e:
             logger.error(f"Error loading/moving tensors for train batch {i}: {load_e}")
             continue # Skip this batch

        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                # Ensure model gets correct inputs
                outputs = model(voxel_input=voxel_inputs, scaled_temp=scaled_temps)
                loss = criterion(outputs, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss encountered in train batch {i}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True) # Clear gradients if loss is invalid
                continue

        except Exception as forward_e:
            logger.exception(f"Error during train forward pass batch {i}: {forward_e}")
            continue # Skip batch on forward error

        try:
            if scaler is not None: # AMP enabled
                scaler.scale(loss).backward()
                # Unscale before clipping
                scaler.unscale_(optimizer)
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else: # No AMP
                loss.backward()
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        except Exception as backward_e:
            logger.exception(f"Error during train backward/step batch {i}: {backward_e}")
            optimizer.zero_grad(set_to_none=True) # Clear potentially corrupt gradients
            continue # Skip batch on backward/step error

        # Accumulate results for epoch metrics
        running_loss_sum += loss.item() * current_batch_size
        # Detach, move to CPU, convert to numpy *before* appending
        batch_preds_all.append(outputs.detach().cpu().numpy())
        batch_targets_all.append(targets.detach().cpu().numpy())
        num_samples_processed += current_batch_size

        if progress: progress.update(i + 1)
        # Explicitly delete tensors to potentially free memory sooner
        del voxel_inputs, scaled_temps, targets, outputs, loss
        if i % 100 == 0 and device.type == 'cuda': torch.cuda.empty_cache()

    if progress: progress.finish()

    # Calculate epoch metrics
    avg_loss = running_loss_sum / num_samples_processed if num_samples_processed > 0 else 0.0
    epoch_pearson = 0.0
    if num_samples_processed > 1 and batch_preds_all and batch_targets_all:
        try:
            # Concatenate results from all batches
            all_preds_flat = np.concatenate([b.flatten() for b in batch_preds_all])
            all_targets_flat = np.concatenate([b.flatten() for b in batch_targets_all])

            # Ensure finite values and sufficient standard deviation for correlation
            valid_mask = np.isfinite(all_preds_flat) & np.isfinite(all_targets_flat)
            preds_valid = all_preds_flat[valid_mask]
            targets_valid = all_targets_flat[valid_mask]

            if len(preds_valid) > 1 and np.std(preds_valid) > 1e-6 and np.std(targets_valid) > 1e-6:
                corr, _ = pearsonr(preds_valid, targets_valid)
                epoch_pearson = corr if not np.isnan(corr) else 0.0
            else:
                logger.debug(f"Train Epoch {epoch+1}: Insufficient valid data or variance for correlation calculation.")
        except Exception as e:
            logger.exception(f"Error calculating train epoch correlation: {e}")

    return avg_loss, epoch_pearson


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[float, float]: # Returns avg_loss, avg_pearson
    """Validate model using preprocessed validation batches."""
    model.eval() # Set model to evaluation mode
    running_loss_sum = 0.0
    batch_preds_all: List[np.ndarray] = []
    batch_targets_all: List[np.ndarray] = []
    num_samples_processed = 0
    autocast_enabled = config['training'].get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda'
    show_progress = config['logging'].get('show_progress_bars', True)

    progress = None
    if show_progress and val_loader is not None and len(val_loader) > 0:
        progress = EnhancedProgressBar(len(val_loader), prefix="Validation")

    if val_loader is None or len(val_loader) == 0:
        logger.warning("Validation loader is empty. Skipping validation.")
        # Return high loss, zero correlation to avoid breaking logic that expects floats
        return float('inf'), 0.0

    with torch.no_grad(): # Disable gradient calculations
        for i, batch_dict in enumerate(val_loader):
            if not batch_dict:
                logger.warning(f"Skipping empty batch dict at index {i} in val_loader.")
                continue

            try:
                if not all(k in batch_dict for k in ['voxels', 'scaled_temps', 'targets']):
                     logger.error(f"Validation batch {i} dict missing required keys: {batch_dict.keys()}. Skipping.")
                     continue

                voxel_inputs = batch_dict['voxels'].to(device, non_blocking=True)
                scaled_temps = batch_dict['scaled_temps'].to(device, non_blocking=True)
                targets = batch_dict['targets'].to(device, non_blocking=True)

                if voxel_inputs.numel() == 0:
                    logger.warning(f"Skipping empty tensor batch at index {i} in val_loader.")
                    continue
                current_batch_size = voxel_inputs.size(0)

            except Exception as load_e:
                logger.error(f"Error loading/moving tensors for validation batch {i}: {load_e}")
                continue # Skip this batch

            try:
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                    outputs = model(voxel_input=voxel_inputs, scaled_temp=scaled_temps)
                    loss = criterion(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss encountered in validation batch {i}. Skipping batch metrics.")
                    continue # Skip accumulation if loss is invalid

                # Accumulate results
                running_loss_sum += loss.item() * current_batch_size
                batch_preds_all.append(outputs.cpu().numpy()) # Move to CPU before numpy conversion
                batch_targets_all.append(targets.cpu().numpy())
                num_samples_processed += current_batch_size

            except Exception as e:
                logger.exception(f"Error during validation forward pass batch {i}: {e}")
                # Continue to next batch

            if progress: progress.update(i + 1)
            # Explicitly delete tensors
            del voxel_inputs, scaled_temps, targets, outputs, loss
            if i % 100 == 0 and device.type == 'cuda': torch.cuda.empty_cache()

    if progress: progress.finish()

    # Calculate validation metrics
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
            else:
                logger.debug(f"Validation: Insufficient valid data or variance for correlation calculation.")

        except Exception as e:
            logger.exception(f"Error calculating validation epoch correlation: {e}")

    return avg_loss, epoch_pearson


# --- Get Optimizer/Scheduler ---
def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Creates an optimizer based on the configuration."""
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    # Add more optimizer types if needed (e.g., SGD)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer created: AdamW (LR={lr:.2e}, WeightDecay={weight_decay:.2e})")
    return optimizer

def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], num_epochs: int):
    """Creates a learning rate scheduler based on the configuration."""
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'reduce_on_plateau').lower()
    monitor_metric = scheduler_config.get('monitor_metric', 'val_pearson')
    if monitor_metric not in ['val_loss', 'val_pearson']:
        logger.warning(f"Invalid scheduler monitor metric '{monitor_metric}'. Defaulting to 'val_pearson'.")
        monitor_metric = 'val_pearson'
    # Determine mode based on metric if not specified or invalid
    mode = scheduler_config.get('mode')
    if mode not in ['min', 'max']:
        mode = 'max' if 'pearson' in monitor_metric else 'min'
        logger.debug(f"Scheduler mode auto-set to '{mode}' based on metric '{monitor_metric}'.")

    scheduler = None
    if scheduler_type == 'reduce_on_plateau':
        patience=int(scheduler_config.get('patience', 5))
        factor=float(scheduler_config.get('factor', 0.5))
        min_lr=float(scheduler_config.get('min_lr', 1e-7))
        threshold=float(scheduler_config.get('threshold', 0.001))
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr, verbose=True)
        logger.info(f"Using ReduceLROnPlateau scheduler (Metric: {monitor_metric}, Mode: {mode}, Patience: {patience}, Factor: {factor})")
    elif scheduler_type == 'cosine_annealing':
        T_max = int(scheduler_config.get('T_max', num_epochs))
        eta_min = float(scheduler_config.get('eta_min', 1e-7))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        logger.info(f"Using CosineAnnealingLR scheduler (T_max={T_max}, eta_min={eta_min:.1e})")
    elif scheduler_type == 'step':
        step_size = int(scheduler_config.get('step_size', 10))
        gamma = float(scheduler_config.get('gamma', 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Using StepLR scheduler (Step Size: {step_size}, Gamma: {gamma})")
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")

    return scheduler, monitor_metric, mode


# --- Main Training Function (Simplified for Preprocessed Data) ---

def train_model(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, List[float]]]]:
    """
    Main function to train the model using preprocessed data batches.
    Assumes preprocessing has already been run and .meta files exist.

    Args:
        config: The configuration dictionary.

    Returns:
        A tuple containing:
            - Path to the best model checkpoint file (or latest if no improvement).
            - The training history dictionary.
        Returns (None, None) on fatal error.
    """
    # --- Setup ---
    run_output_dir = config["output"]["run_dir"]
    models_dir = config["output"]["models_dir"]
    ensure_dir(models_dir) # Ensure models dir within run dir exists

    log_section_header(logger, f"STARTING TRAINING RUN: {config['output']['run_name']}")
    log_memory_usage(logger)
    # check_system_resources can be added back if needed

    device = get_device(config["system_utilization"].get("adjust_for_gpu", True))
    set_num_threads(config["training"].get("num_workers")) # Set PyTorch threads based on dataloader workers
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        # Optional: Enable deterministic algorithms for reproducibility (can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Load Datasets & DataLoaders ---
    with log_stage("DATA_PREPARATION", "Loading preprocessed datasets"):
        try:
            train_meta_path = config["data"]["processed_train_meta"]
            val_meta_path = config["data"]["processed_val_meta"]
            processed_dir = config["data"]["processed_dir"]

            if not os.path.exists(train_meta_path):
                 raise FileNotFoundError(f"Preprocessed training metadata file not found: {train_meta_path}. Run 'preprocess' first.")
            if not os.path.exists(val_meta_path):
                 raise FileNotFoundError(f"Preprocessed validation metadata file not found: {val_meta_path}. Run 'preprocess' first.")

            train_dataset = PreprocessedVoxelFlexDataset(train_meta_path, processed_dir)
            val_dataset = PreprocessedVoxelFlexDataset(val_meta_path, processed_dir)

            if len(train_dataset) == 0:
                 raise ValueError("Preprocessed training dataset is empty. Check .meta file and .pt files.")
            if len(val_dataset) == 0:
                 raise ValueError("Preprocessed validation dataset is empty. Check .meta file and .pt files.")

            # Create DataLoaders directly from datasets that yield batches
            train_batch_size = config['training']['batch_size'] # Used for reference, actual batching is per .pt file
            num_workers = config['training'].get('num_workers', 0)
            num_workers = adjust_workers_for_memory(num_workers) # Adjust based on memory
            pin_memory = config['training'].get('pin_memory', True) and (device.type == 'cuda')
            persistent_workers = pin_memory and num_workers > 0 and config['training'].get('persistent_workers', True)

            logger.info(f"Train DataLoader: {len(train_dataset)} batches, Workers={num_workers}, PinMemory={pin_memory}, PersistentWorkers={persistent_workers}")
            # Shuffle=True shuffles the *order* of the batch files loaded each epoch
            train_loader = DataLoader(
                train_dataset,
                batch_size=None, # Load one preprocessed batch file at a time
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=lambda x: x[0] if x else None # Handle potential empty batches from dataset __getitem__
            )

            logger.info(f"Val DataLoader: {len(val_dataset)} batches, Workers={num_workers}, PinMemory={pin_memory}, PersistentWorkers={persistent_workers}")
            val_loader = DataLoader(
                val_dataset,
                batch_size=None, # Load one preprocessed batch file at a time
                shuffle=False, # No need to shuffle validation batches
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=lambda x: x[0] if x else None # Handle potential empty batches
            )

        except Exception as e:
            logger.exception(f"Fatal error loading preprocessed data or creating DataLoaders: {e}")
            return None, None

    # --- Temperature Scaler (Load info for saving with checkpoint) ---
    temp_scaling_params: Optional[Dict[str, float]] = None
    try:
        scaling_params_path = config["data"]["temp_scaling_params_file"]
        if not os.path.exists(scaling_params_path):
             raise FileNotFoundError(f"Temperature scaling file not found at expected location: {scaling_params_path}")
        temp_scaling_params = load_json(scaling_params_path)
        if 'temp_min' not in temp_scaling_params or 'temp_max' not in temp_scaling_params:
             raise ValueError("Temperature scaling file is missing 'temp_min' or 'temp_max'.")
        logger.info(f"Loaded temperature scaling params from {scaling_params_path} for checkpoint saving.")
    except Exception as e:
         logger.warning(f"Could not load temp scaling params ({e}). Checkpoint won't include them. Inference might fail if params differ.")
         temp_scaling_params = None # Ensure it's None if loading failed


    # --- Model & Optimizer ---
    with log_stage("MODEL_CREATION", "Creating model and optimizer"):
        try:
             # Determine input shape from the first batch
             logger.info("Loading one batch to determine input shape...")
             input_shape = None
             sample_batch = None
             for batch in train_loader:
                 if batch and 'voxels' in batch and batch['voxels'].numel() > 0:
                     sample_batch = batch
                     break # Found a valid batch

             if sample_batch is None:
                  # Try validation loader if train loader failed (less ideal)
                  for batch in val_loader:
                     if batch and 'voxels' in batch and batch['voxels'].numel() > 0:
                         sample_batch = batch
                         break
             if sample_batch is None:
                 raise RuntimeError("Could not load any valid batch from train or val loader to determine input shape.")

             # Shape is (Batch, Channels, D, H, W), we need (Channels, D, H, W)
             input_shape = tuple(sample_batch['voxels'].shape[1:])
             logger.info(f"Determined model input voxel shape: {input_shape}")
             del sample_batch; gc.collect() # Free memory from sample batch

             # Create model instance
             model = get_model(config['model'], input_shape=input_shape)
             model.to(device)
             total_params = sum(p.numel() for p in model.parameters())
             trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
             logger.info(f"Model '{config['model']['architecture']}' created.")
             logger.info(f"  Total Parameters: {total_params:,}")
             logger.info(f"  Trainable Parameters: {trainable_params:,}")

             optimizer = get_optimizer(model, config)
             num_epochs = config['training']['num_epochs']
             scheduler, monitor_metric, scheduler_mode = get_scheduler(optimizer, config, num_epochs)

        except Exception as e:
            logger.exception(f"Fatal error creating model or optimizer: {e}")
            return None, None

    # --- Resume / Training Prep ---
    start_epoch = 0
    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'train_pearson': [], 'val_pearson': [], 'lr': []}
    best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
    best_epoch = -1
    resume_path = config['training'].get('resume_checkpoint')

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Attempting to resume training from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            # Load model state dict cautiously
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys: logger.warning(f"Resume Warning: Missing keys in model state_dict: {missing_keys}")
            if unexpected_keys: logger.warning(f"Resume Warning: Unexpected keys in model state_dict: {unexpected_keys}")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1 # Start from next epoch
            history = checkpoint.get('history', history)
            # Load best metric value correctly based on monitor_metric from config, not checkpoint's saved metric name
            # This handles cases where the monitored metric changed between runs
            best_metric_value = checkpoint.get(f'best_{monitor_metric}', best_metric_value)
            best_epoch = checkpoint.get('best_epoch', -1)
            # Load scheduler state if available and types match
            if scheduler and 'scheduler_state_dict' in checkpoint and type(scheduler).__name__ == checkpoint.get('scheduler_type'):
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 logger.info("Resuming scheduler state.")
            elif scheduler and 'scheduler_state_dict' in checkpoint:
                 logger.warning("Scheduler type in checkpoint does not match current scheduler. Cannot resume scheduler state.")

            logger.info(f"Resumed from epoch {start_epoch}. Best '{monitor_metric}': {best_metric_value:.6f} @ epoch {best_epoch}")
            del checkpoint; gc.collect()
        except Exception as e:
            logger.exception(f"Failed to load checkpoint '{resume_path}': {e}. Starting training from scratch.")
            start_epoch = 0 # Reset if checkpoint load fails
            history = {'train_loss': [], 'val_loss': [], 'train_pearson': [], 'val_pearson': [], 'lr': []}
            best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
            best_epoch = -1
    else:
        logger.info("No resume checkpoint specified or found. Starting training from scratch.")


    # --- Training Loop Setup ---
    criterion = nn.MSELoss() # Use Mean Squared Error for RMSF regression
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['mixed_precision']['enabled'] and device.type == 'cuda')
    if scaler.is_enabled(): logger.info("Using Automatic Mixed Precision (AMP).")

    early_stopping_cfg = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_cfg.get('enabled', True)
    early_stopping_patience = early_stopping_cfg.get('patience', 10)
    early_stopping_delta = early_stopping_cfg.get('min_delta', 0.001)
    early_stopping_counter = 0
    # Ensure early stopping monitors the same metric as scheduler/best model saving
    early_stopping_metric = early_stopping_cfg.get('monitor_metric', monitor_metric)
    early_stopping_mode = early_stopping_cfg.get('mode', scheduler_mode)
    if early_stopping_metric != monitor_metric or early_stopping_mode != scheduler_mode:
         logger.warning(f"Early stopping metric/mode ({early_stopping_metric}/{early_stopping_mode}) differs from scheduler/best model metric/mode ({monitor_metric}/{scheduler_mode}). Using scheduler/best model settings for consistency.")
         early_stopping_metric = monitor_metric
         early_stopping_mode = scheduler_mode

    save_best_metric = config['training'].get('save_best_metric', monitor_metric) # Metric to trigger saving 'best_model.pt'
    save_best_mode = config['training'].get('save_best_mode', scheduler_mode) # 'min' or 'max'

    # --- SIMPLIFIED TRAINING LOOP ---
    start_train_loop = time.time()
    log_section_header(logger, f"STARTING TRAINING (Epochs {start_epoch+1} to {num_epochs})")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Train one epoch
        avg_epoch_train_loss, avg_epoch_train_corr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler
        )
        # Basic check for NaN/Inf return values (should ideally not happen)
        if np.isnan(avg_epoch_train_loss) or np.isinf(avg_epoch_train_loss):
             avg_epoch_train_loss = history['train_loss'][-1] if history['train_loss'] else 1e9 # Use last valid or large value
             logger.error(f"Epoch {epoch+1}: Invalid training loss returned. Using fallback value: {avg_epoch_train_loss:.4f}")
        if np.isnan(avg_epoch_train_corr) or np.isinf(avg_epoch_train_corr):
             avg_epoch_train_corr = history['train_pearson'][-1] if history['train_pearson'] else 0.0 # Use last valid or zero
             logger.error(f"Epoch {epoch+1}: Invalid training correlation returned. Using fallback value: {avg_epoch_train_corr:.4f}")

        history['train_loss'].append(avg_epoch_train_loss)
        history['train_pearson'].append(avg_epoch_train_corr)

        # Validate one epoch
        avg_epoch_val_loss, avg_epoch_val_corr = validate(
            model, val_loader, criterion, device, config
        )
        # Basic check for NaN/Inf return values
        if np.isnan(avg_epoch_val_loss) or np.isinf(avg_epoch_val_loss):
             avg_epoch_val_loss = history['val_loss'][-1] if history['val_loss'] else 1e9 # Use last valid or large value
             logger.error(f"Epoch {epoch+1}: Invalid validation loss returned. Using fallback value: {avg_epoch_val_loss:.4f}")
        if np.isnan(avg_epoch_val_corr) or np.isinf(avg_epoch_val_corr):
             avg_epoch_val_corr = history['val_pearson'][-1] if history['val_pearson'] else 0.0 # Use last valid or zero
             logger.error(f"Epoch {epoch+1}: Invalid validation correlation returned. Using fallback value: {avg_epoch_val_corr:.4f}")

        history['val_loss'].append(avg_epoch_val_loss)
        history['val_pearson'].append(avg_epoch_val_corr)
        history['lr'].append(optimizer.param_groups[0]['lr']) # Log current learning rate

        epoch_duration = time.time() - epoch_start_time

        # Logging epoch results
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} Completed --- | Time: {epoch_duration:.1f}s | LR: {history['lr'][-1]:.2e}")
        logger.info(f"  Train -> Loss: {avg_epoch_train_loss:.6f} | Pearson: {avg_epoch_train_corr:.6f}")
        logger.info(f"  Valid -> Loss: {avg_epoch_val_loss:.6f} | Pearson: {avg_epoch_val_corr:.6f}")

        # Scheduler Step
        current_metric_val = avg_epoch_val_corr if monitor_metric == 'val_pearson' else avg_epoch_val_loss
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_metric_val)
            else: # For CosineAnnealingLR, StepLR etc.
                scheduler.step()

        # Checkpointing and Best Model Saving
        # Determine if current epoch is the best based on configured metric and mode
        is_best = False
        if save_best_mode == 'max' and current_metric_val > best_metric_value + early_stopping_delta:
            is_best = True
            improvement = current_metric_val - best_metric_value
        elif save_best_mode == 'min' and current_metric_val < best_metric_value - early_stopping_delta:
            is_best = True
            improvement = best_metric_value - current_metric_val # Positive improvement
        # Handle cases where metric is NaN (should be rare after checks above)
        elif np.isnan(current_metric_val):
             logger.warning(f"Epoch {epoch+1}: Monitored metric '{save_best_metric}' is NaN. Cannot determine improvement.")

        if is_best:
            logger.info(f"  >>> New best {save_best_metric}: {current_metric_val:.6f} (Improvement: {improvement:.6f}). Saving best model...")
            best_metric_value = current_metric_val
            best_epoch = epoch + 1 # Use 1-based epoch number
            early_stopping_counter = 0 # Reset counter on improvement

            best_model_path = os.path.join(models_dir, "best_model.pt")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'best_{save_best_metric}': best_metric_value, # Store the specific best metric value
                'best_epoch': best_epoch,
                'config': config, # Save config used for this training run
                'input_shape': input_shape, # Save input shape used by the model
                'temp_scaling_params': temp_scaling_params, # Save scaling params
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scheduler_type': type(scheduler).__name__ if scheduler else None
            }
            try:
                torch.save(checkpoint_data, best_model_path)
            except Exception as save_e:
                logger.error(f"Failed to save best model checkpoint: {save_e}")

        else:
            early_stopping_counter += 1
            logger.info(f"  {save_best_metric} did not improve. Best: {best_metric_value:.6f} @ Ep {best_epoch}. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # Periodic Checkpointing (save full state including history)
        chkpt_interval = config['training'].get('checkpoint_interval', 0)
        if chkpt_interval > 0 and (epoch + 1) % chkpt_interval == 0:
             chkpt_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1}.pt")
             logger.info(f"Saving periodic checkpoint to {chkpt_path}")
             periodic_checkpoint_data = {
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'history': history, # Include history in periodic checkpoints
                 f'best_{save_best_metric}': best_metric_value, # Keep track of best so far
                 'best_epoch': best_epoch,
                 'config': config,
                 'input_shape': input_shape,
                 'temp_scaling_params': temp_scaling_params,
                 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                 'scheduler_type': type(scheduler).__name__ if scheduler else None
             }
             try:
                  torch.save(periodic_checkpoint_data, chkpt_path)
             except Exception as save_e:
                  logger.error(f"Failed to save periodic checkpoint {chkpt_path}: {save_e}")

        # Always save latest model state (minimal info, overwrites)
        latest_model_path = os.path.join(models_dir, "latest_model.pt")
        latest_checkpoint_data = {
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'config': config, # Include config for easier reloading/info
             'input_shape': input_shape,
             'temp_scaling_params': temp_scaling_params,
             'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
             'scheduler_type': type(scheduler).__name__ if scheduler else None,
             'history': history # Include history here too for convenience
        }
        try:
             torch.save(latest_checkpoint_data, latest_model_path)
        except Exception as save_e:
             logger.error(f"Failed to save latest model state: {save_e}")


        # Early Stopping Check
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1} after {early_stopping_patience} epochs without improvement based on '{early_stopping_metric}'.")
            break # Exit training loop

        # Clear memory at end of epoch
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))

    # --- Finalization ---
    training_duration = time.time() - start_train_loop
    log_section_header(logger, "TRAINING FINISHED")
    logger.info(f"Total Training Time: {training_duration:.2f}s ({training_duration/60:.1f} mins)")

    final_model_path = None
    if best_epoch != -1:
        logger.info(f"Best {save_best_metric}: {best_metric_value:.6f} achieved at epoch {best_epoch}")
        final_model_path = os.path.join(models_dir, "best_model.pt")
        # Verify best model file exists
        if not os.path.exists(final_model_path):
             logger.error("Best model was recorded but checkpoint file not found! Using latest model.")
             final_model_path = os.path.join(models_dir, "latest_model.pt")
    else:
         logger.warning("No improvement observed during training based on monitored metric.")
         final_model_path = os.path.join(models_dir, "latest_model.pt")
         logger.info(f"Using latest model from epoch {epoch+1}: {final_model_path}")

    # Verify final model path exists before returning
    if not final_model_path or not os.path.exists(final_model_path):
         logger.error("Could not determine or find the final model checkpoint file.")
         final_model_path = None


    # Save final history
    history_path = os.path.join(run_output_dir, "training_history.json")
    try:
        save_json(history, history_path)
        logger.info(f"Full training history saved to {history_path}")
    except Exception as e:
        logger.error(f"Failed to save training history: {e}")


    # Optional: Generate final plots automatically
    try:
        from voxelflex.cli.commands.visualize import create_loss_curve, create_correlation_curve
        viz_dir = config["output"]["visualizations_dir"]
        logger.info("Generating final training plots...")
        if history.get('train_loss') and history.get('val_loss'):
            create_loss_curve(history, viz_dir, save_format=config['visualization']['save_format'], dpi=config['visualization']['dpi'])
        if history.get('train_pearson') and history.get('val_pearson'):
            create_correlation_curve(history, viz_dir, save_format=config['visualization']['save_format'], dpi=config['visualization']['dpi'])
    except ImportError:
        logger.warning("Could not import visualization functions for final plots. Run 'visualize' command separately.")
    except Exception as plot_e:
        logger.error(f"Failed to generate final training plots: {plot_e}")

    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    return final_model_path, history
EOF

# --- src/voxelflex/cli/commands/predict.py ---
cat << 'EOF' > src/voxelflex/cli/commands/predict.py
# src/voxelflex/cli/commands/predict.py
"""
Prediction command for VoxelFlex (Temperature-Aware).

Loads a trained model and predicts RMSF for specified domains at a given temperature.
Uses on-demand loading of raw voxel data.
"""
import os
import time
import json
import logging
import gc
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.predict")

# Project imports
from voxelflex.data.data_loader import (
    PredictionDataset,
    load_process_voxels_from_hdf5, # Use the robust HDF5 loader
    load_aggregated_rmsf_data,
    create_domain_mapping
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import EnhancedProgressBar, log_memory_usage, log_stage, log_section_header
from voxelflex.utils.file_utils import ensure_dir, resolve_path, save_json, load_json # Import load_json
from voxelflex.utils.system_utils import get_device, clear_memory, check_memory_usage
from voxelflex.utils.temp_scaling import get_temperature_scaler

def predict_rmsf(
    config: Dict[str, Any],
    model_path: str,
    target_temperature: float,
    domain_ids_to_predict: Optional[List[str]] = None,
    output_csv_filename: Optional[str] = None # Allow specifying output filename
 ) -> Optional[str]:
    """
    Runs prediction for specified domains at a target temperature.

    Args:
        config: Configuration dictionary.
        model_path: Path to the trained model checkpoint (.pt).
        target_temperature: The temperature (in K) for which to predict RMSF.
        domain_ids_to_predict: Optional list of specific HDF5 domain keys to predict.
                                If None, uses domains from the test split file defined in config.
        output_csv_filename: Optional base filename for the output CSV.

    Returns:
        Path to the saved predictions CSV file, or None on failure.
    """
    run_output_dir = config["output"]["run_dir"]
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)

    # Determine output filename
    if output_csv_filename is None:
        # Generate default name if not provided
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_str = f"{target_temperature:.0f}K"
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_csv_filename = f"predictions_{model_name}_{temp_str}_{timestamp}.csv"

    # Construct full path
    predictions_path = os.path.join(metrics_dir, output_csv_filename)

    log_section_header(logger, f"STARTING PREDICTION at {target_temperature:.1f}K")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output CSV: {predictions_path}")
    log_memory_usage(logger)
    device = get_device(config["system_utilization"].get("adjust_for_gpu", True))
    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))

    model = None # Ensure model is defined for finally block
    try:
        # --- Load Model & Temperature Scaler ---
        with log_stage("SETUP", "Loading model and temperature scaler"):
            try:
                logger.info(f"Loading checkpoint from: {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu') # Load to CPU first

                # Extract necessary info from checkpoint
                model_cfg_from_ckpt = checkpoint.get('config', {}).get('model', {})
                input_shape = checkpoint.get('input_shape')
                scaling_params_from_ckpt = checkpoint.get('temp_scaling_params')
                train_cfg_from_ckpt = checkpoint.get('config', {}).get('training', {}) # For mixed precision setting

                if not model_cfg_from_ckpt: raise ValueError("Model config section missing from checkpoint.")
                if not input_shape: raise ValueError("Model input_shape missing from checkpoint.")
                if not scaling_params_from_ckpt:
                     logger.warning("Temperature scaling parameters missing from checkpoint. Attempting to load from config path...")
                     # Try loading from the path specified in the *current* config, assuming it matches the run
                     scaler_path_from_config = config["data"]["temp_scaling_params_file"]
                     if os.path.exists(scaler_path_from_config):
                         scaling_params_from_ckpt = load_json(scaler_path_from_config)
                         logger.info(f"Loaded scaling params from config path: {scaler_path_from_config}")
                     else:
                          raise ValueError("Temperature scaling parameters missing from checkpoint and config path.")

                # Create model instance
                model = get_model(model_cfg_from_ckpt, input_shape=input_shape)
                # Load weights
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys: logger.warning(f"Prediction Load Warning: Missing keys in model state_dict: {missing_keys}")
                if unexpected_keys: logger.warning(f"Prediction Load Warning: Unexpected keys in model state_dict: {unexpected_keys}")

                model.to(device)
                model.eval() # Set to evaluation mode
                logger.info(f"Loaded model '{model_cfg_from_ckpt.get('architecture')}' to {device}.")

                # Get temperature scaler function
                temp_scaler_func = get_temperature_scaler(params=scaling_params_from_ckpt)
                scaled_target_temp_value = temp_scaler_func(target_temperature)
                logger.info(f"Target temp {target_temperature}K scaled to: {scaled_target_temp_value:.4f}")
                # Create tensor for prediction - can be expanded per batch later
                scaled_target_temp_tensor_base = torch.tensor([[scaled_target_temp_value]], dtype=torch.float32).to(device)

                del checkpoint; gc.collect() # Free checkpoint memory

            except Exception as e:
                logger.exception(f"Error loading model or temperature scaler: {e}")
                return None


        # --- Prepare Domain List ---
        with log_stage("SETUP", "Preparing domain list for prediction"):
            if domain_ids_to_predict is None:
                logger.info("No specific domains provided. Using domains from test split file.")
                test_split_file = config['input'].get('test_split_file')
                if not test_split_file or not os.path.exists(test_split_file):
                    logger.error(f"Test split file not found or specified ({test_split_file}). Cannot determine domains to predict.")
                    return None
                domain_ids_to_predict = load_list_from_file(test_split_file)
                if not domain_ids_to_predict:
                     logger.error(f"Test split file ({test_split_file}) is empty. No domains to predict.")
                     return None
                logger.info(f"Loaded {len(domain_ids_to_predict)} domains from test split file.")

            # Ensure the list contains unique domains
            domain_ids_to_predict = sorted(list(set(domain_ids_to_predict)))
            logger.info(f"Predicting for {len(domain_ids_to_predict)} unique domains.")


        # --- Identify Samples to Predict (Domain + Residue from HDF5) ---
        with log_stage("SETUP", "Identifying target residues from HDF5"):
            samples_to_predict: List[Tuple[str, str]] = []
            try:
                 with h5py.File(config["input"]["voxel_file"], 'r') as f_h5:
                      for domain_id in domain_ids_to_predict:
                           if domain_id not in f_h5:
                                logger.warning(f"Requested prediction domain '{domain_id}' not found in HDF5. Skipping.")
                                continue
                           # Find residue group (same logic as preprocess/data_loader)
                           domain_group = f_h5[domain_id]
                           residue_group = None
                           potential_chain_keys = sorted([k for k in domain_group.keys() if isinstance(domain_group[k], h5py.Group)])
                           for chain_key in potential_chain_keys:
                                try:
                                     potential_residue_group = domain_group[chain_key]
                                     if any(key.isdigit() for key in potential_residue_group.keys()):
                                          residue_group = potential_residue_group; break
                                except Exception: continue

                           if residue_group:
                                for resid_str in residue_group.keys():
                                     if resid_str.isdigit():
                                          samples_to_predict.append((domain_id, resid_str))
                           else:
                                logger.warning(f"No residue group found for domain '{domain_id}'. Skipping.")
            except Exception as e:
                 logger.exception(f"Error reading HDF5 to identify target residues: {e}")
                 return None

            if not samples_to_predict:
                 logger.error("No valid residues found in HDF5 for the specified domains. Cannot run prediction.")
                 return None
            logger.info(f"Identified {len(samples_to_predict)} target residues from HDF5.")


        # --- Create Dataset & DataLoader ---
        # PredictionDataset now handles pre-loading/filtering based on voxel availability
        with log_stage("SETUP", "Creating Prediction Dataset and DataLoader"):
            try:
                pred_dataset = PredictionDataset(
                    samples_to_load=samples_to_predict,
                    voxel_hdf5_path=config["input"]["voxel_file"],
                    expected_channels=model_cfg_from_ckpt['input_channels'], # Use channels from loaded model config
                    target_shape_chw=input_shape # Validate against loaded model shape
                )

                if len(pred_dataset) == 0:
                    logger.error("Prediction dataset is empty after attempting to load voxels. Cannot proceed.")
                    return None

                # Use configured prediction batch size
                pred_batch_size = config.get('prediction', {}).get('batch_size', 128)
                pred_loader = DataLoader(
                    pred_dataset,
                    batch_size=pred_batch_size,
                    shuffle=False, # Important for consistent output order if needed
                    num_workers=0, # Keep prediction simple, load in main thread
                    pin_memory=False # Not typically needed for CPU->GPU transfer here
                )
                logger.info(f"Prediction DataLoader created: {len(pred_loader)} batches, BatchSize={pred_batch_size}.")

            except Exception as e:
                logger.exception(f"Error creating prediction dataset/loader: {e}")
                return None

        # --- Load Optional Resname Information (for output CSV) ---
        resname_lookup: Dict[Tuple[str, int], str] = {}
        try:
             with log_stage("SETUP", "Loading residue name information (optional)"):
                  rmsf_df_for_names = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
                  # We only need domain_id, resid, resname
                  resname_df = rmsf_df_for_names[['domain_id', 'resid', 'resname']].copy()
                  resname_df['resid'] = pd.to_numeric(resname_df['resid'], errors='coerce').astype('Int64')
                  resname_df.dropna(inplace=True)
                  resname_df.drop_duplicates(subset=['domain_id', 'resid'], keep='first', inplace=True)
                  resname_map = resname_df.set_index(['domain_id', 'resid'])['resname'].to_dict()

                  # Create lookup using HDF5 key -> RMSF mapping if needed (or assume direct match?)
                  # Let's assume prediction uses HDF5 keys, so we need mapping
                  hdf5_keys_in_dataset = list(set(s[0] for s in pred_dataset.samples))
                  domain_mapping_pred = create_domain_mapping(hdf5_keys_in_dataset, resname_df['domain_id'].unique().tolist())

                  # Populate the lookup for samples that are actually in the dataset
                  for domain_id, resid_str in pred_dataset.samples:
                       try:
                            resid_int = int(resid_str)
                            mapped_rmsf_id = domain_mapping_pred.get(domain_id, domain_id) # Fallback to original ID
                            resname = resname_map.get((mapped_rmsf_id, resid_int))
                            if resname is None: # Try base name if mapped failed
                                 base_rmsf_id = mapped_rmsf_id.split('_')[0]
                                 if base_rmsf_id != mapped_rmsf_id:
                                      resname = resname_map.get((base_rmsf_id, resid_int))

                            resname_lookup[(domain_id, resid_int)] = resname if resname else "UNK"
                       except ValueError: continue # Skip if resid_str is not int

                  logger.info(f"Residue name lookup created with {len(resname_lookup)} entries.")
                  del rmsf_df_for_names, resname_df, resname_map, domain_mapping_pred; gc.collect()
        except Exception as e:
             logger.warning(f"Could not load residue names from RMSF file: {e}. Resnames will be 'UNK'.")


        # --- Run Prediction Loop ---
        with log_stage("PREDICTION", "Running inference loop"):
            logger.info("Starting prediction loop...")
            results_list = []
            progress = EnhancedProgressBar(len(pred_loader), prefix=f"Predict {target_temperature:.0f}K")
            # Use mixed precision if enabled during training
            autocast_enabled = train_cfg_from_ckpt.get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda'

            with torch.no_grad(): # Ensure no gradients are calculated
                for i, batch_data in enumerate(pred_loader):
                    # batch_data is (list_of_domain_ids, list_of_resid_strs, voxel_tensor)
                    if not batch_data or len(batch_data) != 3:
                        logger.warning(f"Skipping invalid batch data at index {i}.")
                        continue

                    batch_domains, batch_resids_str, batch_voxels = batch_data
                    # Check if batch is effectively empty
                    if not batch_domains or batch_voxels.numel() == 0:
                        logger.warning(f"Skipping empty batch {i+1}.")
                        continue

                    try:
                        batch_voxels = batch_voxels.to(device, non_blocking=True)
                        current_batch_size = batch_voxels.size(0)
                        # Expand the base scaled temperature tensor for the current batch size
                        batch_scaled_temps = scaled_target_temp_tensor_base.expand(current_batch_size, 1)

                        # Run model inference
                        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                             batch_outputs = model(voxel_input=batch_voxels, scaled_temp=batch_scaled_temps)

                        # Process outputs
                        preds_np = batch_outputs.detach().cpu().numpy().flatten()

                        # Collect results for this batch
                        for j in range(current_batch_size):
                             domain_id = batch_domains[j]
                             resid_str = batch_resids_str[j]
                             try:
                                  resid_int = int(resid_str)
                                  resname = resname_lookup.get((domain_id, resid_int), "UNK") # Get resname
                                  results_list.append({
                                       'domain_id': domain_id, # Use HDF5 key as identifier
                                       'resid': resid_int,
                                       'resname': resname,
                                       'predicted_rmsf': preds_np[j],
                                       'prediction_temperature': target_temperature
                                  })
                             except ValueError:
                                  logger.warning(f"Invalid residue ID format encountered in prediction output: {domain_id}:{resid_str}")
                             except IndexError:
                                  logger.warning(f"Index error accessing prediction output for batch {i+1}, item {j}")

                    except Exception as e:
                         logger.exception(f"Error predicting batch {i+1}: {e}")
                         # Continue to next batch if one fails

                    progress.update(i + 1)
                    del batch_voxels, batch_scaled_temps, batch_outputs, preds_np # Clean up batch tensors
                    if i % 50 == 0: clear_memory(force_gc=False, clear_cuda=(device.type=='cuda'))

            progress.finish()
            logger.info(f"Prediction loop finished. Collected {len(results_list)} results.")


        # --- Save Results ---
        with log_stage("OUTPUT", "Saving prediction results"):
            if results_list:
                try:
                    results_df = pd.DataFrame(results_list)
                    # Ensure correct order of columns
                    output_cols = ['domain_id', 'resid', 'resname', 'prediction_temperature', 'predicted_rmsf']
                    results_df = results_df[output_cols]
                    results_df.to_csv(predictions_path, index=False, float_format='%.6f')
                    logger.info(f"Predictions saved successfully to: {predictions_path}")
                except Exception as e:
                    logger.exception(f"Failed to save predictions CSV to {predictions_path}: {e}")
                    return None # Indicate failure
            else:
                logger.warning("No results generated during prediction loop. Output file will not be created.")
                return None # Indicate no results

        logger.info("Prediction command finished successfully.")
        return predictions_path # Return path on success

    finally:
        # Cleanup
        del model # Ensure model is deleted
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
        log_memory_usage(logger)

EOF

# --- src/voxelflex/cli/commands/evaluate.py ---
cat << 'EOF' > src/voxelflex/cli/commands/evaluate.py
# src/voxelflex/cli/commands/evaluate.py
"""
Evaluation command for VoxelFlex (Temperature-Aware).

Calculates performance metrics by comparing predictions against ground truth
from the aggregated RMSF data. Includes stratification and permutation importance.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.evaluate")

# Project imports
from voxelflex.utils.logging_utils import (
    get_logger, log_stage, log_memory_usage, log_section_header, EnhancedProgressBar
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, resolve_path
from voxelflex.utils.system_utils import clear_memory, check_memory_usage, get_device
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.temp_scaling import get_temperature_scaler
from voxelflex.data.data_loader import (
    PredictionDataset,
    load_process_voxels_from_hdf5, # Use robust HDF5 loader for permutation importance
    load_aggregated_rmsf_data,
    create_domain_mapping # Needed for resname lookup
)

# --- Metric Calculation Helpers ---

def safe_metric(metric_func: Callable, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Safely compute metric, handling NaNs, Infs, and insufficient data."""
    try:
        # Ensure inputs are numpy arrays and float64 for precision
        x_np = np.asarray(y_true).astype(np.float64)
        y_np = np.asarray(y_pred).astype(np.float64)

        # Filter out non-finite values
        valid_mask = np.isfinite(x_np) & np.isfinite(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]

        # Check for sufficient data points
        if len(x_clean) < 2:
            # logger.debug(f"Metric {metric_func.__name__}: Insufficient data points ({len(x_clean)}). Returning NaN.")
            return np.nan

        # Check for zero variance in correlation metrics
        is_correlation = 'pearsonr' in metric_func.__name__ or 'spearmanr' in metric_func.__name__
        if is_correlation:
            if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
                # logger.debug(f"Metric {metric_func.__name__}: Zero variance detected. Returning NaN.")
                return np.nan

        # Calculate metric
        if 'pearsonr' in metric_func.__name__:
             result, _ = metric_func(x_clean, y_clean)
        elif 'spearmanr' in metric_func.__name__:
             result, _ = metric_func(x_clean, y_clean)
        else:
             result = metric_func(x_clean, y_clean, **kwargs)

        # Return result only if finite
        return float(result) if np.isfinite(result) else np.nan

    except Exception as e:
        logger.debug(f"Error calculating metric {metric_func.__name__}: {e}")
        return np.nan # Return NaN on any exception


def calculate_metrics(df: pd.DataFrame, label: str = "Overall") -> Dict[str, float]:
    """Calculate standard regression metrics from a DataFrame."""
    metrics = {}
    count = len(df)
    metrics['count'] = int(count)

    if count < 2:
        logger.warning(f"Metrics ({label}): Insufficient samples ({count}) for calculation.")
        # Return dict with count and NaNs for other metrics
        nan_metrics = ['pearson', 'spearman', 'r2', 'mse', 'rmse', 'mae',
                       'mean_absolute_relative_error', 'median_absolute_relative_error',
                       'cv_rmse_percent']
        for key in nan_metrics: metrics[key] = np.nan
        return metrics

    if 'target_rmsf' not in df or 'predicted_rmsf' not in df:
        logger.warning(f"Metrics ({label}): Missing target_rmsf or predicted_rmsf columns.")
        return metrics # Should not happen if merge worked, but safety check

    y_true = df['target_rmsf'].values
    y_pred = df['predicted_rmsf'].values

    # Calculate standard metrics safely
    metrics['pearson'] = safe_metric(pearsonr, y_true, y_pred)
    metrics['spearman'] = safe_metric(spearmanr, y_true, y_pred)
    metrics['r2'] = safe_metric(r2_score, y_true, y_pred)
    metrics['mse'] = safe_metric(mean_squared_error, y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse']) if pd.notna(metrics['mse']) else np.nan
    metrics['mae'] = safe_metric(mean_absolute_error, y_true, y_pred)

    # Calculate relative errors safely
    try:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        if len(y_true_clean) > 0:
             abs_error = np.abs(y_true_clean - y_pred_clean)
             # Avoid division by zero or very small numbers for relative error
             safe_y_true = np.maximum(np.abs(y_true_clean), 1e-6)
             relative_error = abs_error / safe_y_true
             # Filter NaNs/Infs that might arise from division
             finite_rel_err = relative_error[np.isfinite(relative_error)]
             if len(finite_rel_err) > 0:
                  metrics['mean_absolute_relative_error'] = float(np.mean(finite_rel_err)) * 100
                  metrics['median_absolute_relative_error'] = float(np.median(finite_rel_err)) * 100
             else:
                  metrics['mean_absolute_relative_error'] = np.nan
                  metrics['median_absolute_relative_error'] = np.nan
             # Calculate CV(RMSE)
             mean_true = np.mean(y_true_clean)
             if pd.notna(metrics['rmse']) and abs(mean_true) > 1e-6:
                 metrics['cv_rmse_percent'] = (metrics['rmse'] / mean_true) * 100
             else:
                 metrics['cv_rmse_percent'] = np.nan
        else:
            metrics['mean_absolute_relative_error'] = np.nan
            metrics['median_absolute_relative_error'] = np.nan
            metrics['cv_rmse_percent'] = np.nan

    except Exception as e:
         logger.warning(f"Error calculating relative/CV metrics ({label}): {e}")
         metrics['mean_absolute_relative_error'] = np.nan
         metrics['median_absolute_relative_error'] = np.nan
         metrics['cv_rmse_percent'] = np.nan

    return metrics

# --- Permutation Importance ---
def perform_permutation_importance(
    model: nn.Module,
    perm_dataset: PredictionDataset, # Dataset containing samples for permutation
    temp_scaler_func: Callable[[float], float],
    target_temperature: float, # The original temperature (used to find baseline performance)
    baseline_metrics: Dict[str, float], # Baseline performance on the same dataset
    device: torch.device,
    config: Dict[str, Any],
    n_repeats: int = 5
) -> Dict[str, float]:
    """
    Performs permutation importance analysis for the temperature feature.

    Args:
        model: The trained model.
        perm_dataset: PredictionDataset containing the samples to evaluate on.
        temp_scaler_func: Function to scale raw temperatures.
        target_temperature: The original temperature the baseline metrics correspond to.
        baseline_metrics: Dictionary of baseline metrics (e.g., pearson, rmse).
        device: The device to run inference on.
        config: Configuration dictionary.
        n_repeats: Number of times to repeat the permutation.

    Returns:
        Dictionary containing permutation importance scores (e.g., 'pearson_drop').
    """
    logger.info(f"Performing Permutation Importance for Temperature (n_repeats={n_repeats})...")
    model.eval() # Ensure model is in eval mode
    batch_size = config.get('prediction', {}).get('batch_size', 128)
    rng = np.random.default_rng(config['training']['seed'])

    # Create a shuffling pool of scaled temperatures from the *training* data distribution
    # This avoids biasing the permutation test with only the target temperature
    scaled_shuffling_pool = []
    try:
        # Try loading the full RMSF file to get all unique temperatures
        rmsf_df_full = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
        all_temps_unique = rmsf_df_full['temperature_feature'].dropna().unique()
        scaled_shuffling_pool = [temp_scaler_func(t) for t in all_temps_unique]
        logger.info(f"Created temperature shuffling pool with {len(scaled_shuffling_pool)} unique scaled values.")
        del rmsf_df_full; gc.collect()
    except Exception as e:
        logger.error(f"Could not load temps for shuffling pool from {config['input']['aggregated_rmsf_file']}: {e}. Using default [0, 1] range.")
        # Fallback to a simple range if file loading fails
        scaled_shuffling_pool = np.linspace(0.0, 1.0, 20).tolist() # Use 20 points between 0 and 1

    if not scaled_shuffling_pool:
        logger.warning("Temp shuffling pool is empty. Using default [0, 1] range.")
        scaled_shuffling_pool = np.linspace(0.0, 1.0, 20).tolist()


    # Get baseline predictions (needed if not already available) - reuse perm_dataset
    # For simplicity, assume baseline_metrics were calculated on the exact same samples as in perm_dataset
    baseline_pearson = baseline_metrics.get('pearson', np.nan)
    baseline_rmse = baseline_metrics.get('rmse', np.nan)

    if pd.isna(baseline_pearson) or pd.isna(baseline_rmse):
         logger.warning("Baseline metrics (Pearson, RMSE) are missing or NaN. Cannot calculate importance drop/increase.")
         # We can still report the average permuted score

    # Store metrics from each permutation repeat
    permuted_metrics_repeats: Dict[str, List[float]] = {'pearson': [], 'rmse': []}
    # Store predictions from the first repeat for potential later use if needed
    # first_repeat_preds: Optional[np.ndarray] = None

    # Get the ground truth targets aligned with the dataset order
    # This requires access to the merged eval_df created earlier. Pass it? Or recalculate?
    # Let's assume eval_df is available or passed indirectly via baseline_metrics source.
    # For robustness, we should ideally re-fetch targets based on perm_dataset.samples
    aligned_targets_np = np.full(len(perm_dataset), np.nan, dtype=np.float64) # Initialize with NaNs
    try:
        gt_df = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
        gt_df['resid'] = pd.to_numeric(gt_df['resid'], errors='coerce').astype('Int64')
        gt_df['temperature_feature'] = pd.to_numeric(gt_df['temperature_feature'], errors='coerce')
        gt_df_filtered = gt_df[np.isclose(gt_df['temperature_feature'], target_temperature)].copy()
        gt_lookup = gt_df_filtered.set_index(['domain_id', 'resid'])['target_rmsf']
        # Create domain mapping just for this lookup
        hdf5_keys_in_dataset = list(set(s[0] for s in perm_dataset.samples))
        domain_mapping_perm = create_domain_mapping(hdf5_keys_in_dataset, gt_df_filtered['domain_id'].unique().tolist())

        for i, (domain_id, resid_str) in enumerate(perm_dataset.samples):
             try:
                  resid_int = int(resid_str)
                  mapped_id = domain_mapping_perm.get(domain_id, domain_id)
                  target = gt_lookup.get((mapped_id, resid_int))
                  if target is None: # Try base name
                      base_id = mapped_id.split('_')[0]
                      if base_id != mapped_id: target = gt_lookup.get((base_id, resid_int))
                  if target is not None and not pd.isna(target):
                       aligned_targets_np[i] = float(target)
             except (ValueError, TypeError, KeyError): continue # Ignore errors for specific samples
        del gt_df, gt_df_filtered, gt_lookup, domain_mapping_perm; gc.collect()
        if np.isnan(aligned_targets_np).all():
             raise ValueError("Could not align any ground truth targets for permutation importance samples.")
    except Exception as e:
         logger.error(f"Failed to get aligned ground truth targets for permutation importance: {e}")
         return {"error": "Failed to get ground truth targets"}


    # Use mixed precision if enabled during training (get from config)
    autocast_enabled = config.get('training',{}).get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda'

    # --- Permutation Loop ---
    for n in range(n_repeats):
        logger.info(f"  Permutation importance repeat {n+1}/{n_repeats}...")
        permuted_preds_batches: List[np.ndarray] = [] # Store predictions for this repeat
        # Create a new DataLoader for each repeat? Or shuffle temps within batch?
        # Shuffling temps within batch is more efficient if dataset is large
        permuted_loader = DataLoader(perm_dataset, batch_size=batch_size, sampler=SequentialSampler(perm_dataset), num_workers=0)
        progress = EnhancedProgressBar(len(permuted_loader), prefix=f"  Permutation {n+1}")

        with torch.no_grad():
            for i, batch_data in enumerate(permuted_loader):
                # _, _, batch_voxels = batch_data
                if not batch_data or len(batch_data) != 3: continue
                _, _, batch_voxels = batch_data

                if batch_voxels.numel() == 0: continue
                batch_voxels = batch_voxels.to(device, non_blocking=True)
                current_batch_size = batch_voxels.size(0)

                # --- Permute Temperature Feature ---
                # Create permuted temperature tensor for this batch
                shuffled_scaled_values = rng.choice(scaled_shuffling_pool, size=current_batch_size, replace=True)
                batch_shuffled_temps = torch.tensor(shuffled_scaled_values, device=device, dtype=torch.float32).unsqueeze(1)

                try:
                    # Run inference with permuted temperature
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        batch_outputs = model(voxel_input=batch_voxels, scaled_temp=batch_shuffled_temps)
                    permuted_preds_batches.append(batch_outputs.detach().cpu().numpy().flatten())
                except Exception as e:
                    logger.warning(f"Error predicting with permuted temps (batch {i}): {e}")
                    # Append NaNs if prediction fails for a batch
                    permuted_preds_batches.append(np.full(current_batch_size, np.nan))

                progress.update(i + 1)
                del batch_voxels, batch_shuffled_temps, batch_outputs
                if i % 100 == 0: clear_memory(force_gc=False, clear_cuda=True)
        progress.finish()

        # Concatenate predictions for the current repeat
        if permuted_preds_batches:
             perm_preds_np = np.concatenate(permuted_preds_batches)
             if len(perm_preds_np) != len(aligned_targets_np):
                  logger.error(f"Permutation repeat {n+1}: Prediction length mismatch ({len(perm_preds_np)}) vs target length ({len(aligned_targets_np)}). Skipping metrics for this repeat.")
                  continue

            # Calculate metrics for this repeat using the permuted predictions
             perm_pearson = safe_metric(pearsonr, aligned_targets_np, perm_preds_np)
             perm_mse = safe_metric(mean_squared_error, aligned_targets_np, perm_preds_np)
             perm_rmse = np.sqrt(perm_mse) if pd.notna(perm_mse) else np.nan

             if pd.notna(perm_pearson): permuted_metrics_repeats['pearson'].append(perm_pearson)
             if pd.notna(perm_rmse): permuted_metrics_repeats['rmse'].append(perm_rmse)

             # Store first repeat predictions if needed later
             # if n == 0: first_repeat_preds = perm_preds_np
        else:
             logger.warning(f"No predictions generated for permutation repeat {n+1}.")


    # --- Calculate Final Importance Scores ---
    importance_scores: Dict[str, float] = {}
    if permuted_metrics_repeats['pearson'] and pd.notna(baseline_pearson):
        mean_perm_pearson = float(np.mean(permuted_metrics_repeats['pearson']))
        std_perm_pearson = float(np.std(permuted_metrics_repeats['pearson']))
        importance_scores['pearson_permuted_mean'] = mean_perm_pearson
        importance_scores['pearson_permuted_std'] = std_perm_pearson
        importance_scores['pearson_drop'] = baseline_pearson - mean_perm_pearson
    else:
         logger.warning("Could not calculate Pearson drop for permutation importance (missing baseline or permuted values).")

    if permuted_metrics_repeats['rmse'] and pd.notna(baseline_rmse):
        mean_perm_rmse = float(np.mean(permuted_metrics_repeats['rmse']))
        std_perm_rmse = float(np.std(permuted_metrics_repeats['rmse']))
        importance_scores['rmse_permuted_mean'] = mean_perm_rmse
        importance_scores['rmse_permuted_std'] = std_perm_rmse
        importance_scores['rmse_increase'] = mean_perm_rmse - baseline_rmse
    else:
         logger.warning("Could not calculate RMSE increase for permutation importance (missing baseline or permuted values).")

    logger.info("--- Permutation Importance Results ---")
    if importance_scores:
         for key, val in importance_scores.items():
              logger.info(f"  {key}: {val:.4f}")
    else:
         logger.info("  No valid permutation importance scores calculated.")

    return importance_scores


# --- Main Evaluation Function ---
def evaluate_model(
    config: Dict[str, Any],
    model_path: str,
    predictions_path: str,
) -> Optional[str]:
    """
    Evaluates model performance using a predictions file and ground truth data.

    Args:
        config: Configuration dictionary.
        model_path: Path to the trained model checkpoint (.pt).
        predictions_path: Path to the predictions CSV file generated by 'predict'.

    Returns:
        Path to the saved evaluation metrics JSON file, or None on failure.
    """
    run_output_dir = config["output"]["run_dir"]
    metrics_dir = config["output"]["metrics_dir"]; ensure_dir(metrics_dir)

    # Construct metrics filename based on predictions filename
    pred_basename = os.path.splitext(os.path.basename(predictions_path))[0]
    metrics_filename = f"evaluation_metrics_{pred_basename}.json"
    metrics_path = os.path.join(metrics_dir, metrics_filename)

    log_section_header(logger, "MODEL EVALUATION")
    logger.info(f"Predictions file: {predictions_path}")
    logger.info(f"Model file: {model_path}")
    logger.info(f"Output metrics JSON: {metrics_path}")
    log_memory_usage(logger)

    all_metrics: Dict[str, Any] = { # Initialize results dict
        'overall': {},
        'stratified': {},
        'permutation_importance': {},
        'evaluation_temperature': None,
        'model_path': model_path,
        'predictions_path': predictions_path,
        'input_data': {
            'voxel_file': config['input'].get('voxel_file'),
            'aggregated_rmsf_file': config['input'].get('aggregated_rmsf_file'),
            'test_split_file': config['input'].get('test_split_file', 'N/A')
        }
    }

    # Use try-finally to ensure cleanup actions happen
    model = None # Define outside try block for finally
    perm_dataset = None
    try:
        # --- Load Predictions ---
        with log_stage("EVAL_SETUP", "Loading predictions"):
            try:
                preds_df = pd.read_csv(predictions_path, dtype={'domain_id': str, 'resname': str})
                # Validate required columns and types
                preds_df['resid'] = pd.to_numeric(preds_df['resid'], errors='coerce').astype('Int64')
                preds_df['predicted_rmsf'] = pd.to_numeric(preds_df['predicted_rmsf'], errors='coerce')
                preds_df['prediction_temperature'] = pd.to_numeric(preds_df['prediction_temperature'], errors='coerce')
                preds_df.dropna(subset=['domain_id', 'resid', 'predicted_rmsf', 'prediction_temperature'], inplace=True)

                if preds_df.empty: raise ValueError("Predictions file contains no valid rows after validation.")

                # Determine evaluation temperature from predictions file
                unique_temps = preds_df['prediction_temperature'].unique()
                if len(unique_temps) > 1:
                    logger.warning(f"Predictions file contains multiple temperatures ({unique_temps}). Evaluation will compare against ground truth at the first temperature found: {unique_temps[0]:.1f}K")
                elif len(unique_temps) == 0:
                     raise ValueError("Cannot determine evaluation temperature from predictions file.")
                eval_temp = unique_temps[0]
                all_metrics['evaluation_temperature'] = float(eval_temp) # Store eval temp
                logger.info(f"Evaluating predictions made at temperature: {eval_temp:.1f}K")

            except Exception as e:
                logger.exception(f"Failed to load or validate predictions CSV '{predictions_path}': {e}")
                return None # Exit if predictions fail to load


        # --- Load Ground Truth ---
        with log_stage("EVAL_SETUP", "Loading and filtering ground truth data"):
            try:
                gt_df_raw = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
                # Select necessary columns, including optional ones for stratification
                cols_to_keep = ['domain_id', 'resid', 'resname', 'target_rmsf', 'temperature_feature']
                optional_cols = ['relative_accessibility', 'dssp', 'secondary_structure_encoded']
                available_optional = [col for col in optional_cols if col in gt_df_raw.columns]
                cols_to_keep.extend(available_optional)

                gt_df = gt_df_raw[cols_to_keep].copy()
                gt_df['resid'] = pd.to_numeric(gt_df['resid'], errors='coerce').astype('Int64')
                gt_df['temperature_feature'] = pd.to_numeric(gt_df['temperature_feature'], errors='coerce')
                # Drop rows where essential GT info is missing BEFORE filtering by temp
                gt_df.dropna(subset=['domain_id', 'resid', 'target_rmsf', 'temperature_feature'], inplace=True)

                # Filter ground truth to match the evaluation temperature
                gt_df_filtered = gt_df[np.isclose(gt_df['temperature_feature'], eval_temp)].copy()
                logger.info(f"Loaded and filtered ground truth to {len(gt_df_filtered)} rows matching temp ~{eval_temp:.1f}K.")

                if gt_df_filtered.empty:
                    raise ValueError(f"No ground truth data matches evaluation temperature {eval_temp:.1f}K.")
                del gt_df_raw, gt_df; gc.collect() # Free memory
            except Exception as e:
                logger.exception(f"Failed to load or process ground truth data: {e}")
                return None # Exit if GT fails


        # --- Merge Predictions and Ground Truth ---
        with log_stage("EVAL_SETUP", "Merging predictions and ground truth"):
            try:
                logger.info("Merging predictions with filtered ground truth data...")
                # Ensure 'resid' is compatible type for merging (int after dropna)
                preds_df['resid'] = preds_df['resid'].astype(int)
                gt_df_filtered['resid'] = gt_df_filtered['resid'].astype(int)

                eval_df = pd.merge(
                    preds_df,
                    gt_df_filtered.drop(columns=['temperature_feature']), # Don't need temp column from GT anymore
                    on=['domain_id', 'resid'], # Merge keys
                    how='inner', # Only keep overlapping entries
                    suffixes=('_pred', '_gt') # Suffixes for potentially overlapping columns like resname
                )

                # Handle potential duplicate resname columns after merge
                if 'resname_pred' in eval_df.columns and 'resname_gt' in eval_df.columns:
                    # Optionally check if they differ and warn
                    # mismatches = eval_df[eval_df['resname_pred'] != eval_df['resname_gt']]
                    # if not mismatches.empty:
                    #     logger.warning(f"Resname mismatch between predictions and ground truth found for {len(mismatches)} entries.")
                    eval_df['resname'] = eval_df['resname_gt'] # Prioritize ground truth resname
                    eval_df.drop(columns=['resname_pred', 'resname_gt'], inplace=True)
                elif 'resname_gt' in eval_df.columns: # Rename if only GT suffix exists
                     eval_df.rename(columns={'resname_gt': 'resname'}, inplace=True)
                elif 'resname_pred' in eval_df.columns: # Rename if only Pred suffix exists
                     eval_df.rename(columns={'resname_pred': 'resname'}, inplace=True)
                elif 'resname' not in eval_df.columns: # If neither exists
                     eval_df['resname'] = 'UNK' # Add placeholder

                logger.info(f"Merged data contains {len(eval_df)} overlapping entries.")
                if eval_df.empty:
                    logger.error("Merge resulted in empty DataFrame. Check domain/resid matching between predictions and ground truth.")
                    return None

                # Final check for NaNs in prediction/target columns
                eval_df.dropna(subset=['predicted_rmsf', 'target_rmsf'], inplace=True)
                logger.info(f"{len(eval_df)} valid entries remaining after RMSF NaN check.")
                if eval_df.empty:
                    logger.error("No valid overlapping entries after RMSF NaN check.")
                    return None
            except Exception as e:
                logger.exception(f"Error merging prediction and ground truth data: {e}")
                return None # Exit on merge error


        # --- Calculate Overall Metrics ---
        with log_stage("EVALUATION", "Calculating overall metrics"):
            overall_metrics = calculate_metrics(eval_df, label="Overall")
            all_metrics['overall'] = overall_metrics
            logger.info("--- Overall Performance ---")
            if overall_metrics:
                for key, val in overall_metrics.items():
                    logger.info(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
            else:
                logger.warning("Overall metrics could not be calculated.")


        # --- Stratified Metrics ---
        if config['evaluation'].get('calculate_stratified_metrics', True):
            with log_stage("EVALUATION", "Calculating stratified metrics"):
                all_metrics['stratified'] = {} # Ensure key exists

                # By Secondary Structure
                ss_col = next((c for c in ['dssp', 'secondary_structure_encoded'] if c in eval_df.columns), None)
                if ss_col:
                    logger.info(f"Stratifying metrics by Secondary Structure ('{ss_col}')...")
                    strat_ss_metrics = {}
                    for ss_type, group in eval_df.groupby(ss_col):
                        if isinstance(group, pd.DataFrame): # Ensure group is DataFrame
                             metrics = calculate_metrics(group, label=f"SS={ss_type}")
                             if metrics.get('count', 0) > 10: # Only report if enough samples
                                 strat_ss_metrics[str(ss_type)] = metrics
                             else:
                                 logger.debug(f"Skipping SS type '{ss_type}' due to insufficient samples ({metrics.get('count', 0)}).")
                    all_metrics['stratified']['secondary_structure'] = strat_ss_metrics
                else:
                    logger.warning("Secondary structure column ('dssp' or 'secondary_structure_encoded') not found for stratification.")

                # By Solvent Accessibility
                sasa_col = 'relative_accessibility'
                if sasa_col in eval_df.columns:
                    logger.info(f"Stratifying metrics by Relative Accessibility ('{sasa_col}')...")
                    sasa_bins = config['evaluation'].get('sasa_bins', [0.0, 0.1, 0.4, 1.01])
                    bin_labels = [f"{sasa_bins[i]:.1f}-{sasa_bins[i+1]:.1f}" for i in range(len(sasa_bins)-1)]
                    strat_sasa_metrics = {}
                    try:
                        # Ensure column is numeric and handle potential errors during binning
                        eval_df[sasa_col] = pd.to_numeric(eval_df[sasa_col], errors='coerce')
                        eval_df['sasa_bin'] = pd.cut(eval_df[sasa_col], bins=sasa_bins, labels=bin_labels, right=False, include_lowest=True)

                        for bin_label, group in eval_df.groupby('sasa_bin', observed=False): # Use observed=False for categorical
                            if isinstance(group, pd.DataFrame):
                                metrics = calculate_metrics(group, label=f"SASA={bin_label}")
                                if metrics.get('count', 0) > 10:
                                    strat_sasa_metrics[str(bin_label)] = metrics
                                else:
                                    logger.debug(f"Skipping SASA bin '{bin_label}' due to insufficient samples ({metrics.get('count', 0)}).")
                        all_metrics['stratified']['sasa'] = strat_sasa_metrics
                    except Exception as sasa_e:
                        logger.error(f"Error during SASA binning/stratification: {sasa_e}")
                else:
                    logger.warning("SASA column ('relative_accessibility') not found for stratification.")

                # Can add stratification by residue type etc. here if needed


        # --- Permutation Importance Calculation ---
        if config['evaluation'].get('calculate_permutation_importance', True):
            with log_stage("EVALUATION", "Calculating permutation importance"):
                perm_importance_scores = {} # Initialize scores dict
                try:
                    logger.info("Loading resources for permutation importance...")
                    device = get_device(config["system_utilization"]["adjust_for_gpu"])
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model_config = checkpoint.get('config', {}).get('model', {})
                    input_shape = checkpoint.get('input_shape')
                    scaling_params = checkpoint.get('temp_scaling_params')
                    if not model_config or not input_shape: raise ValueError("Model config or shape missing from checkpoint.")
                    if not scaling_params: raise ValueError("Temperature scaling params missing from checkpoint.")

                    model = get_model(model_config, input_shape=input_shape)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    model.to(device); model.eval(); del checkpoint; gc.collect()

                    temp_scaler_func = get_temperature_scaler(params=scaling_params)

                    # Get samples present in the final evaluation dataframe
                    perm_samples_to_load = list(eval_df[['domain_id', 'resid']].apply(lambda x: (x['domain_id'], str(x['resid'])), axis=1).unique())
                    if not perm_samples_to_load:
                         raise ValueError("No samples available in eval_df for permutation importance.")

                    logger.info(f"Creating permutation dataset for {len(perm_samples_to_load)} unique residues...")
                    perm_dataset = PredictionDataset(
                         samples_to_load=perm_samples_to_load,
                         voxel_hdf5_path=config['input']['voxel_file'],
                         expected_channels=model_config['input_channels'],
                         target_shape_chw=input_shape
                    )

                    if len(perm_dataset) == 0:
                         raise ValueError("Permutation dataset is empty after loading voxels.")
                    if len(perm_dataset) != len(eval_df):
                         logger.warning(f"Permutation dataset size ({len(perm_dataset)}) differs from eval_df size ({len(eval_df)}). Ensure samples match.")
                         # This might happen if some voxels failed loading for eval_df samples
                         # Need to use the intersection for fair comparison.
                         # Filter eval_df to only include samples in perm_dataset
                         perm_dataset_keys = set(perm_dataset.samples)
                         eval_df_keys = set(eval_df[['domain_id', 'resid']].apply(lambda x: (x['domain_id'], str(x['resid'])), axis=1))
                         common_keys = perm_dataset_keys.intersection(eval_df_keys)
                         eval_df_filtered_for_perm = eval_df[eval_df[['domain_id', 'resid']].apply(lambda x: (x['domain_id'], str(x['resid'])), axis=1).isin(common_keys)].copy()
                         logger.info(f"Filtered eval_df to {len(eval_df_filtered_for_perm)} samples matching permutation dataset.")
                         if len(eval_df_filtered_for_perm) != len(perm_dataset):
                              # This should not happen if logic is correct
                              raise RuntimeError("Mismatch between filtered eval_df and permutation dataset sizes.")
                         # Recalculate baseline on the filtered set for fair comparison
                         filtered_baseline_metrics = calculate_metrics(eval_df_filtered_for_perm, label="Permutation Baseline")
                    else:
                         # Sizes match, use original overall metrics
                         filtered_baseline_metrics = overall_metrics

                    n_repeats = config['evaluation'].get('permutation_n_repeats', 5)

                    # Perform permutation importance
                    perm_importance_scores = perform_permutation_importance(
                         model, perm_dataset, temp_scaler_func, eval_temp, filtered_baseline_metrics, device, config, n_repeats
                    )

                except Exception as perm_e:
                    logger.exception(f"Failed to perform permutation importance: {perm_e}")
                    perm_importance_scores = {"error": str(perm_e)}
                finally:
                    # Clean up permutation specific resources
                    del model # Ensure model loaded for perm is deleted
                    del perm_dataset
                    clear_memory(force_gc=True, clear_cuda=True)

                all_metrics['permutation_importance'] = perm_importance_scores


        # --- Save All Metrics ---
        with log_stage("OUTPUT", "Saving evaluation metrics"):
            try:
                # Helper function to convert numpy types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, (np.int_, np.integer)): return int(obj)
                    elif isinstance(obj, (np.float_, np.floating)): return float(obj) if np.isfinite(obj) else None # Convert non-finite floats to None
                    elif isinstance(obj, (np.bool_)): return bool(obj)
                    elif isinstance(obj, (np.void)): return None
                    elif isinstance(obj, np.ndarray): return [convert_numpy_types(item) for item in obj] # Recursively convert array elements
                    elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)): return [convert_numpy_types(i) for i in obj]
                    elif pd.isna(obj): return None # Handle pandas NaNs
                    else: return obj # Keep other types as is

                metrics_serializable = convert_numpy_types(all_metrics)
                save_json(metrics_serializable, metrics_path)
                logger.info(f"Evaluation metrics saved successfully to {metrics_path}")
            except Exception as e:
                logger.exception(f"Failed to save metrics JSON to {metrics_path}: {e}")
                return None # Indicate failure to save

        log_memory_usage(logger)
        logger.info("Evaluation finished successfully.")
        return metrics_path # Return path on success

    finally:
        # Final cleanup
        clear_memory(force_gc=True)

EOF

# --- src/voxelflex/cli/commands/visualize.py ---
cat << 'EOF' > src/voxelflex/cli/commands/visualize.py
# src/voxelflex/cli/commands/visualize.py
"""
Visualization command for VoxelFlex (Temperature-Aware).

Generates plots for model performance analysis, optionally saving plot data.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union # Add Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
# Ensure 'viridis' is imported correctly if used directly
from matplotlib.cm import ScalarMappable, viridis #, viridis_r if needed
from scipy.stats import gaussian_kde, pearsonr

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.visualize")

# Project imports
from voxelflex.utils.logging_utils import get_logger, log_stage, log_section_header # Add log_section_header
from voxelflex.utils.file_utils import ensure_dir, load_json, save_json, resolve_path
from voxelflex.data.data_loader import load_aggregated_rmsf_data # Needed for merging GT
# Import the metric calculation from evaluate for consistency
from voxelflex.cli.commands.evaluate import calculate_metrics

# --- Plotting Functions ---

def _save_plot_and_data(
    fig: Figure,
    plot_df: Optional[pd.DataFrame],
    base_filename: str,
    output_dir: str,
    save_format: str,
    dpi: int,
    save_data: bool
) -> Optional[str]:
    """Helper function to save plot image and optionally its data."""
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"{base_filename}_{timestamp}.{save_format}")
    csv_path = os.path.join(output_dir, f"{base_filename}_{timestamp}_data.csv")

    plot_saved = False
    try:
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_path}")
        plot_saved = True
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")
    finally:
        plt.close(fig) # Close figure to free memory, regardless of save success

    if save_data and plot_df is not None and not plot_df.empty:
        try:
            plot_df.to_csv(csv_path, index=False, float_format='%.6f')
            logger.info(f"Saved plot data: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save plot data {csv_path}: {e}")

    return plot_path if plot_saved else None

def create_metric_curve(
    history: Dict[str, List[float]],
    metric_key: str, # e.g., 'loss' or 'pearson'
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True
) -> Optional[str]:
    """Creates a plot of training and validation metrics over epochs."""
    train_key = f'train_{metric_key}'
    val_key = f'val_{metric_key}'
    lr_key = 'lr'

    if train_key not in history or val_key not in history or not history[train_key] or not history[val_key]:
        logger.warning(f"History data missing for '{metric_key}'. Skipping {metric_key} curve plot.")
        return None

    logger.info(f"Creating {metric_key} curve plot...")
    train_values = history[train_key]
    val_values = history[val_key]
    epochs = range(1, len(train_values) + 1)

    plot_df_dict: Dict[str, List[Union[int, float]]] = { # Use Union for types
        'epoch': list(epochs),
        train_key: train_values,
        val_key: val_values
    }
    # Handle potential length mismatch for LR if resuming incomplete epoch
    if lr_key in history and len(history[lr_key]) >= len(epochs):
        plot_df_dict[lr_key] = history[lr_key][:len(epochs)]
    elif lr_key in history:
         logger.warning(f"Length mismatch for '{lr_key}' in history. Cannot plot LR.")

    plot_df = pd.DataFrame(plot_df_dict)


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_values, 'o-', color='royalblue', label=f'Training {metric_key.capitalize()}', markersize=4, alpha=0.8)
    ax.plot(epochs, val_values, 's-', color='orangered', label=f'Validation {metric_key.capitalize()}', markersize=4, alpha=0.8)

    # Determine best epoch based on metric (loss=min, corr=max)
    try:
        is_loss = 'loss' in metric_key.lower()
        valid_val_values = [v for v in val_values if pd.notna(v)] # Filter NaNs for argmin/argmax
        if not valid_val_values: raise ValueError("No valid validation values found.")

        if is_loss:
            best_val_epoch_idx = np.nanargmin(val_values) # Use nanargmin
        else:
            best_val_epoch_idx = np.nanargmax(val_values) # Use nanargmax

        best_val_epoch = best_val_epoch_idx + 1
        best_val_value = val_values[best_val_epoch_idx]
        ax.axvline(best_val_epoch, linestyle='--', color='gray', alpha=0.7, label=f'Best Val @ Ep {best_val_epoch} ({best_val_value:.4f})')
    except (ValueError, IndexError) as e:
         logger.warning(f"Could not determine best epoch for {metric_key} curve: {e}")


    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_key.capitalize(), fontsize=12)
    ax.set_title(f'Training and Validation {metric_key.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Optional: Add Learning Rate on secondary axis if plotting correlation
    if not is_loss and lr_key in plot_df.columns:
        ax2 = ax.twinx()
        ax2.plot(epochs, plot_df[lr_key], 'd--', color='green', label='Learning Rate', markersize=3, alpha=0.5)
        ax2.set_ylabel('Learning Rate', color='green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='green', labelsize=9)
        # Use log scale if LR varies significantly and has no zeros/negatives
        lr_vals = plot_df[lr_key].dropna()
        if len(lr_vals.unique()) > 2 and (lr_vals > 0).all():
            try:
                 ax2.set_yscale('log')
            except ValueError as e: # Catch potential issues with non-positive values if check failed
                 logger.warning(f"Could not set log scale for LR axis: {e}")
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best', fontsize=9)


    fig.tight_layout()
    base_filename = f"{metric_key}_curve"
    # Pass the DataFrame used for plotting
    return _save_plot_and_data(fig, plot_df, base_filename, output_dir, save_format, dpi, save_data)

# Alias specific curve functions
def create_loss_curve(history, output_dir, save_format='png', dpi=150, save_data=True):
    return create_metric_curve(history, 'loss', output_dir, save_format, dpi, save_data)

def create_correlation_curve(history, output_dir, save_format='png', dpi=150, save_data=True):
    return create_metric_curve(history, 'pearson', output_dir, save_format, dpi, save_data)

def create_prediction_scatter(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    max_points: int = 1000,
    save_data: bool = True
) -> Optional[str]:
    """Creates Predicted vs. Actual RMSF scatter plot."""
    if eval_df.empty or 'target_rmsf' not in eval_df or 'predicted_rmsf' not in eval_df:
        logger.warning("Cannot create prediction scatter: DataFrame empty or missing columns.")
        return None

    logger.info("Creating prediction scatter plot...")
    # Use calculate_metrics for consistency
    metrics = calculate_metrics(eval_df, label="Scatter")

    y_true = eval_df['target_rmsf'].values
    y_pred = eval_df['predicted_rmsf'].values

    # Filter only finite values for plotting and metrics text
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_plot = y_true[valid_mask]
    y_pred_plot = y_pred[valid_mask]
    plot_df_full = eval_df[valid_mask].copy() # Dataframe with only valid points

    if len(y_true_plot) == 0:
        logger.warning("No finite data points for prediction scatter plot.")
        return None

    # Sample points *after* filtering NaNs if needed
    if len(y_true_plot) > max_points:
        logger.debug(f"Sampling {max_points} points for scatter plot from {len(y_true_plot)} valid points.")
        indices = np.random.choice(len(y_true_plot), max_points, replace=False)
        y_true_sampled, y_pred_sampled = y_true_plot[indices], y_pred_plot[indices]
        # Data to save should be the sampled points if sampling occurred
        plot_df_to_save = plot_df_full.iloc[indices][['target_rmsf', 'predicted_rmsf']].copy()
    else:
        y_true_sampled, y_pred_sampled = y_true_plot, y_pred_plot
        plot_df_to_save = plot_df_full[['target_rmsf', 'predicted_rmsf']].copy() # Save all valid points

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_sampled, y_pred_sampled, c='steelblue', s=20, alpha=0.5, edgecolors='none', label='Predictions')

    # Add identity line
    # Calculate limits based on the *sampled* data for the plot view
    min_val = min(y_true_sampled.min(), y_pred_sampled.min()) - 0.1 * abs(min(y_true_sampled.min(), y_pred_sampled.min())) # Add padding
    max_val = max(y_true_sampled.max(), y_pred_sampled.max()) + 0.1 * abs(max(y_true_sampled.max(), y_pred_sampled.max()))
    lims = [min_val, max_val]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    # Use metrics calculated on the *full* valid dataset for annotation
    metrics_text = (
        f"Pearson: {metrics.get('pearson', np.nan):.3f}\n"
        f"R²: {metrics.get('r2', np.nan):.3f}\n"
        f"RMSE: {metrics.get('rmse', np.nan):.3f}\n"
        f"MAE: {metrics.get('mae', np.nan):.3f}\n"
        f"N: {metrics.get('count', 0):,}" # Count is from full valid dataset
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_title('Predicted vs. Actual RMSF', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual RMSF', fontsize=12)
    ax.set_ylabel('Predicted RMSF', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    base_filename = "prediction_scatter"
    # Save the potentially sampled data
    return _save_plot_and_data(fig, plot_df_to_save, base_filename, output_dir, save_format, dpi, save_data)


def create_predicted_vs_validation_scatter_density(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True # Save the underlying x, y data
) -> Optional[str]:
    """Creates Predicted vs. Actual RMSF scatter plot with density coloring."""
    if eval_df.empty or 'target_rmsf' not in eval_df or 'predicted_rmsf' not in eval_df:
        logger.warning("Cannot create density scatter: DataFrame empty or missing columns.")
        return None

    logger.info("Creating prediction density scatter plot...")
    y_true = eval_df['target_rmsf'].values
    y_pred = eval_df['predicted_rmsf'].values

    # Filter out NaNs/Infs before KDE
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    plot_df = eval_df[valid_mask][['target_rmsf', 'predicted_rmsf']].copy() # DataFrame for saving

    if len(y_true_clean) < 5: # Need points for KDE
        logger.warning("Too few valid points (<5) for density estimation. Skipping density scatter.")
        return None

    # Calculate metrics on valid points only
    metrics = calculate_metrics(plot_df, label="Density Scatter")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate the point density
    try:
        xy = np.vstack([y_true_clean, y_pred_clean])
        # Handle potential singular matrix in KDE
        try:
             z = gaussian_kde(xy)(xy)
        except np.linalg.LinAlgError:
             logger.warning("Singular matrix in KDE calculation. Adding small jitter.")
             jitter_scale = 1e-6 * (np.max(xy, axis=1) - np.min(xy, axis=1))
             jitter = jitter_scale[:, np.newaxis] * np.random.randn(*xy.shape)
             z = gaussian_kde(xy + jitter)(xy + jitter)

        # Sort points by density, so dense points are plotted last
        idx = z.argsort()
        x_plot, y_plot, z_plot = y_true_clean[idx], y_pred_clean[idx], z[idx]

        scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=10, cmap=viridis, alpha=0.7, edgecolors='none')
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label('Point Density', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    except Exception as kde_e:
         logger.warning(f"KDE calculation failed ({kde_e}). Falling back to simple scatter.")
         ax.scatter(y_true_clean, y_pred_clean, c='steelblue', s=10, alpha=0.5, edgecolors='none')


    # Add identity line
    min_val = min(y_true_clean.min(), y_pred_clean.min()) - 0.1 * abs(min(y_true_clean.min(), y_pred_clean.min()))
    max_val = max(y_true_clean.max(), y_pred_clean.max()) + 0.1 * abs(max(y_true_clean.max(), y_pred_clean.max()))
    lims = [min_val, max_val]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label="y=x") # Ensure line is visible
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    metrics_text = (
        f"Pearson: {metrics.get('pearson', np.nan):.3f}\n"
        f"R²: {metrics.get('r2', np.nan):.3f}\n"
        f"RMSE: {metrics.get('rmse', np.nan):.3f}\n"
        f"MAE: {metrics.get('mae', np.nan):.3f}\n"
        f"N: {metrics.get('count', 0):,}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_title('Predicted vs. Actual RMSF (Density Scatter)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual RMSF', fontsize=12)
    ax.set_ylabel('Predicted RMSF', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    base_filename = "prediction_density_scatter"
    return _save_plot_and_data(fig, plot_df, base_filename, output_dir, save_format, dpi, save_data)


def create_error_distribution(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True
) -> Optional[str]:
    """Creates histogram of prediction errors (Predicted - Actual)."""
    if eval_df.empty or 'target_rmsf' not in eval_df or 'predicted_rmsf' not in eval_df:
        logger.warning("Cannot create error distribution: DataFrame empty or missing columns.")
        return None

    logger.info("Creating error distribution plot...")
    eval_df_copy = eval_df.copy() # Work on a copy
    eval_df_copy['error'] = eval_df_copy['predicted_rmsf'] - eval_df_copy['target_rmsf']
    errors = eval_df_copy['error'].dropna()
    if errors.empty:
        logger.warning("No valid error values found for distribution plot.")
        return None

    plot_df = pd.DataFrame({'error': errors}) # Data for saving

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=50, ax=ax, color='coral', edgecolor='black', alpha=0.7, stat="density")

    mean_error = errors.mean()
    median_error = errors.median()
    std_error = errors.std()

    ax.axvline(mean_error, color='k', linestyle='--', linewidth=1.5, label=f'Mean: {mean_error:.3f}')
    ax.axvline(median_error, color='k', linestyle=':', linewidth=1.5, label=f'Median: {median_error:.3f}')
    # ax.axvspan(mean_error - std_error, mean_error + std_error, alpha=0.15, color='gray', label=f'StdDev: {std_error:.3f}')

    ax.set_title('Distribution of Prediction Errors (Predicted - Actual)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Density', fontsize=12) # Use Density since kde=True
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    base_filename = "error_distribution"
    return _save_plot_and_data(fig, plot_df, base_filename, output_dir, save_format, dpi, save_data)


def create_stratified_error_boxplot(
    eval_df: pd.DataFrame,
    stratify_by_col: str,
    plot_title: str,
    x_label: str,
    output_dir: str,
    base_filename: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True,
    min_samples_per_group: int = 10
) -> Optional[str]:
    """Creates a boxplot of absolute errors stratified by a given column."""
    if eval_df.empty or stratify_by_col not in eval_df.columns:
        logger.warning(f"Cannot create stratified boxplot: DataFrame empty or column '{stratify_by_col}' missing.")
        return None

    logger.info(f"Creating stratified error boxplot by '{stratify_by_col}'...")
    eval_df_copy = eval_df.copy()
    if 'error' not in eval_df_copy.columns:
         eval_df_copy['error'] = eval_df_copy['predicted_rmsf'] - eval_df_copy['target_rmsf']
    eval_df_copy['abs_error'] = np.abs(eval_df_copy['error'])

    # Ensure stratification column is suitable type (e.g., string or category) and handle NaNs
    eval_df_copy = eval_df_copy.dropna(subset=[stratify_by_col, 'abs_error'])
    eval_df_copy[stratify_by_col] = eval_df_copy[stratify_by_col].astype(str) # Convert to string for consistent grouping

    # Filter groups with enough samples
    group_counts = eval_df_copy[stratify_by_col].value_counts()
    valid_groups = group_counts[group_counts >= min_samples_per_group].index.tolist()
    plot_df = eval_df_copy[eval_df_copy[stratify_by_col].isin(valid_groups)].copy()

    if plot_df.empty:
        logger.warning(f"No groups in '{stratify_by_col}' met the minimum sample requirement ({min_samples_per_group}). Skipping boxplot.")
        return None

    # Sort categories for consistent plotting
    # Try numeric sort first if they look like numbers/bins, else alphabetical
    try:
        # Attempt to extract leading number for sorting (e.g., for SASA bins)
        categories_sorted = sorted(valid_groups, key=lambda x: float(x.split('-')[0]))
    except (ValueError, IndexError):
        # Fallback to alphabetical sort if numeric extraction fails
        categories_sorted = sorted(valid_groups)


    fig, ax = plt.subplots(figsize=(max(8, len(categories_sorted)*0.5), 6)) # Adjust width based on # categories
    sns.boxplot(x=stratify_by_col, y='abs_error', data=plot_df, ax=ax, order=categories_sorted,
                palette='viridis', fliersize=2, linewidth=1.0, showfliers=False) # Hide outliers for clarity?

    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Absolute Prediction Error', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10, ha='right')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')

    # Add counts below boxes
    y_min, y_max = ax.get_ylim() # Get current y-limits AFTER plotting boxes
    # y_range = y_max - y_min if y_max > y_min else 1.0 # Avoid zero range
    # Place text slightly below the minimum y-axis value shown
    text_y_pos = y_min - (y_max - y_min) * 0.05 # Position below plot area

    for i, cat in enumerate(categories_sorted):
        count = group_counts[cat]
        ax.text(i, text_y_pos, f"n={count}", ha='center', va='top', fontsize=8, color='gray')

    # Adjust y-limit slightly to make space for text
    ax.set_ylim(bottom=text_y_pos - (y_max - y_min) * 0.02)

    fig.tight_layout()
    # Data to save includes the stratification column and the absolute error
    plot_data_to_save = plot_df[[stratify_by_col, 'abs_error']].copy()
    return _save_plot_and_data(fig, plot_data_to_save, base_filename, output_dir, save_format, dpi, save_data)


# Specific wrappers for stratified plots
def create_residue_type_analysis(eval_df, output_dir, save_format='png', dpi=150, save_data=True):
    return create_stratified_error_boxplot(
        eval_df, 'resname', 'Absolute Error by Residue Type', 'Residue Type',
        output_dir, 'residue_type_error_boxplot', save_format, dpi, save_data
    )

def create_sasa_error_analysis(eval_df, output_dir, sasa_bins, save_format='png', dpi=150, save_data=True):
    sasa_col = 'relative_accessibility'
    if sasa_col not in eval_df.columns:
        logger.warning(f"SASA column '{sasa_col}' not found. Skipping SASA error analysis.")
        return None
    eval_df_copy = eval_df.copy()
    # Ensure bin column exists
    if 'sasa_bin' not in eval_df_copy.columns:
        bin_labels = [f"{sasa_bins[i]:.1f}-{sasa_bins[i+1]:.1f}" for i in range(len(sasa_bins)-1)]
        try:
            # Ensure numeric conversion before cutting
            eval_df_copy[sasa_col] = pd.to_numeric(eval_df_copy[sasa_col], errors='coerce')
            eval_df_copy['sasa_bin'] = pd.cut(eval_df_copy[sasa_col], bins=sasa_bins, labels=bin_labels, right=False, include_lowest=True)
        except Exception as e:
             logger.error(f"Failed to create SASA bins for plotting: {e}")
             return None
    # Pass the dataframe with the 'sasa_bin' column
    return create_stratified_error_boxplot(
        eval_df_copy, 'sasa_bin', 'Absolute Error by Relative Solvent Accessibility', 'SASA Bin',
        output_dir, 'sasa_error_boxplot', save_format, dpi, save_data
    )

def create_ss_error_analysis(eval_df, output_dir, save_format='png', dpi=150, save_data=True):
    ss_col = next((c for c in ['dssp', 'secondary_structure_encoded'] if c in eval_df.columns), None)
    if ss_col is None:
        logger.warning("No secondary structure column found. Skipping SS error analysis.")
        return None
    return create_stratified_error_boxplot(
        eval_df, ss_col, 'Absolute Error by Secondary Structure', f'SS Type ({ss_col})',
        output_dir, 'ss_error_boxplot', save_format, dpi, save_data
    )


def create_amino_acid_performance(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True
) -> Optional[str]:
    """Creates bar plots of various metrics grouped by amino acid type."""
    if eval_df.empty or 'resname' not in eval_df.columns:
        logger.warning("Cannot create AA performance plot: DataFrame empty or 'resname' missing.")
        return None

    logger.info("Creating amino acid performance plots...")
    aa_metrics_list = []
    # Ensure resname is string and handle potential NaNs dropped by earlier steps
    eval_df_copy = eval_df.dropna(subset=['resname']).copy()
    eval_df_copy['resname'] = eval_df_copy['resname'].astype(str)

    grouped = eval_df_copy.groupby('resname')
    # Define standard AA order if possible, otherwise sort alphabetically
    aa_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    resnames_present = eval_df_copy['resname'].unique()
    resnames_sorted = [aa for aa in aa_order if aa in resnames_present] + sorted([aa for aa in resnames_present if aa not in aa_order])

    for resname in resnames_sorted:
        group = grouped.get_group(resname)
        if len(group) >= 2: # Need min 2 points for most metrics
             metrics = calculate_metrics(group, label=f"AA={resname}")
             # Only add if key metrics were calculable
             if pd.notna(metrics.get('pearson')) or pd.notna(metrics.get('rmse')):
                 metrics['resname'] = resname
                 aa_metrics_list.append(metrics)
        # else: logger.debug(f"Skipping AA '{resname}' due to insufficient samples ({len(group)})")

    if not aa_metrics_list:
         logger.warning("No amino acid groups had sufficient valid samples for performance plotting.")
         return None

    metrics_df = pd.DataFrame(aa_metrics_list)
    plot_df_to_save = metrics_df.copy() # Data to save

    # Plotting setup
    metrics_to_plot = ['pearson', 'rmse', 'mae', 'count']
    titles = ['Pearson Correlation', 'RMSE', 'MAE', 'Sample Count']
    palettes = ['coolwarm', 'viridis_r', 'magma_r', 'crest']
    num_metrics = len(metrics_to_plot)
    ncols = 2
    nrows = (num_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True) # Share x-axis
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if metric not in metrics_df.columns:
            logger.warning(f"Metric '{metric}' not found in calculated AA metrics. Skipping its plot.")
            # Disable the unused subplot
            if i < len(axes): axes[i].set_visible(False)
            continue

        # Filter out NaN values for the specific metric before plotting
        plot_data = metrics_df.dropna(subset=[metric])
        if plot_data.empty:
            logger.warning(f"No valid data for metric '{metric}' after dropping NaNs. Skipping plot.")
            if i < len(axes): axes[i].set_visible(False)
            continue

        sns.barplot(x='resname', y=metric, data=plot_data, ax=axes[i], palette=palettes[i % len(palettes)], order=resnames_sorted)
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel(None) # Remove redundant x-label from upper plots if sharing x
        axes[i].set_ylabel(metric.upper() if metric != 'count' else 'Count', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45, labelsize=9, ha='right')
        axes[i].tick_params(axis='y', labelsize=9)
        axes[i].grid(True, linestyle=':', alpha=0.6, axis='y')

    # Set x-label only on the bottom plots
    for i in range(ncols * (nrows - 1), len(axes)):
        if axes[i].get_visible(): # Only if plot was actually drawn
             axes[i].set_xlabel("Residue Type", fontsize=11)

    # Remove any unused subplots if num_metrics < nrows*ncols
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)


    fig.suptitle("Performance Metrics by Amino Acid Type", fontsize=16, fontweight='bold')#, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap

    base_filename = "amino_acid_performance"
    return _save_plot_and_data(fig, plot_df_to_save, base_filename, output_dir, save_format, dpi, save_data)


# --- Main Visualization Function ---

def create_visualizations(
    config: Dict[str, Any],
    predictions_path: str,
    history_path: Optional[str] = None # Allow passing history file path
) -> List[str]:
    """
    Generate and save performance visualizations based on config settings.

    Args:
        config: Configuration dictionary.
        predictions_path: Path to the prediction CSV file.
        history_path: Optional path to the training history JSON file.

    Returns:
        List of paths to the generated plot files.
    """
    log_section_header(logger, "GENERATING VISUALIZATIONS")
    predictions_path = resolve_path(predictions_path)
    if not os.path.exists(predictions_path):
        logger.error(f"Predictions file not found: {predictions_path}")
        return []
    history_path = resolve_path(history_path) if history_path else None
    viz_output_dir = config["output"]["visualizations_dir"]
    ensure_dir(viz_output_dir) # Ensure output directory exists

    viz_config = config.get("visualization", {})
    save_format = viz_config.get("save_format", "png")
    dpi = viz_config.get("dpi", 150)
    save_plot_data = viz_config.get("save_plot_data", True)
    max_scatter = viz_config.get("max_scatter_points", 1000)
    sasa_bins = config.get('evaluation', {}).get('sasa_bins', [0.0, 0.1, 0.4, 1.01]) # Get bins from eval config

    generated_plots: List[str] = []
    eval_df: Optional[pd.DataFrame] = None # Initialize

    # --- Load Data ---
    try:
        with log_stage("VIS_SETUP", "Loading prediction and ground truth data"):
            logger.info(f"Loading predictions from: {predictions_path}")
            preds_df = pd.read_csv(predictions_path, dtype={'domain_id': str, 'resname': str})
            preds_df['resid'] = pd.to_numeric(preds_df['resid'], errors='coerce').astype('Int64')
            preds_df['predicted_rmsf'] = pd.to_numeric(preds_df['predicted_rmsf'], errors='coerce')
            preds_df['prediction_temperature'] = pd.to_numeric(preds_df['prediction_temperature'], errors='coerce')
            preds_df.dropna(subset=['domain_id', 'resid', 'predicted_rmsf', 'prediction_temperature'], inplace=True)

            if preds_df.empty:
                 logger.error("Predictions file is empty or contains no valid rows. Cannot generate comparison visualizations.")
                 return [] # Stop if no predictions

            # Load ground truth for comparison plots
            gt_file_path = config['input'].get('aggregated_rmsf_file')
            if not gt_file_path:
                logger.warning("Aggregated RMSF file path not specified in config. Cannot generate comparison plots.")
                eval_df = preds_df.copy() # Can only plot prediction distribution
            else:
                logger.info(f"Loading ground truth from: {gt_file_path}")
                gt_df_raw = load_aggregated_rmsf_data(gt_file_path)
                # Select necessary columns, including optional ones
                cols_to_keep = ['domain_id', 'resid', 'resname', 'target_rmsf', 'temperature_feature']
                optional_cols = ['relative_accessibility', 'dssp', 'secondary_structure_encoded']
                available_optional = [col for col in optional_cols if col in gt_df_raw.columns]
                cols_to_keep.extend(available_optional)
                gt_df = gt_df_raw[cols_to_keep].copy()
                gt_df['resid'] = pd.to_numeric(gt_df['resid'], errors='coerce').astype('Int64')
                gt_df['temperature_feature'] = pd.to_numeric(gt_df['temperature_feature'], errors='coerce')
                gt_df.dropna(subset=['domain_id', 'resid', 'target_rmsf', 'temperature_feature'], inplace=True)

                # Merge predictions and ground truth based on prediction temperature
                pred_temp = preds_df['prediction_temperature'].iloc[0]
                gt_df_filtered = gt_df[np.isclose(gt_df['temperature_feature'], pred_temp)].copy()
                logger.info(f"Merging predictions with {len(gt_df_filtered)} ground truth entries for temp ~{pred_temp:.1f}K...")

                preds_df['resid'] = preds_df['resid'].astype(int) # Ensure int for merge
                gt_df_filtered['resid'] = gt_df_filtered['resid'].astype(int)
                eval_df = pd.merge(preds_df, gt_df_filtered.drop(columns=['temperature_feature']), on=['domain_id', 'resid'], how='inner', suffixes=('_pred', '_gt'))

                # Handle resname column merge issues
                if 'resname_pred' in eval_df.columns and 'resname_gt' in eval_df.columns:
                    eval_df['resname'] = eval_df['resname_gt']
                    eval_df.drop(columns=['resname_pred', 'resname_gt'], inplace=True)
                elif 'resname_gt' in eval_df.columns: eval_df.rename(columns={'resname_gt': 'resname'}, inplace=True)
                elif 'resname_pred' in eval_df.columns: eval_df.rename(columns={'resname_pred': 'resname'}, inplace=True)
                elif 'resname' not in eval_df.columns: eval_df['resname'] = 'UNK'

                eval_df.dropna(subset=['predicted_rmsf', 'target_rmsf'], inplace=True)
                logger.info(f"Merged data for plots contains {len(eval_df)} entries.")

                if eval_df.empty:
                    logger.error("No overlapping data found between predictions and ground truth. Cannot generate comparison plots.")
                    eval_df = None # Reset eval_df if merge failed

    except Exception as e:
        logger.exception(f"Error loading data for visualization: {e}")
        return [] # Cannot proceed without data

    # --- Load History ---
    train_history = None
    if history_path and os.path.exists(history_path):
        logger.info(f"Loading training history from: {history_path}")
        try:
             train_history = load_json(history_path)
        except Exception as e:
             logger.error(f"Failed to load history file {history_path}: {e}")
    elif viz_config.get("plot_loss") or viz_config.get("plot_correlation"):
        logger.warning("Training history file not found or specified. Skipping metric curve plots.")


    # --- Generate Plots ---
    with log_stage("VISUALIZATION", "Creating plots"):
        # Training Curves (require history)
        if train_history:
            if viz_config.get("plot_loss", False):
                 path = create_loss_curve(train_history, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_correlation", False):
                 if 'train_pearson' in train_history and 'val_pearson' in train_history:
                      path = create_correlation_curve(train_history, viz_output_dir, save_format, dpi, save_plot_data)
                      if path: generated_plots.append(path)
                 else: logger.warning("Pearson correlation data not found in history. Skipping correlation curve.")
        else:
             if viz_config.get("plot_loss") or viz_config.get("plot_correlation"):
                 logger.info("Skipping loss/correlation curves as history file was not provided or loaded.")

        # Comparison Plots (require merged eval_df)
        if eval_df is not None and not eval_df.empty:
            if viz_config.get("plot_predictions", False):
                path = create_prediction_scatter(eval_df, viz_output_dir, save_format, dpi, max_scatter, save_plot_data)
                if path: generated_plots.append(path)
            if viz_config.get("plot_density_scatter", False):
                 path = create_predicted_vs_validation_scatter_density(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_error_distribution", False):
                path = create_error_distribution(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                if path: generated_plots.append(path)
            if viz_config.get("plot_residue_type_analysis", False):
                path = create_residue_type_analysis(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                if path: generated_plots.append(path)
            if viz_config.get("plot_sasa_error_analysis", False):
                 path = create_sasa_error_analysis(eval_df, viz_output_dir, sasa_bins, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_ss_error_analysis", False):
                 path = create_ss_error_analysis(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_amino_acid_performance", False):
                 path = create_amino_acid_performance(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
        elif eval_df is None:
             logger.warning("Skipping comparison plots as merged evaluation data could not be created.")


    logger.info(f"Finished generating {len(generated_plots)} plots.")
    return generated_plots
EOF

# --- src/voxelflex/cli/commands/evaluate.py ---
# (Content already generated above)

# --- src/voxelflex/cli/commands/predict.py ---
# (Content already generated above)

# --- src/voxelflex/cli/commands/preprocess.py ---
# (Content already generated above)

# --- Packaging Files ---

# --- pyproject.toml ---
cat << 'EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "voxelflex"
# Increment version for refactoring, consider using dynamic versioning later
version = "0.3.0"
authors = [
  # Replace with your actual name and email
  { name="Your Name", email="your.email@example.com" },
]
description = "Temperature-aware protein flexibility (RMSF) prediction from 3D voxel data using CNNs and a preprocessing workflow."
readme = "README.md"
requires-python = ">=3.9" # Based on type hints and f-strings
license = { file = "LICENSE" } # Add a LICENSE file (e.g., MIT)
classifiers = [
    "Development Status :: 3 - Alpha", # Update as project matures
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License", # Choose your license
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.21", # Check specific version needs
    "pandas>=1.3",
    "torch>=1.10.0", # Consider torch version carefully based on hardware/CUDA
    "scipy",
    "scikit-learn",
    "h5py",
    "pyyaml",
    "matplotlib", # Ensure compatible version if issues arise
    "seaborn",
    "psutil",
    "tqdm",
    # Add specific versions if needed for reproducibility
    # e.g., "torch==2.0.1", "pandas==2.0.3"
]

[project.urls]
"Homepage" = "https://github.com/yourusername/voxelflex" # Replace with your repo URL
"Bug Tracker" = "https://github.com/yourusername/voxelflex/issues" # Replace

# Define the command-line script entry point
[project.scripts]
voxelflex = "voxelflex.cli.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

# Optional: Tool configurations (can be added later)
# [tool.black]
# line-length = 88
# target-version = ['py310']

# [tool.isort]
# profile = "black"

# [tool.mypy]
# python_version = "3.10"
# warn_return_any = true
# warn_unused_configs = true
# ignore_missing_imports = true # Initially, relax this if needed
EOF

# --- README.md ---
cat << 'EOF' > README.md
# VoxelFlex (Temperature-Aware - Preprocessing Workflow)

Predicting temperature-dependent protein flexibility (RMSF) from 3D voxel data using deep learning, optimized with a robust preprocessing pipeline.

## Overview

This package provides tools to train and use 3D Convolutional Neural Networks (CNNs) for predicting per-residue Root Mean Square Fluctuation (RMSF) values. This version is **temperature-aware**, meaning it takes the simulation temperature as an input feature, allowing a single model to predict flexibility across different temperatures.

It utilizes voxelized representations of protein structures (e.g., from Aposteriori) and RMSF data derived from Molecular Dynamics (MD) simulations (e.g., from the mdCATH dataset).

**Key Feature:** This version implements an optimized **preprocessing workflow**. Raw voxel and RMSF data are converted into batched PyTorch tensor files (`.pt`) before training. This significantly simplifies the training loop, improves performance, enhances robustness, and allows for efficient handling of very large datasets within defined memory limits.

## Features

*   Temperature-aware 3D CNN models (MultipathRMSFNet, DenseNet3D, DilatedResNet3D).
*   **Robust preprocessing pipeline:** Converts raw HDF5/CSV to optimized `.pt` batch files.
    *   Handles HDF5 boolean dtype casting and shape transposition.
    *   Manages memory via batch-wise HDF5 loading and optional caching.
    *   Scales temperature feature based on training set statistics.
    *   Generates metadata files (`.meta`) for easy loading.
*   **Simplified and efficient training:** Uses preprocessed batches directly in the DataLoader.
*   Support for mixed-precision training and standard optimizers/schedulers.
*   Rigorous evaluation including stratified metrics and permutation feature importance for temperature.
*   Visualization tools for analyzing training progress and model performance.
*   Command-line interface (`preprocess`, `train`, `predict`, `evaluate`, `visualize`).
*   Designed for use with large-scale MD datasets like mdCATH.

## Installation

```bash
# Recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows

# Install the package from the project root directory
pip install .

# Or for development (changes in src/ reflect immediately):
pip install -e .
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The package is primarily used via the command line interface.

**1. Preprocess Data:** (Run this first!)
```bash
voxelflex preprocess --config path/to/your/config.yaml [-v|-vv]
```
This reads raw data specified in the config, performs processing and scaling, and saves batches to `input_data/processed/` (or as configured in `data.processed_dir`).

**2. Train Model:**
```bash
voxelflex train --config path/to/your/config.yaml [-v|-vv] [--force_preprocess]
```
This loads the preprocessed data and trains the model, saving outputs (checkpoints, logs) to `outputs/<run_name>/`. Use `--force_preprocess` to re-run preprocessing before training.

**3. Predict RMSF:**
```bash
voxelflex predict --config path/to/config.yaml --model path/to/model.pt --temperature 320 [-v|-vv] [--domains ID1 ID2 ...] [--output_csv filename.csv]
```
Loads the trained model and predicts RMSF for specific domains (or test split domains) at the given temperature. Saves results to `outputs/<run_name>/metrics/`.

**4. Evaluate Model:**
```bash
voxelflex evaluate --config path/to/config.yaml --model path/to/model.pt --predictions path/to/preds.csv [-v|-vv]
```
Compares predictions against ground truth (from the aggregated RMSF file) and calculates performance metrics, saving results to `outputs/<run_name>/metrics/`.

**5. Visualize Results:**
```bash
voxelflex visualize --config path/to/config.yaml --predictions path/to/preds.csv [-v|-vv] [--history path/to/history.json]
```
Generates performance plots based on prediction/evaluation data and optional training history, saving them to `outputs/<run_name>/visualizations/`.

Use `-v` for INFO level logging and `-vv` for DEBUG level logging for any command.

## Configuration

Modify the `src/voxelflex/config/default_config.yaml` file or create a copy and adjust parameters. Key sections:

*   `input`: Paths to raw voxel HDF5, aggregated RMSF CSV, and domain split files (`.txt`).
*   `data`: Paths for processed data output, preprocessing batch size, cache limit.
*   `output`: Base directory for run outputs (logs, models, metrics, visualizations).
*   `model`: CNN architecture choice and hyperparameters.
*   `training`: Epochs, batch size (for loading `.pt` files), learning rate, optimizer, scheduler, etc.
*   `prediction`: Batch size for inference.
*   `evaluation`: Settings for stratified metrics and permutation importance.
*   `logging`: Logging levels and progress bar visibility.
*   `visualization`: Toggles for different plots and output settings.
*   `system_utilization`: GPU preference.

## Data Preparation

1.  **Voxel Data (HDF5):** Place your HDF5 file (e.g., `mdcath_voxelized.hdf5`) in `input_data/voxel/`. Ensure it follows the expected structure (`DomainID -> ChainID -> ResidueID -> Dataset`) and that datasets are predominantly `bool` type with shape `(X, Y, Z, Channels)` where `Channels=5`.
2.  **RMSF Data (CSV):** Create an aggregated CSV file (e.g., `aggregated_rmsf_all_temps.csv`) containing RMSF data from all relevant temperatures. Required columns: `domain_id` (must be mappable to HDF5 keys), `resid` (integer), `resname` (string), `temperature_feature` (float, Kelvin), `target_rmsf` (float). Optional columns for evaluation: `relative_accessibility`, `dssp` (or `secondary_structure_encoded`). Place in `input_data/rmsf/`.
3.  **Splits (.txt):** Generate plain text files listing HDF5 domain keys for training, validation, and testing (one ID per line). Place these in `input_data/`. Using splits generated by tools like `mdcath-sampling` is highly recommended to mitigate homology bias.

## Contributing

[Optional: Add guidelines for contributions if applicable]

## License

[Specify License, e.g., MIT License] - Remember to add a LICENSE file.
EOF

# --- requirements.txt ---
cat << 'EOF' > requirements.txt
# VoxelFlex Core Dependencies
# Pin versions for better reproducibility, update as needed.
numpy>=1.21
pandas>=1.3
# PyTorch: Choose version compatible with your CUDA toolkit if using GPU
# Example for CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Example for CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Example for CPU only: pip install torch torchvision torchaudio
torch>=1.10.0
scipy
scikit-learn
h5py
pyyaml
matplotlib
seaborn
psutil
tqdm

# Optional but recommended for development/testing:
# pytest
# flake8
# black
# isort
# mypy
EOF

# --- .gitignore ---
cat << 'EOF' > .gitignore
# Python cache files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# *.manifest
# *.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE files
.idea/
.vscode/
*.sublime-project
*.sublime-workspace

# Output directories (important!)
outputs/
input_data/processed/ # Exclude generated processed data

# Temporary files
*.log
*.tmp
*.bak
*.swp
*~
*.patch
*.diff

# OS generated files
.DS_Store
Thumbs.db

# Memmap temp dirs (if pattern used, though cleanup should handle)
# voxelflex_memmap_*/ # Better handled by cleanup
EOF

# --- Placeholder test file ---
cat << 'EOF' > tests/test_placeholder.py
import pytest

def test_placeholder():
    """Placeholder test."""
    assert True
EOF

# --- Placeholder LICENSE file (MIT Example) ---
cat << 'EOF' > LICENSE
MIT License

Copyright (c) $(date +%Y) [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF


echo "--------------------------------------"
echo "VoxelFlex package structure created."
echo "--------------------------------------"
echo "Next steps:"
echo "1. Review the generated files, especially:"
echo "   - src/voxelflex/config/default_config.yaml (Adjust paths and parameters)"
echo "   - requirements.txt (Ensure versions are appropriate)"
echo "   - pyproject.toml (Update author, URL, license if needed)"
echo "   - LICENSE (Replace placeholder content with your chosen license)"
echo "2. Place your input data in the 'input_data/' directory."
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Install the package: pip install -e ."
echo "5. Run commands: voxelflex preprocess --config your_config.yaml"
echo "--------------------------------------"

exit 0

