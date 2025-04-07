"""
Utility functions for temperature scaling.
"""

import json
import os
import logging
from typing import Callable, Tuple, List, Dict, Any, Optional

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
