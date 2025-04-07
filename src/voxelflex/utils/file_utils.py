"""
File system utilities for VoxelFlex.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import os

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
