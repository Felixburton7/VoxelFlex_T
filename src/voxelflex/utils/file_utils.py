"""
File system utilities for VoxelFlex.

Provides functions for file path resolution, directory creation, and JSON handling.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import os

logger = logging.getLogger("voxelflex.utils.file")

def resolve_path(path: Union[str, Path]) -> str:
    """
    Resolves a path relative to CWD or user home.
    
    Args:
        path: Input path string or Path object
        
    Returns:
        Resolved absolute path as string
    """
    if path is None:
        return None
        
    p = Path(str(path)).expanduser()
    
    # Resolve relative paths based on the current working directory
    if not p.is_absolute():
        p = Path.cwd() / p
        
    # Return absolute path
    try:
        # Use the absolute path to avoid issues with non-existent path components
        return str(p.absolute())
    except Exception as e:
        logger.warning(f"Could not fully resolve path {p}: {e}. Returning absolute path.")
        return str(p.absolute())


def ensure_dir(dir_path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Directory path to ensure exists
        
    Raises:
        NotADirectoryError: If path exists but is not a directory
    """
    if dir_path:
        path = Path(dir_path)
        if not path.exists():
            logger.debug(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")


def save_json(data: Union[Dict, List], file_path: Union[str, Path], indent: int = 4) -> None:
    """
    Save dictionary or list to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        file_path: Path where JSON will be saved
        indent: Indentation level for pretty printing
        
    Raises:
        Exception: If saving fails
    """
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
    """
    Load dictionary or list from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data structure from JSON
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid JSON
        Exception: For other loading errors
    """
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
    """
    Load a list of strings from a file, one item per line.
    
    Args:
        file_path: Path to text file
        
    Returns:
        List of strings from file (empty list if file not found or error)
    """
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
        return []  # Return empty list on error


def save_list_to_file(data: List[str], file_path: Union[str, Path]) -> None:
    """
    Save a list of strings to a file, one item per line.
    
    Args:
        data: List of strings to save
        file_path: Path where file will be saved
        
    Raises:
        Exception: If saving fails
    """
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


def remove_file(file_path: Union[str, Path], ignore_errors: bool = False) -> bool:
    """
    Safely remove a file if it exists.
    
    Args:
        file_path: Path to file to remove
        ignore_errors: Whether to ignore errors during removal
        
    Returns:
        True if file was removed or didn't exist, False on error when ignore_errors=False
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return True
        
    try:
        file_path.unlink()
        logger.debug(f"Removed file: {file_path}")
        return True
    except Exception as e:
        if ignore_errors:
            logger.debug(f"Ignored error removing file {file_path}: {e}")
            return True
        else:
            logger.error(f"Failed to remove file {file_path}: {e}")
            return False


def is_file_empty(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists and is empty.
    
    Args:
        file_path: Path to file to check
        
    Returns:
        True if file doesn't exist or is empty, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return True
        
    return file_path.stat().st_size == 0