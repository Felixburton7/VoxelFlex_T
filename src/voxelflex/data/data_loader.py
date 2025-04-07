# src/voxelflex/data/data_loader.py (Optimized)
"""
Data loading module for VoxelFlex (Temperature-Aware).

Defines:
- SafeDomainCache: Thread-safe caching system for domain data
- VoxelDataset: Loads sample metadata from file and reads/processes HDF5 voxels on demand.
- PredictionDataset: Minimal dataset for prediction, loading raw voxels on demand.
- Helper functions for loading raw RMSF data and creating lookups/mappings.
- Robust function for loading raw voxel data from HDF5.
"""

import os
import logging
import time
import gc
import h5py
import json
import psutil
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from collections import defaultdict, OrderedDict, namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Check for PyArrow and import
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger("voxelflex.data") # Use parent logger name

# Internal imports
from voxelflex.data.validators import validate_aggregated_rmsf_data
from voxelflex.utils.file_utils import resolve_path, ensure_dir, load_json, save_json
from voxelflex.utils.logging_utils import EnhancedProgressBar

# Define master samples filename constant
MASTER_SAMPLES_FILENAME = "master_samples.parquet"

# Define a global HDF5 file handle for worker processes
worker_h5_file = None

# --- Domain Cache Implementation ---
class SafeDomainCache:
    """
    Thread-safe, memory-safe cache for domain voxel data.
    Uses LRU (Least Recently Used) strategy with memory monitoring.
    """
    def __init__(self, max_size_mb=2000, monitor_memory=True, memory_limit_percent=75):
        """
        Initialize the cache with memory safety limits.
        
        Args:
            max_size_mb: Maximum cache size in MB (default 2000MB/2GB)
            monitor_memory: Whether to monitor system memory usage
            memory_limit_percent: Emergency clear if memory exceeds this percentage
        """
        self.cache = {}
        self.domain_sizes = {}
        self.access_order = []
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.monitor_memory = monitor_memory
        self.memory_limit_percent = memory_limit_percent
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.emergency_clears = 0
        
        logger.info(f"Initialized SafeDomainCache: Max size: {max_size_mb}MB, Memory monitoring: {monitor_memory}")
    
    def get(self, domain_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get a domain from the cache.
        
        Args:
            domain_id: The domain ID to retrieve
            
        Returns:
            The cached domain data or None if not in cache
        """
        domain_data = self.cache.get(domain_id)
        
        if domain_data is not None:
            # Update access order for LRU
            if domain_id in self.access_order:
                self.access_order.remove(domain_id)
            self.access_order.append(domain_id)
            self.hits += 1
            return domain_data
        
        self.misses += 1
        return None
    
    def add(self, domain_id: str, domain_data: Dict[str, np.ndarray]) -> bool:
        """
        Add a domain to the cache with memory safety checks.
        
        Args:
            domain_id: The domain ID to cache
            domain_data: Dict mapping residue_ids to voxel arrays
            
        Returns:
            True if domain was added, False if skipped for safety
        """
        # Check for memory pressure if monitoring enabled
        if self.monitor_memory and psutil.virtual_memory().percent > self.memory_limit_percent:
            logger.warning(f"System memory usage ({psutil.virtual_memory().percent}%) exceeds limit ({self.memory_limit_percent}%). Emergency cache clear.")
            self.clear()
            self.emergency_clears += 1
            return False
        
        # Skip if already in cache
        if domain_id in self.cache:
            # Update access order
            if domain_id in self.access_order:
                self.access_order.remove(domain_id)
            self.access_order.append(domain_id)
            return True
        
        # Calculate size of domain data
        domain_size = sum(arr.nbytes for arr in domain_data.values() if hasattr(arr, 'nbytes'))
        
        # Skip caching overly large domains (>25% of total cache)
        if domain_size > (self.max_size_bytes * 0.25):
            logger.info(f"Domain {domain_id} too large ({domain_size/1024/1024:.1f}MB) for cache")
            return False
        
        # Make room in cache if needed
        while self.current_size_bytes + domain_size > self.max_size_bytes and self.access_order:
            self._remove_oldest()
        
        # Add to cache
        self.cache[domain_id] = domain_data
        self.domain_sizes[domain_id] = domain_size
        self.access_order.append(domain_id)
        self.current_size_bytes += domain_size
        
        return True
    
    def _remove_oldest(self) -> None:
        """Remove the least recently used domain from cache."""
        if not self.access_order:
            return
        
        oldest_key = self.access_order.pop(0)
        
        if oldest_key in self.cache:
            size = self.domain_sizes.pop(oldest_key, 0)
            self.cache.pop(oldest_key)
            self.current_size_bytes -= size
            self.evictions += 1
    
    def clear(self) -> None:
        """Completely clear the cache."""
        self.cache.clear()
        self.domain_sizes.clear()
        self.access_order.clear()
        self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        return {
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": (self.current_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
            "num_domains": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
            "evictions": self.evictions,
            "emergency_clears": self.emergency_clears
        }

# Create a namedtuple for samples to avoid pandas overhead during training
Sample = namedtuple('Sample', ['hdf5_domain_id', 'resid_str', 'resid_int', 'raw_temp', 'target_rmsf', 'split'])

class VoxelDataset(Dataset):
    """
    Dataset that loads metadata from a master sample file (CSV or Parquet)
    and reads/processes corresponding voxel data from HDF5 on demand.
    """
    def __init__(
        self,
        processed_dir: str,
        split: str, # 'train', 'val', or 'test'
        voxel_hdf5_path: str,
        config: Dict[str, Any],
        # Voxel shape/channel info needed by load_process_voxels_from_hdf5
        expected_channels: int = 5,
        target_shape_chw: Optional[Tuple[int, int, int, int]] = (5, 21, 21, 21)
    ):
        """
        Initializes the dataset by loading and filtering the master sample list.

        Args:
            processed_dir: Directory containing the master sample file.
            split: Which dataset split to load ('train', 'val', or 'test').
            voxel_hdf5_path: Path to the raw HDF5 voxel file.
            config: Configuration dictionary
            expected_channels: Expected channels for voxel processing.
            target_shape_chw: Expected final voxel shape for validation.
        """
        self.processed_dir = resolve_path(processed_dir)
        self.split = split
        self.voxel_hdf5_path = resolve_path(voxel_hdf5_path)
        self.expected_channels = expected_channels
        self.target_shape_chw = target_shape_chw
        self.samples = []  # Store as list of namedtuples instead of DataFrame
        self.domain_mapping = {}  # Cache domain mapping
        
        # Initialize domain cache
        cache_size_mb = config.get('data', {}).get('domain_cache_mb', 2000)
        monitor_memory = config.get('data', {}).get('monitor_memory', True)
        self.domain_cache = SafeDomainCache(max_size_mb=cache_size_mb, monitor_memory=monitor_memory)
        
        # Hard-coded temperature scaling constants based on dataset
        # These would typically come from a preprocessing analysis
        self.temp_min = 280.0
        self.temp_max = 360.0

        master_file_path = os.path.join(self.processed_dir, MASTER_SAMPLES_FILENAME)

        if not os.path.exists(master_file_path):
            logger.error(f"Master sample file not found: {master_file_path}. Dataset for split '{split}' will be empty.")
            return
        if not os.path.exists(self.voxel_hdf5_path):
             logger.error(f"Voxel HDF5 file not found: {self.voxel_hdf5_path}. Cannot load voxels.")
             return # Cannot function without voxels

        try:
            logger.info(f"Loading master samples from {master_file_path} for split '{split}'...")
            t0 = time.time()
            
            # Load only necessary columns
            if MASTER_SAMPLES_FILENAME.endswith(".parquet"):
                if not PYARROW_AVAILABLE: raise ImportError("PyArrow needed to read .parquet master file.")
                cols_to_read = ['hdf5_domain_id', 'resid_str', 'raw_temp', 'target_rmsf', 'split']
                master_df = pd.read_parquet(master_file_path, columns=cols_to_read)
            else: # Assume CSV
                master_df = pd.read_csv(master_file_path, low_memory=False, usecols=['hdf5_domain_id', 'resid_str', 'raw_temp', 'target_rmsf', 'split'])

            t1 = time.time()
            logger.info(f"Loaded {len(master_df)} total samples in {t1-t0:.2f}s.")

            # Filter for the required split
            filtered_df = master_df[master_df['split'] == self.split].copy()
            
            # Convert DataFrame to list of namedtuples (more efficient for __getitem__)
            for _, row in filtered_df.iterrows():
                self.samples.append(Sample(
                    hdf5_domain_id=str(row['hdf5_domain_id']),  # Ensure string type
                    resid_str=str(row['resid_str']),            # Ensure string type
                    resid_int=int(row['resid_str']),            # Pre-convert to int
                    raw_temp=float(row['raw_temp']),            # Ensure float type
                    target_rmsf=float(row['target_rmsf']),      # Ensure float type
                    split=self.split
                ))
            
            logger.info(f"Filtered to {len(self.samples)} samples for split '{split}'.")
            
            # Load domain mapping if available (serialized during preprocessing)
            mapping_path = os.path.join(self.processed_dir, "domain_mapping.json")
            if os.path.exists(mapping_path):
                try:
                    self.domain_mapping = load_json(mapping_path)
                    logger.info(f"Loaded domain mapping with {len(self.domain_mapping)} entries")
                except Exception as e:
                    logger.warning(f"Could not load domain mapping: {e}")

            if not self.samples and len(master_df) > 0:
                 logger.warning(f"No samples found for split '{split}' in {master_file_path}.")
            elif self.samples:
                 logger.info(f"Initialized VoxelDataset for split '{split}'.")

        except Exception as e:
            logger.exception(f"Error loading or filtering master sample file {master_file_path} for split '{split}': {e}")
            self.samples = []  # Ensure samples is empty on error

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves sample metadata, loads/processes voxel data on demand,
        and returns the sample dictionary for the DataLoader.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary {'voxels': Tensor, 'scaled_temps': Tensor, 'targets': Tensor},
            or None if retrieval/voxel loading fails for this sample.
        """
        if not self.samples or idx >= len(self.samples):
            return None # Return None if index out of range or dataset is empty

        try:
            # Get sample metadata (now from namedtuple)
            sample = self.samples[idx]
            hdf5_domain_id = sample.hdf5_domain_id
            resid_str = sample.resid_str
            raw_temp = sample.raw_temp
            target_rmsf = sample.target_rmsf

            # Try to get domain data from cache first
            domain_data = self.domain_cache.get(hdf5_domain_id)
            
            # If not in cache, load and process from HDF5
            if domain_data is None:
                # Use persistent worker HDF5 handle if available
                global worker_h5_file
                use_persistent = worker_h5_file is not None
                
                if use_persistent:
                    # Use the persistent connection
                    domain_data = load_process_domain_from_handle(
                        worker_h5_file,
                        hdf5_domain_id,
                        expected_channels=self.expected_channels,
                        target_shape_chw=self.target_shape_chw
                    )
                else:
                    # Fall back to opening the file
                    domain_data = load_process_voxels_from_hdf5(
                        self.voxel_hdf5_path,
                        domain_ids=[hdf5_domain_id],
                        expected_channels=self.expected_channels,
                        target_shape_chw=self.target_shape_chw,
                        log_per_residue=False
                    ).get(hdf5_domain_id, {})
                
                # Add to cache if loading succeeded
                if domain_data:
                    self.domain_cache.add(hdf5_domain_id, domain_data)

            # Extract the specific residue's processed array
            voxel_np = None if domain_data is None else domain_data.get(resid_str)

            if voxel_np is None:
                return None # Skip this sample if voxel loading failed

            # --- Process other features ---
            # Scale temperature directly (no function call)
            scaled_temp = (raw_temp - self.temp_min) / (self.temp_max - self.temp_min)
            scaled_temp = min(max(scaled_temp, 0.0), 1.0)  # Clip to [0, 1]

            # Convert numpy array to tensor (CPU)
            voxel_tensor = torch.from_numpy(voxel_np)

            # Create tensors for temp and target
            scaled_temp_tensor = torch.tensor([scaled_temp], dtype=torch.float32) # Shape (1,)
            target_tensor = torch.tensor(target_rmsf, dtype=torch.float32) # Scalar

            return {
                'voxels': voxel_tensor,
                'scaled_temps': scaled_temp_tensor,
                'targets': target_tensor
            }

        except Exception as e:
            # Log error with index for traceability
            logger.debug(f"Error in sample index {idx}: {e}")
            return None # Return None if any step fails

    def get_cache_stats(self):
        """Return cache statistics for logging."""
        return self.domain_cache.get_stats()

def load_process_domain_from_handle(
    h5_handle,
    domain_id: str,
    expected_channels: int = 5,
    target_shape_chw: Optional[Tuple[int, int, int, int]] = (5, 21, 21, 21),
    log_per_residue: bool = False
) -> Dict[str, np.ndarray]:
    """
    Process domain data from an already-open HDF5 file handle.
    
    Args:
        h5_handle: Open h5py File handle
        domain_id: Domain ID to process
        expected_channels: Expected voxel channels
        target_shape_chw: Target shape for validation
        log_per_residue: Whether to log per-residue processing
        
    Returns:
        Dictionary mapping residue IDs to processed voxel arrays
    """
    domain_data_dict = {}
    
    try:
        if domain_id not in h5_handle:
            return {}
            
        domain_group = h5_handle[domain_id]
        residue_group = None
        
        # Find the first valid residue group
        potential_chain_keys = sorted([k for k in domain_group.keys() 
                                      if isinstance(domain_group[k], h5py.Group)])
        
        for chain_key in potential_chain_keys:
            try:
                potential_res_group = domain_group[chain_key]
                if any(key.isdigit() for key in potential_res_group.keys()):
                    residue_group = potential_res_group
                    break
            except Exception:
                continue
                
        if residue_group is None:
            return {}
            
        # Process residues in the group
        for resid_str in residue_group.keys():
            if not resid_str.isdigit():
                continue
                
            try:
                voxel_dataset = residue_group[resid_str]
                if not isinstance(voxel_dataset, h5py.Dataset):
                    continue
                    
                voxel_raw = voxel_dataset[:]
                # Process the array once
                if voxel_raw.dtype == bool:
                    processed_array = voxel_raw.astype(np.float32)
                elif np.issubdtype(voxel_raw.dtype, np.floating):
                    processed_array = voxel_raw.astype(np.float32, copy=False)
                elif np.issubdtype(voxel_raw.dtype, np.integer):
                    processed_array = voxel_raw.astype(np.float32)
                else:
                    continue
                    
                # Transpose if needed
                if processed_array.ndim == 4 and processed_array.shape[-1] == expected_channels:
                    processed_array = np.transpose(processed_array, (3, 0, 1, 2))
                
                if np.isnan(processed_array).any() or np.isinf(processed_array).any():
                    continue
                    
                domain_data_dict[resid_str] = processed_array
            except Exception:
                continue
                
        return domain_data_dict
        
    except Exception:
        return {}

# Worker initialization function to create a persistent HDF5 connection
def worker_init_fn(worker_id):
    global worker_h5_file
    # Get the HDF5 file path from the worker's dataset
    try:
        # Get the dataset from the worker's DataLoader
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            if hasattr(dataset, 'voxel_hdf5_path'):
                hdf5_path = dataset.voxel_hdf5_path
                worker_h5_file = h5py.File(hdf5_path, 'r')
                # print(f"Worker {worker_id}: opened HDF5 file {hdf5_path}")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: failed to open HDF5 file: {e}")

# Create a simpler collate function with less error handling
def simple_collate_fn(batch):
    """A simpler collate function with minimal error handling."""
    # Filter None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
        
    # Direct batch construction
    return {
        'voxels': torch.stack([item['voxels'] for item in batch]),
        'scaled_temps': torch.stack([item['scaled_temps'] for item in batch]),
        'targets': torch.stack([item['targets'] for item in batch])
    }

# --- Keep other existing functions but optimize where possible ---

# Retain load_process_voxels_from_hdf5 for compatibility but simplify
def load_process_voxels_from_hdf5(
    voxel_hdf5_path: str,
    domain_ids: List[str],
    expected_channels: int = 5,
    target_shape_chw: Optional[Tuple[int, int, int, int]] = (5, 21, 21, 21),
    log_per_residue: bool = False
) -> Dict[str, Dict[str, np.ndarray]]:
    voxel_hdf5_path = resolve_path(voxel_hdf5_path)
    log_level = logging.DEBUG if len(domain_ids) == 1 else logging.INFO
    logger.log(log_level, f"Loading voxels for {len(domain_ids)} domains")
    processed_voxels = {}
    
    try:
        with h5py.File(voxel_hdf5_path, 'r') as f_h5:
            for domain_id in domain_ids:
                domain_data = load_process_domain_from_handle(
                    f_h5, domain_id, expected_channels, target_shape_chw, log_per_residue
                )
                if domain_data:
                    processed_voxels[domain_id] = domain_data
    except Exception as e:
        logger.exception(f"Error processing HDF5: {e}")
        
    return processed_voxels

# Optimize PredictionDataset to use the same cache system
class PredictionDataset(Dataset):
    def __init__(
        self, 
        samples_to_load: List[Tuple[str, str]], 
        voxel_hdf5_path: str, 
        config: Dict[str, Any] = None,
        expected_channels: int = 5, 
        target_shape_chw: Optional[Tuple[int, int, int, int]] = (5, 21, 21, 21)
    ):
        self.voxel_hdf5_path = resolve_path(voxel_hdf5_path)
        self.expected_channels = expected_channels
        self.target_shape_chw = target_shape_chw
        self.samples = []
        self._dummy_shape = None
        
        # Initialize domain cache
        cache_size_mb = config.get('data', {}).get('domain_cache_mb', 2000) if config else 2000
        monitor_memory = config.get('data', {}).get('monitor_memory', True) if config else True
        self.domain_cache = SafeDomainCache(max_size_mb=cache_size_mb, monitor_memory=monitor_memory)
        
        logger.info(f"Initializing PredictionDataset: Loading voxels for {len(samples_to_load)} samples...")
        
        # Group samples by domain for more efficient loading
        domains_needed = defaultdict(list)
        for d, r in samples_to_load:
            domains_needed[d].append(r)
            
        loaded_count = 0
        failed_load_samples = 0
        progress = EnhancedProgressBar(len(domains_needed), desc="Prefetching Voxels (Prediction)")
        
        with h5py.File(self.voxel_hdf5_path, 'r') as f_h5:
            for i, (domain_id, resid_list) in enumerate(domains_needed.items()):
                try:
                    # Load domain data with existing handle
                    domain_data = load_process_domain_from_handle(
                        f_h5, domain_id, expected_channels, target_shape_chw, log_per_residue=False
                    )
                    
                    if domain_data:
                        # Add domain to cache
                        self.domain_cache.add(domain_id, domain_data)
                        
                        # Add valid samples to dataset
                        for resid_str in resid_list:
                            if resid_str in domain_data:
                                self.samples.append((domain_id, resid_str))
                                loaded_count += 1
                                
                                if self._dummy_shape is None and domain_data[resid_str] is not None:
                                    self._dummy_shape = domain_data[resid_str].shape
                            else:
                                failed_load_samples += 1
                    else:
                        failed_load_samples += len(resid_list)
                except Exception as e:
                    logger.warning(f"Error pre-fetching {domain_id}: {e}")
                    failed_load_samples += len(resid_list)
                    
                progress.update(i + 1)
                
        progress.finish()
        logger.info(f"PredictionDataset initialized with {loaded_count} samples and {failed_load_samples} failures")
        
        if not self.samples:
            logger.error("PredictionDataset: No valid voxel data loaded.")
            self._set_dummy_shape()
            
    def _set_dummy_shape(self):
        if self.target_shape_chw:
            self._dummy_shape = self.target_shape_chw
        else:
            self._dummy_shape = (self.expected_channels or 5, 21, 21, 21)
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx) -> Tuple[str, str, torch.Tensor]:
        if idx >= len(self.samples):
            return "ERROR", "0", self._get_dummy_voxel()
            
        domain_id, resid_str = self.samples[idx]
        
        try:
            # Try to get from cache first
            domain_data = self.domain_cache.get(domain_id)
            
            if domain_data is None:
                # Reload if needed
                with h5py.File(self.voxel_hdf5_path, 'r') as f_h5:
                    domain_data = load_process_domain_from_handle(
                        f_h5, domain_id, self.expected_channels, self.target_shape_chw
                    )
                    if domain_data:
                        self.domain_cache.add(domain_id, domain_data)
            
            voxel_np = None if domain_data is None else domain_data.get(resid_str)
            
            if voxel_np is None:
                return domain_id, resid_str, self._get_dummy_voxel()
                
            return domain_id, resid_str, torch.from_numpy(voxel_np)
        except Exception as e:
            logger.debug(f"Error retrieving item {idx}: {e}")
            return domain_id, resid_str, self._get_dummy_voxel()
            
    def _get_dummy_voxel(self) -> torch.Tensor:
        if self._dummy_shape is None:
            self._set_dummy_shape()
        return torch.zeros(self._dummy_shape, dtype=torch.float32)
        
    def get_cache_stats(self):
        """Return cache statistics for logging."""
        return self.domain_cache.get_stats()

# --- Optimized Helper Functions for Preprocessing ---

def load_aggregated_rmsf_data(aggregated_rmsf_file: str) -> pd.DataFrame:
    """Load aggregated RMSF data with type conversion optimizations."""
    rmsf_file = resolve_path(aggregated_rmsf_file)
    logger.info(f"Loading RMSF data from: {rmsf_file}")
    if not os.path.exists(rmsf_file):
        raise FileNotFoundError(f"RMSF file not found: {rmsf_file}")
    
    try:
        # Use dtype specifications during loading to avoid later conversions
        dtype_spec = {
            'domain_id': str,
            'resid': 'Int64',  # Allow NA values with pandas Int64
            'temperature_feature': float,
            'target_rmsf': float
        }
        rmsf_df = pd.read_csv(rmsf_file, dtype=dtype_spec, low_memory=False)
        logger.info(f"Loaded {len(rmsf_df)} rows with optimized types")
        return validate_aggregated_rmsf_data(rmsf_df)
    except Exception as e:
        logger.exception(f"Failed to read RMSF CSV: {e}")
        raise

def create_master_rmsf_lookup(rmsf_df: pd.DataFrame) -> Dict[Tuple[str, int], List[Tuple[float, float]]]:
    """Create and serialize RMSF lookup for reuse."""
    logger.info("Creating RMSF lookup...")
    start_time = time.time()
    required_cols = ['domain_id', 'resid', 'temperature_feature', 'target_rmsf']
    
    if not all(col in rmsf_df.columns for col in required_cols):
        raise ValueError(f"RMSF DataFrame missing required columns: {required_cols}")
    
    # Use existing numeric columns without conversion if possible
    lookup = defaultdict(list)
    grouped = rmsf_df.groupby(['domain_id', 'resid'])
    
    for name, group in grouped:
        domain_id = str(name[0])
        resid_int = int(name[1])
        temp_rmsf_pairs = list(zip(
            group['temperature_feature'].astype(float),
            group['target_rmsf'].astype(float)
        ))
        lookup[(domain_id, resid_int)] = temp_rmsf_pairs
    
    # Create base name lookup as before
    base_name_lookup = {}
    for (domain_id, resid_int), pairs in lookup.items():
        base_name = str(domain_id).split('_')[0]
        if base_name != domain_id:
            base_key = (base_name, resid_int)
            if base_key not in lookup and base_key not in base_name_lookup:
                base_name_lookup[base_key] = pairs
    
    original_keys = len(lookup)
    lookup.update(base_name_lookup)
    base_added = len(lookup) - original_keys
    
    duration = time.time() - start_time
    logger.info(f"RMSF lookup created: {len(lookup)} keys ({base_added} base names added) in {duration:.2f}s")
    
    # Serialize the lookup for future use
    try:
        processed_dir = os.path.dirname(os.path.dirname(rmsf_df.iloc[0]['domain_id'])) 
        if processed_dir:
            lookup_path = os.path.join(processed_dir, "rmsf_lookup.json")
            serializable_lookup = {}
            for (domain_id, resid_int), pairs in lookup.items():
                serializable_lookup[f"{domain_id}:{resid_int}"] = pairs
            save_json(serializable_lookup, lookup_path)
            logger.info(f"Serialized RMSF lookup to {lookup_path}")
    except Exception as e:
        logger.warning(f"Could not serialize RMSF lookup: {e}")
    
    return dict(lookup)

def create_domain_mapping(voxel_domain_keys: List[str], rmsf_domain_ids: List[str]) -> Dict[str, str]:
    """Create and serialize domain mapping for reuse."""
    logger.info("Creating domain mapping...")
    if not voxel_domain_keys:
        logger.warning("Voxel keys empty.")
        return {}
    if not rmsf_domain_ids:
        logger.warning("RMSF IDs empty.")
        return {}
    
    voxel_keys_set = set(voxel_domain_keys)
    rmsf_ids_set = set(rmsf_domain_ids)
    mapping = {}
    matches = {'exact': 0, 'base_match': 0}
    
    rmsf_base_to_full = {}
    for rid in rmsf_domain_ids:
        rmsf_base_to_full.setdefault(str(rid).split('_')[0], rid)
    
    for hdf5_key in voxel_keys_set:
        if hdf5_key in rmsf_ids_set:
            mapping[hdf5_key] = hdf5_key
            matches['exact'] += 1
            continue
        base_hdf5 = str(hdf5_key).split('_')[0]
        if base_hdf5 in rmsf_ids_set:
            mapping[hdf5_key] = base_hdf5
            matches['base_match'] += 1
        elif base_hdf5 in rmsf_base_to_full:
            mapping[hdf5_key] = rmsf_base_to_full[base_hdf5]
            matches['base_match'] += 1
    
    total_mapped = len(mapping)
    coverage = (total_mapped / len(voxel_domain_keys)) * 100 if voxel_domain_keys else 0
    logger.info(f"Domain mapping: {total_mapped}/{len(voxel_domain_keys)} keys mapped ({coverage:.1f}%)")
    logger.info(f"  Matches: Exact={matches['exact']}, Base={matches['base_match']}")
    
    # Serialize the mapping for future use
    try:
        processed_dir = os.path.dirname(os.path.dirname(voxel_domain_keys[0])) 
        if processed_dir:
            mapping_path = os.path.join(processed_dir, "domain_mapping.json")
            save_json(mapping, mapping_path)
            logger.info(f"Serialized domain mapping to {mapping_path}")
    except Exception as e:
        logger.warning(f"Could not serialize domain mapping: {e}")
    
    return mapping