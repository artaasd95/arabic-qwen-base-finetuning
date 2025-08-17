"""Common Utility Functions

This module contains common utility functions used across the Arabic Qwen
fine-tuning project.
"""

import logging
import os
import json
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import shutil
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Optional custom log format
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging level set to {log_level}")


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"Data saved to JSON: {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from JSON: {file_path}")
    return data


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]):
    """Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Data saved to YAML: {file_path}")


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    logger.info(f"Data loaded from YAML: {file_path}")
    return data


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_device() -> str:
    """Get the best available device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    
    return device


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information.
    
    Returns:
        Dictionary with GPU memory info
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    device_count = torch.cuda.device_count()
    memory_info = {"gpu_available": True, "device_count": device_count, "devices": []}
    
    for i in range(device_count):
        device_props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        memory_total = device_props.total_memory
        
        device_info = {
            "device_id": i,
            "name": device_props.name,
            "memory_total_gb": memory_total / (1024**3),
            "memory_allocated_gb": memory_allocated / (1024**3),
            "memory_reserved_gb": memory_reserved / (1024**3),
            "memory_free_gb": (memory_total - memory_reserved) / (1024**3)
        }
        memory_info["devices"].append(device_info)
    
    return memory_info


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Get hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """Create a backup of a file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same directory)
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    shutil.copy2(file_path, backup_path)
    logger.info(f"File backed up: {file_path} -> {backup_path}")
    
    return backup_path


def clean_directory(dir_path: Union[str, Path], pattern: str = "*", keep_recent: int = 0):
    """Clean directory by removing files matching pattern.
    
    Args:
        dir_path: Directory to clean
        pattern: File pattern to match (default: all files)
        keep_recent: Number of recent files to keep
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        logger.warning(f"Directory not found: {dir_path}")
        return
    
    files = list(dir_path.glob(pattern))
    files = [f for f in files if f.is_file()]
    
    if keep_recent > 0:
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        files_to_remove = files[keep_recent:]
    else:
        files_to_remove = files
    
    removed_count = 0
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            removed_count += 1
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"Cleaned directory {dir_path}: removed {removed_count} files")


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    logger.info("Configuration validation passed")
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size info
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in bytes (assuming float32)
    model_size_bytes = param_count * 4
    
    return {
        "total_parameters": param_count,
        "trainable_parameters": trainable_param_count,
        "non_trainable_parameters": param_count - trainable_param_count,
        "model_size_mb": model_size_bytes / (1024 * 1024),
        "model_size_gb": model_size_bytes / (1024 * 1024 * 1024)
    }


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Print model information.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    info = get_model_size(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Non-trainable Parameters: {info['non_trainable_parameters']:,}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB ({info['model_size_gb']:.4f} GB)")
    print()


def create_experiment_name(
    method: str,
    model_name: str,
    dataset_name: str,
    timestamp: bool = True
) -> str:
    """Create experiment name.
    
    Args:
        method: Training method
        model_name: Model name
        dataset_name: Dataset name
        timestamp: Whether to include timestamp
        
    Returns:
        Experiment name
    """
    # Clean names
    method = method.lower().replace("-", "_")
    model_name = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_name = dataset_name.split("/")[-1].replace("-", "_").lower()
    
    name_parts = [method, model_name, dataset_name]
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts.append(timestamp_str)
    
    return "_".join(name_parts)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix