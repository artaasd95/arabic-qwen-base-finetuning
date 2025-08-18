# Base Data Loader Documentation

The `BaseDataLoader` class serves as the foundation for all data loading operations in the Arabic Qwen Base Fine-tuning framework. It provides core functionality for loading, processing, and validating data across different training methodologies.

## Class Overview

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class BaseDataLoader(ABC):
    """Base class for all data loaders in the framework."""
```

## Location

**File**: `src/data/base_loader.py`

## Core Features

- **Abstract Base Class**: Defines common interface for all data loaders
- **Caching System**: Efficient data caching for repeated loads
- **Validation Framework**: Comprehensive data validation
- **Format Support**: Multiple input format handling
- **Error Handling**: Robust error management
- **Memory Management**: Efficient memory usage patterns
- **Parallel Processing**: Multi-threaded data processing

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: Any,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs
):
    """
    Initialize the base data loader.
    
    Args:
        config: Data configuration object
        tokenizer: Tokenizer for text processing
        enable_cache: Whether to enable data caching
        cache_dir: Directory for cache storage
        **kwargs: Additional arguments
    """
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | Any | Required | Data configuration object |
| `tokenizer` | Optional[Any] | None | Tokenizer for text processing |
| `enable_cache` | bool | True | Enable data caching |
| `cache_dir` | Optional[str] | None | Cache directory path |

### Abstract Methods

Subclasses must implement these methods:

```python
@abstractmethod
def load_dataset(self, split: str) -> Dataset:
    """Load dataset for specified split."""
    pass

@abstractmethod
def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single data sample."""
    pass

@abstractmethod
def validate_sample(self, sample: Dict[str, Any]) -> bool:
    """Validate a single data sample."""
    pass
```

## Core Methods

### Data Loading

#### `_load_raw_data()`

```python
def _load_raw_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load raw data from file.
    
    Args:
        file_path: Path to data file
        
    Returns:
        List of raw data samples
        
    Raises:
        FileNotFoundError: If file doesn't exist
        DataLoadingError: If file format is unsupported
    """
```

**Supported Formats:**
- JSONL (`.jsonl`, `.json`)
- CSV (`.csv`)
- Parquet (`.parquet`)
- TSV (`.tsv`)

**Example:**
```python
loader = SFTDataLoader(config)
raw_data = loader._load_raw_data("data/train.jsonl")
print(f"Loaded {len(raw_data)} samples")
```

#### `_get_file_path()`

```python
def _get_file_path(self, split: str) -> Path:
    """
    Get file path for specified split.
    
    Args:
        split: Data split name ('train', 'validation', 'test')
        
    Returns:
        Path to data file
        
    Raises:
        ValueError: If split is not supported
        FileNotFoundError: If file doesn't exist
    """
```

### Caching System

#### `_get_cache_key()`

```python
def _get_cache_key(self, file_path: Path, config_hash: str) -> str:
    """
    Generate cache key for data file.
    
    Args:
        file_path: Path to data file
        config_hash: Hash of configuration
        
    Returns:
        Unique cache key
    """
```

#### `_load_from_cache()`

```python
def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load processed data from cache.
    
    Args:
        cache_key: Cache key
        
    Returns:
        Cached data if available, None otherwise
    """
```

#### `_save_to_cache()`

```python
def _save_to_cache(self, cache_key: str, data: List[Dict[str, Any]]) -> None:
    """
    Save processed data to cache.
    
    Args:
        cache_key: Cache key
        data: Processed data to cache
    """
```

### Data Processing

#### `_process_batch()`

```python
def _process_batch(
    self, 
    batch: List[Dict[str, Any]], 
    batch_idx: int
) -> List[Dict[str, Any]]:
    """
    Process a batch of samples.
    
    Args:
        batch: Batch of raw samples
        batch_idx: Batch index for logging
        
    Returns:
        Processed batch
    """
```

#### `_parallel_process()`

```python
def _parallel_process(
    self, 
    data: List[Dict[str, Any]], 
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Process data in parallel.
    
    Args:
        data: Raw data samples
        num_workers: Number of worker processes
        
    Returns:
        Processed data samples
    """
```

### Validation

#### `_validate_data()`

```python
def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate all data samples.
    
    Args:
        data: Data samples to validate
        
    Returns:
        Valid data samples
        
    Raises:
        ValidationError: If validation fails
    """
```

#### `_validate_config()`

```python
def _validate_config(self) -> None:
    """
    Validate data configuration.
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
```

### Utilities

#### `get_dataset_info()`

```python
def get_dataset_info(self, split: str) -> Dict[str, Any]:
    """
    Get information about dataset.
    
    Args:
        split: Data split name
        
    Returns:
        Dataset information dictionary
    """
```

#### `get_sample_by_index()`

```python
def get_sample_by_index(self, split: str, index: int) -> Dict[str, Any]:
    """
    Get specific sample by index.
    
    Args:
        split: Data split name
        index: Sample index
        
    Returns:
        Data sample
    """
```

## Usage Examples

### Basic Usage

```python
from src.data.sft_loader import SFTDataLoader
from src.config.sft_config import SFTConfig

# Load configuration
config = SFTConfig.from_yaml("config/sft_config.yaml")

# Create data loader
loader = SFTDataLoader(config.data)

# Load dataset
train_dataset = loader.load_dataset("train")
print(f"Training samples: {len(train_dataset)}")

# Get dataset info
info = loader.get_dataset_info("train")
print(f"Dataset info: {info}")
```

### Custom Data Loader

```python
from src.data.base_loader import BaseDataLoader
from torch.utils.data import Dataset

class CustomDataLoader(BaseDataLoader):
    """Custom data loader implementation."""
    
    def load_dataset(self, split: str) -> Dataset:
        """Load dataset for specified split."""
        file_path = self._get_file_path(split)
        
        # Load raw data
        raw_data = self._load_raw_data(file_path)
        
        # Process data
        processed_data = []
        for sample in raw_data:
            if self.validate_sample(sample):
                processed_sample = self.process_sample(sample)
                processed_data.append(processed_sample)
        
        return CustomDataset(processed_data)
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data sample."""
        # Custom processing logic
        processed = {
            "input_text": sample["input"],
            "target_text": sample["output"]
        }
        return processed
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a single data sample."""
        required_keys = ["input", "output"]
        return all(key in sample for key in required_keys)

class CustomDataset(Dataset):
    """Custom dataset implementation."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
```

### Caching Configuration

```python
# Enable caching with custom directory
loader = SFTDataLoader(
    config.data,
    enable_cache=True,
    cache_dir="/path/to/cache"
)

# First load: processes and caches data
train_dataset = loader.load_dataset("train")

# Second load: uses cached data (much faster)
train_dataset = loader.load_dataset("train")
```

### Parallel Processing

```python
# Configure parallel processing
config.data.preprocessing_num_workers = 8

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")
```

## Configuration Integration

The base loader integrates with configuration objects:

```python
@dataclass
class BaseDataConfig:
    """Base configuration for data loading."""
    
    # File paths
    train_file: str
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Processing
    preprocessing_num_workers: int = 4
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    
    # Caching
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Validation
    strict_validation: bool = True
    skip_invalid_samples: bool = True
    
    # Memory management
    streaming: bool = False
    buffer_size: int = 1000
    max_samples_in_memory: int = 10000
```

## Error Handling

The base loader provides comprehensive error handling:

### Custom Exceptions

```python
class DataLoadingError(Exception):
    """Raised when data loading fails."""
    
    def __init__(self, message: str, file_path: str = None, suggestions: List[str] = None):
        super().__init__(message)
        self.file_path = file_path
        self.suggestions = suggestions or []

class ValidationError(Exception):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, invalid_samples: List[int] = None):
        super().__init__(message)
        self.invalid_samples = invalid_samples or []

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
```

### Error Handling Example

```python
try:
    loader = SFTDataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except DataLoadingError as e:
    print(f"Failed to load data: {e}")
    if e.file_path:
        print(f"File: {e.file_path}")
    if e.suggestions:
        print("Suggestions:")
        for suggestion in e.suggestions:
            print(f"  - {suggestion}")
except ValidationError as e:
    print(f"Validation failed: {e}")
    if e.invalid_samples:
        print(f"Invalid samples: {e.invalid_samples[:10]}...")  # Show first 10
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Performance Optimization

### Memory Management

```python
# For large datasets, use streaming
config.data.streaming = True
config.data.buffer_size = 1000
config.data.max_samples_in_memory = 5000

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")
```

### Caching Strategies

```python
# Cache processed data for faster subsequent loads
loader = SFTDataLoader(
    config.data,
    enable_cache=True,
    cache_dir="cache/processed_data"
)

# Clear cache if needed
loader.clear_cache()

# Get cache statistics
cache_stats = loader.get_cache_stats()
print(f"Cache hits: {cache_stats['hits']}")
print(f"Cache misses: {cache_stats['misses']}")
```

### Parallel Processing

```python
# Optimize for your system
import multiprocessing

num_cores = multiprocessing.cpu_count()
config.data.preprocessing_num_workers = min(num_cores - 1, 8)

loader = SFTDataLoader(config.data)
```

## Best Practices

### 1. Configuration
- Always validate configuration before loading
- Use appropriate worker counts for your system
- Enable caching for repeated experiments

### 2. Error Handling
- Implement proper exception handling
- Log errors with context information
- Provide helpful error messages

### 3. Performance
- Use parallel processing for large datasets
- Enable streaming for memory-constrained environments
- Monitor memory usage during loading

### 4. Data Quality
- Implement thorough validation
- Handle edge cases gracefully
- Log data quality metrics

## Troubleshooting

### Common Issues

#### File Not Found
```python
# Check file paths in configuration
print(f"Train file: {config.data.train_file}")
print(f"File exists: {Path(config.data.train_file).exists()}")
```

#### Memory Issues
```python
# Reduce memory usage
config.data.streaming = True
config.data.max_samples_in_memory = 1000
config.data.preprocessing_num_workers = 2
```

#### Slow Loading
```python
# Enable caching and parallel processing
config.data.enable_cache = True
config.data.preprocessing_num_workers = 8
```

#### Validation Errors
```python
# Skip invalid samples instead of failing
config.data.skip_invalid_samples = True
config.data.strict_validation = False
```

## See Also

- [SFT Data Loader](sft_loader.md)
- [Preference Data Loader](preference_loader.md)
- [Data Processors](processors/index.md)
- [Data Utilities](utils/index.md)
- [Configuration Documentation](../config/index.md)