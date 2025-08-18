# Environment Configuration Documentation

The `EnvConfig` class provides a centralized system for managing environment variables in the Arabic Qwen Base Fine-tuning framework. It handles automatic detection, type conversion, and validation of environment variables.

## Class Overview

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path

@dataclass
class EnvConfig:
    """Environment variable configuration management."""
```

## Location

**File**: `src/config/env_config.py`

## Core Features

- **Automatic Detection**: Scan environment for Arabic Qwen variables
- **Type Conversion**: Convert string environment variables to appropriate Python types
- **Validation**: Validate environment variable values
- **Default Handling**: Provide sensible defaults when variables are not set
- **Prefix Management**: Consistent naming convention enforcement

## Environment Variable Naming Convention

All environment variables follow the pattern:
```
ARABIC_QWEN_<SECTION>_<PARAMETER>
```

### Examples
- `ARABIC_QWEN_MODEL_NAME` → `model.name`
- `ARABIC_QWEN_TRAINING_LEARNING_RATE` → `training.learning_rate`
- `ARABIC_QWEN_DATA_TRAIN_FILE` → `data.train_file`
- `ARABIC_QWEN_OPTIMIZATION_USE_LORA` → `optimization.use_lora`

## Environment Variable Categories

### Model Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_MODEL_NAME` | str | "Qwen/Qwen-7B" | Model name or path |
| `ARABIC_QWEN_MODEL_TOKENIZER_NAME` | str | None | Tokenizer name (defaults to model name) |
| `ARABIC_QWEN_MODEL_MAX_LENGTH` | int | 2048 | Maximum sequence length |
| `ARABIC_QWEN_MODEL_DEVICE` | str | "auto" | Device specification |
| `ARABIC_QWEN_MODEL_TORCH_DTYPE` | str | "auto" | PyTorch data type |
| `ARABIC_QWEN_MODEL_TRUST_REMOTE_CODE` | bool | false | Trust remote code execution |

### Training Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_TRAINING_LEARNING_RATE` | float | 5e-5 | Learning rate |
| `ARABIC_QWEN_TRAINING_BATCH_SIZE` | int | 4 | Training batch size |
| `ARABIC_QWEN_TRAINING_NUM_EPOCHS` | int | 3 | Number of training epochs |
| `ARABIC_QWEN_TRAINING_WARMUP_STEPS` | int | 100 | Warmup steps |
| `ARABIC_QWEN_TRAINING_SAVE_STEPS` | int | 500 | Save checkpoint every N steps |
| `ARABIC_QWEN_TRAINING_EVAL_STEPS` | int | 500 | Evaluation frequency |
| `ARABIC_QWEN_TRAINING_GRADIENT_ACCUMULATION_STEPS` | int | 1 | Gradient accumulation |
| `ARABIC_QWEN_TRAINING_MAX_GRAD_NORM` | float | 1.0 | Gradient clipping norm |

### Data Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_DATA_TRAIN_FILE` | str | "data/train.jsonl" | Training data file |
| `ARABIC_QWEN_DATA_VALIDATION_FILE` | str | "data/validation.jsonl" | Validation data file |
| `ARABIC_QWEN_DATA_TEST_FILE` | str | None | Test data file |
| `ARABIC_QWEN_DATA_MAX_SAMPLES` | int | None | Maximum samples to use |
| `ARABIC_QWEN_DATA_PREPROCESSING_NUM_WORKERS` | int | 4 | Data preprocessing workers |
| `ARABIC_QWEN_DATA_DATALOADER_NUM_WORKERS` | int | 0 | DataLoader workers |

### Optimization Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_OPTIMIZATION_USE_LORA` | bool | false | Enable LoRA |
| `ARABIC_QWEN_OPTIMIZATION_LORA_RANK` | int | 16 | LoRA rank |
| `ARABIC_QWEN_OPTIMIZATION_LORA_ALPHA` | int | 32 | LoRA alpha |
| `ARABIC_QWEN_OPTIMIZATION_LORA_DROPOUT` | float | 0.1 | LoRA dropout |
| `ARABIC_QWEN_OPTIMIZATION_USE_QUANTIZATION` | bool | false | Enable quantization |
| `ARABIC_QWEN_OPTIMIZATION_QUANTIZATION_BITS` | int | 4 | Quantization bits |
| `ARABIC_QWEN_OPTIMIZATION_USE_GRADIENT_CHECKPOINTING` | bool | false | Gradient checkpointing |

### Preference Optimization Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_DPO_BETA` | float | 0.1 | DPO beta parameter |
| `ARABIC_QWEN_DPO_REFERENCE_FREE` | bool | false | Reference-free DPO |
| `ARABIC_QWEN_KTO_BETA` | float | 0.1 | KTO beta parameter |
| `ARABIC_QWEN_KTO_DESIRABLE_WEIGHT` | float | 1.0 | KTO desirable weight |
| `ARABIC_QWEN_KTO_UNDESIRABLE_WEIGHT` | float | 1.0 | KTO undesirable weight |
| `ARABIC_QWEN_IPO_BETA` | float | 0.1 | IPO beta parameter |
| `ARABIC_QWEN_CPO_BETA` | float | 0.1 | CPO beta parameter |
| `ARABIC_QWEN_CPO_ALPHA` | float | 1.0 | CPO alpha parameter |

### Logging Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_LOGGING_LOG_LEVEL` | str | "INFO" | Logging level |
| `ARABIC_QWEN_LOGGING_LOG_FILE` | str | None | Log file path |
| `ARABIC_QWEN_LOGGING_USE_WANDB` | bool | false | Enable Weights & Biases |
| `ARABIC_QWEN_LOGGING_WANDB_PROJECT` | str | "arabic-qwen-finetuning" | W&B project name |
| `ARABIC_QWEN_LOGGING_WANDB_ENTITY` | str | None | W&B entity |
| `ARABIC_QWEN_LOGGING_USE_TENSORBOARD` | bool | false | Enable TensorBoard |
| `ARABIC_QWEN_LOGGING_TENSORBOARD_LOG_DIR` | str | "logs/tensorboard" | TensorBoard log directory |

### Path Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ARABIC_QWEN_PATHS_OUTPUT_DIR` | str | "output" | Output directory |
| `ARABIC_QWEN_PATHS_CACHE_DIR` | str | "cache" | Cache directory |
| `ARABIC_QWEN_PATHS_CHECKPOINT_DIR` | str | "checkpoints" | Checkpoint directory |
| `ARABIC_QWEN_PATHS_LOG_DIR` | str | "logs" | Log directory |
| `ARABIC_QWEN_PATHS_DATA_DIR` | str | "data" | Data directory |
| `ARABIC_QWEN_PATHS_CONFIG_DIR` | str | "config" | Configuration directory |

## Class Methods

### `get_env_var(key: str, default: Any = None, var_type: Type = str) -> Any`

Retrieve and convert an environment variable.

**Parameters**:
- `key` (str): Environment variable name
- `default` (Any): Default value if variable is not set
- `var_type` (Type): Target type for conversion

**Returns**:
- Converted environment variable value or default

**Example**:
```python
from src.config.env_config import EnvConfig

# Get string value
model_name = EnvConfig.get_env_var("ARABIC_QWEN_MODEL_NAME", "Qwen/Qwen-7B")

# Get integer value
batch_size = EnvConfig.get_env_var("ARABIC_QWEN_BATCH_SIZE", 4, int)

# Get boolean value
use_lora = EnvConfig.get_env_var("ARABIC_QWEN_USE_LORA", False, bool)

# Get float value
learning_rate = EnvConfig.get_env_var("ARABIC_QWEN_LEARNING_RATE", 5e-5, float)
```

### `get_all_env_vars() -> Dict[str, str]`

Retrieve all Arabic Qwen environment variables.

**Returns**:
- Dictionary of all environment variables with the Arabic Qwen prefix

**Example**:
```python
env_vars = EnvConfig.get_all_env_vars()
for key, value in env_vars.items():
    print(f"{key}: {value}")
```

### `validate_env_var(key: str, value: Any, constraints: Dict[str, Any]) -> bool`

Validate an environment variable value against constraints.

**Parameters**:
- `key` (str): Environment variable name
- `value` (Any): Value to validate
- `constraints` (Dict[str, Any]): Validation constraints

**Returns**:
- True if validation passes

**Raises**:
- `ValueError`: If validation fails

**Example**:
```python
constraints = {
    "min_value": 1e-6,
    "max_value": 1e-2,
    "type": float
}

EnvConfig.validate_env_var(
    "ARABIC_QWEN_LEARNING_RATE", 
    5e-5, 
    constraints
)
```

### `set_env_var(key: str, value: Any) -> None`

Set an environment variable with proper type conversion.

**Parameters**:
- `key` (str): Environment variable name
- `value` (Any): Value to set

**Example**:
```python
EnvConfig.set_env_var("ARABIC_QWEN_BATCH_SIZE", 8)
EnvConfig.set_env_var("ARABIC_QWEN_USE_LORA", True)
```

## Type Conversion

### Supported Types

The environment configuration system supports automatic conversion for:

#### String (`str`)
```python
# Environment: ARABIC_QWEN_MODEL_NAME="Qwen/Qwen-7B"
model_name = EnvConfig.get_env_var("ARABIC_QWEN_MODEL_NAME", var_type=str)
# Result: "Qwen/Qwen-7B"
```

#### Integer (`int`)
```python
# Environment: ARABIC_QWEN_BATCH_SIZE="8"
batch_size = EnvConfig.get_env_var("ARABIC_QWEN_BATCH_SIZE", var_type=int)
# Result: 8
```

#### Float (`float`)
```python
# Environment: ARABIC_QWEN_LEARNING_RATE="5e-5"
learning_rate = EnvConfig.get_env_var("ARABIC_QWEN_LEARNING_RATE", var_type=float)
# Result: 5e-5
```

#### Boolean (`bool`)
```python
# Environment: ARABIC_QWEN_USE_LORA="true"
use_lora = EnvConfig.get_env_var("ARABIC_QWEN_USE_LORA", var_type=bool)
# Result: True

# Supported boolean values (case-insensitive):
# True: "true", "1", "yes", "on"
# False: "false", "0", "no", "off"
```

#### List (`List[str]`)
```python
# Environment: ARABIC_QWEN_DEVICES="cuda:0,cuda:1,cuda:2"
devices = EnvConfig.get_env_var("ARABIC_QWEN_DEVICES", var_type=list)
# Result: ["cuda:0", "cuda:1", "cuda:2"]
```

#### Path (`Path`)
```python
# Environment: ARABIC_QWEN_DATA_DIR="/path/to/data"
data_dir = EnvConfig.get_env_var("ARABIC_QWEN_DATA_DIR", var_type=Path)
# Result: Path("/path/to/data")
```

### Custom Type Conversion

```python
from enum import Enum
from typing import Type, Any

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

def convert_log_level(value: str) -> LogLevel:
    """Convert string to LogLevel enum."""
    try:
        return LogLevel(value.upper())
    except ValueError:
        raise ValueError(f"Invalid log level: {value}")

# Usage
log_level_str = EnvConfig.get_env_var("ARABIC_QWEN_LOG_LEVEL", "INFO")
log_level = convert_log_level(log_level_str)
```

## Environment File Support

### `.env` File Format

```bash
# .env file
# Model Configuration
ARABIC_QWEN_MODEL_NAME="Qwen/Qwen-7B"
ARABIC_QWEN_MODEL_MAX_LENGTH=2048
ARABIC_QWEN_MODEL_DEVICE="auto"

# Training Configuration
ARABIC_QWEN_TRAINING_LEARNING_RATE=5e-5
ARABIC_QWEN_TRAINING_BATCH_SIZE=4
ARABIC_QWEN_TRAINING_NUM_EPOCHS=3

# Optimization Configuration
ARABIC_QWEN_OPTIMIZATION_USE_LORA=true
ARABIC_QWEN_OPTIMIZATION_LORA_RANK=16
ARABIC_QWEN_OPTIMIZATION_USE_QUANTIZATION=false

# Logging Configuration
ARABIC_QWEN_LOGGING_LOG_LEVEL="INFO"
ARABIC_QWEN_LOGGING_USE_WANDB=false

# Path Configuration
ARABIC_QWEN_PATHS_OUTPUT_DIR="output"
ARABIC_QWEN_PATHS_CACHE_DIR="cache"
```

### Loading Environment Files

```python
import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_file(env_file: str = ".env") -> None:
    """Load environment variables from file."""
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"Environment file {env_file} not found")

# Load environment file
load_env_file(".env")

# Now environment variables are available
from src.config.sft_config import SFTConfig
config = SFTConfig.from_yaml("config/sft_config.yaml")
```

## Validation System

### Built-in Validators

```python
class EnvValidators:
    """Built-in environment variable validators."""
    
    @staticmethod
    def validate_learning_rate(value: float) -> bool:
        """Validate learning rate range."""
        return 1e-6 <= value <= 1e-2
    
    @staticmethod
    def validate_batch_size(value: int) -> bool:
        """Validate batch size."""
        return value > 0 and value <= 1024
    
    @staticmethod
    def validate_model_name(value: str) -> bool:
        """Validate model name format."""
        # Check if it's a HuggingFace model or local path
        return bool(value and (value.count('/') == 1 or Path(value).exists()))
    
    @staticmethod
    def validate_device(value: str) -> bool:
        """Validate device specification."""
        valid_devices = ["auto", "cpu"] + [f"cuda:{i}" for i in range(8)]
        return value in valid_devices
```

### Custom Validation

```python
from typing import Callable, Any

def create_range_validator(min_val: float, max_val: float) -> Callable[[Any], bool]:
    """Create a range validator function."""
    def validator(value: float) -> bool:
        return min_val <= value <= max_val
    return validator

# Usage
learning_rate_validator = create_range_validator(1e-6, 1e-2)
batch_size_validator = create_range_validator(1, 1024)

# Apply validation
learning_rate = EnvConfig.get_env_var("ARABIC_QWEN_LEARNING_RATE", 5e-5, float)
if not learning_rate_validator(learning_rate):
    raise ValueError(f"Invalid learning rate: {learning_rate}")
```

## Environment Variable Precedence

The system follows this precedence order (highest to lowest):

1. **Explicitly set environment variables**
2. **Environment file (`.env`)**
3. **System environment variables**
4. **Configuration file values**
5. **Default values**

### Example

```python
# 1. System environment
os.environ["ARABIC_QWEN_LEARNING_RATE"] = "1e-4"

# 2. .env file contains:
# ARABIC_QWEN_LEARNING_RATE=2e-4

# 3. config.yaml contains:
# training:
#   learning_rate: 3e-4

# 4. Default value: 5e-5

# Result: 1e-4 (system environment wins)
config = SFTConfig.from_yaml("config.yaml")
print(config.training.learning_rate)  # 1e-4
```

## Development and Testing

### Environment Variable Testing

```python
import unittest
from unittest.mock import patch
from src.config.env_config import EnvConfig

class TestEnvConfig(unittest.TestCase):
    
    @patch.dict('os.environ', {'ARABIC_QWEN_BATCH_SIZE': '8'})
    def test_int_conversion(self):
        """Test integer environment variable conversion."""
        batch_size = EnvConfig.get_env_var("ARABIC_QWEN_BATCH_SIZE", 4, int)
        self.assertEqual(batch_size, 8)
    
    @patch.dict('os.environ', {'ARABIC_QWEN_USE_LORA': 'true'})
    def test_bool_conversion(self):
        """Test boolean environment variable conversion."""
        use_lora = EnvConfig.get_env_var("ARABIC_QWEN_USE_LORA", False, bool)
        self.assertTrue(use_lora)
    
    def test_default_value(self):
        """Test default value when environment variable is not set."""
        value = EnvConfig.get_env_var("ARABIC_QWEN_NONEXISTENT", "default")
        self.assertEqual(value, "default")
    
    def test_validation_error(self):
        """Test validation error handling."""
        with self.assertRaises(ValueError):
            EnvConfig.validate_env_var(
                "ARABIC_QWEN_LEARNING_RATE",
                -1.0,
                {"min_value": 0.0}
            )
```

### Mock Environment Variables

```python
from unittest.mock import patch

# Mock environment variables for testing
test_env = {
    "ARABIC_QWEN_MODEL_NAME": "Qwen/Qwen-1_8B",
    "ARABIC_QWEN_BATCH_SIZE": "2",
    "ARABIC_QWEN_USE_LORA": "true",
    "ARABIC_QWEN_LEARNING_RATE": "1e-4"
}

with patch.dict('os.environ', test_env):
    config = SFTConfig.from_yaml("config/test_config.yaml")
    # Configuration will use mocked environment variables
```

## Best Practices

### 1. Environment Variable Organization

```bash
# Group related variables together
# Model Configuration
ARABIC_QWEN_MODEL_NAME="Qwen/Qwen-7B"
ARABIC_QWEN_MODEL_MAX_LENGTH=2048

# Training Configuration
ARABIC_QWEN_TRAINING_LEARNING_RATE=5e-5
ARABIC_QWEN_TRAINING_BATCH_SIZE=4

# Use comments to document variables
# Learning rate for the optimizer (range: 1e-6 to 1e-2)
ARABIC_QWEN_TRAINING_LEARNING_RATE=5e-5
```

### 2. Sensitive Information

```bash
# Keep sensitive information in environment variables
ARABIC_QWEN_WANDB_API_KEY="your-api-key-here"
ARABIC_QWEN_HF_TOKEN="your-huggingface-token"

# Never commit these to version control
# Add to .gitignore:
# .env
# .env.local
# .env.production
```

### 3. Environment-Specific Configuration

```bash
# .env.development
ARABIC_QWEN_MODEL_NAME="Qwen/Qwen-1_8B"  # Smaller model for dev
ARABIC_QWEN_BATCH_SIZE=2
ARABIC_QWEN_LOG_LEVEL="DEBUG"

# .env.production
ARABIC_QWEN_MODEL_NAME="Qwen/Qwen-7B"   # Full model for production
ARABIC_QWEN_BATCH_SIZE=8
ARABIC_QWEN_LOG_LEVEL="INFO"
```

### 4. Documentation

```python
# Document environment variables in code
class EnvVars:
    """Environment variable definitions."""
    
    # Model configuration
    MODEL_NAME = "ARABIC_QWEN_MODEL_NAME"  # Model name or path
    MODEL_MAX_LENGTH = "ARABIC_QWEN_MODEL_MAX_LENGTH"  # Max sequence length
    
    # Training configuration
    LEARNING_RATE = "ARABIC_QWEN_TRAINING_LEARNING_RATE"  # Learning rate (1e-6 to 1e-2)
    BATCH_SIZE = "ARABIC_QWEN_TRAINING_BATCH_SIZE"  # Batch size (1 to 1024)
```

## Troubleshooting

### Common Issues

1. **Type Conversion Errors**:
   ```python
   # Error: Cannot convert "invalid" to int
   try:
       batch_size = EnvConfig.get_env_var("ARABIC_QWEN_BATCH_SIZE", 4, int)
   except ValueError as e:
       print(f"Invalid batch size: {e}")
       batch_size = 4  # Use default
   ```

2. **Missing Environment Variables**:
   ```python
   # Check if environment variable exists
   import os
   if "ARABIC_QWEN_MODEL_NAME" not in os.environ:
       print("Warning: ARABIC_QWEN_MODEL_NAME not set, using default")
   ```

3. **Boolean Conversion Issues**:
   ```python
   # Valid boolean values
   valid_true = ["true", "1", "yes", "on"]
   valid_false = ["false", "0", "no", "off"]
   
   # Case-insensitive conversion
   use_lora = EnvConfig.get_env_var("ARABIC_QWEN_USE_LORA", False, bool)
   ```

### Debug Mode

```python
import logging
from src.config.env_config import EnvConfig

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Get all environment variables for debugging
env_vars = EnvConfig.get_all_env_vars()
for key, value in env_vars.items():
    print(f"DEBUG: {key} = {value}")
```

## See Also

- [Base Configuration](base_config.md)
- [SFT Configuration](sft_config.md)
- [Configuration Examples](../../config/)
- [Environment Setup Guide](../../README.md#environment-setup)