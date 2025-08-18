# BaseConfig Class Documentation

The `BaseConfig` class serves as the foundation for all configuration classes in the Arabic Qwen Base Fine-tuning framework. It provides core functionality for loading, saving, validating, and managing configuration data.

## Class Overview

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar
from pathlib import Path

@dataclass
class BaseConfig:
    """Base configuration class with YAML support and validation."""
```

## Location

**File**: `src/config/base_config.py`

## Core Features

- **YAML Integration**: Load and save configurations from/to YAML files
- **Environment Variable Support**: Override configuration values with environment variables
- **Validation**: Built-in validation with extensible validation methods
- **Type Safety**: Full type hints and dataclass integration
- **Nested Configuration**: Support for complex nested configuration structures

## Class Methods

### `from_yaml(cls, file_path: str | Path) -> Self`

Load configuration from a YAML file with environment variable overrides.

**Parameters**:
- `file_path` (str | Path): Path to the YAML configuration file

**Returns**:
- Instance of the configuration class

**Raises**:
- `FileNotFoundError`: If the configuration file doesn't exist
- `yaml.YAMLError`: If the YAML file is malformed
- `ValueError`: If configuration validation fails

**Example**:
```python
from src.config.sft_config import SFTConfig

# Load configuration from YAML
config = SFTConfig.from_yaml("config/sft_config.yaml")
print(f"Model: {config.model.name}")
```

### `to_yaml(self, file_path: str | Path) -> None`

Save configuration to a YAML file.

**Parameters**:
- `file_path` (str | Path): Path where to save the configuration

**Raises**:
- `PermissionError`: If unable to write to the specified path
- `yaml.YAMLError`: If serialization fails

**Example**:
```python
config = SFTConfig()
config.to_yaml("output/my_config.yaml")
```

### `validate(self) -> None`

Validate the configuration. Override in subclasses for custom validation.

**Raises**:
- `ValueError`: If validation fails

**Example**:
```python
class CustomConfig(BaseConfig):
    learning_rate: float = 1e-4
    
    def validate(self) -> None:
        super().validate()
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
```

### `to_dict(self) -> Dict[str, Any]`

Convert configuration to a dictionary.

**Returns**:
- Dictionary representation of the configuration

**Example**:
```python
config = SFTConfig()
config_dict = config.to_dict()
print(config_dict)
```

### `from_dict(cls, data: Dict[str, Any]) -> Self`

Create configuration instance from a dictionary.

**Parameters**:
- `data` (Dict[str, Any]): Dictionary containing configuration data

**Returns**:
- Instance of the configuration class

**Example**:
```python
data = {
    "model": {"name": "Qwen/Qwen-7B"},
    "training": {"learning_rate": 5e-5}
}
config = SFTConfig.from_dict(data)
```

### `update(self, **kwargs) -> Self`

Update configuration with new values.

**Parameters**:
- `**kwargs`: Key-value pairs to update

**Returns**:
- Updated configuration instance

**Example**:
```python
config = SFTConfig()
updated_config = config.update(
    model__name="Qwen/Qwen-1_8B",
    training__learning_rate=1e-4
)
```

## Environment Variable Integration

The `BaseConfig` class automatically integrates with environment variables through the `EnvConfig` system.

### Environment Variable Naming Convention

Environment variables follow the pattern:
```
ARABIC_QWEN_<SECTION>_<PARAMETER>
```

### Supported Types

| Python Type | Environment Variable Format | Example |
|-------------|----------------------------|----------|
| `str` | Direct string | `ARABIC_QWEN_MODEL_NAME="Qwen/Qwen-7B"` |
| `int` | Integer string | `ARABIC_QWEN_BATCH_SIZE="8"` |
| `float` | Float string | `ARABIC_QWEN_LEARNING_RATE="5e-5"` |
| `bool` | "true"/"false" (case-insensitive) | `ARABIC_QWEN_USE_LORA="true"` |
| `List[str]` | Comma-separated | `ARABIC_QWEN_DEVICES="cuda:0,cuda:1"` |
| `Path` | Path string | `ARABIC_QWEN_DATA_DIR="/path/to/data"` |

### Environment Variable Override Example

```python
import os
from src.config.sft_config import SFTConfig

# Set environment variables
os.environ["ARABIC_QWEN_MODEL_NAME"] = "Qwen/Qwen-1_8B"
os.environ["ARABIC_QWEN_LEARNING_RATE"] = "1e-4"
os.environ["ARABIC_QWEN_USE_LORA"] = "true"

# Load configuration (environment variables override YAML)
config = SFTConfig.from_yaml("config/sft_config.yaml")

print(f"Model: {config.model.name}")  # Qwen/Qwen-1_8B
print(f"LR: {config.training.learning_rate}")  # 1e-4
print(f"LoRA: {config.optimization.use_lora}")  # True
```

## Validation System

### Built-in Validation

The base class provides basic validation:
- Type checking for all fields
- Required field validation
- Basic range checking for numeric values

### Custom Validation

Subclasses can implement custom validation by overriding the `validate()` method:

```python
@dataclass
class TrainingConfig(BaseConfig):
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    
    def validate(self) -> None:
        """Validate training configuration."""
        super().validate()
        
        # Learning rate validation
        if not (1e-6 <= self.learning_rate <= 1e-2):
            raise ValueError(
                f"Learning rate {self.learning_rate} must be between 1e-6 and 1e-2"
            )
        
        # Batch size validation
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Epochs validation
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
```

### Validation Decorators

For complex validation, you can use validation decorators:

```python
from functools import wraps
from typing import Callable

def validate_range(min_val: float, max_val: float):
    """Decorator for range validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, value):
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Value {value} must be between {min_val} and {max_val}"
                )
            return func(self, value)
        return wrapper
    return decorator

@dataclass
class ModelConfig(BaseConfig):
    learning_rate: float = 5e-5
    
    @validate_range(1e-6, 1e-2)
    def set_learning_rate(self, value: float):
        self.learning_rate = value
```

## YAML Integration

### YAML File Format

Configuration files use standard YAML format with nested structures:

```yaml
# config/example_config.yaml
model:
  name: "Qwen/Qwen-7B"
  tokenizer_name: "Qwen/Qwen-7B"
  max_length: 2048
  device: "auto"

training:
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 3
  warmup_steps: 100
  save_steps: 500

data:
  train_file: "data/train.jsonl"
  validation_file: "data/validation.jsonl"
  max_samples: null

optimization:
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  use_quantization: false

logging:
  log_level: "INFO"
  log_file: "logs/training.log"
  use_wandb: false
```

### YAML Loading Process

1. **File Reading**: Load YAML content from file
2. **Environment Override**: Apply environment variable overrides
3. **Type Conversion**: Convert values to appropriate Python types
4. **Object Creation**: Create configuration object from processed data
5. **Validation**: Run validation on the created object

### YAML Saving Process

1. **Serialization**: Convert configuration object to dictionary
2. **Type Conversion**: Convert Python types to YAML-compatible formats
3. **File Writing**: Write YAML content to file with proper formatting

## Error Handling

### Common Exceptions

#### `ConfigurationError`
```python
class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass
```

#### `ValidationError`
```python
class ValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    pass
```

#### `FileNotFoundError`
```python
# Built-in exception for missing configuration files
try:
    config = SFTConfig.from_yaml("nonexistent.yaml")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
```

### Error Handling Best Practices

```python
from src.config.sft_config import SFTConfig
from src.config.base_config import ConfigurationError

def load_config_safely(config_path: str) -> SFTConfig:
    """Safely load configuration with proper error handling."""
    try:
        config = SFTConfig.from_yaml(config_path)
        return config
    except FileNotFoundError:
        print(f"Configuration file '{config_path}' not found")
        print("Using default configuration")
        return SFTConfig()
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}")
        raise ConfigurationError(f"Failed to load configuration: {e}")
```

## Advanced Usage

### Configuration Inheritance

```python
@dataclass
class BaseTrainingConfig(BaseConfig):
    """Base training configuration."""
    learning_rate: float = 5e-5
    batch_size: int = 4
    
@dataclass
class SFTTrainingConfig(BaseTrainingConfig):
    """SFT-specific training configuration."""
    max_length: int = 2048
    
@dataclass
class DPOTrainingConfig(BaseTrainingConfig):
    """DPO-specific training configuration."""
    beta: float = 0.1
    reference_free: bool = False
```

### Configuration Composition

```python
@dataclass
class ModelConfig(BaseConfig):
    name: str = "Qwen/Qwen-7B"
    max_length: int = 2048

@dataclass
class TrainingConfig(BaseConfig):
    learning_rate: float = 5e-5
    batch_size: int = 4

@dataclass
class FullConfig(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def validate(self) -> None:
        super().validate()
        self.model.validate()
        self.training.validate()
```

### Dynamic Configuration

```python
def create_config_from_args(args) -> BaseConfig:
    """Create configuration from command line arguments."""
    config_data = {
        "model": {
            "name": args.model_name,
            "max_length": args.max_length
        },
        "training": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size
        }
    }
    
    return SFTConfig.from_dict(config_data)
```

## Performance Considerations

### Configuration Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_cached_config(config_path: str) -> SFTConfig:
    """Load configuration with caching."""
    return SFTConfig.from_yaml(config_path)
```

### Lazy Loading

```python
@dataclass
class LazyConfig(BaseConfig):
    """Configuration with lazy loading of expensive resources."""
    _model_config: Optional[ModelConfig] = None
    
    @property
    def model_config(self) -> ModelConfig:
        if self._model_config is None:
            self._model_config = ModelConfig.from_yaml("model_config.yaml")
        return self._model_config
```

## Testing

### Unit Testing Configuration

```python
import unittest
from unittest.mock import patch
from src.config.sft_config import SFTConfig

class TestSFTConfig(unittest.TestCase):
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SFTConfig()
        self.assertIsInstance(config, SFTConfig)
        self.assertEqual(config.model.name, "Qwen/Qwen-7B")
    
    def test_yaml_loading(self):
        """Test YAML configuration loading."""
        config = SFTConfig.from_yaml("tests/fixtures/test_config.yaml")
        self.assertEqual(config.training.learning_rate, 1e-4)
    
    @patch.dict('os.environ', {'ARABIC_QWEN_LEARNING_RATE': '2e-4'})
    def test_env_override(self):
        """Test environment variable override."""
        config = SFTConfig.from_yaml("tests/fixtures/test_config.yaml")
        self.assertEqual(config.training.learning_rate, 2e-4)
    
    def test_validation_error(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError):
            config = SFTConfig()
            config.training.learning_rate = -1.0
            config.validate()
```

### Integration Testing

```python
def test_config_integration():
    """Test configuration integration with training pipeline."""
    config = SFTConfig.from_yaml("config/test_config.yaml")
    
    # Test that configuration works with actual training components
    from src.training.sft_trainer import SFTTrainer
    trainer = SFTTrainer(config)
    
    assert trainer.config == config
    assert trainer.model_name == config.model.name
```

## See Also

- [Environment Configuration](env_config.md)
- [SFT Configuration](sft_config.md)
- [Training Documentation](../training/index.md)
- [Configuration Examples](../../config/)