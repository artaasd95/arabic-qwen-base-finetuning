# Configuration Module Documentation

The configuration module provides a robust, type-safe configuration management system for the Arabic Qwen Base Fine-tuning framework. It supports YAML-based configuration files, environment variable overrides, and validation.

## Overview

The configuration system is built around a hierarchical structure with a base configuration class and specialized configurations for different training methods.

## Module Structure

```
config/
├── __init__.py          # Module initialization and exports
├── base_config.py       # Base configuration class
├── env_config.py        # Environment variable configuration
├── sft_config.py        # Supervised Fine-Tuning configuration
├── dpo_config.py        # Direct Preference Optimization configuration
├── kto_config.py        # Kahneman-Tversky Optimization configuration
├── ipo_config.py        # Identity Preference Optimization configuration
└── cpo_config.py        # Conservative Preference Optimization configuration
```

## Core Components

### [BaseConfig](base_config.md)
The foundation class that provides common configuration functionality:
- YAML file loading and saving
- Environment variable integration
- Configuration validation
- Nested configuration support

### [EnvConfig](env_config.md)
Environment variable configuration management:
- Automatic environment variable detection
- Type conversion and validation
- Default value handling

### Training Method Configurations

| Configuration | Purpose | Documentation |
|---------------|---------|---------------|
| [SFTConfig](sft_config.md) | Supervised Fine-Tuning | Complete SFT configuration options |
| [DPOConfig](dpo_config.md) | Direct Preference Optimization | DPO-specific parameters |
| [KTOConfig](kto_config.md) | Kahneman-Tversky Optimization | KTO configuration and parameters |
| [IPOConfig](ipo_config.md) | Identity Preference Optimization | IPO-specific settings |
| [CPOConfig](cpo_config.md) | Conservative Preference Optimization | CPO configuration options |

## Quick Start

### Loading Configuration from YAML

```python
from src.config.sft_config import SFTConfig

# Load from YAML file
config = SFTConfig.from_yaml("config/sft_config.yaml")

# Access configuration values
print(f"Model name: {config.model.name}")
print(f"Learning rate: {config.training.learning_rate}")
```

### Environment Variable Override

```python
import os
from src.config.sft_config import SFTConfig

# Set environment variable
os.environ["ARABIC_QWEN_LEARNING_RATE"] = "1e-4"

# Load configuration (environment variable will override YAML)
config = SFTConfig.from_yaml("config/sft_config.yaml")
print(f"Learning rate: {config.training.learning_rate}")  # Will be 1e-4
```

### Creating Configuration Programmatically

```python
from src.config.sft_config import SFTConfig, ModelConfig, TrainingConfig

# Create configuration objects
model_config = ModelConfig(
    name="Qwen/Qwen-7B",
    tokenizer_name="Qwen/Qwen-7B",
    max_length=2048
)

training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    num_epochs=3
)

# Create main configuration
config = SFTConfig(
    model=model_config,
    training=training_config
)

# Save to file
config.to_yaml("my_config.yaml")
```

## Configuration Hierarchy

```
BaseConfig
├── ModelConfig
│   ├── name: str
│   ├── tokenizer_name: str
│   ├── max_length: int
│   └── device: str
├── TrainingConfig
│   ├── learning_rate: float
│   ├── batch_size: int
│   ├── num_epochs: int
│   ├── warmup_steps: int
│   └── save_steps: int
├── DataConfig
│   ├── train_file: str
│   ├── validation_file: str
│   ├── test_file: str
│   └── max_samples: int
├── OptimizationConfig
│   ├── use_lora: bool
│   ├── lora_rank: int
│   ├── use_quantization: bool
│   └── quantization_bits: int
└── LoggingConfig
    ├── log_level: str
    ├── log_file: str
    ├── use_wandb: bool
    └── wandb_project: str
```

## Environment Variable Mapping

Environment variables follow the pattern `ARABIC_QWEN_<SECTION>_<PARAMETER>`:

| Environment Variable | Configuration Path | Type | Description |
|---------------------|-------------------|------|-------------|
| `ARABIC_QWEN_MODEL_NAME` | `model.name` | str | Model name or path |
| `ARABIC_QWEN_LEARNING_RATE` | `training.learning_rate` | float | Learning rate |
| `ARABIC_QWEN_BATCH_SIZE` | `training.batch_size` | int | Training batch size |
| `ARABIC_QWEN_USE_LORA` | `optimization.use_lora` | bool | Enable LoRA |
| `ARABIC_QWEN_LOG_LEVEL` | `logging.log_level` | str | Logging level |

## Validation

All configuration classes include comprehensive validation:

```python
from src.config.sft_config import SFTConfig

try:
    config = SFTConfig.from_yaml("invalid_config.yaml")
except ValueError as e:
    print(f"Configuration validation error: {e}")
```

### Common Validation Rules

- **Model names**: Must be valid HuggingFace model identifiers or local paths
- **Learning rates**: Must be positive floats, typically between 1e-6 and 1e-3
- **Batch sizes**: Must be positive integers
- **File paths**: Must exist and be readable
- **Device specifications**: Must be valid PyTorch device strings

## Configuration Templates

The framework provides several configuration templates:

### Development Configuration
```yaml
# config/dev_config.yaml
model:
  name: "Qwen/Qwen-1_8B"  # Smaller model for development
  max_length: 512

training:
  batch_size: 2
  num_epochs: 1
  learning_rate: 1e-4

data:
  max_samples: 1000  # Limited data for quick testing

logging:
  log_level: "DEBUG"
```

### Production Configuration
```yaml
# config/prod_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  batch_size: 8
  num_epochs: 3
  learning_rate: 5e-5

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  quantization_bits: 4

logging:
  log_level: "INFO"
  use_wandb: true
```

## Best Practices

### 1. Configuration Organization
- Use separate configuration files for different environments (dev, staging, prod)
- Keep sensitive information in environment variables
- Use meaningful configuration file names

### 2. Environment Variables
- Set environment variables in `.env` files for local development
- Use CI/CD systems to manage environment variables in production
- Document all required environment variables

### 3. Validation
- Always validate configurations before training
- Use type hints for better IDE support
- Implement custom validation for domain-specific constraints

### 4. Documentation
- Document all configuration options
- Provide examples for common use cases
- Keep configuration schemas up to date

## Extending Configuration

### Adding New Configuration Options

1. **Add to the appropriate config class**:
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    batch_size: int = 4
    # Add new option
    gradient_clipping: float = 1.0
```

2. **Update validation**:
```python
def validate(self) -> None:
    super().validate()
    if self.gradient_clipping <= 0:
        raise ValueError("Gradient clipping must be positive")
```

3. **Add environment variable mapping**:
```python
# In env_config.py
GRADIENT_CLIPPING = "ARABIC_QWEN_GRADIENT_CLIPPING"
```

4. **Update documentation**:
- Add to configuration reference
- Update examples
- Add to environment variable table

### Creating Custom Configuration Classes

```python
from dataclasses import dataclass
from src.config.base_config import BaseConfig

@dataclass
class CustomConfig(BaseConfig):
    """Custom configuration for specialized training."""
    
    custom_parameter: str = "default_value"
    custom_flag: bool = False
    
    def validate(self) -> None:
        """Validate custom configuration."""
        super().validate()
        
        if not self.custom_parameter:
            raise ValueError("Custom parameter cannot be empty")
```

## Troubleshooting

### Common Issues

1. **Configuration file not found**:
   - Check file path is correct
   - Ensure file has proper YAML syntax
   - Verify file permissions

2. **Environment variable not recognized**:
   - Check variable name follows naming convention
   - Ensure variable is set in current environment
   - Verify type conversion is supported

3. **Validation errors**:
   - Check all required fields are provided
   - Verify data types match expectations
   - Ensure values are within valid ranges

### Debug Mode

```python
import logging
from src.config.sft_config import SFTConfig

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load configuration with debug output
config = SFTConfig.from_yaml("config/sft_config.yaml")
```

## See Also

- [Training Documentation](../training/index.md)
- [Data Loading Documentation](../data_loader/index.md)
- [Environment Setup Guide](../../README.md#environment-setup)
- [Configuration Examples](../../config/)