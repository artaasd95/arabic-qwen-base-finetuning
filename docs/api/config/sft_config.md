# SFT Configuration Documentation

The `SFTConfig` class provides comprehensive configuration management for Supervised Fine-Tuning (SFT) in the Arabic Qwen Base Fine-tuning framework. It extends the base configuration system with SFT-specific parameters and validation.

## Class Overview

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from src.config.base_config import BaseConfig

@dataclass
class SFTConfig(BaseConfig):
    """Configuration for Supervised Fine-Tuning (SFT)."""
```

## Location

**File**: `src/config/sft_config.py`

## Configuration Structure

The SFT configuration is organized into several nested configuration classes:

```
SFTConfig
├── model: ModelConfig
├── training: TrainingConfig
├── data: DataConfig
├── optimization: OptimizationConfig
├── logging: LoggingConfig
├── paths: PathConfig
├── evaluation: EvaluationConfig
└── experiment: ExperimentConfig
```

## Configuration Classes

### ModelConfig

Defines model-related configuration parameters.

```python
@dataclass
class ModelConfig:
    """Model configuration for SFT."""
    name: str = "Qwen/Qwen-7B"
    tokenizer_name: Optional[str] = None
    max_length: int = 2048
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    padding_side: str = "right"
    truncation_side: str = "right"
    add_eos_token: bool = True
    add_bos_token: bool = False
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "Qwen/Qwen-7B" | Model name or path |
| `tokenizer_name` | Optional[str] | None | Tokenizer name (defaults to model name) |
| `max_length` | int | 2048 | Maximum sequence length |
| `device` | str | "auto" | Device specification |
| `torch_dtype` | str | "auto" | PyTorch data type |
| `trust_remote_code` | bool | False | Trust remote code execution |
| `use_fast_tokenizer` | bool | True | Use fast tokenizer implementation |
| `padding_side` | str | "right" | Padding side ("left" or "right") |
| `truncation_side` | str | "right" | Truncation side ("left" or "right") |
| `add_eos_token` | bool | True | Add end-of-sequence token |
| `add_bos_token` | bool | False | Add beginning-of-sequence token |

### TrainingConfig

Defines training-related configuration parameters.

```python
@dataclass
class TrainingConfig:
    """Training configuration for SFT."""
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler_type: str = "cosine"
    num_cycles: float = 0.5
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    label_smoothing_factor: float = 0.0
    seed: int = 42
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 5e-5 | Learning rate for optimizer |
| `batch_size` | int | 4 | Training batch size |
| `num_epochs` | int | 3 | Number of training epochs |
| `warmup_steps` | int | 100 | Number of warmup steps |
| `warmup_ratio` | float | 0.1 | Warmup ratio (alternative to warmup_steps) |
| `save_steps` | int | 500 | Save checkpoint every N steps |
| `eval_steps` | int | 500 | Evaluation frequency |
| `logging_steps` | int | 10 | Logging frequency |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `max_grad_norm` | float | 1.0 | Maximum gradient norm for clipping |
| `weight_decay` | float | 0.01 | Weight decay coefficient |
| `adam_beta1` | float | 0.9 | Adam optimizer beta1 parameter |
| `adam_beta2` | float | 0.999 | Adam optimizer beta2 parameter |
| `adam_epsilon` | float | 1e-8 | Adam optimizer epsilon parameter |
| `lr_scheduler_type` | str | "cosine" | Learning rate scheduler type |
| `num_cycles` | float | 0.5 | Number of cycles for cosine scheduler |
| `dataloader_num_workers` | int | 0 | Number of DataLoader workers |
| `dataloader_pin_memory` | bool | True | Pin memory for DataLoader |
| `remove_unused_columns` | bool | True | Remove unused columns from dataset |
| `label_smoothing_factor` | float | 0.0 | Label smoothing factor |
| `seed` | int | 42 | Random seed for reproducibility |

### DataConfig

Defines data-related configuration parameters.

```python
@dataclass
class DataConfig:
    """Data configuration for SFT."""
    train_file: str = "data/train.jsonl"
    validation_file: Optional[str] = "data/validation.jsonl"
    test_file: Optional[str] = None
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    preprocessing_num_workers: int = 4
    text_column: str = "text"
    prompt_column: str = "prompt"
    response_column: str = "response"
    system_column: Optional[str] = "system"
    conversation_template: str = "default"
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    return_overflowing_tokens: bool = False
    stride: int = 0
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_file` | str | "data/train.jsonl" | Training data file path |
| `validation_file` | Optional[str] | "data/validation.jsonl" | Validation data file path |
| `test_file` | Optional[str] | None | Test data file path |
| `max_samples` | Optional[int] | None | Maximum samples to use (None for all) |
| `validation_split_percentage` | float | 0.1 | Validation split percentage |
| `preprocessing_num_workers` | int | 4 | Number of preprocessing workers |
| `text_column` | str | "text" | Text column name |
| `prompt_column` | str | "prompt" | Prompt column name |
| `response_column` | str | "response" | Response column name |
| `system_column` | Optional[str] | "system" | System message column name |
| `conversation_template` | str | "default" | Conversation template type |
| `add_special_tokens` | bool | True | Add special tokens during tokenization |
| `truncation` | bool | True | Enable truncation |
| `padding` | str | "max_length" | Padding strategy |
| `return_overflowing_tokens` | bool | False | Return overflowing tokens |
| `stride` | int | 0 | Stride for overflowing tokens |

### OptimizationConfig

Defines optimization-related configuration parameters.

```python
@dataclass
class OptimizationConfig:
    """Optimization configuration for SFT."""
    # LoRA Configuration
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    
    # Quantization Configuration
    use_quantization: bool = False
    quantization_bits: int = 4
    quantization_type: str = "nf4"
    use_double_quantization: bool = True
    quantization_compute_dtype: str = "float16"
    
    # Memory Optimization
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # Mixed Precision
    fp16: bool = False
    bf16: bool = False
    tf32: bool = True
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lora` | bool | False | Enable LoRA (Low-Rank Adaptation) |
| `lora_rank` | int | 16 | LoRA rank |
| `lora_alpha` | int | 32 | LoRA alpha parameter |
| `lora_dropout` | float | 0.1 | LoRA dropout rate |
| `lora_target_modules` | Optional[List[str]] | None | Target modules for LoRA |
| `lora_bias` | str | "none" | LoRA bias configuration |
| `lora_task_type` | str | "CAUSAL_LM" | LoRA task type |
| `use_quantization` | bool | False | Enable quantization |
| `quantization_bits` | int | 4 | Number of quantization bits |
| `quantization_type` | str | "nf4" | Quantization type |
| `use_double_quantization` | bool | True | Use double quantization |
| `quantization_compute_dtype` | str | "float16" | Compute dtype for quantization |
| `use_gradient_checkpointing` | bool | False | Enable gradient checkpointing |
| `use_flash_attention` | bool | False | Enable Flash Attention |
| `use_deepspeed` | bool | False | Enable DeepSpeed |
| `deepspeed_config` | Optional[str] | None | DeepSpeed configuration file |
| `fp16` | bool | False | Enable FP16 mixed precision |
| `bf16` | bool | False | Enable BF16 mixed precision |
| `tf32` | bool | True | Enable TF32 on Ampere GPUs |

### LoggingConfig

Defines logging and monitoring configuration parameters.

```python
@dataclass
class LoggingConfig:
    """Logging configuration for SFT."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "arabic-qwen-sft"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    # TensorBoard
    use_tensorboard: bool = False
    tensorboard_log_dir: str = "logs/tensorboard"
    
    # MLflow
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "arabic-qwen-sft"
    
    # Reporting
    report_to: List[str] = field(default_factory=list)
    logging_first_step: bool = True
    logging_nan_inf_filter: bool = True
```

### PathConfig

Defines path-related configuration parameters.

```python
@dataclass
class PathConfig:
    """Path configuration for SFT."""
    output_dir: str = "output"
    cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data"
    config_dir: str = "config"
    model_cache_dir: Optional[str] = None
    tokenizer_cache_dir: Optional[str] = None
```

### EvaluationConfig

Defines evaluation-related configuration parameters.

```python
@dataclass
class EvaluationConfig:
    """Evaluation configuration for SFT."""
    eval_strategy: str = "steps"
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    eval_delay: float = 0
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    save_strategy: str = "steps"
    save_only_model: bool = False
    prediction_loss_only: bool = False
    include_inputs_for_metrics: bool = False
```

### ExperimentConfig

Defines experiment tracking and reproducibility parameters.

```python
@dataclass
class ExperimentConfig:
    """Experiment configuration for SFT."""
    experiment_name: str = "sft-experiment"
    run_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
```

## Usage Examples

### Basic Configuration

```python
from src.config.sft_config import SFTConfig

# Create default configuration
config = SFTConfig()

# Access nested configurations
print(f"Model: {config.model.name}")
print(f"Learning rate: {config.training.learning_rate}")
print(f"Batch size: {config.training.batch_size}")
```

### Loading from YAML

```python
# Load configuration from YAML file
config = SFTConfig.from_yaml("config/sft_config.yaml")

# Configuration will include environment variable overrides
print(f"Output directory: {config.paths.output_dir}")
```

### Creating Custom Configuration

```python
from src.config.sft_config import (
    SFTConfig, ModelConfig, TrainingConfig, OptimizationConfig
)

# Create custom model configuration
model_config = ModelConfig(
    name="Qwen/Qwen-1_8B",
    max_length=1024,
    device="cuda:0"
)

# Create custom training configuration
training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=5,
    warmup_steps=200
)

# Create optimization configuration with LoRA
optimization_config = OptimizationConfig(
    use_lora=True,
    lora_rank=32,
    lora_alpha=64,
    use_quantization=True,
    quantization_bits=4
)

# Create full configuration
config = SFTConfig(
    model=model_config,
    training=training_config,
    optimization=optimization_config
)

# Save to file
config.to_yaml("my_sft_config.yaml")
```

### Environment Variable Override

```python
import os
from src.config.sft_config import SFTConfig

# Set environment variables
os.environ["ARABIC_QWEN_MODEL_NAME"] = "Qwen/Qwen-7B"
os.environ["ARABIC_QWEN_TRAINING_LEARNING_RATE"] = "2e-5"
os.environ["ARABIC_QWEN_OPTIMIZATION_USE_LORA"] = "true"
os.environ["ARABIC_QWEN_OPTIMIZATION_LORA_RANK"] = "64"

# Load configuration (environment variables override YAML)
config = SFTConfig.from_yaml("config/sft_config.yaml")

print(f"Model: {config.model.name}")  # Qwen/Qwen-7B
print(f"Learning rate: {config.training.learning_rate}")  # 2e-5
print(f"Use LoRA: {config.optimization.use_lora}")  # True
print(f"LoRA rank: {config.optimization.lora_rank}")  # 64
```

## YAML Configuration Example

```yaml
# config/sft_config.yaml
model:
  name: "Qwen/Qwen-7B"
  tokenizer_name: null
  max_length: 2048
  device: "auto"
  torch_dtype: "auto"
  trust_remote_code: false
  use_fast_tokenizer: true
  padding_side: "right"
  truncation_side: "right"
  add_eos_token: true
  add_bos_token: false

training:
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 3
  warmup_steps: 100
  warmup_ratio: 0.1
  save_steps: 500
  eval_steps: 500
  logging_steps: 10
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  lr_scheduler_type: "cosine"
  num_cycles: 0.5
  dataloader_num_workers: 0
  dataloader_pin_memory: true
  remove_unused_columns: true
  label_smoothing_factor: 0.0
  seed: 42

data:
  train_file: "data/train.jsonl"
  validation_file: "data/validation.jsonl"
  test_file: null
  max_samples: null
  validation_split_percentage: 0.1
  preprocessing_num_workers: 4
  text_column: "text"
  prompt_column: "prompt"
  response_column: "response"
  system_column: "system"
  conversation_template: "default"
  add_special_tokens: true
  truncation: true
  padding: "max_length"
  return_overflowing_tokens: false
  stride: 0

optimization:
  # LoRA Configuration
  use_lora: false
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules: null
  lora_bias: "none"
  lora_task_type: "CAUSAL_LM"
  
  # Quantization Configuration
  use_quantization: false
  quantization_bits: 4
  quantization_type: "nf4"
  use_double_quantization: true
  quantization_compute_dtype: "float16"
  
  # Memory Optimization
  use_gradient_checkpointing: false
  use_flash_attention: false
  use_deepspeed: false
  deepspeed_config: null
  
  # Mixed Precision
  fp16: false
  bf16: false
  tf32: true

logging:
  log_level: "INFO"
  log_file: null
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # Weights & Biases
  use_wandb: false
  wandb_project: "arabic-qwen-sft"
  wandb_entity: null
  wandb_run_name: null
  wandb_tags: null
  
  # TensorBoard
  use_tensorboard: false
  tensorboard_log_dir: "logs/tensorboard"
  
  # MLflow
  use_mlflow: false
  mlflow_tracking_uri: null
  mlflow_experiment_name: "arabic-qwen-sft"
  
  # Reporting
  report_to: []
  logging_first_step: true
  logging_nan_inf_filter: true

paths:
  output_dir: "output"
  cache_dir: "cache"
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  data_dir: "data"
  config_dir: "config"
  model_cache_dir: null
  tokenizer_cache_dir: null

evaluation:
  eval_strategy: "steps"
  eval_steps: 500
  eval_accumulation_steps: null
  eval_delay: 0
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  load_best_model_at_end: true
  save_total_limit: 3
  save_strategy: "steps"
  save_only_model: false
  prediction_loss_only: false
  include_inputs_for_metrics: false

experiment:
  experiment_name: "sft-experiment"
  run_name: null
  description: null
  tags: null
  notes: null
  group: null
  job_type: "train"
  resume_from_checkpoint: null
  ignore_data_skip: false
  push_to_hub: false
  hub_model_id: null
  hub_strategy: "every_save"
  hub_token: null
```

## Validation

The SFT configuration includes comprehensive validation:

```python
class SFTConfig(BaseConfig):
    def validate(self) -> None:
        """Validate SFT configuration."""
        super().validate()
        
        # Validate model configuration
        self._validate_model_config()
        
        # Validate training configuration
        self._validate_training_config()
        
        # Validate data configuration
        self._validate_data_config()
        
        # Validate optimization configuration
        self._validate_optimization_config()
        
        # Validate paths
        self._validate_paths()
    
    def _validate_model_config(self) -> None:
        """Validate model configuration."""
        if not self.model.name:
            raise ValueError("Model name cannot be empty")
        
        if self.model.max_length <= 0:
            raise ValueError("Max length must be positive")
        
        if self.model.padding_side not in ["left", "right"]:
            raise ValueError("Padding side must be 'left' or 'right'")
    
    def _validate_training_config(self) -> None:
        """Validate training configuration."""
        if not (1e-6 <= self.training.learning_rate <= 1e-2):
            raise ValueError("Learning rate must be between 1e-6 and 1e-2")
        
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.training.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if not (0.0 <= self.training.warmup_ratio <= 1.0):
            raise ValueError("Warmup ratio must be between 0.0 and 1.0")
    
    def _validate_data_config(self) -> None:
        """Validate data configuration."""
        from pathlib import Path
        
        if not Path(self.data.train_file).exists():
            raise ValueError(f"Training file not found: {self.data.train_file}")
        
        if self.data.validation_file and not Path(self.data.validation_file).exists():
            raise ValueError(f"Validation file not found: {self.data.validation_file}")
        
        if not (0.0 < self.data.validation_split_percentage < 1.0):
            raise ValueError("Validation split percentage must be between 0.0 and 1.0")
    
    def _validate_optimization_config(self) -> None:
        """Validate optimization configuration."""
        if self.optimization.use_lora:
            if self.optimization.lora_rank <= 0:
                raise ValueError("LoRA rank must be positive")
            
            if self.optimization.lora_alpha <= 0:
                raise ValueError("LoRA alpha must be positive")
        
        if self.optimization.use_quantization:
            if self.optimization.quantization_bits not in [4, 8]:
                raise ValueError("Quantization bits must be 4 or 8")
```

## Configuration Templates

### Development Configuration

```yaml
# config/dev_sft_config.yaml
model:
  name: "Qwen/Qwen-1_8B"  # Smaller model for development
  max_length: 512

training:
  batch_size: 2
  num_epochs: 1
  learning_rate: 1e-4
  save_steps: 100
  eval_steps: 100
  logging_steps: 5

data:
  max_samples: 1000  # Limited data for quick testing

optimization:
  use_lora: true
  lora_rank: 8
  use_quantization: true
  quantization_bits: 4

logging:
  log_level: "DEBUG"
  use_tensorboard: true
```

### Production Configuration

```yaml
# config/prod_sft_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  batch_size: 8
  num_epochs: 3
  learning_rate: 5e-5
  gradient_accumulation_steps: 4
  warmup_steps: 500

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  quantization_bits: 4
  use_gradient_checkpointing: true
  bf16: true

logging:
  log_level: "INFO"
  use_wandb: true
  wandb_project: "arabic-qwen-production"
```

### High-Performance Configuration

```yaml
# config/high_perf_sft_config.yaml
model:
  name: "Qwen/Qwen-14B"
  max_length: 4096

training:
  batch_size: 16
  gradient_accumulation_steps: 8
  learning_rate: 3e-5
  dataloader_num_workers: 8
  dataloader_pin_memory: true

optimization:
  use_lora: true
  lora_rank: 32
  lora_alpha: 64
  use_quantization: true
  quantization_bits: 4
  use_gradient_checkpointing: true
  use_flash_attention: true
  use_deepspeed: true
  deepspeed_config: "config/deepspeed_config.json"
  bf16: true

logging:
  use_wandb: true
  use_tensorboard: true
  use_mlflow: true
```

## Best Practices

### 1. Configuration Organization

- Use separate configuration files for different environments
- Keep sensitive information in environment variables
- Use meaningful configuration file names
- Document configuration changes in version control

### 2. Performance Optimization

- Enable LoRA for memory efficiency
- Use quantization for large models
- Enable gradient checkpointing for memory savings
- Use appropriate batch sizes and gradient accumulation

### 3. Reproducibility

- Set random seeds consistently
- Document all configuration parameters
- Use version control for configuration files
- Track experiments with proper naming

### 4. Monitoring and Logging

- Enable appropriate logging levels
- Use experiment tracking tools (W&B, MLflow)
- Monitor training metrics regularly
- Save checkpoints frequently

## See Also

- [Base Configuration](base_config.md)
- [Environment Configuration](env_config.md)
- [DPO Configuration](dpo_config.md)
- [Training Documentation](../training/index.md)
- [SFT Trainer Documentation](../training/sft_trainer.md)