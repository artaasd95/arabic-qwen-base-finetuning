# DPO Configuration Documentation

The `DPOConfig` class provides comprehensive configuration management for Direct Preference Optimization (DPO) in the Arabic Qwen Base Fine-tuning framework. It extends the base configuration system with DPO-specific parameters and validation.

## Class Overview

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from src.config.base_config import BaseConfig

@dataclass
class DPOConfig(BaseConfig):
    """Configuration for Direct Preference Optimization (DPO)."""
```

## Location

**File**: `src/config/dpo_config.py`

## DPO Overview

Direct Preference Optimization (DPO) is a method for training language models to align with human preferences without requiring a separate reward model. It directly optimizes the policy using preference data consisting of chosen and rejected responses.

### Key Concepts

- **Preference Data**: Pairs of (prompt, chosen_response, rejected_response)
- **Beta Parameter**: Controls the strength of the preference optimization
- **Reference Model**: Optional reference model for KL divergence regularization
- **Reference-Free**: Option to train without a reference model

## Configuration Structure

The DPO configuration extends the base configuration with DPO-specific parameters:

```
DPOConfig
├── model: ModelConfig
├── training: DPOTrainingConfig
├── data: DPODataConfig
├── dpo: DPOSpecificConfig
├── optimization: OptimizationConfig
├── logging: LoggingConfig
├── paths: PathConfig
├── evaluation: EvaluationConfig
└── experiment: ExperimentConfig
```

## DPO-Specific Configuration Classes

### DPOSpecificConfig

Defines DPO algorithm-specific parameters.

```python
@dataclass
class DPOSpecificConfig:
    """DPO-specific configuration parameters."""
    beta: float = 0.1
    reference_free: bool = False
    reference_model_name: Optional[str] = None
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    use_weighting: bool = False
    weighting_temperature: float = 1.0
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_target_length: int = 1024
    truncation_mode: str = "keep_end"
    precompute_ref_log_probs: bool = False
    model_init_kwargs: Optional[Dict[str, Any]] = None
    ref_model_init_kwargs: Optional[Dict[str, Any]] = None
    force_use_ref_model: bool = False
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 0.1 | DPO beta parameter (KL regularization strength) |
| `reference_free` | bool | False | Whether to use reference-free DPO |
| `reference_model_name` | Optional[str] | None | Reference model name (defaults to main model) |
| `loss_type` | str | "sigmoid" | Loss function type ("sigmoid", "hinge", "ipo") |
| `label_smoothing` | float | 0.0 | Label smoothing factor |
| `use_weighting` | bool | False | Use importance weighting |
| `weighting_temperature` | float | 1.0 | Temperature for importance weighting |
| `max_length` | int | 2048 | Maximum sequence length |
| `max_prompt_length` | int | 1024 | Maximum prompt length |
| `max_target_length` | int | 1024 | Maximum target length |
| `truncation_mode` | str | "keep_end" | Truncation mode ("keep_end", "keep_start") |
| `precompute_ref_log_probs` | bool | False | Precompute reference log probabilities |
| `model_init_kwargs` | Optional[Dict] | None | Model initialization arguments |
| `ref_model_init_kwargs` | Optional[Dict] | None | Reference model initialization arguments |
| `force_use_ref_model` | bool | False | Force using separate reference model |

### DPOTrainingConfig

Extends the base training configuration with DPO-specific training parameters.

```python
@dataclass
class DPOTrainingConfig:
    """Training configuration for DPO."""
    learning_rate: float = 5e-7  # Lower learning rate for DPO
    batch_size: int = 4
    num_epochs: int = 1
    warmup_steps: int = 150
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
    remove_unused_columns: bool = False  # Keep all columns for DPO
    label_smoothing_factor: float = 0.0
    seed: int = 42
    
    # DPO-specific training parameters
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_checkpointing: bool = False
    dataloader_drop_last: bool = True
    eval_accumulation_steps: Optional[int] = None
    save_safetensors: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
```

### DPODataConfig

Defines data configuration specific to DPO preference datasets.

```python
@dataclass
class DPODataConfig:
    """Data configuration for DPO."""
    train_file: str = "data/dpo_train.jsonl"
    validation_file: Optional[str] = "data/dpo_validation.jsonl"
    test_file: Optional[str] = None
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    preprocessing_num_workers: int = 4
    
    # Column names for preference data
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    system_column: Optional[str] = "system"
    
    # Data processing parameters
    conversation_template: str = "default"
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    return_overflowing_tokens: bool = False
    stride: int = 0
    
    # DPO-specific data parameters
    sanity_check: bool = False
    sanity_check_size: int = 100
    ignore_bias_buffers: bool = False
    disable_dropout: bool = True
    generate_during_eval: bool = False
    is_encoder_decoder: bool = False
    preprocess_logits_for_metrics: Optional[callable] = None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_file` | str | "data/dpo_train.jsonl" | Training preference data file |
| `validation_file` | Optional[str] | "data/dpo_validation.jsonl" | Validation preference data file |
| `test_file` | Optional[str] | None | Test preference data file |
| `prompt_column` | str | "prompt" | Column name for prompts |
| `chosen_column` | str | "chosen" | Column name for chosen responses |
| `rejected_column` | str | "rejected" | Column name for rejected responses |
| `system_column` | Optional[str] | "system" | Column name for system messages |
| `sanity_check` | bool | False | Enable sanity check mode |
| `sanity_check_size` | int | 100 | Number of samples for sanity check |
| `ignore_bias_buffers` | bool | False | Ignore bias buffers during training |
| `disable_dropout` | bool | True | Disable dropout during training |
| `generate_during_eval` | bool | False | Generate responses during evaluation |

## Usage Examples

### Basic DPO Configuration

```python
from src.config.dpo_config import DPOConfig

# Create default DPO configuration
config = DPOConfig()

# Access DPO-specific parameters
print(f"Beta parameter: {config.dpo.beta}")
print(f"Reference-free: {config.dpo.reference_free}")
print(f"Loss type: {config.dpo.loss_type}")
```

### Loading from YAML

```python
# Load DPO configuration from YAML file
config = DPOConfig.from_yaml("config/dpo_config.yaml")

# Configuration includes environment variable overrides
print(f"Model: {config.model.name}")
print(f"Beta: {config.dpo.beta}")
print(f"Learning rate: {config.training.learning_rate}")
```

### Creating Custom DPO Configuration

```python
from src.config.dpo_config import (
    DPOConfig, ModelConfig, DPOSpecificConfig, DPOTrainingConfig
)

# Create DPO-specific configuration
dpo_config = DPOSpecificConfig(
    beta=0.2,
    reference_free=False,
    loss_type="sigmoid",
    max_length=2048,
    max_prompt_length=1024
)

# Create training configuration optimized for DPO
training_config = DPOTrainingConfig(
    learning_rate=1e-6,  # Lower learning rate for DPO
    batch_size=2,
    num_epochs=1,
    gradient_accumulation_steps=8
)

# Create model configuration
model_config = ModelConfig(
    name="Qwen/Qwen-7B",
    max_length=2048
)

# Create full DPO configuration
config = DPOConfig(
    model=model_config,
    training=training_config,
    dpo=dpo_config
)

# Save configuration
config.to_yaml("my_dpo_config.yaml")
```

### Environment Variable Override

```python
import os
from src.config.dpo_config import DPOConfig

# Set DPO-specific environment variables
os.environ["ARABIC_QWEN_DPO_BETA"] = "0.3"
os.environ["ARABIC_QWEN_DPO_REFERENCE_FREE"] = "true"
os.environ["ARABIC_QWEN_DPO_LOSS_TYPE"] = "hinge"
os.environ["ARABIC_QWEN_TRAINING_LEARNING_RATE"] = "1e-6"

# Load configuration with environment overrides
config = DPOConfig.from_yaml("config/dpo_config.yaml")

print(f"Beta: {config.dpo.beta}")  # 0.3
print(f"Reference-free: {config.dpo.reference_free}")  # True
print(f"Loss type: {config.dpo.loss_type}")  # hinge
print(f"Learning rate: {config.training.learning_rate}")  # 1e-6
```

## YAML Configuration Example

```yaml
# config/dpo_config.yaml
model:
  name: "Qwen/Qwen-7B"
  tokenizer_name: null
  max_length: 2048
  device: "auto"
  torch_dtype: "auto"
  trust_remote_code: false

training:
  learning_rate: 5.0e-7  # Lower learning rate for DPO
  batch_size: 4
  num_epochs: 1
  warmup_steps: 150
  warmup_ratio: 0.1
  save_steps: 500
  eval_steps: 500
  logging_steps: 10
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  dataloader_num_workers: 0
  remove_unused_columns: false  # Keep all columns for DPO
  seed: 42
  
  # DPO-specific training parameters
  max_steps: -1
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_checkpointing: false
  dataloader_drop_last: true
  save_safetensors: true

data:
  train_file: "data/dpo_train.jsonl"
  validation_file: "data/dpo_validation.jsonl"
  test_file: null
  max_samples: null
  validation_split_percentage: 0.1
  preprocessing_num_workers: 4
  
  # Column names for preference data
  prompt_column: "prompt"
  chosen_column: "chosen"
  rejected_column: "rejected"
  system_column: "system"
  
  # Data processing parameters
  conversation_template: "default"
  add_special_tokens: true
  truncation: true
  padding: "max_length"
  
  # DPO-specific data parameters
  sanity_check: false
  sanity_check_size: 100
  ignore_bias_buffers: false
  disable_dropout: true
  generate_during_eval: false
  is_encoder_decoder: false

dpo:
  beta: 0.1
  reference_free: false
  reference_model_name: null
  loss_type: "sigmoid"
  label_smoothing: 0.0
  use_weighting: false
  weighting_temperature: 1.0
  max_length: 2048
  max_prompt_length: 1024
  max_target_length: 1024
  truncation_mode: "keep_end"
  precompute_ref_log_probs: false
  model_init_kwargs: null
  ref_model_init_kwargs: null
  force_use_ref_model: false

optimization:
  # LoRA Configuration
  use_lora: true  # Recommended for DPO
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
  
  # Mixed Precision
  fp16: false
  bf16: true  # Recommended for DPO
  tf32: true

logging:
  log_level: "INFO"
  use_wandb: false
  wandb_project: "arabic-qwen-dpo"
  use_tensorboard: false
  tensorboard_log_dir: "logs/tensorboard"
  report_to: []

paths:
  output_dir: "output/dpo"
  cache_dir: "cache"
  checkpoint_dir: "checkpoints/dpo"
  log_dir: "logs/dpo"
  data_dir: "data"

evaluation:
  eval_strategy: "steps"
  eval_steps: 500
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  load_best_model_at_end: true
  save_total_limit: 3
  save_strategy: "steps"

experiment:
  experiment_name: "dpo-experiment"
  run_name: null
  description: "Direct Preference Optimization training"
  tags: ["dpo", "preference-optimization"]
  job_type: "train"
  resume_from_checkpoint: null
```

## Data Format

DPO requires preference data in a specific format:

### JSONL Format

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس.", "rejected": "لا أعرف."}
{"prompt": "اشرح لي مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال في علوم الكمبيوتر يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً.", "rejected": "الذكاء الاصطناعي شيء معقد جداً."}
```

### With System Messages

```jsonl
{"system": "أنت مساعد ذكي ومفيد", "prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد ومركزها السياسي والثقافي.", "rejected": "باريس."}
```

### Conversation Format

```jsonl
{
  "prompt": [
    {"role": "user", "content": "ما هي عاصمة فرنسا؟"}
  ],
  "chosen": [
    {"role": "assistant", "content": "عاصمة فرنسا هي باريس."}
  ],
  "rejected": [
    {"role": "assistant", "content": "لا أعرف."}
  ]
}
```

## DPO Algorithm Parameters

### Beta Parameter

The beta parameter controls the strength of the KL regularization:

- **Low beta (0.01-0.1)**: Stronger regularization, more conservative updates
- **Medium beta (0.1-0.5)**: Balanced approach
- **High beta (0.5-1.0)**: Weaker regularization, more aggressive updates

```python
# Conservative approach
config.dpo.beta = 0.05

# Balanced approach
config.dpo.beta = 0.1

# Aggressive approach
config.dpo.beta = 0.3
```

### Loss Types

#### Sigmoid Loss (Default)
```python
config.dpo.loss_type = "sigmoid"
# Uses sigmoid function for preference modeling
```

#### Hinge Loss
```python
config.dpo.loss_type = "hinge"
# Uses hinge loss for preference modeling
```

#### IPO Loss
```python
config.dpo.loss_type = "ipo"
# Uses Identity Preference Optimization loss
```

### Reference Model Options

#### Using Reference Model
```python
config.dpo.reference_free = False
config.dpo.reference_model_name = "Qwen/Qwen-7B"  # Or None to use main model
```

#### Reference-Free DPO
```python
config.dpo.reference_free = True
# No reference model needed, faster training
```

## Validation

The DPO configuration includes comprehensive validation:

```python
class DPOConfig(BaseConfig):
    def validate(self) -> None:
        """Validate DPO configuration."""
        super().validate()
        
        # Validate DPO-specific parameters
        self._validate_dpo_config()
        
        # Validate data format
        self._validate_dpo_data()
        
        # Validate training parameters
        self._validate_dpo_training()
    
    def _validate_dpo_config(self) -> None:
        """Validate DPO-specific configuration."""
        if not (0.001 <= self.dpo.beta <= 1.0):
            raise ValueError("DPO beta must be between 0.001 and 1.0")
        
        if self.dpo.loss_type not in ["sigmoid", "hinge", "ipo"]:
            raise ValueError("Loss type must be 'sigmoid', 'hinge', or 'ipo'")
        
        if self.dpo.max_prompt_length >= self.dpo.max_length:
            raise ValueError("Max prompt length must be less than max length")
    
    def _validate_dpo_data(self) -> None:
        """Validate DPO data configuration."""
        required_columns = [self.data.prompt_column, self.data.chosen_column, self.data.rejected_column]
        
        # Check if data file exists and has required columns
        from pathlib import Path
        import json
        
        if Path(self.data.train_file).exists():
            with open(self.data.train_file, 'r', encoding='utf-8') as f:
                sample = json.loads(f.readline())
                for col in required_columns:
                    if col not in sample:
                        raise ValueError(f"Required column '{col}' not found in data")
    
    def _validate_dpo_training(self) -> None:
        """Validate DPO training configuration."""
        # DPO typically uses lower learning rates
        if self.training.learning_rate > 1e-4:
            import warnings
            warnings.warn(
                f"Learning rate {self.training.learning_rate} is high for DPO. "
                "Consider using a lower value (1e-6 to 1e-5)."
            )
        
        # DPO typically uses fewer epochs
        if self.training.num_epochs > 3:
            import warnings
            warnings.warn(
                f"Number of epochs {self.training.num_epochs} is high for DPO. "
                "Consider using 1-3 epochs."
            )
```

## Configuration Templates

### Quick Start DPO Configuration

```yaml
# config/quick_dpo_config.yaml
model:
  name: "Qwen/Qwen-1_8B"
  max_length: 1024

training:
  learning_rate: 5e-7
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 4

data:
  train_file: "data/dpo_train_small.jsonl"
  max_samples: 1000

dpo:
  beta: 0.1
  reference_free: true
  max_length: 1024
  max_prompt_length: 512

optimization:
  use_lora: true
  lora_rank: 8
  use_quantization: true
```

### Production DPO Configuration

```yaml
# config/prod_dpo_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  learning_rate: 1e-6
  batch_size: 4
  num_epochs: 1
  gradient_accumulation_steps: 8
  warmup_steps: 100

dpo:
  beta: 0.1
  reference_free: false
  loss_type: "sigmoid"
  max_length: 2048
  max_prompt_length: 1024

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  use_gradient_checkpointing: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-dpo-prod"
```

### Research DPO Configuration

```yaml
# config/research_dpo_config.yaml
model:
  name: "Qwen/Qwen-14B"
  max_length: 4096

training:
  learning_rate: 5e-7
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 16

dpo:
  beta: 0.05  # Conservative beta for research
  reference_free: false
  loss_type: "sigmoid"
  max_length: 4096
  max_prompt_length: 2048
  precompute_ref_log_probs: true

optimization:
  use_lora: true
  lora_rank: 32
  lora_alpha: 64
  use_quantization: true
  use_deepspeed: true

logging:
  use_wandb: true
  use_tensorboard: true
  use_mlflow: true
```

## Best Practices

### 1. Learning Rate Selection

- Start with 5e-7 to 1e-6 for DPO
- Use lower learning rates than SFT
- Monitor training loss carefully

### 2. Beta Parameter Tuning

- Start with 0.1 (default)
- Increase for more aggressive optimization
- Decrease for more conservative updates

### 3. Data Quality

- Ensure high-quality preference pairs
- Balance chosen/rejected examples
- Use diverse prompts and responses

### 4. Memory Optimization

- Use LoRA for memory efficiency
- Enable gradient checkpointing
- Use quantization for large models

### 5. Evaluation

- Monitor both training and validation loss
- Use human evaluation for final assessment
- Compare with baseline models

## Troubleshooting

### Common Issues

1. **High Memory Usage**:
   ```python
   # Enable memory optimizations
   config.optimization.use_lora = True
   config.optimization.use_gradient_checkpointing = True
   config.optimization.use_quantization = True
   ```

2. **Training Instability**:
   ```python
   # Use lower learning rate and smaller beta
   config.training.learning_rate = 1e-7
   config.dpo.beta = 0.05
   ```

3. **Slow Convergence**:
   ```python
   # Increase learning rate and beta slightly
   config.training.learning_rate = 5e-6
   config.dpo.beta = 0.2
   ```

## See Also

- [Base Configuration](base_config.md)
- [SFT Configuration](sft_config.md)
- [KTO Configuration](kto_config.md)
- [DPO Training Documentation](../training/dpo_trainer.md)
- [Preference Optimization Guide](../../guides/preference_optimization.md)