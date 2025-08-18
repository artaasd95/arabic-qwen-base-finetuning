# IPO Configuration Documentation

The `IPOConfig` class provides comprehensive configuration management for Identity Preference Optimization (IPO) in the Arabic Qwen Base Fine-tuning framework. IPO is a preference optimization method that addresses the length bias issue in DPO.

## Class Overview

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from src.config.base_config import BaseConfig

@dataclass
class IPOConfig(BaseConfig):
    """Configuration for Identity Preference Optimization (IPO)."""
```

## Location

**File**: `src/config/ipo_config.py`

## IPO Overview

Identity Preference Optimization (IPO) is an improvement over DPO that addresses the length bias problem. IPO uses a different loss function that is more robust to response length differences and provides better alignment with human preferences.

### Key Features

- **Length Bias Mitigation**: Reduces preference for longer responses
- **Improved Alignment**: Better correlation with human preferences
- **Stable Training**: More stable optimization dynamics
- **Regularization**: Built-in regularization through identity mapping

## IPO-Specific Configuration

### IPOSpecificConfig

```python
@dataclass
class IPOSpecificConfig:
    """IPO-specific configuration parameters."""
    beta: float = 0.1
    tau: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "ipo"
    reference_free: bool = False
    reference_model_name: Optional[str] = None
    
    # IPO-specific parameters
    ipo_grad_type: str = "linear"
    ipo_alpha: float = 1.0
    ipo_beta: float = 0.1
    ipo_tau: float = 0.1
    
    # Length normalization
    length_penalty: float = 0.0
    normalize_by_length: bool = True
    length_normalization_type: str = "sqrt"
    
    # Training parameters
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_target_length: int = 1024
    truncation_mode: str = "keep_end"
    
    # Model parameters
    precompute_ref_log_probs: bool = False
    model_init_kwargs: Optional[Dict[str, Any]] = None
    ref_model_init_kwargs: Optional[Dict[str, Any]] = None
    force_use_ref_model: bool = False
    
    # Advanced IPO parameters
    use_identity_mapping: bool = True
    identity_weight: float = 1.0
    preference_weight: float = 1.0
    regularization_strength: float = 0.01
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 0.1 | IPO beta parameter (KL regularization strength) |
| `tau` | float | 0.1 | IPO tau parameter (temperature) |
| `label_smoothing` | float | 0.0 | Label smoothing factor |
| `ipo_grad_type` | str | "linear" | Gradient type for IPO loss |
| `ipo_alpha` | float | 1.0 | IPO alpha parameter |
| `ipo_beta` | float | 0.1 | IPO beta parameter |
| `ipo_tau` | float | 0.1 | IPO tau parameter |
| `length_penalty` | float | 0.0 | Length penalty coefficient |
| `normalize_by_length` | bool | True | Whether to normalize by response length |
| `length_normalization_type` | str | "sqrt" | Type of length normalization |
| `use_identity_mapping` | bool | True | Use identity mapping in loss |
| `identity_weight` | float | 1.0 | Weight for identity mapping term |
| `preference_weight` | float | 1.0 | Weight for preference term |
| `regularization_strength` | float | 0.01 | Regularization strength |

### IPODataConfig

```python
@dataclass
class IPODataConfig:
    """Data configuration for IPO."""
    train_file: str = "data/ipo_train.jsonl"
    validation_file: Optional[str] = "data/ipo_validation.jsonl"
    test_file: Optional[str] = None
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    preprocessing_num_workers: int = 4
    
    # Column names for IPO data
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    system_column: Optional[str] = "system"
    
    # Data processing parameters
    conversation_template: str = "default"
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    
    # IPO-specific data parameters
    sanity_check: bool = False
    sanity_check_size: int = 100
    ignore_bias_buffers: bool = False
    disable_dropout: bool = True
    generate_during_eval: bool = False
    
    # Length handling
    filter_by_length: bool = True
    min_length_ratio: float = 0.1
    max_length_ratio: float = 10.0
    length_difference_threshold: int = 100
```

## Usage Examples

### Basic IPO Configuration

```python
from src.config.ipo_config import IPOConfig

# Create default IPO configuration
config = IPOConfig()

# Access IPO-specific parameters
print(f"Beta parameter: {config.ipo.beta}")
print(f"Tau parameter: {config.ipo.tau}")
print(f"Length normalization: {config.ipo.normalize_by_length}")
```

### Custom IPO Configuration

```python
from src.config.ipo_config import IPOConfig, IPOSpecificConfig

# Create IPO-specific configuration
ipo_config = IPOSpecificConfig(
    beta=0.2,
    tau=0.05,  # Lower temperature for sharper preferences
    normalize_by_length=True,
    length_normalization_type="log",  # Log normalization
    ipo_alpha=1.2,
    identity_weight=0.8,
    preference_weight=1.2
)

# Create full configuration
config = IPOConfig(ipo=ipo_config)
config.to_yaml("my_ipo_config.yaml")
```

### Loading from YAML

```python
from src.config.ipo_config import IPOConfig

# Load configuration from YAML file
config = IPOConfig.from_yaml("config/ipo_config.yaml")

# Override specific parameters
config.ipo.beta = 0.15
config.training.learning_rate = 1e-6
```

## YAML Configuration Example

```yaml
# config/ipo_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  learning_rate: 1.0e-6
  batch_size: 4
  num_epochs: 1
  gradient_accumulation_steps: 4
  warmup_steps: 100

data:
  train_file: "data/ipo_train.jsonl"
  validation_file: "data/ipo_validation.jsonl"
  prompt_column: "prompt"
  chosen_column: "chosen"
  rejected_column: "rejected"
  filter_by_length: true
  length_difference_threshold: 100

ipo:
  beta: 0.1
  tau: 0.1
  label_smoothing: 0.0
  ipo_grad_type: "linear"
  ipo_alpha: 1.0
  ipo_beta: 0.1
  ipo_tau: 0.1
  
  # Length handling
  length_penalty: 0.0
  normalize_by_length: true
  length_normalization_type: "sqrt"
  
  # Identity mapping
  use_identity_mapping: true
  identity_weight: 1.0
  preference_weight: 1.0
  regularization_strength: 0.01
  
  max_length: 2048
  max_prompt_length: 1024

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-ipo"
```

## Data Format

IPO uses the same pairwise preference data format as DPO:

### JSONL Format

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد.", "rejected": "باريس."}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً.", "rejected": "الذكاء الاصطناعي شيء معقد."}
```

### With System Messages

```jsonl
{"system": "أنت مساعد ذكي ومفيد.", "prompt": "ما هي فوائد التمرين؟", "chosen": "التمرين له فوائد عديدة للصحة الجسدية والنفسية...", "rejected": "التمرين مفيد."}
```

## IPO Algorithm Details

### Length Normalization Types

#### Square Root Normalization
```python
config.ipo.length_normalization_type = "sqrt"
# Normalizes by sqrt(length)
```

#### Logarithmic Normalization
```python
config.ipo.length_normalization_type = "log"
# Normalizes by log(length + 1)
```

#### Linear Normalization
```python
config.ipo.length_normalization_type = "linear"
# Normalizes by length
```

### Identity Mapping Configuration

#### Standard Identity Mapping
```python
config.ipo.use_identity_mapping = True
config.ipo.identity_weight = 1.0
config.ipo.preference_weight = 1.0
```

#### Emphasize Identity Term
```python
config.ipo.identity_weight = 1.5
config.ipo.preference_weight = 0.8
```

#### Emphasize Preference Term
```python
config.ipo.identity_weight = 0.8
config.ipo.preference_weight = 1.5
```

## Configuration Templates

### Quick Start IPO

```yaml
# config/quick_ipo_config.yaml
model:
  name: "Qwen/Qwen-1_8B"
  max_length: 1024

training:
  learning_rate: 1e-6
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 8

data:
  train_file: "data/ipo_train_small.jsonl"
  max_samples: 1000
  filter_by_length: true

ipo:
  beta: 0.1
  tau: 0.1
  normalize_by_length: true
  use_identity_mapping: true

optimization:
  use_lora: true
  lora_rank: 8
  use_quantization: true
```

### Production IPO

```yaml
# config/prod_ipo_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  learning_rate: 5e-7
  batch_size: 4
  num_epochs: 1
  gradient_accumulation_steps: 8

ipo:
  beta: 0.1
  tau: 0.05
  ipo_alpha: 1.2
  normalize_by_length: true
  length_normalization_type: "sqrt"
  use_identity_mapping: true
  identity_weight: 1.0
  preference_weight: 1.2
  regularization_strength: 0.01

optimization:
  use_lora: true
  lora_rank: 16
  use_gradient_checkpointing: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-ipo-prod"
```

### Research IPO

```yaml
# config/research_ipo_config.yaml
model:
  name: "Qwen/Qwen-14B"
  max_length: 4096

training:
  learning_rate: 1e-7
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 16

ipo:
  beta: 0.05
  tau: 0.02
  label_smoothing: 0.1
  ipo_grad_type: "nonlinear"
  ipo_alpha: 1.5
  length_penalty: 0.1
  normalize_by_length: true
  length_normalization_type: "log"
  regularization_strength: 0.05

data:
  filter_by_length: true
  min_length_ratio: 0.2
  max_length_ratio: 5.0
  length_difference_threshold: 50

optimization:
  use_lora: false  # Full fine-tuning
  use_gradient_checkpointing: true
  bf16: true
```

## Best Practices

### 1. Length Bias Mitigation
- Always enable length normalization
- Use appropriate normalization type for your data
- Filter extreme length differences

### 2. Parameter Tuning
- Start with lower tau values for sharper preferences
- Adjust identity/preference weights based on data quality
- Use regularization to prevent overfitting

### 3. Data Quality
- Ensure high-quality preference annotations
- Filter pairs with extreme length differences
- Balance chosen/rejected response lengths

### 4. Training Strategy
- Use lower learning rates than DPO
- Monitor both preference and identity losses
- Use early stopping based on validation metrics

## Troubleshooting

### Common Issues

#### Training Instability
```yaml
# Solution: Lower learning rate and tau
training:
  learning_rate: 5e-7  # Lower LR
ipo:
  tau: 0.05  # Lower temperature
  regularization_strength: 0.02  # Higher regularization
```

#### Length Bias Persistence
```yaml
# Solution: Stronger length normalization
ipo:
  normalize_by_length: true
  length_normalization_type: "log"  # Stronger normalization
  length_penalty: 0.1  # Add length penalty
data:
  filter_by_length: true
  length_difference_threshold: 50  # Stricter filtering
```

#### Poor Convergence
```yaml
# Solution: Adjust identity mapping weights
ipo:
  identity_weight: 1.2  # Emphasize identity term
  preference_weight: 0.8
  regularization_strength: 0.01
```

## See Also

- [DPO Configuration](dpo_config.md)
- [KTO Configuration](kto_config.md)
- [CPO Configuration](cpo_config.md)
- [IPO Training Documentation](../training/ipo_trainer.md)