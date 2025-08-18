# KTO Configuration Documentation

The `KTOConfig` class provides comprehensive configuration management for Kahneman-Tversky Optimization (KTO) in the Arabic Qwen Base Fine-tuning framework. KTO is a preference optimization method inspired by Kahneman-Tversky prospect theory.

## Class Overview

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from src.config.base_config import BaseConfig

@dataclass
class KTOConfig(BaseConfig):
    """Configuration for Kahneman-Tversky Optimization (KTO)."""
```

## Location

**File**: `src/config/kto_config.py`

## KTO Overview

Kahneman-Tversky Optimization (KTO) is a preference optimization method that models human preferences using prospect theory principles. Unlike DPO which requires paired preferences, KTO can work with binary feedback (desirable/undesirable) for individual responses.

### Key Concepts

- **Prospect Theory**: Models human decision-making under uncertainty
- **Binary Feedback**: Uses desirable/undesirable labels instead of pairwise preferences
- **Loss Aversion**: Incorporates asymmetric treatment of gains and losses
- **Reference Point**: Uses a reference model or point for comparison

## KTO-Specific Configuration

### KTOSpecificConfig

```python
@dataclass
class KTOSpecificConfig:
    """KTO-specific configuration parameters."""
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    reference_free: bool = False
    reference_model_name: Optional[str] = None
    loss_type: str = "kto"
    kto_alpha: float = 1.0
    kto_beta_s: float = 1.0
    kto_gamma: float = 0.88
    kto_lambda: float = 2.25
    use_weighting: bool = True
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
| `beta` | float | 0.1 | KTO beta parameter (KL regularization strength) |
| `desirable_weight` | float | 1.0 | Weight for desirable responses |
| `undesirable_weight` | float | 1.0 | Weight for undesirable responses |
| `reference_free` | bool | False | Whether to use reference-free KTO |
| `reference_model_name` | Optional[str] | None | Reference model name |
| `loss_type` | str | "kto" | Loss function type |
| `kto_alpha` | float | 1.0 | KTO alpha parameter (reference point) |
| `kto_beta_s` | float | 1.0 | KTO beta_s parameter (sensitivity) |
| `kto_gamma` | float | 0.88 | KTO gamma parameter (probability weighting) |
| `kto_lambda` | float | 2.25 | KTO lambda parameter (loss aversion) |
| `use_weighting` | bool | True | Use importance weighting |
| `weighting_temperature` | float | 1.0 | Temperature for importance weighting |

### KTODataConfig

```python
@dataclass
class KTODataConfig:
    """Data configuration for KTO."""
    train_file: str = "data/kto_train.jsonl"
    validation_file: Optional[str] = "data/kto_validation.jsonl"
    test_file: Optional[str] = None
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    preprocessing_num_workers: int = 4
    
    # Column names for KTO data
    prompt_column: str = "prompt"
    response_column: str = "response"
    label_column: str = "label"  # "desirable" or "undesirable"
    system_column: Optional[str] = "system"
    
    # Data processing parameters
    conversation_template: str = "default"
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    
    # KTO-specific data parameters
    sanity_check: bool = False
    sanity_check_size: int = 100
    ignore_bias_buffers: bool = False
    disable_dropout: bool = True
    generate_during_eval: bool = False
    balance_labels: bool = True
    label_mapping: Dict[str, int] = field(default_factory=lambda: {
        "desirable": 1,
        "undesirable": 0,
        "good": 1,
        "bad": 0,
        "positive": 1,
        "negative": 0
    })
```

## Usage Examples

### Basic KTO Configuration

```python
from src.config.kto_config import KTOConfig

# Create default KTO configuration
config = KTOConfig()

# Access KTO-specific parameters
print(f"Beta parameter: {config.kto.beta}")
print(f"Desirable weight: {config.kto.desirable_weight}")
print(f"Lambda (loss aversion): {config.kto.kto_lambda}")
```

### Custom KTO Configuration

```python
from src.config.kto_config import KTOConfig, KTOSpecificConfig

# Create KTO-specific configuration
kto_config = KTOSpecificConfig(
    beta=0.2,
    desirable_weight=1.2,  # Slightly higher weight for desirable responses
    undesirable_weight=0.8,  # Lower weight for undesirable responses
    kto_lambda=2.5,  # Higher loss aversion
    kto_gamma=0.9,  # Probability weighting
    use_weighting=True
)

# Create full configuration
config = KTOConfig(kto=kto_config)
config.to_yaml("my_kto_config.yaml")
```

## YAML Configuration Example

```yaml
# config/kto_config.yaml
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
  train_file: "data/kto_train.jsonl"
  validation_file: "data/kto_validation.jsonl"
  prompt_column: "prompt"
  response_column: "response"
  label_column: "label"
  balance_labels: true
  label_mapping:
    desirable: 1
    undesirable: 0
    good: 1
    bad: 0

kto:
  beta: 0.1
  desirable_weight: 1.0
  undesirable_weight: 1.0
  reference_free: false
  kto_alpha: 1.0
  kto_beta_s: 1.0
  kto_gamma: 0.88
  kto_lambda: 2.25
  use_weighting: true
  max_length: 2048
  max_prompt_length: 1024

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-kto"
```

## Data Format

KTO requires data with binary labels for individual responses:

### JSONL Format

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "response": "عاصمة فرنسا هي باريس.", "label": "desirable"}
{"prompt": "ما هي عاصمة فرنسا؟", "response": "لا أعرف.", "label": "undesirable"}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "response": "الذكاء الاصطناعي هو مجال يهدف إلى إنشاء أنظمة ذكية.", "label": "desirable"}
```

### Alternative Label Formats

```jsonl
{"prompt": "سؤال", "response": "إجابة جيدة", "label": "good"}
{"prompt": "سؤال", "response": "إجابة سيئة", "label": "bad"}
{"prompt": "سؤال", "response": "إجابة إيجابية", "label": "positive"}
{"prompt": "سؤال", "response": "إجابة سلبية", "label": "negative"}
```

## KTO Algorithm Parameters

### Prospect Theory Parameters

#### Alpha (Reference Point)
```python
config.kto.kto_alpha = 1.0  # Reference point for utility calculation
```

#### Beta_s (Sensitivity)
```python
config.kto.kto_beta_s = 1.0  # Sensitivity to gains and losses
```

#### Gamma (Probability Weighting)
```python
config.kto.kto_gamma = 0.88  # Probability weighting function parameter
```

#### Lambda (Loss Aversion)
```python
config.kto.kto_lambda = 2.25  # Loss aversion coefficient
# Higher values = more loss averse
```

### Weight Configuration

#### Balanced Weighting
```python
config.kto.desirable_weight = 1.0
config.kto.undesirable_weight = 1.0
```

#### Emphasize Positive Examples
```python
config.kto.desirable_weight = 1.5
config.kto.undesirable_weight = 1.0
```

#### Emphasize Negative Examples
```python
config.kto.desirable_weight = 1.0
config.kto.undesirable_weight = 1.5
```

## Configuration Templates

### Quick Start KTO

```yaml
# config/quick_kto_config.yaml
model:
  name: "Qwen/Qwen-1_8B"
  max_length: 1024

training:
  learning_rate: 1e-6
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 8

data:
  train_file: "data/kto_train_small.jsonl"
  max_samples: 1000
  balance_labels: true

kto:
  beta: 0.1
  reference_free: true
  kto_lambda: 2.0

optimization:
  use_lora: true
  lora_rank: 8
  use_quantization: true
```

### Production KTO

```yaml
# config/prod_kto_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  learning_rate: 5e-7
  batch_size: 4
  num_epochs: 1
  gradient_accumulation_steps: 8

kto:
  beta: 0.1
  desirable_weight: 1.2
  undesirable_weight: 0.8
  kto_lambda: 2.5
  kto_gamma: 0.9
  use_weighting: true

optimization:
  use_lora: true
  lora_rank: 16
  use_gradient_checkpointing: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-kto-prod"
```

## Best Practices

### 1. Data Preparation
- Ensure balanced desirable/undesirable examples
- Use high-quality human annotations
- Consider inter-annotator agreement

### 2. Parameter Tuning
- Start with default prospect theory parameters
- Adjust lambda for loss aversion sensitivity
- Tune weights based on data distribution

### 3. Training Strategy
- Use lower learning rates than SFT
- Monitor both desirable and undesirable loss
- Use early stopping based on validation metrics

## See Also

- [DPO Configuration](dpo_config.md)
- [IPO Configuration](ipo_config.md)
- [KTO Training Documentation](../training/kto_trainer.md)