# CPO Configuration Documentation

The `CPOConfig` class provides comprehensive configuration management for Contrastive Preference Optimization (CPO) in the Arabic Qwen Base Fine-tuning framework. CPO is a preference optimization method that uses contrastive learning principles.

## Class Overview

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from src.config.base_config import BaseConfig

@dataclass
class CPOConfig(BaseConfig):
    """Configuration for Contrastive Preference Optimization (CPO)."""
```

## Location

**File**: `src/config/cpo_config.py`

## CPO Overview

Contrastive Preference Optimization (CPO) is a preference optimization method that leverages contrastive learning to better align language models with human preferences. CPO uses both positive and negative examples to create more robust preference representations.

### Key Features

- **Contrastive Learning**: Uses positive and negative examples for better alignment
- **Robust Training**: More stable training dynamics than standard preference methods
- **Multi-Negative Sampling**: Supports multiple negative examples per positive
- **Temperature Scaling**: Adjustable temperature for contrastive loss

## CPO-Specific Configuration

### CPOSpecificConfig

```python
@dataclass
class CPOSpecificConfig:
    """CPO-specific configuration parameters."""
    beta: float = 0.1
    temperature: float = 0.1
    alpha: float = 1.0
    loss_type: str = "cpo"
    reference_free: bool = False
    reference_model_name: Optional[str] = None
    
    # Contrastive learning parameters
    num_negatives: int = 1
    negative_sampling_strategy: str = "random"
    hard_negative_ratio: float = 0.0
    margin: float = 0.1
    
    # CPO-specific parameters
    cpo_alpha: float = 1.0
    cpo_beta: float = 0.1
    cpo_gamma: float = 1.0
    simcse_dropout: float = 0.1
    
    # Contrastive loss parameters
    contrastive_temperature: float = 0.05
    contrastive_weight: float = 1.0
    preference_weight: float = 1.0
    
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
    
    # Advanced CPO parameters
    use_momentum: bool = False
    momentum_beta: float = 0.999
    queue_size: int = 65536
    use_queue: bool = False
    normalize_embeddings: bool = True
    projection_dim: Optional[int] = None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 0.1 | CPO beta parameter (KL regularization strength) |
| `temperature` | float | 0.1 | Temperature for preference scoring |
| `alpha` | float | 1.0 | CPO alpha parameter |
| `num_negatives` | int | 1 | Number of negative examples per positive |
| `negative_sampling_strategy` | str | "random" | Strategy for sampling negatives |
| `hard_negative_ratio` | float | 0.0 | Ratio of hard negatives to include |
| `margin` | float | 0.1 | Margin for contrastive loss |
| `cpo_alpha` | float | 1.0 | CPO-specific alpha parameter |
| `cpo_beta` | float | 0.1 | CPO-specific beta parameter |
| `cpo_gamma` | float | 1.0 | CPO-specific gamma parameter |
| `simcse_dropout` | float | 0.1 | Dropout rate for SimCSE-style training |
| `contrastive_temperature` | float | 0.05 | Temperature for contrastive loss |
| `contrastive_weight` | float | 1.0 | Weight for contrastive loss term |
| `preference_weight` | float | 1.0 | Weight for preference loss term |
| `use_momentum` | bool | False | Use momentum for contrastive learning |
| `momentum_beta` | float | 0.999 | Momentum coefficient |
| `queue_size` | int | 65536 | Size of negative queue |
| `use_queue` | bool | False | Use queue for negative sampling |
| `normalize_embeddings` | bool | True | Normalize embeddings before contrastive loss |
| `projection_dim` | Optional[int] | None | Dimension for projection head |

### CPODataConfig

```python
@dataclass
class CPODataConfig:
    """Data configuration for CPO."""
    train_file: str = "data/cpo_train.jsonl"
    validation_file: Optional[str] = "data/cpo_validation.jsonl"
    test_file: Optional[str] = None
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    preprocessing_num_workers: int = 4
    
    # Column names for CPO data
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    system_column: Optional[str] = "system"
    
    # Multi-negative support
    multiple_negatives_column: Optional[str] = "negatives"
    hard_negatives_column: Optional[str] = "hard_negatives"
    
    # Data processing parameters
    conversation_template: str = "default"
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    
    # CPO-specific data parameters
    sanity_check: bool = False
    sanity_check_size: int = 100
    ignore_bias_buffers: bool = False
    disable_dropout: bool = True
    generate_during_eval: bool = False
    
    # Contrastive data parameters
    augment_data: bool = False
    augmentation_strategy: str = "paraphrase"
    augmentation_ratio: float = 0.1
    create_hard_negatives: bool = False
    hard_negative_strategy: str = "semantic_search"
```

## Usage Examples

### Basic CPO Configuration

```python
from src.config.cpo_config import CPOConfig

# Create default CPO configuration
config = CPOConfig()

# Access CPO-specific parameters
print(f"Beta parameter: {config.cpo.beta}")
print(f"Temperature: {config.cpo.temperature}")
print(f"Number of negatives: {config.cpo.num_negatives}")
```

### Custom CPO Configuration

```python
from src.config.cpo_config import CPOConfig, CPOSpecificConfig

# Create CPO-specific configuration
cpo_config = CPOSpecificConfig(
    beta=0.2,
    temperature=0.05,  # Lower temperature for sharper contrasts
    num_negatives=3,  # Multiple negatives per positive
    contrastive_temperature=0.02,
    contrastive_weight=1.5,  # Emphasize contrastive learning
    use_momentum=True,
    normalize_embeddings=True
)

# Create full configuration
config = CPOConfig(cpo=cpo_config)
config.to_yaml("my_cpo_config.yaml")
```

### Loading from YAML

```python
from src.config.cpo_config import CPOConfig

# Load configuration from YAML file
config = CPOConfig.from_yaml("config/cpo_config.yaml")

# Override specific parameters
config.cpo.num_negatives = 5
config.training.learning_rate = 1e-6
```

## YAML Configuration Example

```yaml
# config/cpo_config.yaml
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
  train_file: "data/cpo_train.jsonl"
  validation_file: "data/cpo_validation.jsonl"
  prompt_column: "prompt"
  chosen_column: "chosen"
  rejected_column: "rejected"
  multiple_negatives_column: "negatives"
  augment_data: true
  augmentation_ratio: 0.1

cpo:
  beta: 0.1
  temperature: 0.1
  alpha: 1.0
  
  # Contrastive learning
  num_negatives: 2
  negative_sampling_strategy: "random"
  hard_negative_ratio: 0.2
  margin: 0.1
  
  # CPO parameters
  cpo_alpha: 1.0
  cpo_beta: 0.1
  cpo_gamma: 1.0
  simcse_dropout: 0.1
  
  # Contrastive loss
  contrastive_temperature: 0.05
  contrastive_weight: 1.0
  preference_weight: 1.0
  
  # Advanced features
  use_momentum: false
  use_queue: false
  normalize_embeddings: true
  
  max_length: 2048
  max_prompt_length: 1024

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-cpo"
```

## Data Format

CPO supports multiple data formats for different use cases:

### Basic Pairwise Format

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد.", "rejected": "باريس."}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً.", "rejected": "الذكاء الاصطناعي شيء معقد."}
```

### Multi-Negative Format

```jsonl
{
  "prompt": "ما هي عاصمة فرنسا؟",
  "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد ومركزها السياسي والثقافي.",
  "rejected": "باريس.",
  "negatives": [
    "لا أعرف.",
    "فرنسا ليس لها عاصمة.",
    "مدينة ليون."
  ]
}
```

### Hard Negatives Format

```jsonl
{
  "prompt": "اشرح مفهوم الذكاء الاصطناعي",
  "chosen": "الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً مثل التعلم والاستدلال وحل المشكلات.",
  "rejected": "الذكاء الاصطناعي شيء معقد.",
  "hard_negatives": [
    "الذكاء الاصطناعي هو نوع من الكمبيوتر.",
    "الذكاء الاصطناعي يعني الروبوتات الذكية فقط."
  ]
}
```

## CPO Algorithm Configuration

### Negative Sampling Strategies

#### Random Sampling
```python
config.cpo.negative_sampling_strategy = "random"
config.cpo.num_negatives = 2
```

#### Hard Negative Sampling
```python
config.cpo.negative_sampling_strategy = "hard"
config.cpo.hard_negative_ratio = 0.5
config.cpo.num_negatives = 3
```

#### Mixed Sampling
```python
config.cpo.negative_sampling_strategy = "mixed"
config.cpo.hard_negative_ratio = 0.3
config.cpo.num_negatives = 4
```

### Contrastive Learning Configuration

#### Standard Contrastive Learning
```python
config.cpo.contrastive_temperature = 0.05
config.cpo.contrastive_weight = 1.0
config.cpo.normalize_embeddings = True
```

#### Momentum Contrastive Learning
```python
config.cpo.use_momentum = True
config.cpo.momentum_beta = 0.999
config.cpo.queue_size = 65536
config.cpo.use_queue = True
```

### Temperature Scaling

#### Sharp Contrasts
```python
config.cpo.temperature = 0.05
config.cpo.contrastive_temperature = 0.02
```

#### Smooth Contrasts
```python
config.cpo.temperature = 0.2
config.cpo.contrastive_temperature = 0.1
```

## Configuration Templates

### Quick Start CPO

```yaml
# config/quick_cpo_config.yaml
model:
  name: "Qwen/Qwen-1_8B"
  max_length: 1024

training:
  learning_rate: 1e-6
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 8

data:
  train_file: "data/cpo_train_small.jsonl"
  max_samples: 1000

cpo:
  beta: 0.1
  temperature: 0.1
  num_negatives: 1
  contrastive_weight: 1.0

optimization:
  use_lora: true
  lora_rank: 8
  use_quantization: true
```

### Production CPO

```yaml
# config/prod_cpo_config.yaml
model:
  name: "Qwen/Qwen-7B"
  max_length: 2048

training:
  learning_rate: 5e-7
  batch_size: 4
  num_epochs: 1
  gradient_accumulation_steps: 8

cpo:
  beta: 0.1
  temperature: 0.05
  num_negatives: 3
  negative_sampling_strategy: "mixed"
  hard_negative_ratio: 0.3
  contrastive_temperature: 0.02
  contrastive_weight: 1.2
  use_momentum: true
  normalize_embeddings: true

optimization:
  use_lora: true
  lora_rank: 16
  use_gradient_checkpointing: true
  bf16: true

logging:
  use_wandb: true
  wandb_project: "arabic-qwen-cpo-prod"
```

### Research CPO

```yaml
# config/research_cpo_config.yaml
model:
  name: "Qwen/Qwen-14B"
  max_length: 4096

training:
  learning_rate: 1e-7
  batch_size: 2
  num_epochs: 1
  gradient_accumulation_steps: 16

cpo:
  beta: 0.05
  temperature: 0.02
  num_negatives: 5
  negative_sampling_strategy: "hard"
  hard_negative_ratio: 0.6
  margin: 0.2
  
  # Advanced contrastive learning
  contrastive_temperature: 0.01
  contrastive_weight: 2.0
  use_momentum: true
  momentum_beta: 0.9999
  queue_size: 131072
  use_queue: true
  
  # Projection head
  projection_dim: 256
  normalize_embeddings: true

data:
  augment_data: true
  augmentation_ratio: 0.2
  create_hard_negatives: true
  hard_negative_strategy: "semantic_search"

optimization:
  use_lora: false  # Full fine-tuning
  use_gradient_checkpointing: true
  bf16: true
```

## Best Practices

### 1. Negative Sampling
- Start with 2-3 negatives per positive
- Use mixed sampling for better diversity
- Include hard negatives for challenging examples

### 2. Temperature Tuning
- Lower temperatures for sharper contrasts
- Adjust based on data difficulty
- Use different temperatures for preference and contrastive losses

### 3. Contrastive Learning
- Normalize embeddings for stable training
- Use momentum for large-scale training
- Adjust contrastive weight based on data quality

### 4. Data Preparation
- Ensure high-quality preference annotations
- Create diverse negative examples
- Balance positive and negative examples

## Troubleshooting

### Common Issues

#### Training Instability
```yaml
# Solution: Lower temperatures and learning rate
training:
  learning_rate: 1e-7  # Lower LR
cpo:
  temperature: 0.1  # Higher temperature
  contrastive_temperature: 0.05
  contrastive_weight: 0.5  # Lower weight
```

#### Poor Contrastive Learning
```yaml
# Solution: Adjust contrastive parameters
cpo:
  num_negatives: 5  # More negatives
  hard_negative_ratio: 0.4  # More hard negatives
  contrastive_temperature: 0.02  # Lower temperature
  normalize_embeddings: true
```

#### Memory Issues
```yaml
# Solution: Reduce batch size and negatives
training:
  batch_size: 2
  gradient_accumulation_steps: 16
cpo:
  num_negatives: 2  # Fewer negatives
  use_queue: false  # Disable queue
optimization:
  use_gradient_checkpointing: true
```

## See Also

- [DPO Configuration](dpo_config.md)
- [IPO Configuration](ipo_config.md)
- [KTO Configuration](kto_config.md)
- [CPO Training Documentation](../training/cpo_trainer.md)