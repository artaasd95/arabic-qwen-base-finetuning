# Training Modules Documentation

This documentation covers the training system for Arabic Qwen model fine-tuning, including all supported training methods, their implementations, and configuration guides.

## Overview

The training system is built with a modular architecture that supports multiple fine-tuning approaches:

- **Supervised Fine-Tuning (SFT)**: Traditional instruction-following training
- **Direct Preference Optimization (DPO)**: Learning from preference pairs
- **Kahneman-Tversky Optimization (KTO)**: Human feedback optimization
- **Identity Preference Optimization (IPO)**: Length-bias aware preference learning
- **Contrastive Preference Optimization (CPO)**: Multi-negative contrastive learning
- **Simple Preference Optimization (SimPO)**: Length-normalized preference optimization

## Architecture

### Base Trainer (`src/training/base_trainer.py`)

The `BaseTrainer` class provides common functionality for all training methods:

```python
from src.training.base_trainer import BaseTrainer
from src.config import SFTConfig

# All trainers inherit from BaseTrainer
class CustomTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
```

**Key Features:**
- Model and tokenizer loading with quantization support
- LoRA configuration and setup
- Data loader initialization
- Common training utilities
- Model saving and loading

**Core Methods:**
- `setup_model_and_tokenizer()`: Initialize model and tokenizer
- `setup_lora()`: Configure LoRA adapters
- `get_data_loader()`: Get appropriate data loader for training method
- `load_and_validate_dataset()`: Load and validate training data
- `create_trainer()`: Abstract method for trainer creation
- `train()`: Main training pipeline
- `save_model()` / `load_model()`: Model persistence

### Training Method Hierarchy

```
BaseTrainer
├── SFTTrainer (Supervised Fine-Tuning)
├── DPOTrainer (Direct Preference Optimization)
└── PreferenceTrainer (Base for preference methods)
    ├── KTO (Kahneman-Tversky Optimization)
    ├── IPO (Identity Preference Optimization)
    └── CPO (Contrastive Preference Optimization)
```

## Training Methods Comparison

| Method | Data Format | Use Case | Key Benefits | Limitations |
|--------|-------------|----------|--------------|-------------|
| **SFT** | Instruction-response pairs | General instruction following | Simple, stable, fast | Limited preference learning |
| **DPO** | Preference pairs (chosen/rejected) | Preference alignment | No reward model needed | Requires high-quality pairs |
| **KTO** | Binary feedback (desirable/undesirable) | Human feedback optimization | Works with simple feedback | May need more data |
| **IPO** | Length-aware preferences | Mitigating length bias | Handles length bias well | More complex setup |
| **CPO** | Multi-negative preferences | Contrastive learning | Rich negative sampling | Computationally intensive |
| **SimPO** | Preference pairs with length normalization | Quality-focused preference learning | Improved response quality | Requires careful tuning |

## Method-Specific Documentation

### 1. Supervised Fine-Tuning (SFT)

**Location**: `src/training/sft_trainer.py`

**Purpose**: Train models on instruction-response pairs for general instruction following.

**Data Format**:
```json
{
  "instruction": "Translate to Arabic: Hello world",
  "response": "مرحبا بالعالم"
}
```

**Configuration Example**:
```python
from src.config import SFTConfig

config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/sft_arabic.jsonl",
    output_dir="./checkpoints/sft",
    max_seq_length=512,
    packing=True,
    instruction_template="alpaca",
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4
)
```

**Key Features**:
- Instruction template support (Alpaca, ChatML, etc.)
- Sequence packing for efficiency
- Gradient checkpointing
- Perplexity evaluation
- Response generation utilities

**Usage**:
```python
from src.training.sft_trainer import create_sft_trainer

trainer = create_sft_trainer(config)
trainer.train()

# Generate responses
response = trainer.generate_response(
    "Explain machine learning in Arabic",
    max_length=256
)
```

### 2. Direct Preference Optimization (DPO)

**Location**: `src/training/dpo_trainer.py`

**Purpose**: Learn from preference pairs without requiring a separate reward model.

**Data Format**:
```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum bits...",
  "rejected": "Quantum computing is just regular computing..."
}
```

**Configuration Example**:
```python
from src.config import DPOConfig

config = DPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/dpo_preferences.jsonl",
    output_dir="./checkpoints/dpo",
    beta=0.1,  # DPO temperature parameter
    loss_type="sigmoid",
    max_prompt_length=256,
    max_length=512,
    learning_rate=5e-7,  # Lower LR for DPO
    num_train_epochs=1
)
```

**Key Features**:
- Reference model setup and management
- Preference pair processing
- DPO loss computation
- Preference accuracy evaluation
- Model comparison utilities

**Usage**:
```python
from src.training.dpo_trainer import create_dpo_trainer

trainer = create_dpo_trainer(config)
trainer.train()

# Compare responses
comparison = trainer.generate_comparison(
    "What is artificial intelligence?",
    max_length=256
)
print(f"Trained: {comparison['trained_model_response']}")
print(f"Reference: {comparison['reference_model_response']}")
```

### 3. Preference Optimization Methods (KTO, IPO, CPO)

**Location**: `src/training/preference_trainer.py`

**Purpose**: Base class for advanced preference optimization methods.

#### Kahneman-Tversky Optimization (KTO)

**Data Format**:
```json
{
  "prompt": "Write a poem about nature",
  "completion": "The trees whisper secrets...",
  "label": true  // true for desirable, false for undesirable
}
```

**Configuration**:
```python
from src.config import KTOConfig

config = KTOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/kto_feedback.jsonl",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    learning_rate=1e-6
)
```

#### Identity Preference Optimization (IPO)

**Data Format**:
```json
{
  "prompt": "Summarize this article",
  "chosen": "Brief summary...",
  "rejected": "Very long detailed summary...",
  "chosen_length": 50,
  "rejected_length": 200
}
```

**Configuration**:
```python
from src.config import IPOConfig

config = IPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/ipo_preferences.jsonl",
    beta=0.1,
    tau=0.1,  # IPO regularization parameter
    label_smoothing=0.0,
    learning_rate=5e-7
)
```

#### Contrastive Preference Optimization (CPO)

**Data Format**:
```json
{
  "prompt": "Explain climate change",
  "chosen": "Climate change refers to...",
  "rejected": ["Wrong explanation 1", "Wrong explanation 2"],
  "hard_negatives": ["Subtle wrong explanation"]
}
```

**Configuration**:
```python
from src.config import CPOConfig

config = CPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/cpo_preferences.jsonl",
    beta=0.1,
    cpo_alpha=1.0,
    simpo_gamma=1.0,
    loss_type="sigmoid",
    learning_rate=5e-7
)
```

**Usage for Preference Methods**:
```python
from src.training.preference_trainer import (
    create_kto_trainer,
    create_ipo_trainer,
    create_cpo_trainer
)

# KTO Training
kto_trainer = create_kto_trainer(kto_config)
kto_trainer.train()

# IPO Training
ipo_trainer = create_ipo_trainer(ipo_config)
ipo_trainer.train()

# CPO Training
cpo_trainer = create_cpo_trainer(cpo_config)
cpo_trainer.train()
```

## Configuration Guide

### Common Configuration Parameters

All training methods share these base parameters:

```python
# Model and Data
model_name: str = "Qwen/Qwen2.5-7B"
dataset_path: str = "data/training.jsonl"
output_dir: str = "./checkpoints"

# Training Parameters
learning_rate: float = 2e-4
num_train_epochs: int = 3
per_device_train_batch_size: int = 4
per_device_eval_batch_size: int = 8
gradient_accumulation_steps: int = 1

# Model Configuration
max_seq_length: int = 512
torch_dtype: str = "bfloat16"
load_in_4bit: bool = False
load_in_8bit: bool = False

# LoRA Configuration
use_lora: bool = True
lora_r: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.1
lora_target_modules: List[str] = ["q_proj", "v_proj"]

# Training Optimization
gradient_checkpointing: bool = True
warmup_ratio: float = 0.1
weight_decay: float = 0.01
max_grad_norm: float = 1.0

# Evaluation and Saving
evaluation_strategy: str = "steps"
eval_steps: int = 500
save_strategy: str = "steps"
save_steps: int = 1000
save_total_limit: int = 3
load_best_model_at_end: bool = True
metric_for_best_model: str = "eval_loss"
```

### Method-Specific Parameters

#### SFT-Specific
```python
packing: bool = True  # Pack multiple sequences
instruction_template: str = "alpaca"  # Template format
response_template: str = "### Response:\n"  # Response prefix
```

#### DPO-Specific
```python
beta: float = 0.1  # DPO temperature
loss_type: str = "sigmoid"  # Loss function type
label_smoothing: float = 0.0  # Label smoothing factor
max_prompt_length: int = 256  # Maximum prompt length
ref_model_name: str = None  # Reference model (defaults to base model)
```

#### KTO-Specific
```python
beta: float = 0.1  # KTO temperature
desirable_weight: float = 1.0  # Weight for desirable examples
undesirable_weight: float = 1.0  # Weight for undesirable examples
```

#### IPO-Specific
```python
beta: float = 0.1  # IPO temperature
tau: float = 0.1  # Regularization parameter
label_smoothing: float = 0.0  # Label smoothing
```

#### CPO-Specific
```python
beta: float = 0.1  # CPO temperature
cpo_alpha: float = 1.0  # CPO loss weight
simpo_gamma: float = 1.0  # SimPO parameter
loss_type: str = "sigmoid"  # Loss function
```

## Training Pipeline

### Standard Training Flow

```python
# 1. Create configuration
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_instructions.jsonl",
    output_dir="./checkpoints/sft_arabic"
)

# 2. Create trainer
trainer = create_sft_trainer(config)

# 3. Train model
trainer.train()

# 4. Evaluate model
eval_results = trainer.evaluate()
print(f"Evaluation loss: {eval_results['eval_loss']}")

# 5. Save model
trainer.save_model()

# 6. Generate responses
response = trainer.predict("Explain AI in Arabic")
print(response)
```

### Advanced Training with Custom Callbacks

```python
from transformers import EarlyStoppingCallback, TrainerCallback

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, **kwargs):
        # Custom logging logic
        pass

# Add callbacks to config
config.callbacks = [
    EarlyStoppingCallback(early_stopping_patience=3),
    CustomLoggingCallback()
]

trainer = create_sft_trainer(config)
trainer.train()
```

## Performance Optimization

### Memory Optimization

```python
# Use quantization
config.load_in_4bit = True
config.load_in_8bit = False  # Don't use both

# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use smaller batch sizes with gradient accumulation
config.per_device_train_batch_size = 2
config.gradient_accumulation_steps = 8  # Effective batch size = 16

# Use LoRA for parameter efficiency
config.use_lora = True
config.lora_r = 16  # Smaller rank for less memory
```

### Speed Optimization

```python
# Use sequence packing (SFT only)
config.packing = True

# Optimize data loading
config.dataloader_num_workers = 4
config.dataloader_pin_memory = True

# Use mixed precision
config.fp16 = True  # or bf16 = True

# Disable unnecessary evaluations
config.evaluation_strategy = "no"  # For faster training
```

### Multi-GPU Training

```python
# Use DeepSpeed for large models
config.deepspeed = "configs/deepspeed_zero2.json"

# Or use DDP
config.ddp_backend = "nccl"
config.ddp_find_unused_parameters = False
```

## Best Practices

### 1. Data Preparation
- **Quality over Quantity**: Use high-quality, diverse datasets
- **Arabic-Specific**: Include proper Arabic text normalization
- **Balanced Datasets**: Ensure balanced representation across domains
- **Validation Split**: Always keep a validation set for monitoring

### 2. Hyperparameter Tuning
- **Learning Rate**: Start with method-specific defaults, then tune
- **Batch Size**: Balance memory constraints with training stability
- **Sequence Length**: Match your target use case requirements
- **LoRA Rank**: Higher rank for complex tasks, lower for efficiency

### 3. Training Monitoring
- **Loss Curves**: Monitor training and validation loss
- **Gradient Norms**: Watch for gradient explosion/vanishing
- **Learning Rate**: Use learning rate scheduling
- **Early Stopping**: Prevent overfitting with patience

### 4. Model Evaluation
- **Multiple Metrics**: Don't rely on loss alone
- **Human Evaluation**: Include human assessment for quality
- **Domain-Specific**: Test on your specific use cases
- **Comparison**: Compare against baseline models

### 5. Arabic Language Considerations
- **Tokenization**: Ensure proper Arabic tokenization
- **Text Direction**: Handle RTL text correctly
- **Diacritics**: Consider diacritical marks in evaluation
- **Dialects**: Account for different Arabic dialects

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```python
# Solutions:
# 1. Reduce batch size
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16

# 2. Use quantization
config.load_in_4bit = True

# 3. Enable gradient checkpointing
config.gradient_checkpointing = True

# 4. Use smaller model or LoRA
config.use_lora = True
config.lora_r = 8
```

#### Training Instability
```python
# Solutions:
# 1. Lower learning rate
config.learning_rate = 1e-5

# 2. Add gradient clipping
config.max_grad_norm = 0.5

# 3. Use warmup
config.warmup_ratio = 0.1

# 4. Increase batch size
config.per_device_train_batch_size = 8
```

#### Poor Performance
```python
# Solutions:
# 1. Check data quality
# 2. Increase training epochs
config.num_train_epochs = 5

# 3. Tune hyperparameters
# 4. Use different training method
# 5. Increase model size or LoRA rank
```

#### Slow Training
```python
# Solutions:
# 1. Enable packing (SFT)
config.packing = True

# 2. Use mixed precision
config.bf16 = True

# 3. Optimize data loading
config.dataloader_num_workers = 4

# 4. Use gradient accumulation
config.gradient_accumulation_steps = 4
```

## Examples

See the `examples/` directory for complete training examples:
- `examples/sft_training.py`: SFT training example
- `examples/dpo_training.py`: DPO training example
- `examples/preference_training.py`: KTO/IPO/CPO examples
- `examples/multi_method_comparison.py`: Comparing different methods

## API Reference

For detailed API documentation, see:
- [Base Trainer API](base_trainer.md)
- [SFT Trainer API](sft_trainer.md)
- [DPO Trainer API](dpo_trainer.md)
- [Preference Trainer API](preference_trainer.md)
- [CPO Trainer API](cpo_trainer.md)
- [SimPO Trainer API](simpo_trainer.md)
- [Model Merger API](model_merger.md)