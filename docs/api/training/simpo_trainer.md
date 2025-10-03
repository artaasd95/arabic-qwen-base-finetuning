# SimPOTrainer API Documentation

## Overview

The `SimPOTrainer` class implements Simple Preference Optimization (SimPO), a length-normalized preference optimization method that improves response quality by addressing length bias in preference learning.

## Class Definition

```python
from src.training.simpo_trainer import SimPOTrainer
from src.config.simpo_config import SimPOConfig

trainer = SimPOTrainer(config: SimPOConfig)
```

## Key Features

- **Length Normalization**: Addresses length bias in preference optimization
- **Improved Quality**: Focuses on response quality over length
- **Efficient Training**: Optimized for memory and computational efficiency
- **Arabic Support**: Specialized handling for Arabic text characteristics

## Configuration

### SimPOConfig Parameters

```python
from src.config.simpo_config import SimPOConfig

config = SimPOConfig(
    # Model configuration
    model_name="Qwen/Qwen2.5-3B",
    model_revision="main",
    torch_dtype="float16",
    
    # Dataset configuration
    dataset_path="data/arabic_preference_data.jsonl",
    dataset_text_field="text",
    dataset_chosen_field="chosen",
    dataset_rejected_field="rejected",
    
    # SimPO-specific parameters
    beta=2.0,                    # SimPO regularization parameter
    gamma_beta_ratio=0.5,        # Length normalization factor
    loss_type="simpo",           # Loss function type
    
    # Training parameters
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    
    # LoRA configuration
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    
    # Output configuration
    output_dir="./checkpoints/simpo",
    save_steps=500,
    logging_steps=10,
    
    # Optimization
    fp16=True,
    gradient_checkpointing=True,
    warmup_steps=100
)
```

### Required Data Format

SimPO requires preference data with chosen and rejected responses:

```json
{
  "prompt": "اشرح لي مفهوم الذكاء الاصطناعي",
  "chosen": "الذكاء الاصطناعي هو مجال في علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً...",
  "rejected": "الذكاء الاصطناعي شيء معقد جداً ولا يمكن شرحه بسهولة."
}
```

## Methods

### Core Methods

#### `__init__(config: SimPOConfig)`
Initialize the SimPO trainer with configuration.

**Parameters:**
- `config`: SimPOConfig object containing all training parameters

**Example:**
```python
from src.config.simpo_config import SimPOConfig
from src.training.simpo_trainer import SimPOTrainer

config = SimPOConfig.from_yaml("config/simpo_config.yaml")
trainer = SimPOTrainer(config)
```

#### `train(train_dataset, eval_dataset=None)`
Execute the SimPO training process.

**Parameters:**
- `train_dataset`: Training dataset with preference pairs
- `eval_dataset`: Optional evaluation dataset

**Returns:**
- Training results and metrics

**Example:**
```python
from datasets import load_dataset

# Load preference dataset
dataset = load_dataset("json", data_files="data/arabic_preferences.jsonl")
train_data = dataset["train"]

# Train the model
results = trainer.train(train_data)
print(f"Training completed. Final loss: {results.training_loss}")
```

#### `evaluate(eval_dataset)`
Evaluate the model on preference data.

**Parameters:**
- `eval_dataset`: Evaluation dataset

**Returns:**
- Evaluation metrics including preference accuracy and loss

**Example:**
```python
eval_results = trainer.evaluate(eval_dataset)
print(f"Preference accuracy: {eval_results['preference_accuracy']:.3f}")
```

### Utility Methods

#### `setup_model_and_tokenizer()`
Initialize model and tokenizer with SimPO-specific configurations.

#### `setup_lora()`
Configure LoRA adapters for efficient fine-tuning.

#### `create_trainer()`
Create the underlying TRL SimPOTrainer instance.

#### `save_model(output_dir=None)`
Save the trained model and tokenizer.

**Parameters:**
- `output_dir`: Optional custom output directory

#### `load_model(model_path)`
Load a previously trained SimPO model.

**Parameters:**
- `model_path`: Path to the saved model

## Usage Examples

### Basic Training

```python
from src.config.simpo_config import SimPOConfig
from src.training.simpo_trainer import SimPOTrainer
from datasets import load_dataset

# Load configuration
config = SimPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_path="data/arabic_preferences.jsonl",
    output_dir="./models/arabic_simpo",
    beta=2.0,
    gamma_beta_ratio=0.5,
    learning_rate=1e-6,
    num_train_epochs=1
)

# Initialize trainer
trainer = SimPOTrainer(config)

# Load dataset
dataset = load_dataset("json", data_files=config.dataset_path)
train_data = dataset["train"]

# Train model
trainer.train(train_data)

# Save model
trainer.save_model()
```

### Advanced Configuration

```python
# Advanced SimPO configuration for Arabic
config = SimPOConfig(
    # Model settings
    model_name="Qwen/Qwen2.5-7B",
    torch_dtype="float16",
    
    # SimPO parameters
    beta=2.5,                    # Higher regularization
    gamma_beta_ratio=0.3,        # More aggressive length normalization
    loss_type="simpo",
    
    # Training optimization
    learning_rate=5e-7,          # Lower learning rate for stability
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    
    # LoRA for efficiency
    use_lora=True,
    lora_r=32,                   # Higher rank for better capacity
    lora_alpha=64,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Arabic-specific settings
    max_seq_length=1024,         # Longer sequences for Arabic
    dataset_text_field="prompt",
    dataset_chosen_field="chosen_arabic",
    dataset_rejected_field="rejected_arabic",
    
    # Optimization
    fp16=True,
    gradient_checkpointing=True,
    warmup_steps=200,
    save_steps=250,
    eval_steps=250,
    logging_steps=25
)

trainer = SimPOTrainer(config)
```

### Multi-GPU Training

```python
import torch
from src.training.simpo_trainer import SimPOTrainer

# Configure for multi-GPU
config = SimPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/large_arabic_preferences.jsonl",
    
    # Multi-GPU settings
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    dataloader_num_workers=4,
    
    # Distributed training
    ddp_find_unused_parameters=False,
    ddp_backend="nccl",
    
    output_dir="./models/arabic_simpo_multi_gpu"
)

# Initialize trainer
trainer = SimPOTrainer(config)

# Train with automatic multi-GPU detection
trainer.train(train_dataset)
```

### Custom Loss Configuration

```python
# Custom SimPO loss parameters
config = SimPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    
    # SimPO loss customization
    beta=1.5,                    # Lower regularization
    gamma_beta_ratio=0.7,        # Less aggressive length normalization
    loss_type="simpo",
    
    # Additional loss parameters
    label_smoothing=0.1,         # Label smoothing for robustness
    
    # Training stability
    max_grad_norm=1.0,           # Gradient clipping
    learning_rate=2e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1
)
```

## Performance Optimization

### Memory Optimization

```python
# Memory-efficient configuration
config = SimPOConfig(
    # Model efficiency
    torch_dtype="float16",
    gradient_checkpointing=True,
    
    # Batch size optimization
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    
    # LoRA for memory efficiency
    use_lora=True,
    lora_r=16,
    
    # Sequence length management
    max_seq_length=512,          # Shorter sequences
    
    # DataLoader optimization
    dataloader_num_workers=2,
    dataloader_pin_memory=True
)
```

### Speed Optimization

```python
# Speed-optimized configuration
config = SimPOConfig(
    # Compilation optimization
    torch_compile=True,
    
    # Batch processing
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    
    # Mixed precision
    fp16=True,
    
    # Efficient attention
    use_flash_attention_2=True,
    
    # DataLoader optimization
    dataloader_num_workers=8,
    prefetch_factor=2
)
```

## Monitoring and Logging

### Training Metrics

SimPO trainer logs the following metrics:

- `train_loss`: Overall training loss
- `simpo_loss`: SimPO-specific loss component
- `preference_accuracy`: Accuracy on preference pairs
- `chosen_rewards`: Average rewards for chosen responses
- `rejected_rewards`: Average rewards for rejected responses
- `reward_margin`: Margin between chosen and rejected rewards

### Custom Logging

```python
import wandb
from src.training.simpo_trainer import SimPOTrainer

# Initialize with logging
config = SimPOConfig(
    # Logging configuration
    logging_dir="./logs/simpo",
    logging_steps=10,
    
    # Weights & Biases integration
    report_to="wandb",
    run_name="arabic_simpo_experiment",
    
    # Evaluation logging
    eval_steps=100,
    save_steps=500,
    
    # Metric logging
    metric_for_best_model="preference_accuracy",
    greater_is_better=True
)

# Initialize wandb
wandb.init(project="arabic-qwen-simpo")

trainer = SimPOTrainer(config)
```

## Error Handling

### Common Issues and Solutions

#### Memory Errors
```python
# Reduce memory usage
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 64
config.gradient_checkpointing = True
config.max_seq_length = 256
```

#### Convergence Issues
```python
# Improve convergence
config.learning_rate = 5e-7  # Lower learning rate
config.warmup_steps = 500    # More warmup
config.beta = 1.0           # Lower regularization
```

#### Data Format Errors
```python
# Validate data format
def validate_preference_data(dataset):
    required_fields = ["prompt", "chosen", "rejected"]
    for field in required_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Missing required field: {field}")
    
    # Check for empty responses
    for example in dataset:
        if not example["chosen"] or not example["rejected"]:
            raise ValueError("Empty chosen or rejected response found")
```

## Integration with Other Components

### With Model Merging

```python
from src.training.simpo_trainer import SimPOTrainer
from src.training.model_merger import ModelMerger, MergeConfig

# Train SimPO model
simpo_trainer = SimPOTrainer(simpo_config)
simpo_trainer.train(preference_dataset)

# Merge with other models
merge_config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./models/arabic_sft", "weight": 0.6},
        {"path": "./models/arabic_simpo", "weight": 0.4}
    ],
    output_path="./models/merged_arabic"
)

merger = ModelMerger(merge_config)
merged_model = merger.merge()
```

### With Dialect Processing

```python
from src.utils.arabic_dialect_utils import ArabicDialectDatasetProcessor

# Process preference data by dialect
processor = ArabicDialectDatasetProcessor()
dialect_balanced_data = processor.balance_dataset(
    dataset_path="data/preferences.jsonl",
    target_distribution={
        'msa': 0.3,
        'egy': 0.25,
        'glf': 0.25,
        'lev': 0.2
    }
)

# Train with dialect-balanced data
trainer = SimPOTrainer(config)
trainer.train(dialect_balanced_data)
```

This comprehensive API documentation provides all the information needed to effectively use the SimPOTrainer for Arabic language model fine-tuning.