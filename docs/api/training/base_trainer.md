# BaseTrainer API Documentation

The `BaseTrainer` class provides the foundation for all training methods in the Arabic Qwen fine-tuning system.

## Overview

**Location**: `src/training/base_trainer.py`

**Purpose**: Provides common functionality and abstract interface for all training methods including model loading, tokenizer setup, LoRA configuration, and training utilities.

## Class Definition

```python
class BaseTrainer:
    """Base trainer class providing common functionality for all training methods."""
    
    def __init__(self, config):
        """Initialize the base trainer with configuration."""
```

## Initialization

### Constructor

```python
def __init__(self, config)
```

**Parameters:**
- `config`: Training configuration object (SFTConfig, DPOConfig, etc.)

**Functionality:**
- Sets up logging
- Initializes model and tokenizer to None
- Stores configuration
- Initializes trainer and data loader to None

**Example:**
```python
from src.training.base_trainer import BaseTrainer
from src.config import SFTConfig

config = SFTConfig(model_name="Qwen/Qwen2.5-7B")
trainer = BaseTrainer(config)
```

## Core Methods

### Model and Tokenizer Setup

#### `setup_model_and_tokenizer()`

```python
def setup_model_and_tokenizer(self) -> None
```

**Purpose**: Initialize model and tokenizer with quantization and optimization settings.

**Functionality:**
- Loads model with specified quantization (4-bit/8-bit)
- Sets up tokenizer with proper padding configuration
- Configures model for training (gradient checkpointing, etc.)
- Handles device placement and memory optimization

**Quantization Support:**
- **4-bit quantization**: Uses BitsAndBytesConfig with NF4
- **8-bit quantization**: Uses BitsAndBytesConfig with int8
- **Mixed precision**: Supports bfloat16 and float16

**Example:**
```python
trainer = BaseTrainer(config)
trainer.setup_model_and_tokenizer()
print(f"Model loaded: {trainer.model}")
print(f"Tokenizer vocab size: {len(trainer.tokenizer)}")
```

#### `_setup_padding_token()`

```python
def _setup_padding_token(self) -> None
```

**Purpose**: Configure padding token for the tokenizer.

**Logic:**
1. If no pad_token exists, sets it to eos_token
2. If no eos_token exists, sets it to unk_token
3. Ensures model embedding is resized if needed

### LoRA Configuration

#### `setup_lora()`

```python
def setup_lora(self) -> None
```

**Purpose**: Configure Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

**Configuration Parameters:**
- `lora_r`: Rank of adaptation (default: 16)
- `lora_alpha`: LoRA scaling parameter (default: 32)
- `lora_dropout`: Dropout rate (default: 0.1)
- `lora_target_modules`: Target modules for LoRA (default: ["q_proj", "v_proj"])
- `lora_bias`: Bias handling (default: "none")
- `lora_task_type`: Task type (default: "CAUSAL_LM")

**Example:**
```python
# Configure LoRA in config
config.use_lora = True
config.lora_r = 32
config.lora_alpha = 64
config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

trainer = BaseTrainer(config)
trainer.setup_model_and_tokenizer()
trainer.setup_lora()
```

### Data Loading

#### `get_data_loader()`

```python
def get_data_loader(self)
```

**Purpose**: Get appropriate data loader based on training method.

**Returns**: Data loader instance for the specific training method

**Supported Methods:**
- `"sft"`: SFTDataLoader
- `"dpo"`: DPODataLoader
- `"kto"`: KTODataLoader
- `"ipo"`: IPODataLoader
- `"cpo"`: CPODataLoader

**Example:**
```python
trainer = BaseTrainer(config)
data_loader = trainer.get_data_loader()
print(f"Data loader type: {type(data_loader)}")
```

#### `load_and_validate_dataset()`

```python
def load_and_validate_dataset(self)
```

**Purpose**: Load and validate training dataset.

**Returns**: Loaded dataset (dict or Dataset object)

**Functionality:**
- Uses appropriate data loader for training method
- Validates dataset format and content
- Handles train/validation splits
- Logs dataset statistics

**Example:**
```python
trainer = BaseTrainer(config)
dataset = trainer.load_and_validate_dataset()
print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset.get('validation', []))}")
```

### Training Arguments

#### `get_training_arguments()`

```python
def get_training_arguments(self) -> TrainingArguments
```

**Purpose**: Create TrainingArguments object with method-specific settings.

**Returns**: Configured TrainingArguments instance

**Base Configuration:**
```python
TrainingArguments(
    output_dir=self.config.output_dir,
    learning_rate=self.config.learning_rate,
    num_train_epochs=self.config.num_train_epochs,
    per_device_train_batch_size=self.config.per_device_train_batch_size,
    per_device_eval_batch_size=self.config.per_device_eval_batch_size,
    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
    warmup_ratio=self.config.warmup_ratio,
    weight_decay=self.config.weight_decay,
    logging_steps=self.config.logging_steps,
    evaluation_strategy=self.config.evaluation_strategy,
    eval_steps=self.config.eval_steps,
    save_strategy=self.config.save_strategy,
    save_steps=self.config.save_steps,
    save_total_limit=self.config.save_total_limit,
    load_best_model_at_end=self.config.load_best_model_at_end,
    metric_for_best_model=self.config.metric_for_best_model,
    greater_is_better=False,
    report_to=self.config.report_to,
    run_name=self.config.run_name,
    seed=self.config.seed,
    data_seed=self.config.data_seed,
    bf16=self.config.bf16,
    fp16=self.config.fp16,
    gradient_checkpointing=self.config.gradient_checkpointing,
    max_grad_norm=self.config.max_grad_norm,
    dataloader_num_workers=self.config.dataloader_num_workers,
    dataloader_pin_memory=self.config.dataloader_pin_memory,
    group_by_length=self.config.group_by_length,
    length_column_name=self.config.length_column_name,
    ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
    ddp_backend=self.config.ddp_backend,
    deepspeed=self.config.deepspeed
)
```

### Callbacks

#### `get_callbacks()`

```python
def get_callbacks(self) -> List
```

**Purpose**: Get training callbacks including early stopping.

**Default Callbacks:**
- `EarlyStoppingCallback`: Stops training when validation metric stops improving

**Configuration:**
```python
config.early_stopping_patience = 3  # Stop after 3 evaluations without improvement
config.early_stopping_threshold = 0.001  # Minimum improvement threshold
```

**Custom Callbacks:**
```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed")

config.callbacks = [CustomCallback()]
```

## Abstract Methods

These methods must be implemented by subclasses:

### `create_trainer()`

```python
@abstractmethod
def create_trainer(self, dataset, training_args, callbacks)
```

**Purpose**: Create method-specific trainer instance.

**Parameters:**
- `dataset`: Training dataset
- `training_args`: TrainingArguments object
- `callbacks`: List of callbacks

**Returns**: Trainer instance

### `_get_training_method()`

```python
@abstractmethod
def _get_training_method(self) -> str
```

**Purpose**: Return the training method name.

**Returns**: Method name string ("sft", "dpo", "kto", "ipo", "cpo")

## Training Pipeline

### `train()`

```python
def train(self) -> None
```

**Purpose**: Execute the complete training pipeline.

**Pipeline Steps:**
1. Setup model and tokenizer
2. Setup LoRA (if enabled)
3. Load and validate dataset
4. Get training arguments
5. Get callbacks
6. Create trainer
7. Execute training
8. Save model

**Example:**
```python
trainer = BaseTrainer(config)
trainer.train()  # Executes complete pipeline
```

### Training Flow Diagram

```
┌─────────────────────┐
│ Initialize Trainer  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Setup Model &       │
│ Tokenizer           │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Setup LoRA          │
│ (if enabled)        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Load & Validate     │
│ Dataset             │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Get Training        │
│ Arguments           │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Get Callbacks       │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Create Trainer      │
│ (method-specific)   │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Execute Training    │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Save Model          │
└─────────────────────┘
```

## Model Management

### `save_model()`

```python
def save_model(self, output_dir: Optional[str] = None) -> None
```

**Purpose**: Save trained model and configuration.

**Parameters:**
- `output_dir`: Directory to save model (optional, uses config.output_dir if None)

**Functionality:**
- Saves LoRA adapters or full model based on configuration
- Saves tokenizer
- Saves training configuration
- Creates model card with training details

**LoRA Saving:**
```python
# Saves LoRA adapters only
if config.use_lora:
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
```

**Full Model Saving:**
```python
# Merges LoRA and saves full model
if config.merge_and_save:
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
```

### `load_model()`

```python
def load_model(self, model_path: str) -> None
```

**Purpose**: Load previously trained model.

**Parameters:**
- `model_path`: Path to saved model

**Functionality:**
- Loads LoRA adapters or full model
- Loads tokenizer
- Restores model configuration

**Example:**
```python
trainer = BaseTrainer(config)
trainer.load_model("./checkpoints/sft_arabic")
response = trainer.predict("Hello in Arabic")
```

## Evaluation and Prediction

### `evaluate()`

```python
def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]
```

**Purpose**: Evaluate model performance.

**Parameters:**
- `eval_dataset`: Evaluation dataset (optional)

**Returns**: Dictionary of evaluation metrics

**Example:**
```python
results = trainer.evaluate()
print(f"Evaluation loss: {results['eval_loss']}")
print(f"Perplexity: {results.get('eval_perplexity', 'N/A')}")
```

### `predict()`

```python
def predict(self, text: str, max_length: int = 512, **kwargs) -> str
```

**Purpose**: Generate text using the trained model.

**Parameters:**
- `text`: Input text/prompt
- `max_length`: Maximum generation length
- `**kwargs`: Additional generation parameters

**Returns**: Generated text

**Generation Parameters:**
```python
response = trainer.predict(
    text="Explain machine learning",
    max_length=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1
)
```

## Configuration Integration

### Supported Configuration Types

```python
from src.config import (
    SFTConfig,
    DPOConfig,
    KTOConfig,
    IPOConfig,
    CPOConfig
)

# Each config type provides method-specific parameters
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/sft.jsonl",
    use_lora=True,
    lora_r=16
)
```

### Configuration Validation

The BaseTrainer validates configuration parameters:

```python
def __init__(self, config):
    # Validate required parameters
    if not hasattr(config, 'model_name'):
        raise ValueError("model_name is required")
    
    if not hasattr(config, 'dataset_path'):
        raise ValueError("dataset_path is required")
    
    # Set defaults for optional parameters
    if not hasattr(config, 'use_lora'):
        config.use_lora = True
```

## Error Handling

### Common Exceptions

```python
class TrainingError(Exception):
    """Base exception for training errors."""
    pass

class ModelLoadError(TrainingError):
    """Exception raised when model loading fails."""
    pass

class DatasetError(TrainingError):
    """Exception raised when dataset loading/validation fails."""
    pass

class ConfigurationError(TrainingError):
    """Exception raised when configuration is invalid."""
    pass
```

### Error Handling Examples

```python
try:
    trainer = BaseTrainer(config)
    trainer.train()
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
except DatasetError as e:
    logger.error(f"Dataset error: {e}")
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

## Performance Optimization

### Memory Optimization

```python
# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use quantization
config.load_in_4bit = True

# Use LoRA for parameter efficiency
config.use_lora = True
config.lora_r = 8  # Smaller rank for less memory

# Optimize batch size
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16
```

### Speed Optimization

```python
# Use mixed precision
config.bf16 = True

# Optimize data loading
config.dataloader_num_workers = 4
config.dataloader_pin_memory = True

# Use efficient attention
config.use_flash_attention = True
```

## Logging and Monitoring

### Logging Configuration

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom logging in trainer
class BaseTrainer:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
```

### Training Monitoring

```python
# Monitor training progress
config.logging_steps = 10
config.eval_steps = 100
config.save_steps = 500

# Use wandb for experiment tracking
config.report_to = "wandb"
config.run_name = "arabic_qwen_sft"
```

## Best Practices

### 1. Configuration Management
```python
# Use environment variables for sensitive data
import os
config.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B")
config.output_dir = os.getenv("OUTPUT_DIR", "./checkpoints")
```

### 2. Resource Management
```python
# Clear GPU memory after training
import torch
torch.cuda.empty_cache()
```

### 3. Reproducibility
```python
# Set seeds for reproducibility
config.seed = 42
config.data_seed = 42
```

### 4. Validation
```python
# Always use validation set
config.evaluation_strategy = "steps"
config.eval_steps = 500
config.load_best_model_at_end = True
```

## Integration Examples

### Custom Trainer Implementation

```python
from src.training.base_trainer import BaseTrainer
from transformers import Trainer

class CustomTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _get_training_method(self) -> str:
        return "custom"
    
    def create_trainer(self, dataset, training_args, callbacks):
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
```

### Multi-Stage Training

```python
# Stage 1: SFT
sft_config = SFTConfig(model_name="Qwen/Qwen2.5-7B")
sft_trainer = SFTTrainer(sft_config)
sft_trainer.train()

# Stage 2: DPO using SFT model
dpo_config = DPOConfig(model_name="./checkpoints/sft")
dpo_trainer = DPOTrainer(dpo_config)
dpo_trainer.train()
```

This comprehensive documentation covers all aspects of the BaseTrainer class, providing both theoretical understanding and practical examples for effective usage.