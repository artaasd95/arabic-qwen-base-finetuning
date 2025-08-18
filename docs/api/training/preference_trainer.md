# PreferenceTrainer API Documentation

The `PreferenceTrainer` class serves as a base class for preference optimization methods (KTO, IPO, CPO) for Arabic Qwen models, providing a unified interface for different preference learning approaches.

## Overview

**Location**: `src/training/preference_trainer.py`

**Purpose**: Unified base class for preference optimization methods that don't require explicit preference pairs (KTO) or use different optimization objectives (IPO, CPO).

**Inheritance**: `PreferenceTrainer` → `BaseTrainer`

**Supported Methods:**
- **KTO (Kahneman-Tversky Optimization)**: Uses individual preference signals
- **IPO (Identity Preference Optimization)**: Alternative to DPO with identity mapping
- **CPO (Conservative Preference Optimization)**: Conservative approach to preference learning

## Class Definition

```python
class PreferenceTrainer(BaseTrainer):
    """Base trainer for preference optimization methods (KTO, IPO, CPO).
    
    This class provides a unified interface for different preference optimization
    methods that extend beyond traditional DPO approaches.
    """
```

## Initialization

### Constructor

```python
def __init__(self, config: Union[KTOConfig, IPOConfig, CPOConfig])
```

**Parameters:**
- `config`: Configuration object for KTO, IPO, or CPO training

**Raises:**
- `TypeError`: If config is not a supported preference optimization config
- `ValueError`: If preference method is not recognized

**Initialization Process:**
1. Calls parent BaseTrainer constructor
2. Determines preference optimization method from config type
3. Validates configuration
4. Initializes reference model to None
5. Sets up method-specific parameters

**Example:**
```python
from src.training.preference_trainer import PreferenceTrainer
from src.config import KTOConfig, IPOConfig, CPOConfig

# KTO configuration
kto_config = KTOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/kto_data.jsonl",
    output_dir="./checkpoints/kto",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0
)

# IPO configuration
ipo_config = IPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/preference_pairs.jsonl",
    output_dir="./checkpoints/ipo",
    beta=0.1,
    label_smoothing=0.0
)

# CPO configuration
cpo_config = CPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/preference_pairs.jsonl",
    output_dir="./checkpoints/cpo",
    beta=0.1,
    label_smoothing=0.0,
    loss_type="sigmoid"
)

kto_trainer = PreferenceTrainer(kto_config)
ipo_trainer = PreferenceTrainer(ipo_config)
cpo_trainer = PreferenceTrainer(cpo_config)
```

## Core Methods

### Training Method Identification

#### `_get_training_method()`

```python
def _get_training_method(self) -> str
```

**Returns**: Method name ("kto", "ipo", or "cpo")

**Purpose**: Identifies the specific preference optimization method for data loading and processing.

**Method Detection:**
```python
if isinstance(self.config, KTOConfig):
    return "kto"
elif isinstance(self.config, IPOConfig):
    return "ipo"
elif isinstance(self.config, CPOConfig):
    return "cpo"
```

### Reference Model Setup

#### `setup_reference_model()`

```python
def setup_reference_model(self) -> None
```

**Purpose**: Initialize the reference model for preference optimization.

**Method-Specific Behavior:**
- **KTO**: Reference model is optional (can be disabled)
- **IPO/CPO**: Reference model is required

**Process:**
1. Check if reference model is needed for the method
2. Load reference model (usually same as base model)
3. Apply quantization if specified
4. Set model to evaluation mode
5. Disable gradients for reference model
6. Move to appropriate device

**KTO Without Reference Model:**
```python
kto_config = KTOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/kto_data.jsonl",
    use_ref_model=False  # Disable reference model for KTO
)
```

**Example:**
```python
trainer = PreferenceTrainer(config)
trainer.setup_reference_model()

if trainer.ref_model is not None:
    print(f"Reference model loaded: {trainer.ref_model}")
    print(f"Reference model device: {next(trainer.ref_model.parameters()).device}")
else:
    print("Training without reference model (KTO mode)")
```

### Trainer Creation

#### `create_trainer()`

```python
def create_trainer(self, dataset, training_args, callbacks)
```

**Purpose**: Create and configure the appropriate preference optimization trainer.

**Parameters:**
- `dataset`: Training dataset (format depends on method)
- `training_args`: TrainingArguments object
- `callbacks`: List of training callbacks

**Returns**: Configured trainer instance from TRL library

**Dynamic Trainer Selection:**
```python
if self.preference_method == "kto":
    from trl import KTOTrainer
    trainer_class = KTOTrainer
elif self.preference_method == "ipo":
    from trl import DPOTrainer  # IPO uses DPO trainer with different loss
    trainer_class = DPOTrainer
elif self.preference_method == "cpo":
    from trl import CPOTrainer
    trainer_class = CPOTrainer
```

**Dataset Processing:**
Each method requires different data formats and processing:

```python
# Method-specific tokenization
if self.preference_method == "kto":
    # KTO format: individual preference signals
    train_dataset = train_dataset.map(
        self.data_loader.tokenize_and_format_kto,
        batched=True
    )
elif self.preference_method in ["ipo", "cpo"]:
    # IPO/CPO format: preference pairs like DPO
    train_dataset = train_dataset.map(
        self.data_loader.tokenize_and_format,
        batched=True
    )
```

### Method-Specific Arguments

#### `_get_method_specific_args()`

```python
def _get_method_specific_args(self) -> Dict[str, Any]
```

**Purpose**: Get method-specific arguments for trainer initialization.

**Returns**: Dictionary of method-specific parameters

**KTO Arguments:**
```python
{
    "beta": config.beta,
    "desirable_weight": config.desirable_weight,
    "undesirable_weight": config.undesirable_weight,
    "max_length": config.max_length,
    "max_prompt_length": config.max_prompt_length
}
```

**IPO Arguments:**
```python
{
    "beta": config.beta,
    "loss_type": "ipo",  # Specific IPO loss
    "label_smoothing": config.label_smoothing,
    "max_length": config.max_length,
    "max_prompt_length": config.max_prompt_length
}
```

**CPO Arguments:**
```python
{
    "beta": config.beta,
    "loss_type": config.loss_type,
    "label_smoothing": config.label_smoothing,
    "max_length": config.max_length,
    "max_prompt_length": config.max_prompt_length,
    "cpo_alpha": config.cpo_alpha  # CPO-specific parameter
}
```

### Training Arguments

#### `get_training_arguments()`

```python
def get_training_arguments(self) -> TrainingArguments
```

**Purpose**: Get method-specific training arguments.

**Returns**: TrainingArguments with method-specific modifications

**Common Preference Optimization Settings:**
```python
# Memory and computation optimization
training_args.remove_unused_columns = False
training_args.gradient_checkpointing = True

# Method-specific label names
if self.preference_method == "kto":
    training_args.label_names = ["completion_labels"]
elif self.preference_method in ["ipo", "cpo"]:
    training_args.label_names = [
        "chosen_labels", "rejected_labels",
        "chosen_input_ids", "rejected_input_ids",
        "chosen_attention_mask", "rejected_attention_mask"
    ]
```

**Learning Rate Adjustments:**
```python
# Method-specific learning rate optimization
if self.preference_method in ["ipo", "cpo"]:
    # Lower learning rate for IPO/CPO stability
    training_args.learning_rate = min(training_args.learning_rate, 5e-7)
elif self.preference_method == "kto":
    # KTO can handle slightly higher learning rates
    training_args.learning_rate = min(training_args.learning_rate, 1e-6)
```

## Evaluation

### `evaluate_model()`

```python
def evaluate_model(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]
```

**Purpose**: Evaluate the preference optimization model.

**Parameters:**
- `eval_dataset`: Evaluation dataset (optional)

**Returns**: Dictionary containing evaluation metrics

**Method-Specific Metrics:**
- **KTO**: KTO loss, desirable/undesirable accuracy
- **IPO**: IPO loss, preference accuracy
- **CPO**: CPO loss, preference accuracy, conservatism metrics

**Example:**
```python
results = trainer.evaluate_model()

if trainer.preference_method == "kto":
    print(f"KTO Loss: {results['eval_loss']:.4f}")
    print(f"Desirable Accuracy: {results.get('eval_desirable_accuracy', 'N/A')}")
elif trainer.preference_method in ["ipo", "cpo"]:
    print(f"{trainer.preference_method.upper()} Loss: {results['eval_loss']:.4f}")
    print(f"Preference Accuracy: {results.get('eval_preference_accuracy', 'N/A')}")
```

### `compute_preference_metrics()`

```python
def compute_preference_metrics(self, eval_dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Compute method-specific preference metrics.

**Parameters:**
- `eval_dataset`: Evaluation dataset

**Returns**: Dictionary of method-specific metrics

**KTO Metrics:**
```python
{
    "kto_loss": float,
    "desirable_accuracy": float,  # Accuracy on desirable examples
    "undesirable_accuracy": float,  # Accuracy on undesirable examples
    "overall_accuracy": float
}
```

**IPO/CPO Metrics:**
```python
{
    "preference_loss": float,
    "preference_accuracy": float,  # How often chosen > rejected
    "margin": float,  # Average margin between chosen and rejected
    "confidence": float  # Model confidence in preferences
}
```

## Text Generation and Comparison

### `generate_with_comparison()`

```python
def generate_with_comparison(
    self, 
    prompt: str, 
    max_length: int = 512, 
    **kwargs
) -> Dict[str, Any]
```

**Purpose**: Generate responses and compare with reference model (if available).

**Parameters:**
- `prompt`: Input prompt
- `max_length`: Maximum generation length
- `**kwargs`: Additional generation parameters

**Returns**: Dictionary with generation results and comparisons

**KTO Response Format:**
```python
{
    "prompt": "Original prompt",
    "trained_model_response": "Response from trained model",
    "reference_model_response": "Response from reference model (if available)",
    "preference_signal": "desirable/undesirable",  # KTO-specific
    "confidence_score": float
}
```

**IPO/CPO Response Format:**
```python
{
    "prompt": "Original prompt",
    "trained_model_response": "Response from trained model",
    "reference_model_response": "Response from reference model",
    "preference_score": float,  # Relative preference score
    "improvement_detected": bool
}
```

**Example:**
```python
comparison = trainer.generate_with_comparison(
    "اشرح مفهوم الذكاء الاصطناعي بطريقة بسيطة",
    max_length=256,
    temperature=0.7
)

print(f"Prompt: {comparison['prompt']}")
print(f"Trained Model: {comparison['trained_model_response']}")

if comparison.get('reference_model_response'):
    print(f"Reference Model: {comparison['reference_model_response']}")

if trainer.preference_method == "kto":
    print(f"Preference Signal: {comparison['preference_signal']}")
else:
    print(f"Preference Score: {comparison['preference_score']:.3f}")
```

## Model Saving

### `save_model()`

```python
def save_model(self, output_dir: Optional[str] = None) -> None
```

**Purpose**: Save the preference optimization model with method-specific metadata.

**Parameters:**
- `output_dir`: Directory to save model (optional)

**Saved Files:**
- Model weights (LoRA adapters or full model)
- Tokenizer files
- Configuration files
- Method-specific metadata file

**KTO Metadata (`kto_metadata.json`):**
```json
{
  "training_method": "kto",
  "beta": 0.1,
  "desirable_weight": 1.0,
  "undesirable_weight": 1.0,
  "max_prompt_length": 256,
  "max_length": 512,
  "use_ref_model": true,
  "ref_model_name": "Qwen/Qwen2.5-7B",
  "training_completed_at": "2024-01-15T10:30:00Z"
}
```

**IPO Metadata (`ipo_metadata.json`):**
```json
{
  "training_method": "ipo",
  "beta": 0.1,
  "label_smoothing": 0.0,
  "max_prompt_length": 256,
  "max_length": 512,
  "ref_model_name": "Qwen/Qwen2.5-7B",
  "training_completed_at": "2024-01-15T10:30:00Z"
}
```

**CPO Metadata (`cpo_metadata.json`):**
```json
{
  "training_method": "cpo",
  "beta": 0.1,
  "cpo_alpha": 1.0,
  "loss_type": "sigmoid",
  "label_smoothing": 0.0,
  "max_prompt_length": 256,
  "max_length": 512,
  "ref_model_name": "Qwen/Qwen2.5-7B",
  "training_completed_at": "2024-01-15T10:30:00Z"
}
```

## Configuration

### KTO Configuration

```python
from src.config import KTOConfig

kto_config = KTOConfig(
    # Base parameters
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/kto_data.jsonl",
    output_dir="./checkpoints/kto",
    
    # KTO-specific parameters
    beta=0.1,  # KL penalty coefficient
    desirable_weight=1.0,  # Weight for desirable examples
    undesirable_weight=1.0,  # Weight for undesirable examples
    max_prompt_length=256,
    max_length=512,
    use_ref_model=True,  # Whether to use reference model
    
    # Training parameters
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4
)
```

### IPO Configuration

```python
from src.config import IPOConfig

ipo_config = IPOConfig(
    # Base parameters
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/preference_pairs.jsonl",
    output_dir="./checkpoints/ipo",
    
    # IPO-specific parameters
    beta=0.1,  # Regularization strength
    label_smoothing=0.0,  # Label smoothing factor
    max_prompt_length=256,
    max_length=512,
    
    # Training parameters (IPO-optimized)
    learning_rate=5e-7,  # Lower learning rate
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8
)
```

### CPO Configuration

```python
from src.config import CPOConfig

cpo_config = CPOConfig(
    # Base parameters
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/preference_pairs.jsonl",
    output_dir="./checkpoints/cpo",
    
    # CPO-specific parameters
    beta=0.1,  # Regularization strength
    cpo_alpha=1.0,  # Conservative penalty weight
    loss_type="sigmoid",  # Loss function type
    label_smoothing=0.0,
    max_prompt_length=256,
    max_length=512,
    
    # Training parameters
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8
)
```

## Data Formats

### KTO Data Format

**Individual Preference Signals:**
```json
{"prompt": "ما هو الذكاء الاصطناعي؟", "completion": "الذكاء الاصطناعي هو...", "label": true}
{"prompt": "اشرح التعلم الآلي", "completion": "التعلم الآلي صعب...", "label": false}
```

**Extended KTO Format:**
```json
{
  "prompt": "Explain quantum computing",
  "completion": "Quantum computing is a revolutionary technology...",
  "label": true,
  "confidence": 0.9,
  "annotator_id": "expert_1"
}
```

**KTO Dataset Split:**
```json
{
  "train": [
    {"prompt": "...", "completion": "...", "label": true},
    {"prompt": "...", "completion": "...", "label": false}
  ],
  "validation": [
    {"prompt": "...", "completion": "...", "label": true},
    {"prompt": "...", "completion": "...", "label": false}
  ]
}
```

### IPO/CPO Data Format

**Same as DPO - Preference Pairs:**
```json
{"prompt": "ما هو الذكاء الاصطناعي؟", "chosen": "الذكاء الاصطناعي هو مجال...", "rejected": "الذكاء الاصطناعي شيء معقد..."}
```

## Performance Optimization

### Memory Optimization

```python
# Method-specific memory optimization
if config.preference_method == "kto":
    # KTO uses less memory (single completion per example)
    config.per_device_train_batch_size = 4
    config.gradient_accumulation_steps = 4
else:
    # IPO/CPO use more memory (preference pairs)
    config.per_device_train_batch_size = 2
    config.gradient_accumulation_steps = 8

# Common optimizations
config.gradient_checkpointing = True
config.use_lora = True
config.load_in_4bit = True  # For reference model
```

### Speed Optimization

```python
# Mixed precision
config.bf16 = True

# Efficient data loading
config.dataloader_num_workers = 2
config.dataloader_pin_memory = True

# Method-specific evaluation frequency
if config.preference_method == "kto":
    config.eval_steps = 500  # More frequent for KTO
else:
    config.eval_steps = 1000  # Less frequent for IPO/CPO
```

## Usage Examples

### KTO Training

```python
from src.training.preference_trainer import create_kto_trainer
from src.config import KTOConfig

# Configure KTO training
config = KTOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_kto.jsonl",
    output_dir="./checkpoints/kto_arabic",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=4
)

# Create and train
trainer = create_kto_trainer(config)
trainer.train()

# Evaluate
results = trainer.evaluate_model()
print(f"KTO Loss: {results['eval_loss']:.4f}")
print(f"Desirable Accuracy: {results.get('eval_desirable_accuracy', 'N/A')}")
```

### IPO Training

```python
from src.training.preference_trainer import create_ipo_trainer
from src.config import IPOConfig

# Configure IPO training
config = IPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_preferences.jsonl",
    output_dir="./checkpoints/ipo_arabic",
    beta=0.1,
    label_smoothing=0.0,
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2
)

# Create and train
trainer = create_ipo_trainer(config)
trainer.train()

# Evaluate
results = trainer.evaluate_model()
print(f"IPO Loss: {results['eval_loss']:.4f}")
```

### CPO Training

```python
from src.training.preference_trainer import create_cpo_trainer
from src.config import CPOConfig

# Configure CPO training
config = CPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_preferences.jsonl",
    output_dir="./checkpoints/cpo_arabic",
    beta=0.1,
    cpo_alpha=1.0,
    loss_type="sigmoid",
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2
)

# Create and train
trainer = create_cpo_trainer(config)
trainer.train()

# Evaluate
results = trainer.evaluate_model()
print(f"CPO Loss: {results['eval_loss']:.4f}")
```

### Multi-Method Comparison

```python
# Train multiple preference optimization methods
methods = {
    "kto": create_kto_trainer(kto_config),
    "ipo": create_ipo_trainer(ipo_config),
    "cpo": create_cpo_trainer(cpo_config)
}

results = {}
for method_name, trainer in methods.items():
    print(f"Training {method_name.upper()}...")
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate_model()
    results[method_name] = eval_results
    
    # Generate comparison
    comparison = trainer.generate_with_comparison(
        "اشرح أهمية التعليم في المجتمع",
        max_length=256
    )
    
    print(f"{method_name.upper()} Response: {comparison['trained_model_response']}")
    print("-" * 50)

# Compare results
for method, result in results.items():
    print(f"{method.upper()} Loss: {result['eval_loss']:.4f}")
```

## Factory Functions

### `create_preference_trainer()`

```python
def create_preference_trainer(
    config: Union[KTOConfig, IPOConfig, CPOConfig]
) -> PreferenceTrainer
```

**Purpose**: General factory function for any preference optimization method.

### `create_kto_trainer()`

```python
def create_kto_trainer(config: KTOConfig) -> PreferenceTrainer
```

**Purpose**: Factory function specifically for KTO training.

### `create_ipo_trainer()`

```python
def create_ipo_trainer(config: IPOConfig) -> PreferenceTrainer
```

**Purpose**: Factory function specifically for IPO training.

### `create_cpo_trainer()`

```python
def create_cpo_trainer(config: CPOConfig) -> PreferenceTrainer
```

**Purpose**: Factory function specifically for CPO training.

**Example:**
```python
from src.training.preference_trainer import (
    create_preference_trainer,
    create_kto_trainer,
    create_ipo_trainer,
    create_cpo_trainer
)

# Using general factory
trainer = create_preference_trainer(config)  # Auto-detects method

# Using specific factories
kto_trainer = create_kto_trainer(kto_config)
ipo_trainer = create_ipo_trainer(ipo_config)
cpo_trainer = create_cpo_trainer(cpo_config)
```

## Best Practices

### 1. Method Selection
- **Use KTO** when you have individual preference signals rather than pairs
- **Use IPO** as an alternative to DPO with potentially better stability
- **Use CPO** when you want conservative preference learning

### 2. Data Preparation
- **KTO**: Ensure balanced distribution of desirable/undesirable examples
- **IPO/CPO**: Use high-quality preference pairs with clear distinctions
- **All methods**: Include diverse prompts and responses

### 3. Hyperparameter Tuning
- Start with beta=0.1 for all methods
- Adjust method-specific parameters based on results
- Monitor both loss and method-specific metrics
- Use lower learning rates for stability

### 4. Training Monitoring
- Track method-specific metrics (desirable accuracy for KTO, preference accuracy for IPO/CPO)
- Monitor generation quality through comparisons
- Use early stopping based on validation metrics
- Log sample generations during training

### 5. Arabic-Specific Considerations
- Ensure preference data reflects Arabic language nuances
- Include both formal and informal Arabic examples
- Consider cultural context in preference judgments
- Test with various Arabic dialects

This comprehensive documentation provides all the information needed to effectively use the PreferenceTrainer for various preference optimization methods with Arabic language models.