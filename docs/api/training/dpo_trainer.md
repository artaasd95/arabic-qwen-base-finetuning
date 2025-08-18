# DPOTrainer API Documentation

The `DPOTrainer` class implements Direct Preference Optimization (DPO) for Arabic Qwen models using preference pairs without requiring a separate reward model.

## Overview

**Location**: `src/training/dpo_trainer.py`

**Purpose**: Train models using preference pairs (chosen/rejected responses) to align model outputs with human preferences through direct optimization.

**Inheritance**: `DPOTrainer` → `BaseTrainer`

**Key Advantages:**
- No separate reward model required
- Direct optimization on preference data
- Stable training process
- Memory efficient compared to RLHF

## Class Definition

```python
class DPOTrainer(BaseTrainer):
    """Trainer for Direct Preference Optimization (DPO).
    
    This class handles the complete DPO training pipeline including
    preference data loading, model setup, and DPO training execution.
    """
```

## Initialization

### Constructor

```python
def __init__(self, config: DPOConfig)
```

**Parameters:**
- `config`: DPOConfig object containing DPO training configuration

**Raises:**
- `TypeError`: If config is not a DPOConfig instance

**Initialization Process:**
1. Calls parent BaseTrainer constructor
2. Validates configuration type
3. Initializes reference model to None
4. Sets up logging

**Example:**
```python
from src.training.dpo_trainer import DPOTrainer
from src.config import DPOConfig

config = DPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/preference_pairs.jsonl",
    output_dir="./checkpoints/dpo",
    beta=0.1
)

trainer = DPOTrainer(config)
```

## Core Methods

### Training Method Identification

#### `_get_training_method()`

```python
def _get_training_method(self) -> str
```

**Returns**: `"dpo"`

**Purpose**: Identifies this trainer as using the DPO method for data loading and processing.

### Reference Model Setup

#### `setup_reference_model()`

```python
def setup_reference_model(self) -> None
```

**Purpose**: Initialize the reference model for DPO training.

**Process:**
1. Load reference model (usually same as base model)
2. Apply quantization if specified
3. Set model to evaluation mode
4. Disable gradients for reference model
5. Move to appropriate device

**Configuration:**
```python
# Reference model configuration
config.ref_model_name = "Qwen/Qwen2.5-7B"  # Can be different from main model
config.load_in_4bit = True  # Apply quantization to reference model
```

**Quantization Support:**
- **4-bit quantization**: Uses BitsAndBytesConfig with NF4
- **8-bit quantization**: Uses BitsAndBytesConfig with int8
- **Mixed precision**: Supports bfloat16 and float16

**Example:**
```python
trainer = DPOTrainer(config)
trainer.setup_reference_model()
print(f"Reference model loaded: {trainer.ref_model}")
print(f"Reference model device: {next(trainer.ref_model.parameters()).device}")
```

### Trainer Creation

#### `create_trainer()`

```python
def create_trainer(self, dataset, training_args, callbacks)
```

**Purpose**: Create and configure the DPO trainer instance using TRL library.

**Parameters:**
- `dataset`: Training dataset with preference pairs
- `training_args`: TrainingArguments object
- `callbacks`: List of training callbacks

**Returns**: Configured `DPOTrainer` instance from TRL library

**Process:**
1. Import TRL DPOTrainer
2. Setup reference model
3. Prepare and tokenize datasets
4. Get DPO-specific arguments
5. Create and return DPO trainer

**Dataset Processing:**
```python
# Tokenization and formatting
train_dataset = train_dataset.map(
    self.data_loader.tokenize_and_format,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training data"
)
```

**Example:**
```python
# Internal usage - called automatically during training
trainer_instance = dpo_trainer.create_trainer(dataset, training_args, callbacks)
print(f"DPO trainer created with {len(dataset['train'])} samples")
```

### Training Arguments

#### `get_training_arguments()`

```python
def get_training_arguments(self) -> TrainingArguments
```

**Purpose**: Get DPO-specific training arguments.

**Returns**: TrainingArguments with DPO-specific modifications

**DPO-Specific Settings:**
```python
# Memory and computation optimization
training_args.remove_unused_columns = False  # Keep all columns for DPO
training_args.gradient_checkpointing = True   # Enable for memory efficiency

# DPO-specific label names
training_args.label_names = [
    "chosen_labels", "rejected_labels", 
    "chosen_input_ids", "rejected_input_ids",
    "chosen_attention_mask", "rejected_attention_mask"
]

# Lower learning rate for stability
if not hasattr(config, 'learning_rate_override'):
    training_args.learning_rate = min(training_args.learning_rate, 5e-7)
```

**Learning Rate Adjustment:**
DPO typically requires much lower learning rates than SFT to maintain training stability.

## Evaluation

### `evaluate_model()`

```python
def evaluate_model(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]
```

**Purpose**: Evaluate the DPO model performance.

**Parameters:**
- `eval_dataset`: Evaluation dataset (optional)

**Returns**: Dictionary containing evaluation metrics

**Metrics:**
- `eval_loss`: DPO loss on evaluation set
- `eval_runtime`: Evaluation time
- `eval_samples_per_second`: Evaluation speed
- Additional DPO-specific metrics from TRL

**Example:**
```python
results = trainer.evaluate_model()
print(f"DPO Loss: {results['eval_loss']:.4f}")
print(f"Evaluation time: {results['eval_runtime']:.2f}s")
```

### `compute_preference_accuracy()`

```python
def compute_preference_accuracy(self, eval_dataset: Dataset) -> float
```

**Purpose**: Compute preference accuracy on evaluation dataset.

**Parameters:**
- `eval_dataset`: Evaluation dataset with preference pairs

**Returns**: Preference accuracy (0-1)

**Calculation:**
Measures how often the model prefers the chosen response over the rejected response.

**Example:**
```python
accuracy = trainer.compute_preference_accuracy(eval_dataset)
print(f"Preference accuracy: {accuracy:.2%}")
```

## Text Generation and Comparison

### `generate_comparison()`

```python
def generate_comparison(
    self, 
    prompt: str, 
    max_length: int = 512, 
    **kwargs
) -> Dict[str, str]
```

**Purpose**: Generate responses from both trained and reference models for comparison.

**Parameters:**
- `prompt`: Input prompt
- `max_length`: Maximum generation length (default: 512)
- `**kwargs`: Additional generation parameters

**Returns**: Dictionary with responses from both models

**Response Format:**
```python
{
    "prompt": "Original prompt",
    "trained_model_response": "Response from trained model",
    "reference_model_response": "Response from reference model"
}
```

**Generation Parameters:**
```python
comparison = trainer.generate_comparison(
    prompt="اشرح مفهوم الذكاء الاصطناعي",
    max_length=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)
```

**Example:**
```python
comparison = trainer.generate_comparison(
    "What are the benefits of renewable energy?",
    max_length=512
)

print(f"Prompt: {comparison['prompt']}")
print(f"\nTrained Model: {comparison['trained_model_response']}")
print(f"\nReference Model: {comparison['reference_model_response']}")

# Analyze improvement
if len(comparison['trained_model_response']) > len(comparison['reference_model_response']):
    print("\nTrained model provides more detailed response")
```

## Model Saving

### `save_model()`

```python
def save_model(self, output_dir: Optional[str] = None) -> None
```

**Purpose**: Save the DPO model with metadata.

**Parameters:**
- `output_dir`: Directory to save model (optional, uses config.output_dir if None)

**Saved Files:**
- Model weights (LoRA adapters or full model)
- Tokenizer files
- Configuration files
- `dpo_metadata.json`: DPO-specific metadata

**DPO Metadata:**
```json
{
  "training_method": "dpo",
  "beta": 0.1,
  "loss_type": "sigmoid",
  "label_smoothing": 0.0,
  "max_prompt_length": 256,
  "max_length": 512,
  "ref_model_name": "Qwen/Qwen2.5-7B",
  "training_completed_at": "2024-01-15T10:30:00Z"
}
```

**Example:**
```python
# Save to default directory
trainer.save_model()

# Save to custom directory
trainer.save_model("./models/arabic_dpo_v1")
```

## Configuration

### DPO-Specific Configuration Parameters

```python
from src.config import DPOConfig

config = DPOConfig(
    # Base parameters
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/preference_pairs.jsonl",
    output_dir="./checkpoints/dpo",
    
    # DPO-specific parameters
    beta=0.1,  # DPO temperature parameter (higher = more conservative)
    loss_type="sigmoid",  # Loss function type
    label_smoothing=0.0,  # Label smoothing factor
    max_prompt_length=256,  # Maximum prompt length
    max_length=512,  # Maximum total sequence length
    ref_model_name=None,  # Reference model (defaults to base model)
    
    # Training parameters (DPO-optimized)
    learning_rate=5e-7,  # Lower learning rate for stability
    num_train_epochs=1,  # Usually fewer epochs needed
    per_device_train_batch_size=2,  # Smaller batches due to memory usage
    gradient_accumulation_steps=8,  # Compensate for smaller batches
    
    # Optimization
    gradient_checkpointing=True,
    remove_unused_columns=False,  # Keep all columns for DPO
    
    # LoRA configuration
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

### Key DPO Parameters

#### Beta (β)
- **Range**: 0.01 - 1.0
- **Default**: 0.1
- **Effect**: Controls the strength of the KL penalty
- **Higher values**: More conservative, closer to reference model
- **Lower values**: More aggressive optimization

#### Loss Type
- **sigmoid**: Standard DPO loss (recommended)
- **hinge**: Hinge loss variant
- **ipo**: Identity Preference Optimization loss

#### Label Smoothing
- **Range**: 0.0 - 0.1
- **Default**: 0.0
- **Effect**: Smooths the preference labels to reduce overconfidence

## Data Format

### Expected Input Format

**JSONL Format:**
```json
{"prompt": "ما هو الذكاء الاصطناعي؟", "chosen": "الذكاء الاصطناعي هو مجال...", "rejected": "الذكاء الاصطناعي شيء معقد..."}
{"prompt": "اشرح التعلم الآلي", "chosen": "التعلم الآلي هو فرع من...", "rejected": "التعلم الآلي صعب جداً..."}
```

**Extended Format with Metadata:**
```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing is a revolutionary technology...",
  "rejected": "Quantum computing is just regular computing...",
  "chosen_score": 8.5,
  "rejected_score": 3.2,
  "annotator_id": "expert_1"
}
```

**Dataset Split:**
```json
{
  "train": [
    {"prompt": "...", "chosen": "...", "rejected": "..."},
    {"prompt": "...", "chosen": "...", "rejected": "..."}
  ],
  "validation": [
    {"prompt": "...", "chosen": "...", "rejected": "..."},
    {"prompt": "...", "chosen": "...", "rejected": "..."}
  ]
}
```

### Data Processing

**Tokenization Process:**
1. Tokenize prompt, chosen, and rejected responses separately
2. Create attention masks for each sequence
3. Pad sequences to maximum length
4. Create labels for loss computation
5. Format for DPO trainer consumption

**Memory Considerations:**
DPO requires storing both chosen and rejected sequences, effectively doubling memory usage compared to SFT.

## Performance Optimization

### Memory Optimization

```python
# Use gradient checkpointing
config.gradient_checkpointing = True

# Smaller batch sizes
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16

# Use LoRA for parameter efficiency
config.use_lora = True
config.lora_r = 8  # Smaller rank for less memory

# Quantize reference model
config.load_in_4bit = True
```

### Speed Optimization

```python
# Use mixed precision
config.bf16 = True

# Optimize data loading
config.dataloader_num_workers = 2  # Lower for DPO
config.dataloader_pin_memory = True

# Efficient evaluation
config.eval_steps = 1000  # Less frequent evaluation
```

### Multi-GPU Considerations

```python
# DPO-specific multi-GPU settings
config.per_device_train_batch_size = 1  # Per GPU
config.gradient_accumulation_steps = 8
config.ddp_find_unused_parameters = False

# Reference model handling
config.ref_model_device = "cuda:0"  # Pin reference model to specific GPU
```

## Usage Examples

### Basic DPO Training

```python
from src.training.dpo_trainer import create_dpo_trainer
from src.config import DPOConfig

# Configure DPO training
config = DPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_preferences.jsonl",
    output_dir="./checkpoints/dpo_arabic",
    beta=0.1,
    max_prompt_length=256,
    max_length=512,
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2
)

# Create and train
trainer = create_dpo_trainer(config)
trainer.train()

# Evaluate
results = trainer.evaluate_model()
print(f"DPO Loss: {results['eval_loss']:.4f}")
```

### Advanced DPO Training

```python
config = DPOConfig(
    model_name="./checkpoints/sft_arabic",  # Start from SFT model
    dataset_path="data/high_quality_preferences.jsonl",
    output_dir="./checkpoints/dpo_advanced",
    
    # DPO parameters
    beta=0.05,  # More aggressive optimization
    loss_type="sigmoid",
    label_smoothing=0.01,
    max_prompt_length=512,
    max_length=1024,
    
    # Training optimization
    learning_rate=1e-7,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_ratio=0.05,
    
    # Advanced LoRA
    use_lora=True,
    lora_r=32,
    lora_alpha=64,
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    
    # Evaluation and monitoring
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Experiment tracking
    report_to="wandb",
    run_name="arabic_dpo_advanced"
)

trainer = create_dpo_trainer(config)
trainer.train()
```

### Model Comparison and Analysis

```python
# Load trained DPO model
trainer = DPOTrainer(config)
trainer.load_model("./checkpoints/dpo_arabic")

# Compare responses
prompts = [
    "ما هي أفضل طريقة لتعلم البرمجة؟",
    "اشرح مفهوم الاستدامة البيئية",
    "كيف يمكن تحسين التعليم في المدارس؟"
]

for prompt in prompts:
    comparison = trainer.generate_comparison(
        prompt,
        max_length=256,
        temperature=0.7
    )
    
    print(f"Prompt: {prompt}")
    print(f"Trained: {comparison['trained_model_response']}")
    print(f"Reference: {comparison['reference_model_response']}")
    print("-" * 50)

# Compute preference accuracy
accuracy = trainer.compute_preference_accuracy(eval_dataset)
print(f"Preference Accuracy: {accuracy:.2%}")
```

### Custom DPO Implementation

```python
class CustomDPOTrainer(DPOTrainer):
    def create_trainer(self, dataset, training_args, callbacks):
        # Custom preprocessing
        def custom_format_dpo(examples):
            prompts = examples["prompt"]
            chosen = examples["chosen"]
            rejected = examples["rejected"]
            
            # Add custom formatting
            formatted_prompts = [f"السؤال: {p}\nالجواب: " for p in prompts]
            
            return {
                "prompt": formatted_prompts,
                "chosen": chosen,
                "rejected": rejected
            }
        
        # Apply custom formatting
        if isinstance(dataset, dict):
            dataset["train"] = dataset["train"].map(
                custom_format_dpo,
                batched=True
            )
        
        return super().create_trainer(dataset, training_args, callbacks)
    
    def compute_preference_accuracy(self, eval_dataset):
        # Custom accuracy computation
        accuracy = super().compute_preference_accuracy(eval_dataset)
        
        # Add custom metrics
        self.logger.info(f"Custom preference accuracy: {accuracy:.4f}")
        
        return accuracy
```

## Factory Function

### `create_dpo_trainer()`

```python
def create_dpo_trainer(config: DPOConfig) -> DPOTrainer
```

**Purpose**: Factory function to create a DPO trainer instance.

**Parameters:**
- `config`: DPOConfig object

**Returns**: DPOTrainer instance

**Example:**
```python
from src.training.dpo_trainer import create_dpo_trainer

trainer = create_dpo_trainer(config)
```

## Best Practices

### 1. Data Quality
- Use high-quality preference pairs with clear distinctions
- Ensure balanced representation of different response types
- Include diverse prompts covering various domains
- Validate preference consistency across annotators

### 2. Hyperparameter Tuning
- Start with beta=0.1 and adjust based on results
- Use lower learning rates (1e-7 to 5e-7)
- Monitor both loss and preference accuracy
- Consider fewer epochs (1-2) to prevent overfitting

### 3. Memory Management
- Use gradient checkpointing for large models
- Implement smaller batch sizes with gradient accumulation
- Consider quantizing the reference model
- Use LoRA for parameter-efficient training

### 4. Training Monitoring
- Track both DPO loss and preference accuracy
- Monitor generation quality through comparisons
- Use early stopping based on validation metrics
- Log sample generations during training

### 5. Arabic-Specific Considerations
- Ensure preference pairs reflect Arabic language nuances
- Include both formal and informal Arabic examples
- Consider cultural context in preference judgments
- Test with various Arabic dialects

### 6. Model Evaluation
- Compare against reference model regularly
- Test on held-out preference pairs
- Conduct human evaluation of generated responses
- Monitor for potential degradation in general capabilities

## Troubleshooting

### Common Issues

#### Training Instability
```python
# Solutions:
# 1. Lower learning rate
config.learning_rate = 1e-7

# 2. Increase beta for more conservative training
config.beta = 0.2

# 3. Add gradient clipping
config.max_grad_norm = 0.3

# 4. Use label smoothing
config.label_smoothing = 0.01
```

#### Memory Issues
```python
# Solutions:
# 1. Reduce batch size
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 32

# 2. Use quantization
config.load_in_4bit = True

# 3. Reduce sequence length
config.max_length = 256
config.max_prompt_length = 128
```

#### Poor Preference Learning
```python
# Solutions:
# 1. Check data quality
# 2. Adjust beta parameter
config.beta = 0.05  # More aggressive

# 3. Increase training data
# 4. Improve preference pair quality
# 5. Use different loss type
config.loss_type = "hinge"
```

This comprehensive documentation provides all the information needed to effectively use the DPOTrainer for Arabic language model preference optimization.