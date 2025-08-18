# SFTTrainer API Documentation

The `SFTTrainer` class implements Supervised Fine-Tuning (SFT) for Arabic Qwen models using instruction-response pairs.

## Overview

**Location**: `src/training/sft_trainer.py`

**Purpose**: Train models on instruction-response datasets to improve instruction-following capabilities for Arabic language tasks.

**Inheritance**: `SFTTrainer` → `BaseTrainer`

## Class Definition

```python
class SFTTrainer(BaseTrainer):
    """Trainer for Supervised Fine-Tuning (SFT).
    
    This class handles the complete SFT training pipeline including
    instruction data loading, model setup, and SFT training execution.
    """
```

## Initialization

### Constructor

```python
def __init__(self, config: SFTConfig)
```

**Parameters:**
- `config`: SFTConfig object containing training configuration

**Raises:**
- `TypeError`: If config is not an SFTConfig instance

**Example:**
```python
from src.training.sft_trainer import SFTTrainer
from src.config import SFTConfig

config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_instructions.jsonl",
    output_dir="./checkpoints/sft"
)

trainer = SFTTrainer(config)
```

## Core Methods

### Training Method Identification

#### `_get_training_method()`

```python
def _get_training_method(self) -> str
```

**Returns**: `"sft"`

**Purpose**: Identifies this trainer as using the SFT method for data loading and processing.

### Trainer Creation

#### `create_trainer()`

```python
def create_trainer(self, dataset, training_args, callbacks)
```

**Purpose**: Create and configure the SFT trainer instance.

**Parameters:**
- `dataset`: Training dataset (dict with 'train' and optional 'validation' keys)
- `training_args`: TrainingArguments object
- `callbacks`: List of training callbacks

**Returns**: Configured `Trainer` instance from Transformers library

**Process:**
1. Prepare training and evaluation datasets
2. Apply tokenization and formatting
3. Get data collator for SFT
4. Create and return Trainer instance

**Example:**
```python
# Internal usage - called automatically during training
trainer = sft_trainer.create_trainer(dataset, training_args, callbacks)
```

### Training Arguments

#### `get_training_arguments()`

```python
def get_training_arguments(self) -> TrainingArguments
```

**Purpose**: Get SFT-specific training arguments.

**Returns**: TrainingArguments with SFT-specific modifications

**SFT-Specific Settings:**
```python
# Enable gradient checkpointing for memory efficiency
training_args.gradient_checkpointing = True

# Set label names for SFT
training_args.label_names = ["labels"]

# Remove unused columns to save memory
training_args.remove_unused_columns = True

# Drop last incomplete batch
training_args.dataloader_drop_last = True

# Group by length if packing is disabled
if not config.packing:
    training_args.group_by_length = True
```

## Loss Computation

### `compute_loss()`

```python
def compute_loss(self, model, inputs, return_outputs=False)
```

**Purpose**: Compute SFT loss using causal language modeling objective.

**Parameters:**
- `model`: The model being trained
- `inputs`: Batch of training inputs
- `return_outputs`: Whether to return model outputs (default: False)

**Returns**: 
- Loss tensor (if return_outputs=False)
- Tuple of (loss, outputs) (if return_outputs=True)

**Loss Calculation:**
```python
# Forward pass
outputs = model(**inputs)
logits = outputs.logits

# Shift logits and labels for next-token prediction
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

# Compute cross-entropy loss
loss_fct = CrossEntropyLoss()
loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1)
)
```

**Example:**
```python
# Custom loss computation (if needed)
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss logic here
        return super().compute_loss(model, inputs, return_outputs)
```

## Evaluation

### `evaluate_model()`

```python
def evaluate_model(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]
```

**Purpose**: Evaluate the SFT model and compute perplexity.

**Parameters:**
- `eval_dataset`: Evaluation dataset (optional)

**Returns**: Dictionary containing evaluation metrics

**Metrics Computed:**
- `eval_loss`: Cross-entropy loss on evaluation set
- `eval_perplexity`: Perplexity (exp(loss))
- `eval_runtime`: Evaluation time
- `eval_samples_per_second`: Evaluation speed

**Example:**
```python
results = trainer.evaluate_model()
print(f"Evaluation Loss: {results['eval_loss']:.4f}")
print(f"Perplexity: {results['eval_perplexity']:.2f}")
print(f"Samples/sec: {results['eval_samples_per_second']:.2f}")
```

## Text Generation

### `generate_response()`

```python
def generate_response(
    self, 
    instruction: str, 
    max_length: int = 512, 
    **kwargs
) -> str
```

**Purpose**: Generate response for a single instruction using the trained model.

**Parameters:**
- `instruction`: Input instruction text
- `max_length`: Maximum generation length (default: 512)
- `**kwargs`: Additional generation parameters

**Returns**: Generated response text

**Generation Process:**
1. Apply instruction template formatting
2. Tokenize input
3. Generate response using model
4. Decode and clean output

**Instruction Templates:**
- **Alpaca**: `"### Instruction:\n{instruction}\n\n### Response:\n"`
- **ChatML**: `"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"`
- **Custom**: User-defined template

**Example:**
```python
# Basic generation
response = trainer.generate_response(
    "اشرح لي مفهوم الذكاء الاصطناعي",
    max_length=256
)

# Advanced generation with parameters
response = trainer.generate_response(
    "Write a poem about nature in Arabic",
    max_length=512,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1
)
```

### `batch_generate()`

```python
def batch_generate(
    self, 
    instructions: List[str], 
    max_length: int = 512, 
    batch_size: int = 4,
    **kwargs
) -> List[str]
```

**Purpose**: Generate responses for multiple instructions efficiently.

**Parameters:**
- `instructions`: List of instruction texts
- `max_length`: Maximum generation length (default: 512)
- `batch_size`: Batch size for processing (default: 4)
- `**kwargs`: Additional generation parameters

**Returns**: List of generated responses

**Example:**
```python
instructions = [
    "ما هو الذكاء الاصطناعي؟",
    "اكتب قصيدة عن الطبيعة",
    "اشرح مفهوم التعلم الآلي"
]

responses = trainer.batch_generate(
    instructions,
    max_length=256,
    batch_size=2
)

for instruction, response in zip(instructions, responses):
    print(f"Q: {instruction}")
    print(f"A: {response}\n")
```

## Model Saving

### `save_model()`

```python
def save_model(self, output_dir: Optional[str] = None) -> None
```

**Purpose**: Save the SFT model with metadata.

**Parameters:**
- `output_dir`: Directory to save model (optional, uses config.output_dir if None)

**Saved Files:**
- Model weights (LoRA adapters or full model)
- Tokenizer files
- Configuration files
- `sft_metadata.json`: SFT-specific metadata

**SFT Metadata:**
```json
{
  "training_method": "sft",
  "instruction_template": "alpaca",
  "response_template": "### Response:\n",
  "packing": true,
  "max_seq_length": 512,
  "model_name": "Qwen/Qwen2.5-7B",
  "dataset_path": "data/arabic_instructions.jsonl",
  "training_completed_at": "2024-01-15T10:30:00Z"
}
```

**Example:**
```python
# Save to default directory
trainer.save_model()

# Save to custom directory
trainer.save_model("./models/arabic_sft_v1")
```

## Configuration

### SFT-Specific Configuration Parameters

```python
from src.config import SFTConfig

config = SFTConfig(
    # Base parameters
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/sft_arabic.jsonl",
    output_dir="./checkpoints/sft",
    
    # SFT-specific parameters
    packing=True,  # Pack multiple sequences for efficiency
    instruction_template="alpaca",  # Instruction formatting
    response_template="### Response:\n",  # Response prefix
    max_seq_length=512,  # Maximum sequence length
    
    # Training parameters
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    
    # Optimization
    gradient_checkpointing=True,
    group_by_length=True,  # Group similar lengths
    dataloader_drop_last=True,  # Drop incomplete batches
    
    # LoRA configuration
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

### Instruction Templates

#### Alpaca Template
```python
template = "### Instruction:\n{instruction}\n\n### Response:\n"
```

#### ChatML Template
```python
template = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
```

#### Custom Template
```python
config.instruction_template = "custom"
config.custom_template = "User: {instruction}\nAssistant: "
```

## Data Format

### Expected Input Format

**JSONL Format:**
```json
{"instruction": "ما هو الذكاء الاصطناعي؟", "response": "الذكاء الاصطناعي هو..."}
{"instruction": "اكتب قصيدة عن الربيع", "response": "في فصل الربيع تتفتح الأزهار..."}
```

**CSV Format:**
```csv
instruction,response
"ما هو الذكاء الاصطناعي؟","الذكاء الاصطناعي هو..."
"اكتب قصيدة عن الربيع","في فصل الربيع تتفتح الأزهار..."
```

**Dataset Split:**
```json
{
  "train": [
    {"instruction": "...", "response": "..."},
    {"instruction": "...", "response": "..."}
  ],
  "validation": [
    {"instruction": "...", "response": "..."},
    {"instruction": "...", "response": "..."}
  ]
}
```

### Data Processing

**Tokenization Process:**
1. Apply instruction template
2. Concatenate instruction and response
3. Tokenize combined text
4. Add special tokens (BOS, EOS)
5. Create attention masks
6. Set labels for loss computation

**Packing (Optional):**
- Combines multiple short sequences into single training examples
- Improves GPU utilization
- Reduces padding overhead

## Performance Optimization

### Memory Optimization

```python
# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use sequence packing
config.packing = True

# Optimize batch size
config.per_device_train_batch_size = 2
config.gradient_accumulation_steps = 8  # Effective batch size = 16

# Use LoRA for parameter efficiency
config.use_lora = True
config.lora_r = 8  # Smaller rank for less memory
```

### Speed Optimization

```python
# Use mixed precision
config.bf16 = True

# Group sequences by length
config.group_by_length = True

# Optimize data loading
config.dataloader_num_workers = 4
config.dataloader_pin_memory = True

# Drop incomplete batches
config.dataloader_drop_last = True
```

## Usage Examples

### Basic SFT Training

```python
from src.training.sft_trainer import create_sft_trainer
from src.config import SFTConfig

# Configure training
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_instructions.jsonl",
    output_dir="./checkpoints/sft_arabic",
    max_seq_length=512,
    packing=True,
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4
)

# Create and train
trainer = create_sft_trainer(config)
trainer.train()

# Evaluate
results = trainer.evaluate_model()
print(f"Final perplexity: {results['eval_perplexity']:.2f}")
```

### Advanced SFT Training with Custom Settings

```python
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/arabic_instructions.jsonl",
    output_dir="./checkpoints/sft_advanced",
    
    # Advanced settings
    instruction_template="chatml",
    max_seq_length=1024,
    packing=True,
    
    # Training optimization
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    
    # LoRA configuration
    use_lora=True,
    lora_r=32,
    lora_alpha=64,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
    
    # Monitoring
    report_to="wandb",
    run_name="arabic_sft_advanced"
)

trainer = create_sft_trainer(config)
trainer.train()
```

### Multi-GPU Training

```python
# Configure for multi-GPU
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data/large_arabic_dataset.jsonl",
    output_dir="./checkpoints/sft_multigpu",
    
    # Multi-GPU settings
    per_device_train_batch_size=8,  # Per GPU
    gradient_accumulation_steps=2,
    ddp_find_unused_parameters=False,
    ddp_backend="nccl",
    
    # DeepSpeed (optional)
    deepspeed="configs/deepspeed_zero2.json"
)

# Run with torchrun or accelerate
# torchrun --nproc_per_node=4 train_sft.py
```

### Custom Data Processing

```python
class CustomSFTTrainer(SFTTrainer):
    def create_trainer(self, dataset, training_args, callbacks):
        # Custom data processing
        def custom_formatting(examples):
            instructions = examples["instruction"]
            responses = examples["response"]
            
            # Custom formatting logic
            formatted_texts = []
            for inst, resp in zip(instructions, responses):
                # Add custom preprocessing
                formatted_text = f"السؤال: {inst}\nالجواب: {resp}"
                formatted_texts.append(formatted_text)
            
            return {"text": formatted_texts}
        
        # Apply custom formatting
        if isinstance(dataset, dict):
            dataset["train"] = dataset["train"].map(
                custom_formatting,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
        
        return super().create_trainer(dataset, training_args, callbacks)
```

### Inference and Generation

```python
# Load trained model
trainer = SFTTrainer(config)
trainer.load_model("./checkpoints/sft_arabic")

# Single generation
response = trainer.generate_response(
    "اشرح لي مفهوم التعلم العميق",
    max_length=512,
    temperature=0.7,
    top_p=0.9
)
print(response)

# Batch generation
instructions = [
    "ما هي فوائد الرياضة؟",
    "كيف يمكنني تعلم البرمجة؟",
    "اكتب قصة قصيرة عن الصداقة"
]

responses = trainer.batch_generate(
    instructions,
    max_length=256,
    batch_size=2
)

for i, (inst, resp) in enumerate(zip(instructions, responses)):
    print(f"Example {i+1}:")
    print(f"Q: {inst}")
    print(f"A: {resp}\n")
```

## Factory Function

### `create_sft_trainer()`

```python
def create_sft_trainer(config: SFTConfig) -> SFTTrainer
```

**Purpose**: Factory function to create an SFT trainer instance.

**Parameters:**
- `config`: SFTConfig object

**Returns**: SFTTrainer instance

**Example:**
```python
from src.training.sft_trainer import create_sft_trainer

trainer = create_sft_trainer(config)
```

## Best Practices

### 1. Data Quality
- Use high-quality, diverse instruction-response pairs
- Ensure proper Arabic text normalization
- Balance dataset across different domains and tasks
- Include both formal and informal Arabic

### 2. Hyperparameter Tuning
- Start with default learning rates (2e-4 for full fine-tuning, 1e-4 for LoRA)
- Use warmup for stable training
- Monitor validation loss to prevent overfitting
- Adjust sequence length based on your data

### 3. Memory Management
- Use gradient checkpointing for large models
- Enable sequence packing for efficiency
- Use LoRA for parameter-efficient training
- Optimize batch size and gradient accumulation

### 4. Evaluation
- Always use a validation set
- Monitor perplexity and generation quality
- Test on diverse Arabic tasks
- Include human evaluation for quality assessment

### 5. Arabic-Specific Considerations
- Handle diacritics appropriately
- Consider different Arabic dialects
- Test with both MSA and dialectal Arabic
- Ensure proper tokenization for Arabic text

This comprehensive documentation provides all the information needed to effectively use the SFTTrainer for Arabic language model fine-tuning.