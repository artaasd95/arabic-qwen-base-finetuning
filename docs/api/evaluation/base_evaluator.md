# BaseEvaluator API Documentation

The `BaseEvaluator` class provides the foundation for all evaluation methods in the Arabic Qwen fine-tuning system.

## Overview

**Location**: `src/evaluation/base_evaluator.py`

**Purpose**: Provides common evaluation functionality and abstract interface for all evaluation methods including model loading, tokenizer setup, basic metrics computation, and evaluation utilities.

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

class BaseEvaluator(ABC):
    """Base class for model evaluation.
    
    This class provides common evaluation functionalities that can be
    extended by specific evaluation implementations.
    """
```

## Initialization

### Constructor

```python
def __init__(
    self,
    model: Optional[Union[str, PreTrainedModel]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `model`: Model instance or path to model (optional)
- `tokenizer`: Tokenizer instance or path to tokenizer (optional)
- `device`: Device to run evaluation on (optional, auto-detects CUDA)

**Functionality:**
- Automatically detects and sets device (CUDA if available, else CPU)
- Loads model and tokenizer if provided
- Sets up logging for evaluation process

**Example:**
```python
from src.evaluation.base_evaluator import BaseEvaluator

# Initialize with model path
evaluator = BaseEvaluator(
    model="Qwen/Qwen2.5-7B",
    tokenizer="Qwen/Qwen2.5-7B",
    device="cuda"
)

# Initialize empty (load later)
evaluator = BaseEvaluator()
```

## Core Methods

### Model and Tokenizer Loading

#### `load_model(model)`

```python
def load_model(self, model: Union[str, PreTrainedModel]) -> None
```

**Purpose**: Load model for evaluation.

**Parameters:**
- `model`: Model instance or path to model

**Functionality:**
- Loads model from path or accepts existing instance
- Moves model to specified device
- Sets model to evaluation mode
- Handles trust_remote_code for custom models

**Example:**
```python
# Load from path
evaluator.load_model("Qwen/Qwen2.5-7B")

# Load existing model instance
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
evaluator.load_model(model)
```

#### `load_tokenizer(tokenizer)`

```python
def load_tokenizer(self, tokenizer: Union[str, PreTrainedTokenizer]) -> None
```

**Purpose**: Load tokenizer for evaluation.

**Parameters:**
- `tokenizer`: Tokenizer instance or path to tokenizer

**Functionality:**
- Loads tokenizer from path or accepts existing instance
- Sets pad token if not present (uses eos_token)
- Handles trust_remote_code for custom tokenizers

**Example:**
```python
# Load from path
evaluator.load_tokenizer("Qwen/Qwen2.5-7B")

# Load existing tokenizer instance
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
evaluator.load_tokenizer(tokenizer)
```

## Abstract Methods

### `evaluate(dataset, **kwargs)`

```python
@abstractmethod
def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
    """Evaluate the model on the given dataset.
    
    Args:
        dataset: Dataset to evaluate on
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dictionary containing evaluation metrics
    """
    pass
```

**Purpose**: Main evaluation method that must be implemented by subclasses.

**Implementation Requirements:**
- Must accept a Dataset object
- Must return a dictionary of metrics
- Should handle additional keyword arguments
- Should log evaluation progress
## Core Evaluation Methods

### `compute_perplexity(dataset, batch_size=8)`

```python
def compute_perplexity(
    self, 
    dataset: Dataset, 
    batch_size: int = 8
) -> float
```

**Purpose**: Compute perplexity on a dataset.

**Parameters:**
- `dataset`: Dataset to evaluate
- `batch_size`: Batch size for evaluation

**Returns:**
- `float`: Perplexity value

**Example:**
```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": ["Hello world", "How are you?", "Fine, thank you"]
})

perplexity = evaluator.compute_perplexity(dataset, batch_size=4)
print(f"Perplexity: {perplexity:.4f}")
```

### `generate_text(prompts, max_length=512, **kwargs)`

```python
def generate_text(
    self,
    prompts: Union[str, List[str]],
    max_length: int = 512,
    **kwargs
) -> Union[str, List[str]]
```

**Purpose**: Generate text from prompts.

**Parameters:**
- `prompts`: Single prompt or list of prompts
- `max_length`: Maximum generation length
- `**kwargs`: Additional generation parameters

**Returns:**
- Generated text(s)

**Example:**
```python
# Single prompt
response = evaluator.generate_text(
    "What is the capital of Egypt?",
    max_length=100,
    temperature=0.7,
    do_sample=True
)

# Multiple prompts
prompts = [
    "Translate to Arabic: Hello",
    "What is 2+2?",
    "Write a short poem"
]
responses = evaluator.generate_text(prompts, max_length=150)
```

### `compute_basic_metrics(predictions, references)`

```python
def compute_basic_metrics(
    self, 
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]
```

**Purpose**: Compute basic text generation metrics.

**Parameters:**
- `predictions`: List of predicted texts
- `references`: List of reference texts

**Returns:**
- Dictionary of basic metrics:
  - `avg_prediction_length`: Average word count in predictions
  - `avg_reference_length`: Average word count in references
  - `length_ratio`: Ratio of prediction to reference length
  - `avg_prediction_char_length`: Average character count in predictions
  - `avg_reference_char_length`: Average character count in references
  - `prediction_diversity`: Ratio of unique predictions

**Example:**
```python
predictions = [
    "The capital of Egypt is Cairo.",
    "2+2 equals 4.",
    "Roses are red, violets are blue."
]
references = [
    "Cairo is the capital of Egypt.",
    "The answer is 4.",
    "A simple poem about flowers."
]

metrics = evaluator.compute_basic_metrics(predictions, references)
print(f"Length ratio: {metrics['length_ratio']:.2f}")
print(f"Diversity: {metrics['prediction_diversity']:.2f}")
```

## Utility Methods

### `save_evaluation_results(results, output_path)`

```python
def save_evaluation_results(
    self, 
    results: Dict[str, Any], 
    output_path: Union[str, Path]
) -> None
```

**Purpose**: Save evaluation results to JSON file.

**Parameters:**
- `results`: Evaluation results dictionary
- `output_path`: Path to save results

**Example:**
```python
results = {
    "perplexity": 15.2,
    "avg_length": 25.4,
    "diversity": 0.85
}

evaluator.save_evaluation_results(
    results, 
    "reports/evaluation_results.json"
)
```

### `load_evaluation_results(input_path)`

```python
def load_evaluation_results(
    self, 
    input_path: Union[str, Path]
) -> Dict[str, Any]
```

**Purpose**: Load evaluation results from JSON file.

**Parameters:**
- `input_path`: Path to load results from

**Returns:**
- Evaluation results dictionary

**Example:**
```python
results = evaluator.load_evaluation_results(
    "reports/evaluation_results.json"
)
print(f"Previous perplexity: {results['perplexity']}")
```

### `compare_models(other_evaluator, dataset, metrics_to_compare=None)`

```python
def compare_models(
    self,
    other_evaluator: 'BaseEvaluator',
    dataset: Dataset,
    metrics_to_compare: Optional[List[str]] = None
) -> Dict[str, Any]
```

**Purpose**: Compare this model with another model.

**Parameters:**
- `other_evaluator`: Another evaluator instance
- `dataset`: Dataset to evaluate both models on
- `metrics_to_compare`: Specific metrics to compare (optional)

**Returns:**
- Comparison results with model results and differences

**Example:**
```python
# Create two evaluators
evaluator1 = SFTEvaluator(model="model1")
evaluator2 = SFTEvaluator(model="model2")

# Compare models
comparison = evaluator1.compare_models(
    evaluator2, 
    dataset,
    metrics_to_compare=["perplexity", "avg_length"]
)

print(f"Model 1 perplexity: {comparison['model_1_results']['perplexity']}")
print(f"Model 2 perplexity: {comparison['model_2_results']['perplexity']}")
print(f"Difference: {comparison['comparison']['perplexity']['difference']}")
```

## Configuration Integration

The BaseEvaluator integrates with the configuration system:

```python
from src.config import SFTConfig
from src.evaluation import SFTEvaluator

# Load from config
config = SFTConfig.from_yaml("config/sft_config.yaml")
evaluator = SFTEvaluator(
    model=config.model_name,
    tokenizer=config.model_name,
    device=config.device
)
```

## Error Handling

The BaseEvaluator includes comprehensive error handling:

```python
try:
    evaluator = BaseEvaluator(model="invalid/model")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

try:
    results = evaluator.evaluate(dataset)
except RuntimeError as e:
    logger.error(f"Evaluation failed: {e}")
```

## Performance Optimization

### Memory Management

```python
# Use gradient checkpointing for large models
evaluator.model.gradient_checkpointing_enable()

# Clear cache periodically
torch.cuda.empty_cache()
```

### Batch Processing

```python
# Process in smaller batches for memory efficiency
for batch in dataset.iter(batch_size=4):
    batch_results = evaluator.evaluate(batch)
```

## Logging and Monitoring

The BaseEvaluator provides comprehensive logging:

```python
import logging

# Set logging level
logging.getLogger('src.evaluation').setLevel(logging.INFO)

# Evaluation progress is automatically logged
evaluator.evaluate(dataset)  # Logs progress and results
```

## Best Practices

### 1. Resource Management

```python
# Always specify device explicitly
evaluator = BaseEvaluator(device="cuda:0")

# Monitor memory usage
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### 2. Evaluation Strategy

```python
# Use appropriate batch sizes
batch_size = 8 if torch.cuda.is_available() else 2

# Save intermediate results
for i, batch in enumerate(batches):
    results = evaluator.evaluate(batch)
    if i % 10 == 0:
        evaluator.save_evaluation_results(
            results, f"checkpoint_{i}.json"
        )
```

### 3. Model Comparison

```python
# Compare multiple models systematically
models = ["model1", "model2", "model3"]
evaluators = [BaseEvaluator(model=m) for m in models]

for i, eval1 in enumerate(evaluators):
    for j, eval2 in enumerate(evaluators[i+1:], i+1):
        comparison = eval1.compare_models(eval2, dataset)
        print(f"Model {i} vs Model {j}: {comparison}")
```

## Integration Examples

### With Training Pipeline

```python
from src.training import SFTTrainer
from src.evaluation import SFTEvaluator

# Train model
trainer = SFTTrainer(config)
trainer.train()

# Evaluate trained model
evaluator = SFTEvaluator(
    model=trainer.model,
    tokenizer=trainer.tokenizer
)
results = evaluator.evaluate(test_dataset)
```

### With Custom Metrics

```python
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, dataset, **kwargs):
        # Call parent methods
        basic_metrics = self.compute_basic_metrics(
            predictions, references
        )
        
        # Add custom metrics
        custom_metrics = self.compute_custom_metrics(
            predictions, references
        )
        
        return {**basic_metrics, **custom_metrics}
    
    def compute_custom_metrics(self, predictions, references):
        # Implement custom evaluation logic
        return {"custom_score": 0.85}
```

## String Representation

The BaseEvaluator provides informative string representation:

```python
print(evaluator)
# Output: BaseEvaluator(model=Qwen/Qwen2.5-7B, tokenizer=Qwen/Qwen2.5-7B, device=cuda)
```

## See Also

- [SFTEvaluator Documentation](sft_evaluator.md) - Supervised Fine-Tuning evaluation
- [PreferenceEvaluator Documentation](preference_evaluator.md) - Preference optimization evaluation
- [Evaluation System Overview](index.md) - Complete evaluation system guide
- [Training Documentation](../training/index.md) - Training system integration
4. Initialize metric computation modules
5. Set up logging
6. Prepare evaluation state

**Example:**
```python
from src.evaluation.base_evaluator import BaseEvaluator
from src.config import EvaluationConfig
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Create evaluation config
config = EvaluationConfig(
    batch_size=8,
    max_length=512,
    metrics=["perplexity", "bleu", "rouge"]
)

# Initialize evaluator
evaluator = BaseEvaluator(model, tokenizer, config)
print(f"Evaluator initialized on device: {evaluator.device}")
```

## Core Methods

### Main Evaluation Method

#### `evaluate()`

```python
def evaluate(
    self,
    dataset: Union[Dataset, DataLoader],
    metrics: Optional[List[str]] = None,
    return_predictions: bool = False,
    save_results: bool = True
) -> Dict[str, Any]
```

**Purpose**: Main evaluation method that orchestrates the entire evaluation process.

**Parameters:**
- `dataset`: Dataset or DataLoader to evaluate on
- `metrics`: List of metrics to compute (uses config.metrics if None)
- `return_predictions`: Whether to return model predictions
- `save_results`: Whether to save evaluation results to disk

**Returns**: Dictionary containing evaluation results

**Process:**
1. Validate inputs and setup
2. Prepare dataset for evaluation
3. Run model inference
4. Compute specified metrics
5. Format and return results
6. Optionally save results

**Example:**
```python
from datasets import load_dataset

# Load evaluation dataset
dataset = load_dataset("arabic_qa", split="test")

# Run evaluation
results = evaluator.evaluate(
    dataset=dataset,
    metrics=["perplexity", "bleu", "rouge"],
    return_predictions=True,
    save_results=True
)

print(f"Perplexity: {results['perplexity']:.2f}")
print(f"BLEU Score: {results['bleu']:.3f}")
print(f"ROUGE-L: {results['rouge']['rougeL']:.3f}")

if 'predictions' in results:
    print(f"Generated {len(results['predictions'])} predictions")
```

### Model Inference

#### `generate_predictions()`

```python
def generate_predictions(
    self,
    dataset: Dataset,
    batch_size: Optional[int] = None,
    **generation_kwargs
) -> List[str]
```

**Purpose**: Generate model predictions for the given dataset.

**Parameters:**
- `dataset`: Dataset to generate predictions for
- `batch_size`: Batch size for generation (uses config.batch_size if None)
- `**generation_kwargs`: Additional generation parameters

**Returns**: List of generated text predictions

**Generation Parameters:**
```python
default_generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.eos_token_id
}
```

**Example:**
```python
# Generate predictions with custom parameters
predictions = evaluator.generate_predictions(
    dataset=test_dataset,
    batch_size=4,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)

print(f"Generated {len(predictions)} predictions")
for i, pred in enumerate(predictions[:3]):
    print(f"Prediction {i+1}: {pred}")
```

#### `compute_logprobs()`

```python
def compute_logprobs(
    self,
    dataset: Dataset,
    batch_size: Optional[int] = None
) -> List[float]
```

**Purpose**: Compute log probabilities for dataset sequences.

**Parameters:**
- `dataset`: Dataset to compute log probabilities for
- `batch_size`: Batch size for computation

**Returns**: List of log probabilities

**Usage**: Primarily for perplexity calculation and preference optimization evaluation.

**Example:**
```python
# Compute log probabilities
logprobs = evaluator.compute_logprobs(
    dataset=test_dataset,
    batch_size=8
)

# Calculate perplexity
import math
perplexity = math.exp(-sum(logprobs) / len(logprobs))
print(f"Perplexity: {perplexity:.2f}")
```

### Metric Computation

#### `compute_metrics()`

```python
def compute_metrics(
    self,
    predictions: List[str],
    references: List[str],
    metrics: List[str]
) -> Dict[str, Any]
```

**Purpose**: Compute specified metrics given predictions and references.

**Parameters:**
- `predictions`: List of model predictions
- `references`: List of reference texts
- `metrics`: List of metric names to compute

**Returns**: Dictionary containing computed metrics

**Supported Metrics:**
- `perplexity`: Language modeling perplexity
- `bleu`: BLEU score (1-4 grams)
- `rouge`: ROUGE scores (1, 2, L)
- `bertscore`: BERTScore (precision, recall, F1)
- `exact_match`: Exact string match accuracy
- `f1`: Token-level F1 score
- `arabic_specific`: Arabic language-specific metrics

**Example:**
```python
# Compute multiple metrics
metrics = evaluator.compute_metrics(
    predictions=generated_texts,
    references=reference_texts,
    metrics=["bleu", "rouge", "bertscore"]
)

print(f"BLEU-4: {metrics['bleu']['bleu']:.3f}")
print(f"ROUGE-1: {metrics['rouge']['rouge1']:.3f}")
print(f"ROUGE-2: {metrics['rouge']['rouge2']:.3f}")
print(f"ROUGE-L: {metrics['rouge']['rougeL']:.3f}")
print(f"BERTScore F1: {metrics['bertscore']['f1']:.3f}")
```

#### `compute_perplexity()`

```python
def compute_perplexity(
    self,
    dataset: Dataset,
    batch_size: Optional[int] = None
) -> float
```

**Purpose**: Compute perplexity on the given dataset.

**Parameters:**
- `dataset`: Dataset to compute perplexity on
- `batch_size`: Batch size for computation

**Returns**: Perplexity value (lower is better)

**Implementation:**
1. Compute log probabilities for all sequences
2. Calculate average negative log likelihood
3. Return exponential of average NLL

**Example:**
```python
# Compute perplexity
perplexity = evaluator.compute_perplexity(
    dataset=validation_dataset,
    batch_size=16
)

print(f"Model perplexity: {perplexity:.2f}")

# Interpret results
if perplexity < 10:
    print("Excellent language modeling performance")
elif perplexity < 50:
    print("Good language modeling performance")
else:
    print("Room for improvement in language modeling")
```

### Data Processing

#### `prepare_dataset()`

```python
def prepare_dataset(
    self,
    dataset: Dataset,
    tokenize: bool = True,
    add_special_tokens: bool = True
) -> Dataset
```

**Purpose**: Prepare dataset for evaluation by tokenizing and formatting.

**Parameters:**
- `dataset`: Raw dataset to prepare
- `tokenize`: Whether to tokenize the text
- `add_special_tokens`: Whether to add special tokens

**Returns**: Prepared dataset ready for evaluation

**Processing Steps:**
1. Tokenize input texts
2. Add special tokens if specified
3. Pad sequences to maximum length
4. Create attention masks
5. Format for model input

**Example:**
```python
# Prepare dataset for evaluation
prepared_dataset = evaluator.prepare_dataset(
    dataset=raw_dataset,
    tokenize=True,
    add_special_tokens=True
)

print(f"Prepared {len(prepared_dataset)} examples")
print(f"Sample input_ids shape: {prepared_dataset[0]['input_ids'].shape}")
```

#### `batch_process()`

```python
def batch_process(
    self,
    dataset: Dataset,
    process_fn: Callable,
    batch_size: Optional[int] = None,
    desc: str = "Processing"
) -> List[Any]
```

**Purpose**: Process dataset in batches with progress tracking.

**Parameters:**
- `dataset`: Dataset to process
- `process_fn`: Function to apply to each batch
- `batch_size`: Batch size for processing
- `desc`: Description for progress bar

**Returns**: List of processing results

**Example:**
```python
def custom_processing(batch):
    # Custom processing logic
    return model.generate(**batch)

# Process in batches
results = evaluator.batch_process(
    dataset=test_dataset,
    process_fn=custom_processing,
    batch_size=8,
    desc="Generating responses"
)

print(f"Processed {len(results)} batches")
```

### Result Management

#### `format_results()`

```python
def format_results(
    self,
    raw_results: Dict[str, Any],
    include_metadata: bool = True
) -> Dict[str, Any]
```

**Purpose**: Format raw evaluation results into a standardized format.

**Parameters:**
- `raw_results`: Raw evaluation results
- `include_metadata`: Whether to include evaluation metadata

**Returns**: Formatted results dictionary

**Formatted Structure:**
```python
{
    "metrics": {
        "perplexity": 15.2,
        "bleu": {"bleu": 0.45, "precisions": [0.7, 0.5, 0.3, 0.2]},
        "rouge": {"rouge1": 0.6, "rouge2": 0.4, "rougeL": 0.55}
    },
    "metadata": {
        "model_name": "Qwen/Qwen2.5-7B",
        "dataset_size": 1000,
        "evaluation_time": "2024-01-15T10:30:00Z",
        "config": {...}
    },
    "summary": {
        "overall_score": 0.52,
        "best_metric": "rouge1",
        "worst_metric": "bleu"
    }
}
```

#### `save_results()`

```python
def save_results(
    self,
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    format: str = "json"
) -> str
```

**Purpose**: Save evaluation results to disk.

**Parameters:**
- `results`: Results dictionary to save
- `output_path`: Path to save results (auto-generated if None)
- `format`: Output format ("json", "yaml", "csv")

**Returns**: Path where results were saved

**Example:**
```python
# Save results in different formats
json_path = evaluator.save_results(results, format="json")
yaml_path = evaluator.save_results(results, format="yaml")
csv_path = evaluator.save_results(results, format="csv")

print(f"Results saved to:")
print(f"  JSON: {json_path}")
print(f"  YAML: {yaml_path}")
print(f"  CSV: {csv_path}")
```

#### `load_results()`

```python
def load_results(
    self,
    results_path: str
) -> Dict[str, Any]
```

**Purpose**: Load previously saved evaluation results.

**Parameters:**
- `results_path`: Path to saved results file

**Returns**: Loaded results dictionary

**Example:**
```python
# Load previous results
previous_results = evaluator.load_results("./results/evaluation_2024-01-15.json")

# Compare with current results
current_results = evaluator.evaluate(test_dataset)

print("Performance comparison:")
for metric in ["perplexity", "bleu"]:
    prev_val = previous_results["metrics"][metric]
    curr_val = current_results["metrics"][metric]
    change = curr_val - prev_val
    print(f"{metric}: {prev_val:.3f} → {curr_val:.3f} ({change:+.3f})")
```

### Utility Methods

#### `get_device()`

```python
def get_device(self) -> str
```

**Purpose**: Get the device being used for evaluation.

**Returns**: Device string ("cuda", "cpu", etc.)

#### `set_seed()`

```python
def set_seed(self, seed: int) -> None
```

**Purpose**: Set random seed for reproducible evaluation.

**Parameters:**
- `seed`: Random seed value

**Example:**
```python
# Set seed for reproducible results
evaluator.set_seed(42)

# Run evaluation multiple times with same results
results1 = evaluator.evaluate(test_dataset)
results2 = evaluator.evaluate(test_dataset)

# Results should be identical
assert results1["perplexity"] == results2["perplexity"]
```

#### `get_memory_usage()`

```python
def get_memory_usage(self) -> Dict[str, float]
```

**Purpose**: Get current memory usage statistics.

**Returns**: Dictionary with memory usage information

**Example:**
```python
# Monitor memory usage
memory_before = evaluator.get_memory_usage()
results = evaluator.evaluate(large_dataset)
memory_after = evaluator.get_memory_usage()

print(f"Memory usage:")
print(f"  Before: {memory_before['allocated']:.2f} GB")
print(f"  After: {memory_after['allocated']:.2f} GB")
print(f"  Peak: {memory_after['max_allocated']:.2f} GB")
```

## Configuration Integration

### EvaluationConfig

```python
from src.config import EvaluationConfig

config = EvaluationConfig(
    # Basic settings
    batch_size=8,
    max_length=512,
    device="auto",  # Auto-detect device
    seed=42,
    
    # Metrics configuration
    metrics=[
        "perplexity",
        "bleu",
        "rouge",
        "bertscore"
    ],
    
    # Generation settings
    generation_config={
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1
    },
    
    # Output settings
    output_dir="./evaluation_results",
    save_predictions=True,
    save_detailed_results=True,
    
    # Performance settings
    use_cache=True,
    fp16=True,
    dataloader_num_workers=2
)
```

### Dynamic Configuration

```python
# Update configuration at runtime
evaluator.config.batch_size = 16
evaluator.config.metrics.append("exact_match")

# Apply configuration changes
evaluator.update_config()

# Verify changes
print(f"New batch size: {evaluator.config.batch_size}")
print(f"Metrics: {evaluator.config.metrics}")
```

## Error Handling

### Exception Types

```python
from src.evaluation.exceptions import (
    EvaluationError,
    MetricComputationError,
    DatasetPreparationError,
    ModelInferenceError
)

try:
    results = evaluator.evaluate(dataset)
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
except MetricComputationError as e:
    print(f"Metric computation failed: {e}")
except DatasetPreparationError as e:
    print(f"Dataset preparation failed: {e}")
except ModelInferenceError as e:
    print(f"Model inference failed: {e}")
```

### Graceful Degradation

```python
# Evaluation with fallback metrics
try:
    results = evaluator.evaluate(
        dataset=test_dataset,
        metrics=["perplexity", "bleu", "rouge", "bertscore"]
    )
except MetricComputationError as e:
    print(f"Some metrics failed: {e}")
    # Retry with basic metrics only
    results = evaluator.evaluate(
        dataset=test_dataset,
        metrics=["perplexity", "bleu"]
    )
    print("Evaluation completed with reduced metrics")
```

## Performance Optimization

### Memory Optimization

```python
# Optimize for memory usage
config = EvaluationConfig(
    batch_size=4,  # Smaller batches
    max_length=256,  # Shorter sequences
    fp16=True,  # Mixed precision
    gradient_checkpointing=True,  # For large models
    use_cache=False  # Disable caching for memory
)

evaluator = BaseEvaluator(model, tokenizer, config)

# Clear cache between evaluations
import torch
torch.cuda.empty_cache()
```

### Speed Optimization

```python
# Optimize for speed
config = EvaluationConfig(
    batch_size=16,  # Larger batches
    dataloader_num_workers=4,  # Parallel data loading
    use_cache=True,  # Enable caching
    fp16=True,  # Mixed precision
    compile_model=True  # Model compilation (PyTorch 2.0+)
)

evaluator = BaseEvaluator(model, tokenizer, config)
```

## Extension Examples

### Custom Evaluator

```python
class CustomArabicEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        self.arabic_metrics = ArabicMetrics()
    
    def evaluate(self, dataset, metrics=None, **kwargs):
        # Call parent evaluation
        results = super().evaluate(dataset, metrics, **kwargs)
        
        # Add custom Arabic metrics
        if "arabic_custom" in (metrics or self.config.metrics):
            arabic_results = self.evaluate_arabic_specific(dataset)
            results["arabic_custom"] = arabic_results
        
        return results
    
    def evaluate_arabic_specific(self, dataset):
        # Custom Arabic evaluation logic
        predictions = self.generate_predictions(dataset)
        references = [example["reference"] for example in dataset]
        
        return {
            "diacritization_accuracy": self.arabic_metrics.diacritization_accuracy(
                predictions, references
            ),
            "dialect_consistency": self.arabic_metrics.dialect_consistency(
                predictions
            ),
            "cultural_appropriateness": self.arabic_metrics.cultural_appropriateness(
                predictions
            )
        }

# Use custom evaluator
custom_evaluator = CustomArabicEvaluator(model, tokenizer, config)
results = custom_evaluator.evaluate(
    dataset=arabic_test_dataset,
    metrics=["perplexity", "bleu", "arabic_custom"]
)
```

### Metric Plugin System

```python
class MetricPlugin:
    def __init__(self, name: str):
        self.name = name
    
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        raise NotImplementedError

class CustomBLEUPlugin(MetricPlugin):
    def __init__(self):
        super().__init__("custom_bleu")
    
    def compute(self, predictions, references):
        # Custom BLEU implementation
        return {"custom_bleu": 0.5}  # Placeholder

# Register plugin
evaluator.register_metric_plugin(CustomBLEUPlugin())

# Use in evaluation
results = evaluator.evaluate(
    dataset=test_dataset,
    metrics=["perplexity", "custom_bleu"]
)
```

## Best Practices

### 1. Evaluation Design
- Use multiple complementary metrics
- Include both automatic and human evaluation when possible
- Test on diverse datasets
- Consider domain-specific metrics

### 2. Performance Considerations
- Choose appropriate batch sizes for your hardware
- Use mixed precision when available
- Cache results for repeated evaluations
- Monitor memory usage for large datasets

### 3. Reproducibility
- Always set random seeds
- Document evaluation configurations
- Save detailed evaluation logs
- Version control evaluation scripts

### 4. Arabic Language Considerations
- Test with different Arabic dialects
- Evaluate diacritization quality
- Check cultural appropriateness
- Assess grammatical correctness

### 5. Error Handling
- Implement graceful degradation for failed metrics
- Log detailed error information
- Provide fallback evaluation strategies
- Validate inputs before processing

The BaseEvaluator provides a robust foundation for all evaluation tasks in the Arabic Qwen fine-tuning framework, ensuring consistent and reliable model assessment across different training methods and use cases.