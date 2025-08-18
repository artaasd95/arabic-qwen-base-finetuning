# SFTEvaluator API Documentation

The `SFTEvaluator` class provides specialized evaluation methods for Supervised Fine-Tuning (SFT) models with instruction-following capabilities.

## Overview

**Location**: `src/evaluation/sft_evaluator.py`

**Purpose**: Evaluates instruction-following models trained with SFT, providing metrics specific to instruction-response quality, format compliance, and Arabic language capabilities.

**Inheritance**: `SFTEvaluator` → `BaseEvaluator`

## Class Definition

```python
from typing import Dict, List, Optional, Any, Union
from datasets import Dataset
import torch
import numpy as np
from .base_evaluator import BaseEvaluator

class SFTEvaluator(BaseEvaluator):
    """Evaluator for Supervised Fine-Tuning models.
    
    This class provides evaluation methods specifically designed for
    instruction-following models trained with SFT.
    """
```

## Initialization

### Constructor

```python
def __init__(
    self,
    model=None,
    tokenizer=None,
    device=None,
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n",
    response_template: str = "{response}"
)
```

**Parameters:**
- `model`: Model instance or path to model (optional)
- `tokenizer`: Tokenizer instance or path to tokenizer (optional)
- `device`: Device to run evaluation on (optional, auto-detects CUDA)
- `instruction_template`: Template for formatting instructions
- `response_template`: Template for formatting responses

**Functionality:**
- Inherits all BaseEvaluator functionality
- Sets up instruction and response templates
- Configures SFT-specific evaluation parameters

**Example:**
```python
from src.evaluation.sft_evaluator import SFTEvaluator

# Initialize with default templates
evaluator = SFTEvaluator(
    model="Qwen/Qwen2.5-7B",
    tokenizer="Qwen/Qwen2.5-7B",
    device="cuda"
)

# Initialize with custom Arabic templates
evaluator = SFTEvaluator(
    model="./checkpoints/arabic_sft_model",
    instruction_template="### التعليمات:\n{instruction}\n\n### الإجابة:\n",
    response_template="{response}"
)
```

## Core Evaluation Methods

### `evaluate(dataset, **kwargs)`

```python
def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]
```

**Purpose**: Main evaluation method for SFT models.

**Parameters:**
- `dataset`: Dataset containing instruction-response pairs
- `**kwargs`: Additional evaluation parameters
  - `batch_size`: Batch size for evaluation (default: 8)
  - `max_length`: Maximum generation length (default: 512)
  - `temperature`: Generation temperature (default: 0.7)
  - `do_sample`: Whether to use sampling (default: True)

**Returns:**
- Dictionary containing comprehensive evaluation metrics

**Expected Dataset Format:**
```python
# Required columns
dataset = Dataset.from_dict({
    "instruction": ["What is the capital of Egypt?", "Translate: Hello"],
    "response": ["The capital of Egypt is Cairo.", "مرحبا"]
})

# Optional columns
dataset = Dataset.from_dict({
    "instruction": [...],
    "response": [...],
    "input": [...],  # Additional context
    "category": [...]  # Instruction category
})
```

**Example:**
```python
from datasets import Dataset

# Create evaluation dataset
eval_data = Dataset.from_dict({
    "instruction": [
        "What is machine learning?",
        "Translate to Arabic: Good morning",
        "Write a short poem about the desert"
    ],
    "response": [
        "Machine learning is a subset of AI...",
        "صباح الخير",
        "Golden sands stretch far and wide..."
    ]
})

# Evaluate model
results = evaluator.evaluate(
    eval_data,
    batch_size=4,
    max_length=256,
    temperature=0.8
)

print(f"Overall Score: {results['overall_score']:.3f}")
print(f"Instruction Following: {results['instruction_following_score']:.3f}")
print(f"Response Quality: {results['response_quality_score']:.3f}")
```

### `generate_instruction_response(instruction, **kwargs)`

```python
def generate_instruction_response(
    self,
    instruction: str,
    **kwargs
) -> str
```

**Purpose**: Generate response for a single instruction.

**Parameters:**
- `instruction`: Input instruction text
- `**kwargs`: Generation parameters (max_length, temperature, etc.)

**Returns:**
- Generated response string

**Example:**
```python
# Generate response for instruction
instruction = "Explain the concept of neural networks in simple terms."
response = evaluator.generate_instruction_response(
    instruction,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

print(f"Instruction: {instruction}")
print(f"Response: {response}")
```

### `evaluate_single_instruction(instruction, reference_response=None)`

```python
def evaluate_single_instruction(
    self,
    instruction: str,
    reference_response: Optional[str] = None
) -> Dict[str, Any]
```

**Purpose**: Evaluate model response to a single instruction.

**Parameters:**
- `instruction`: Input instruction
- `reference_response`: Optional reference response for comparison

**Returns:**
- Dictionary with detailed evaluation metrics for the single instruction

**Example:**
```python
instruction = "What are the benefits of renewable energy?"
reference = "Renewable energy reduces pollution and is sustainable..."

result = evaluator.evaluate_single_instruction(
    instruction,
    reference_response=reference
)

print(f"Generated Response: {result['generated_response']}")
print(f"Instruction Type: {result['instruction_type']}")
print(f"Response Length: {result['response_length']}")
print(f"Format Compliance: {result['format_compliance']}")
```

## Instruction Classification

### `_classify_instruction_type(instruction)`

```python
def _classify_instruction_type(self, instruction: str) -> str
```

**Purpose**: Classify instruction into categories for targeted evaluation.

**Categories:**
- `"question"`: Direct questions requiring factual answers
- `"task"`: Task-oriented instructions (translate, summarize, etc.)
- `"creative"`: Creative writing requests
- `"conversational"`: Conversational prompts
- `"analytical"`: Analysis or reasoning tasks

**Example:**
```python
# Different instruction types
question = "What is the capital of France?"
task = "Translate this text to Arabic: Hello world"
creative = "Write a poem about the ocean"

print(evaluator._classify_instruction_type(question))    # "question"
print(evaluator._classify_instruction_type(task))       # "task"
print(evaluator._classify_instruction_type(creative))   # "creative"
```

## Quality Assessment Methods

### `_evaluate_format_compliance(response)`

```python
def _evaluate_format_compliance(self, response: str) -> Dict[str, bool]
```

**Purpose**: Evaluate response format and structure quality.

**Returns:**
- Dictionary with format compliance metrics:
  - `has_punctuation`: Proper punctuation usage
  - `proper_capitalization`: Appropriate capitalization
  - `reasonable_length`: Response length within reasonable bounds
  - `coherent_structure`: Logical structure and flow

**Example:**
```python
response = "The capital of Egypt is Cairo. It is a historic city."
compliance = evaluator._evaluate_format_compliance(response)

print(f"Has punctuation: {compliance['has_punctuation']}")
print(f"Proper capitalization: {compliance['proper_capitalization']}")
print(f"Reasonable length: {compliance['reasonable_length']}")
```

### `_compute_instruction_following_score(instruction, response)`

```python
def _compute_instruction_following_score(
    self,
    instruction: str,
    response: str
) -> float
```

**Purpose**: Compute how well the response follows the given instruction.

**Returns:**
- Float score between 0.0 and 1.0

**Example:**
```python
instruction = "List three benefits of exercise"
response = "1. Improves health 2. Increases energy 3. Reduces stress"

score = evaluator._compute_instruction_following_score(instruction, response)
print(f"Instruction following score: {score:.3f}")
```

## Specialized Evaluation Metrics

### SFT-Specific Metrics

The SFTEvaluator computes several specialized metrics:

#### Instruction Following Metrics
- **Instruction Following Score**: How well responses address instructions
- **Task Completion Rate**: Percentage of tasks completed successfully
- **Format Adherence**: Compliance with requested output formats

#### Response Quality Metrics
- **Coherence Score**: Logical flow and consistency
- **Relevance Score**: Relevance to the instruction
- **Completeness Score**: Thoroughness of the response

#### Language-Specific Metrics
- **Arabic Grammar Score**: Grammar correctness for Arabic responses
- **Code-Switching Detection**: Mixed language usage analysis
- **Dialect Consistency**: Consistency in Arabic dialect usage

**Example:**
```python
# Comprehensive evaluation with all metrics
results = evaluator.evaluate(dataset)

# Instruction following metrics
print(f"Instruction Following: {results['instruction_following_score']:.3f}")
print(f"Task Completion Rate: {results['task_completion_rate']:.3f}")
print(f"Format Adherence: {results['format_adherence_score']:.3f}")

# Response quality metrics
print(f"Coherence: {results['coherence_score']:.3f}")
print(f"Relevance: {results['relevance_score']:.3f}")
print(f"Completeness: {results['completeness_score']:.3f}")

# Arabic-specific metrics
if 'arabic_grammar_score' in results:
    print(f"Arabic Grammar: {results['arabic_grammar_score']:.3f}")
    print(f"Dialect Consistency: {results['dialect_consistency_score']:.3f}")
```

## Batch Evaluation

### `batch_evaluate_instructions(instructions, reference_responses=None)`

```python
def batch_evaluate_instructions(
    self,
    instructions: List[str],
    reference_responses: Optional[List[str]] = None,
    batch_size: int = 8
) -> List[Dict[str, Any]]
```

**Purpose**: Evaluate multiple instructions efficiently in batches.

**Parameters:**
- `instructions`: List of instructions to evaluate
- `reference_responses`: Optional list of reference responses
- `batch_size`: Batch size for processing

**Returns:**
- List of evaluation results for each instruction

**Example:**
```python
instructions = [
    "What is artificial intelligence?",
    "Translate to Arabic: Good morning",
    "Write a haiku about technology"
]

references = [
    "AI is the simulation of human intelligence...",
    "صباح الخير",
    "Silicon dreams flow / Through circuits of endless light / Future awakens"
]

results = evaluator.batch_evaluate_instructions(
    instructions,
    reference_responses=references,
    batch_size=4
)

for i, result in enumerate(results):
    print(f"Instruction {i+1}: {result['instruction_following_score']:.3f}")
```

## Configuration Integration

The SFTEvaluator integrates with the configuration system:

```python
from src.config import SFTConfig
from src.evaluation import SFTEvaluator

# Load from SFT config
config = SFTConfig.from_yaml("config/sft_config.yaml")
evaluator = SFTEvaluator(
    model=config.model.model_name,
    tokenizer=config.model.model_name,
    device=config.training.device,
    instruction_template=config.data.instruction_template,
    response_template=config.data.response_template
)
```

## Performance Optimization

### Memory Management

```python
# Enable gradient checkpointing for large models
evaluator.model.gradient_checkpointing_enable()

# Use smaller batch sizes for memory efficiency
results = evaluator.evaluate(
    dataset,
    batch_size=4  # Reduce if OOM errors occur
)

# Clear cache periodically
import torch
torch.cuda.empty_cache()
```

### Speed Optimization

```python
# Use mixed precision for faster evaluation
with torch.cuda.amp.autocast():
    results = evaluator.evaluate(dataset)

# Reduce generation length for faster evaluation
results = evaluator.evaluate(
    dataset,
    max_length=128,  # Shorter responses
    do_sample=False  # Greedy decoding
)
```

## Usage Examples

### Basic SFT Evaluation

```python
from src.evaluation import SFTEvaluator
from datasets import Dataset

# Initialize evaluator
evaluator = SFTEvaluator(
    model="./checkpoints/arabic_sft_model",
    device="cuda"
)

# Load evaluation dataset
eval_dataset = Dataset.from_json("data/eval_instructions.jsonl")

# Run evaluation
results = evaluator.evaluate(eval_dataset)

# Print results
print(f"Overall Score: {results['overall_score']:.3f}")
print(f"Instruction Following: {results['instruction_following_score']:.3f}")
print(f"Response Quality: {results['response_quality_score']:.3f}")
```

### Advanced Evaluation with Custom Templates

```python
# Custom Arabic templates
arabic_instruction_template = """
### التعليمات:
{instruction}

### السياق:
{input}

### الإجابة:
"""

arabic_response_template = "{response}"

# Initialize with custom templates
evaluator = SFTEvaluator(
    model="./models/arabic_qwen_sft",
    instruction_template=arabic_instruction_template,
    response_template=arabic_response_template
)

# Evaluate with custom parameters
results = evaluator.evaluate(
    dataset,
    batch_size=6,
    max_length=300,
    temperature=0.8,
    do_sample=True
)
```

### Model Comparison

```python
# Compare two SFT models
evaluator1 = SFTEvaluator(model="model1")
evaluator2 = SFTEvaluator(model="model2")

# Run comparison
comparison = evaluator1.compare_models(
    evaluator2,
    dataset,
    metrics_to_compare=[
        "instruction_following_score",
        "response_quality_score",
        "coherence_score"
    ]
)

print("Model Comparison Results:")
for metric, values in comparison["comparison"].items():
    print(f"{metric}:")
    print(f"  Model 1: {values['model_1']:.3f}")
    print(f"  Model 2: {values['model_2']:.3f}")
    print(f"  Difference: {values['difference']:.3f}")
```

### Integration with Training

```python
from src.training import SFTTrainer
from src.evaluation import SFTEvaluator

# Train model
trainer = SFTTrainer(config)
trainer.train()

# Evaluate trained model
evaluator = SFTEvaluator(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    instruction_template=config.data.instruction_template
)

# Run evaluation on test set
test_results = evaluator.evaluate(test_dataset)

# Save results
evaluator.save_evaluation_results(
    test_results,
    "reports/sft_evaluation_results.json"
)
```

## Factory Function

### `create_sft_evaluator(**kwargs)`

```python
def create_sft_evaluator(
    model=None,
    tokenizer=None,
    device=None,
    **kwargs
) -> SFTEvaluator
```

**Purpose**: Factory function to create SFT evaluator instances.

**Parameters:**
- `model`: Model instance or path
- `tokenizer`: Tokenizer instance or path
- `device`: Device to use
- `**kwargs`: Additional evaluator parameters

**Returns:**
- SFTEvaluator instance

**Example:**
```python
from src.evaluation import create_sft_evaluator

# Create evaluator using factory function
evaluator = create_sft_evaluator(
    model="Qwen/Qwen2.5-7B",
    device="cuda",
    instruction_template="### Instruction: {instruction}\n### Response:"
)
```

## Best Practices

### 1. Template Design

```python
# Use clear, consistent templates
instruction_template = """
### Instruction:
{instruction}

### Response:
"""

# For Arabic models, use Arabic templates
arabic_template = """
### التعليمات:
{instruction}

### الإجابة:
"""
```

### 2. Evaluation Strategy

```python
# Use diverse evaluation datasets
eval_datasets = {
    "general": load_general_instructions(),
    "arabic": load_arabic_instructions(),
    "technical": load_technical_instructions(),
    "creative": load_creative_instructions()
}

# Evaluate on each category
for category, dataset in eval_datasets.items():
    results = evaluator.evaluate(dataset)
    print(f"{category}: {results['overall_score']:.3f}")
```

### 3. Resource Management

```python
# Monitor memory usage
import torch

print(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
results = evaluator.evaluate(dataset, batch_size=4)
print(f"GPU memory after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

## Error Handling

```python
try:
    evaluator = SFTEvaluator(model="invalid/model")
except Exception as e:
    logger.error(f"Failed to load SFT evaluator: {e}")

try:
    results = evaluator.evaluate(dataset)
except RuntimeError as e:
    logger.error(f"SFT evaluation failed: {e}")
    # Reduce batch size and retry
    results = evaluator.evaluate(dataset, batch_size=2)
```

## See Also

- [BaseEvaluator Documentation](base_evaluator.md) - Base evaluation functionality
- [PreferenceEvaluator Documentation](preference_evaluator.md) - Preference optimization evaluation
- [SFTTrainer Documentation](../training/sft_trainer.md) - SFT training implementation
- [Evaluation System Overview](index.md) - Complete evaluation system guide