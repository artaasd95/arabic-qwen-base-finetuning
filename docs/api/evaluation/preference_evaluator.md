# PreferenceEvaluator API Documentation

The `PreferenceEvaluator` class provides specialized evaluation methods for preference optimization models trained with DPO, KTO, IPO, and CPO methods.

## Overview

**Location**: `src/evaluation/preference_evaluator.py`

**Purpose**: Evaluates preference optimization models by computing preference-specific metrics, comparing model responses against reference models, and measuring alignment with human preferences.

**Inheritance**: `PreferenceEvaluator` → `BaseEvaluator`

**Supported Methods**: DPO, KTO, IPO, CPO

## Class Definition

```python
from typing import Dict, List, Optional, Any, Union, Tuple
from datasets import Dataset
import torch
import numpy as np
from .base_evaluator import BaseEvaluator

class PreferenceEvaluator(BaseEvaluator):
    """Evaluator for preference optimization methods.
    
    This class provides evaluation methods for models trained with
    preference optimization techniques like DPO, KTO, IPO, and CPO.
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
    ref_model=None,
    method: str = "dpo"
)
```

**Parameters:**
- `model`: Trained preference model instance or path (optional)
- `tokenizer`: Tokenizer instance or path (optional)
- `device`: Device to run evaluation on (optional, auto-detects CUDA)
- `ref_model`: Reference model for comparison (optional but recommended)
- `method`: Preference optimization method ("dpo", "kto", "ipo", "cpo")

**Raises:**
- `ValueError`: If unsupported method is specified

**Functionality:**
- Inherits all BaseEvaluator functionality
- Sets up reference model for comparison
- Configures method-specific evaluation parameters
- Validates method compatibility

**Example:**
```python
from src.evaluation.preference_evaluator import PreferenceEvaluator

# Initialize DPO evaluator
evaluator = PreferenceEvaluator(
    model="./checkpoints/arabic_dpo_model",
    tokenizer="Qwen/Qwen2.5-7B",
    ref_model="Qwen/Qwen2.5-7B",
    method="dpo",
    device="cuda"
)

# Initialize KTO evaluator
kto_evaluator = PreferenceEvaluator(
    model="./checkpoints/arabic_kto_model",
    ref_model="./checkpoints/arabic_sft_model",
    method="kto"
)

# Initialize without reference model (some metrics will be unavailable)
evaluator_no_ref = PreferenceEvaluator(
    model="./checkpoints/arabic_ipo_model",
    method="ipo"
)
```

## Core Evaluation Methods

### `evaluate(dataset, **kwargs)`

```python
def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]
```

**Purpose**: Main evaluation method for preference optimization models.

**Parameters:**
- `dataset`: Dataset containing preference data
- `**kwargs`: Additional evaluation parameters
  - `batch_size`: Batch size for evaluation (default: 8)
  - `max_length`: Maximum generation length (default: 512)
  - `temperature`: Generation temperature (default: 0.7)
  - `do_sample`: Whether to use sampling (default: True)
  - `compute_win_rate`: Whether to compute win rate against reference (default: True)

**Returns:**
- Dictionary containing comprehensive preference evaluation metrics

**Expected Dataset Formats:**

#### DPO/IPO/CPO Format:
```python
# Pairwise preference data
dataset = Dataset.from_dict({
    "prompt": ["What is AI?", "Translate: Hello"],
    "chosen": ["AI is artificial intelligence...", "مرحبا"],
    "rejected": ["AI is a computer", "Hello in Arabic"]
})
```

#### KTO Format:
```python
# Binary preference data
dataset = Dataset.from_dict({
    "prompt": ["What is AI?", "Translate: Hello"],
    "completion": ["AI is artificial intelligence...", "مرحبا"],
    "label": [True, True]  # True for desirable, False for undesirable
})
```

**Example:**
```python
from datasets import Dataset

# DPO evaluation dataset
dpo_data = Dataset.from_dict({
    "prompt": [
        "Explain quantum computing",
        "What are the benefits of renewable energy?",
        "How does machine learning work?"
    ],
    "chosen": [
        "Quantum computing uses quantum mechanics principles...",
        "Renewable energy reduces carbon emissions...",
        "Machine learning uses algorithms to find patterns..."
    ],
    "rejected": [
        "Quantum computing is just fast computing",
        "Renewable energy is expensive",
        "Machine learning is magic"
    ]
})

# Evaluate DPO model
results = evaluator.evaluate(
    dpo_data,
    batch_size=4,
    max_length=256,
    compute_win_rate=True
)

print(f"Preference Accuracy: {results['preference_accuracy']:.3f}")
print(f"Win Rate vs Reference: {results['win_rate']:.3f}")
print(f"Average Reward: {results['average_reward']:.3f}")
```

### `compute_preference_accuracy(dataset)`

```python
def compute_preference_accuracy(self, dataset: Dataset) -> float
```

**Purpose**: Compute how often the model prefers the chosen response over rejected.

**Parameters:**
- `dataset`: Dataset with preference pairs

**Returns:**
- Float between 0.0 and 1.0 representing preference accuracy

**Example:**
```python
# Compute preference accuracy
accuracy = evaluator.compute_preference_accuracy(dpo_data)
print(f"Model correctly prefers chosen responses {accuracy:.1%} of the time")
```

### `compute_win_rate(dataset, ref_model=None)`

```python
def compute_win_rate(
    self,
    dataset: Dataset,
    ref_model=None
) -> Dict[str, float]
```

**Purpose**: Compute win rate of trained model against reference model.

**Parameters:**
- `dataset`: Evaluation dataset
- `ref_model`: Reference model (uses self.ref_model if None)

**Returns:**
- Dictionary with win rate statistics

**Example:**
```python
# Compute win rate against reference
win_stats = evaluator.compute_win_rate(eval_dataset)

print(f"Win Rate: {win_stats['win_rate']:.3f}")
print(f"Tie Rate: {win_stats['tie_rate']:.3f}")
print(f"Loss Rate: {win_stats['loss_rate']:.3f}")
print(f"Total Comparisons: {win_stats['total_comparisons']}")
```

## Method-Specific Evaluation

### DPO Evaluation

#### `evaluate_dpo_loss(dataset)`

```python
def evaluate_dpo_loss(self, dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Compute DPO-specific loss and metrics.

**Returns:**
- Dictionary with DPO loss components:
  - `dpo_loss`: Overall DPO loss
  - `chosen_rewards`: Average rewards for chosen responses
  - `rejected_rewards`: Average rewards for rejected responses
  - `reward_margin`: Difference between chosen and rejected rewards
  - `kl_divergence`: KL divergence from reference model

**Example:**
```python
# Evaluate DPO-specific metrics
dpo_metrics = evaluator.evaluate_dpo_loss(dpo_dataset)

print(f"DPO Loss: {dpo_metrics['dpo_loss']:.4f}")
print(f"Reward Margin: {dpo_metrics['reward_margin']:.4f}")
print(f"KL Divergence: {dpo_metrics['kl_divergence']:.4f}")
```

### KTO Evaluation

#### `evaluate_kto_metrics(dataset)`

```python
def evaluate_kto_metrics(self, dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Compute KTO-specific metrics based on prospect theory.

**Returns:**
- Dictionary with KTO metrics:
  - `kto_loss`: KTO loss value
  - `desirable_accuracy`: Accuracy on desirable examples
  - `undesirable_accuracy`: Accuracy on undesirable examples
  - `utility_gain`: Average utility for desirable responses
  - `utility_loss`: Average utility for undesirable responses

**Example:**
```python
# KTO evaluation dataset
kto_data = Dataset.from_dict({
    "prompt": ["Explain AI", "What is Python?"],
    "completion": ["AI is artificial intelligence", "Python is a programming language"],
    "label": [True, True]
})

# Evaluate KTO metrics
kto_metrics = evaluator.evaluate_kto_metrics(kto_data)

print(f"KTO Loss: {kto_metrics['kto_loss']:.4f}")
print(f"Desirable Accuracy: {kto_metrics['desirable_accuracy']:.3f}")
print(f"Utility Gain: {kto_metrics['utility_gain']:.4f}")
```

### IPO Evaluation

#### `evaluate_ipo_metrics(dataset)`

```python
def evaluate_ipo_metrics(self, dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Compute IPO-specific metrics with length bias mitigation.

**Returns:**
- Dictionary with IPO metrics:
  - `ipo_loss`: IPO loss value
  - `length_normalized_accuracy`: Accuracy normalized for length bias
  - `short_response_accuracy`: Accuracy on shorter responses
  - `long_response_accuracy`: Accuracy on longer responses
  - `length_bias_score`: Measure of length bias in preferences

**Example:**
```python
# Evaluate IPO metrics
ipo_metrics = evaluator.evaluate_ipo_metrics(ipo_dataset)

print(f"IPO Loss: {ipo_metrics['ipo_loss']:.4f}")
print(f"Length Normalized Accuracy: {ipo_metrics['length_normalized_accuracy']:.3f}")
print(f"Length Bias Score: {ipo_metrics['length_bias_score']:.4f}")
```

### CPO Evaluation

#### `evaluate_cpo_metrics(dataset)`

```python
def evaluate_cpo_metrics(self, dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Compute CPO-specific contrastive learning metrics.

**Returns:**
- Dictionary with CPO metrics:
  - `cpo_loss`: CPO contrastive loss
  - `contrastive_accuracy`: Accuracy in contrastive setting
  - `hard_negative_accuracy`: Accuracy on hard negative examples
  - `representation_quality`: Quality of learned representations

**Example:**
```python
# Evaluate CPO metrics
cpo_metrics = evaluator.evaluate_cpo_metrics(cpo_dataset)

print(f"CPO Loss: {cpo_metrics['cpo_loss']:.4f}")
print(f"Contrastive Accuracy: {cpo_metrics['contrastive_accuracy']:.3f}")
print(f"Hard Negative Accuracy: {cpo_metrics['hard_negative_accuracy']:.3f}")
```

## Response Generation and Comparison

### `generate_preference_comparison(prompt, **kwargs)`

```python
def generate_preference_comparison(
    self,
    prompt: str,
    **kwargs
) -> Dict[str, Any]
```

**Purpose**: Generate and compare responses from trained and reference models.

**Parameters:**
- `prompt`: Input prompt for generation
- `**kwargs`: Generation parameters

**Returns:**
- Dictionary with comparison results:
  - `prompt`: Original prompt
  - `trained_response`: Response from trained model
  - `reference_response`: Response from reference model (if available)
  - `preference_score`: Preference score for trained model
  - `quality_metrics`: Quality assessment metrics

**Example:**
```python
# Generate comparison
comparison = evaluator.generate_preference_comparison(
    "What are the advantages of renewable energy?",
    max_length=200,
    temperature=0.8
)

print(f"Prompt: {comparison['prompt']}")
print(f"\nTrained Model: {comparison['trained_response']}")
print(f"\nReference Model: {comparison['reference_response']}")
print(f"\nPreference Score: {comparison['preference_score']:.3f}")
```

### `batch_preference_comparison(prompts, **kwargs)`

```python
def batch_preference_comparison(
    self,
    prompts: List[str],
    batch_size: int = 8,
    **kwargs
) -> List[Dict[str, Any]]
```

**Purpose**: Generate preference comparisons for multiple prompts efficiently.

**Parameters:**
- `prompts`: List of prompts for comparison
- `batch_size`: Batch size for processing
- `**kwargs`: Generation parameters

**Returns:**
- List of comparison results for each prompt

**Example:**
```python
prompts = [
    "Explain machine learning",
    "What is quantum computing?",
    "How does blockchain work?"
]

comparisons = evaluator.batch_preference_comparison(
    prompts,
    batch_size=4,
    max_length=150
)

for i, comp in enumerate(comparisons):
    print(f"Prompt {i+1}: Score = {comp['preference_score']:.3f}")
```

## Specialized Metrics

### Arabic-Specific Preference Metrics

#### `compute_arabic_preference_metrics(dataset)`

```python
def compute_arabic_preference_metrics(self, dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Compute Arabic language-specific preference metrics.

**Returns:**
- Dictionary with Arabic-specific metrics:
  - `arabic_fluency_preference`: Preference for fluent Arabic responses
  - `dialect_consistency_preference`: Preference for consistent dialect usage
  - `cultural_appropriateness`: Cultural appropriateness of preferred responses
  - `code_switching_preference`: Handling of Arabic-English code-switching

**Example:**
```python
# Arabic preference evaluation
arabic_metrics = evaluator.compute_arabic_preference_metrics(arabic_dataset)

print(f"Arabic Fluency Preference: {arabic_metrics['arabic_fluency_preference']:.3f}")
print(f"Dialect Consistency: {arabic_metrics['dialect_consistency_preference']:.3f}")
print(f"Cultural Appropriateness: {arabic_metrics['cultural_appropriateness']:.3f}")
```

### Reward Model Evaluation

#### `evaluate_reward_model(dataset)`

```python
def evaluate_reward_model(self, dataset: Dataset) -> Dict[str, float]
```

**Purpose**: Evaluate the implicit reward model learned by preference optimization.

**Returns:**
- Dictionary with reward model metrics:
  - `reward_accuracy`: Accuracy of reward predictions
  - `reward_correlation`: Correlation with human preferences
  - `reward_calibration`: Calibration of reward scores
  - `reward_consistency`: Consistency across similar prompts

**Example:**
```python
# Evaluate implicit reward model
reward_metrics = evaluator.evaluate_reward_model(preference_dataset)

print(f"Reward Accuracy: {reward_metrics['reward_accuracy']:.3f}")
print(f"Reward Correlation: {reward_metrics['reward_correlation']:.3f}")
print(f"Reward Calibration: {reward_metrics['reward_calibration']:.3f}")
```

## Model Comparison and Analysis

### `compare_preference_models(other_evaluator, dataset)`

```python
def compare_preference_models(
    self,
    other_evaluator: 'PreferenceEvaluator',
    dataset: Dataset
) -> Dict[str, Any]
```

**Purpose**: Compare two preference optimization models.

**Parameters:**
- `other_evaluator`: Another PreferenceEvaluator instance
- `dataset`: Evaluation dataset

**Returns:**
- Dictionary with detailed comparison results

**Example:**
```python
# Compare DPO and KTO models
dpo_evaluator = PreferenceEvaluator(model="./dpo_model", method="dpo")
kto_evaluator = PreferenceEvaluator(model="./kto_model", method="kto")

comparison = dpo_evaluator.compare_preference_models(
    kto_evaluator,
    eval_dataset
)

print(f"DPO vs KTO Comparison:")
print(f"Preference Accuracy: {comparison['preference_accuracy']['model_1']:.3f} vs {comparison['preference_accuracy']['model_2']:.3f}")
print(f"Win Rate: {comparison['win_rate']['model_1']:.3f} vs {comparison['win_rate']['model_2']:.3f}")
```

### `analyze_preference_patterns(dataset)`

```python
def analyze_preference_patterns(self, dataset: Dataset) -> Dict[str, Any]
```

**Purpose**: Analyze patterns in model preferences and behavior.

**Returns:**
- Dictionary with pattern analysis:
  - `preference_consistency`: Consistency of preferences
  - `length_bias_analysis`: Analysis of length bias
  - `topic_preference_analysis`: Preferences by topic/category
  - `quality_vs_preference`: Relationship between quality and preference

**Example:**
```python
# Analyze preference patterns
patterns = evaluator.analyze_preference_patterns(eval_dataset)

print(f"Preference Consistency: {patterns['preference_consistency']:.3f}")
print(f"Length Bias Score: {patterns['length_bias_analysis']['bias_score']:.3f}")
print(f"Topic Preferences: {patterns['topic_preference_analysis']}")
```

## Configuration Integration

The PreferenceEvaluator integrates with the configuration system:

```python
from src.config import DPOConfig, KTOConfig, IPOConfig, CPOConfig
from src.evaluation import PreferenceEvaluator

# Load from DPO config
dpo_config = DPOConfig.from_yaml("config/dpo_config.yaml")
evaluator = PreferenceEvaluator(
    model=dpo_config.model.model_name,
    tokenizer=dpo_config.model.model_name,
    ref_model=dpo_config.training.ref_model_name,
    method="dpo",
    device=dpo_config.training.device
)

# Load from KTO config
kto_config = KTOConfig.from_yaml("config/kto_config.yaml")
kto_evaluator = PreferenceEvaluator(
    model=kto_config.model.model_name,
    method="kto"
)
```

## Performance Optimization

### Memory Management

```python
# Enable gradient checkpointing for large models
evaluator.model.gradient_checkpointing_enable()
if evaluator.ref_model:
    evaluator.ref_model.gradient_checkpointing_enable()

# Use smaller batch sizes for memory efficiency
results = evaluator.evaluate(
    dataset,
    batch_size=2  # Reduce for memory constraints
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
    max_length=128,
    do_sample=False  # Greedy decoding is faster
)

# Disable unnecessary computations
results = evaluator.evaluate(
    dataset,
    compute_win_rate=False  # Skip if reference model unavailable
)
```

## Usage Examples

### Basic Preference Evaluation

```python
from src.evaluation import PreferenceEvaluator
from datasets import Dataset

# Initialize evaluator
evaluator = PreferenceEvaluator(
    model="./checkpoints/arabic_dpo_model",
    ref_model="./checkpoints/arabic_sft_model",
    method="dpo",
    device="cuda"
)

# Load preference dataset
preference_dataset = Dataset.from_json("data/preference_eval.jsonl")

# Run evaluation
results = evaluator.evaluate(preference_dataset)

# Print key metrics
print(f"Preference Accuracy: {results['preference_accuracy']:.3f}")
print(f"Win Rate: {results['win_rate']:.3f}")
print(f"Average Reward: {results['average_reward']:.3f}")
```

### Multi-Method Comparison

```python
# Compare different preference optimization methods
methods = ["dpo", "kto", "ipo", "cpo"]
evaluators = {}
results = {}

for method in methods:
    evaluators[method] = PreferenceEvaluator(
        model=f"./checkpoints/arabic_{method}_model",
        ref_model="./checkpoints/arabic_sft_model",
        method=method
    )
    results[method] = evaluators[method].evaluate(eval_dataset)

# Compare results
print("Method Comparison:")
for method in methods:
    print(f"{method.upper()}: {results[method]['preference_accuracy']:.3f}")
```

### Advanced Arabic Evaluation

```python
# Comprehensive Arabic preference evaluation
arabic_evaluator = PreferenceEvaluator(
    model="./models/arabic_preference_model",
    ref_model="./models/arabic_base_model",
    method="dpo"
)

# Load Arabic-specific evaluation data
arabic_dataset = Dataset.from_json("data/arabic_preferences.jsonl")

# Run comprehensive evaluation
results = arabic_evaluator.evaluate(arabic_dataset)
arabic_metrics = arabic_evaluator.compute_arabic_preference_metrics(arabic_dataset)
patterns = arabic_evaluator.analyze_preference_patterns(arabic_dataset)

# Print comprehensive results
print("=== Arabic Preference Evaluation ===")
print(f"Overall Preference Accuracy: {results['preference_accuracy']:.3f}")
print(f"Arabic Fluency Preference: {arabic_metrics['arabic_fluency_preference']:.3f}")
print(f"Dialect Consistency: {arabic_metrics['dialect_consistency_preference']:.3f}")
print(f"Cultural Appropriateness: {arabic_metrics['cultural_appropriateness']:.3f}")
print(f"Preference Consistency: {patterns['preference_consistency']:.3f}")
```

### Integration with Training Pipeline

```python
from src.training import DPOTrainer
from src.evaluation import PreferenceEvaluator

# Train DPO model
trainer = DPOTrainer(config)
trainer.train()

# Evaluate trained model
evaluator = PreferenceEvaluator(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    ref_model=trainer.ref_model,
    method="dpo"
)

# Run evaluation on test set
test_results = evaluator.evaluate(test_dataset)

# Generate comparison examples
comparisons = evaluator.batch_preference_comparison([
    "What is artificial intelligence?",
    "Explain renewable energy benefits",
    "How does machine learning work?"
])

# Save results
evaluator.save_evaluation_results(
    {
        "test_results": test_results,
        "comparisons": comparisons
    },
    "reports/dpo_evaluation_results.json"
)
```

## Factory Function

### `create_preference_evaluator(**kwargs)`

```python
def create_preference_evaluator(
    model=None,
    tokenizer=None,
    device=None,
    ref_model=None,
    method: str = "dpo",
    **kwargs
) -> PreferenceEvaluator
```

**Purpose**: Factory function to create preference evaluator instances.

**Parameters:**
- `model`: Model instance or path
- `tokenizer`: Tokenizer instance or path
- `device`: Device to use
- `ref_model`: Reference model
- `method`: Preference optimization method
- `**kwargs`: Additional evaluator parameters

**Returns:**
- PreferenceEvaluator instance

**Example:**
```python
from src.evaluation import create_preference_evaluator

# Create evaluator using factory function
evaluator = create_preference_evaluator(
    model="./checkpoints/arabic_dpo_model",
    ref_model="Qwen/Qwen2.5-7B",
    method="dpo",
    device="cuda"
)
```

## Best Practices

### 1. Reference Model Selection

```python
# Use appropriate reference models
# For DPO: Use the SFT model as reference
dpo_evaluator = PreferenceEvaluator(
    model="./dpo_model",
    ref_model="./sft_model",  # SFT model as reference
    method="dpo"
)

# For KTO: Can use base model or SFT model
kto_evaluator = PreferenceEvaluator(
    model="./kto_model",
    ref_model="./base_model",  # Base model as reference
    method="kto"
)
```

### 2. Evaluation Strategy

```python
# Use diverse evaluation datasets
eval_datasets = {
    "general": load_general_preferences(),
    "arabic": load_arabic_preferences(),
    "technical": load_technical_preferences(),
    "creative": load_creative_preferences()
}

# Evaluate on each category
for category, dataset in eval_datasets.items():
    results = evaluator.evaluate(dataset)
    print(f"{category}: {results['preference_accuracy']:.3f}")
```

### 3. Method-Specific Considerations

```python
# DPO: Focus on preference accuracy and reward margins
if evaluator.method == "dpo":
    dpo_metrics = evaluator.evaluate_dpo_loss(dataset)
    print(f"Reward Margin: {dpo_metrics['reward_margin']:.4f}")

# KTO: Focus on desirable vs undesirable accuracy
elif evaluator.method == "kto":
    kto_metrics = evaluator.evaluate_kto_metrics(dataset)
    print(f"Desirable Accuracy: {kto_metrics['desirable_accuracy']:.3f}")

# IPO: Focus on length bias mitigation
elif evaluator.method == "ipo":
    ipo_metrics = evaluator.evaluate_ipo_metrics(dataset)
    print(f"Length Bias Score: {ipo_metrics['length_bias_score']:.4f}")
```

### 4. Resource Management

```python
# Monitor memory usage with reference models
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
    evaluator = PreferenceEvaluator(
        model="invalid/model",
        method="invalid_method"
    )
except ValueError as e:
    logger.error(f"Invalid method: {e}")
except Exception as e:
    logger.error(f"Failed to load preference evaluator: {e}")

try:
    results = evaluator.evaluate(dataset)
except RuntimeError as e:
    logger.error(f"Preference evaluation failed: {e}")
    # Reduce batch size and retry
    results = evaluator.evaluate(dataset, batch_size=2)
```

## See Also

- [BaseEvaluator Documentation](base_evaluator.md) - Base evaluation functionality
- [SFTEvaluator Documentation](sft_evaluator.md) - SFT model evaluation
- [DPOTrainer Documentation](../training/dpo_trainer.md) - DPO training implementation
- [PreferenceTrainer Documentation](../training/preference_trainer.md) - KTO/IPO/CPO training
- [Evaluation System Overview](index.md) - Complete evaluation system guide