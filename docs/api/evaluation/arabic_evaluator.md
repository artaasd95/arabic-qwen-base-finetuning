# ArabicEvaluator

The `ArabicEvaluator` class provides specialized evaluation methods for Arabic language models with comprehensive metrics designed specifically for Arabic text analysis.

## Overview

The `ArabicEvaluator` extends the <mcfile name="base_evaluator.py" path="src/evaluation/base_evaluator.py"></mcfile> to provide Arabic-specific evaluation capabilities including:

- **Character-level Analysis**: Arabic character ratio, diacritics coverage, script consistency
- **Linguistic Metrics**: Dialect detection, formality scoring, grammar correctness
- **Cultural Assessment**: Cultural appropriateness and religious sensitivity
- **Quality Metrics**: Text fluency, coherence, and readability
- **Specialized Features**: Diacritization accuracy and dialect consistency

## Class Definition

```python
from src.evaluation import ArabicEvaluator, create_arabic_evaluator

# Direct instantiation
evaluator = ArabicEvaluator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    dialect="msa",
    diacritics_enabled=True
)

# Factory function
evaluator = create_arabic_evaluator(
    model="path/to/model",
    tokenizer="path/to/tokenizer",
    dialect="egyptian",
    diacritics_enabled=False
)
```

## Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` or model object | `None` | Model instance or path to model |
| `tokenizer` | `str` or tokenizer object | `None` | Tokenizer instance or path |
| `device` | `str` | `None` | Device for evaluation ("cuda", "cpu") |
| `dialect` | `str` | `"msa"` | Arabic dialect ("msa", "egyptian", "gulf", "levantine", "maghrebi") |
| `diacritics_enabled` | `bool` | `True` | Whether to evaluate diacritics |

## Core Evaluation Methods

### Main Evaluation

#### `evaluate(dataset, **kwargs)`

Performs comprehensive Arabic language evaluation.

```python
from datasets import Dataset

# Prepare Arabic dataset
dataset = Dataset.from_dict({
    "text": [
        "مرحباً بكم في عالم الذكاء الاصطناعي",
        "كيف يمكنني مساعدتك اليوم؟",
        "هذا نص باللغة العربية الفصحى"
    ]
})

# Evaluate
results = evaluator.evaluate(dataset)
print(f"Arabic character ratio: {results['arabic_character_ratio']:.3f}")
print(f"Dialect detection accuracy: {results['dialect_detection_accuracy']:.3f}")
print(f"Grammar correctness: {results['grammar_correctness']:.3f}")
```

**Returns:**
```python
{
    # Base metrics from BaseEvaluator
    "perplexity": 15.2,
    "bleu_score": 0.85,
    
    # Character-level metrics
    "arabic_character_ratio": 0.95,
    "diacritics_coverage": 0.12,
    "script_consistency": 0.98,
    
    # Linguistic metrics
    "dialect_detection_accuracy": 0.87,
    "formality_score": 0.75,
    "grammar_correctness": 0.82,
    
    # Cultural metrics
    "cultural_appropriateness": 0.95,
    "religious_sensitivity": 0.98,
    
    # Quality metrics
    "text_fluency": 0.78,
    "coherence_score": 0.81,
    "readability_score": 0.73,
    
    # Configuration
    "dialect": "msa",
    "diacritics_enabled": True
}
```

### Arabic-Specific Metrics

#### `compute_arabic_metrics(dataset)`

Computes comprehensive Arabic language metrics.

```python
# Compute only Arabic-specific metrics
arabic_metrics = evaluator.compute_arabic_metrics(dataset)

print(f"Formality score: {arabic_metrics['formality_score']:.3f}")
print(f"Cultural appropriateness: {arabic_metrics['cultural_appropriateness']:.3f}")
```

## Specialized Evaluation Methods

### Diacritization Accuracy

#### `evaluate_diacritization_accuracy(predictions, references)`

Evaluates accuracy of Arabic diacritization.

```python
# Diacritized text evaluation
predictions = ["مَرْحَباً بِكُمْ", "كَيْفَ حَالُكُمْ"]
references = ["مَرْحَباً بِكُمْ", "كَيْفَ حَالُكُمْ"]

accuracy = evaluator.evaluate_diacritization_accuracy(predictions, references)
print(f"Diacritization accuracy: {accuracy:.3f}")
```

### Dialect Consistency

#### `evaluate_dialect_consistency(texts)`

Evaluates consistency of dialect usage across texts.

```python
# Check dialect consistency
texts = [
    "إزيك يا صاحبي؟",  # Egyptian
    "شلونك حبيبي؟",     # Gulf
    "كيف حالك يا صديقي؟" # MSA
]

consistency = evaluator.evaluate_dialect_consistency(texts)
print(f"Dialect consistency: {consistency:.3f}")
```

## Text Generation

### Arabic Response Generation

#### `generate_arabic_response(prompt, **kwargs)`

Generates Arabic responses with proper formatting.

```python
# Generate Arabic response
prompt = "اكتب قصة قصيرة عن الصداقة"
response = evaluator.generate_arabic_response(
    prompt,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

print(f"Generated response: {response}")
```

## Model Comparison

### Arabic Model Comparison

#### `compare_arabic_models(other_evaluator, dataset)`

Compares two Arabic language models.

```python
# Create two evaluators for different models
evaluator_1 = create_arabic_evaluator(
    model="model_1_path",
    dialect="msa"
)

evaluator_2 = create_arabic_evaluator(
    model="model_2_path",
    dialect="msa"
)

# Compare models
comparison = evaluator_1.compare_arabic_models(evaluator_2, dataset)

print("Model Comparison Results:")
for metric, data in comparison["comparison"].items():
    print(f"{metric}: {data['better_model']} wins")
    print(f"  Model 1: {data['model_1']:.3f}")
    print(f"  Model 2: {data['model_2']:.3f}")
    print(f"  Difference: {data['difference']:.3f}")
```

## Supported Arabic Dialects

The evaluator supports detection and evaluation of major Arabic dialects:

| Dialect | Code | Description | Example Markers |
|---------|------|-------------|----------------|
| Modern Standard Arabic | `msa` | Formal Arabic | إن، أن، كان، سوف |
| Egyptian | `egyptian` | Egyptian Arabic | ده، دي، إيه، ازاي |
| Gulf | `gulf` | Gulf Arabic | شلون، وين، شنو |
| Levantine | `levantine` | Levantine Arabic | شو، وين، كيف، هيك |
| Maghrebi | `maghrebi` | Maghrebi Arabic | شنو، فين، كيفاش |

## Configuration Integration

The evaluator integrates with the configuration system:

```python
# Using with configuration
from src.config import EvaluationConfig

config = EvaluationConfig(
    evaluation_method="arabic",
    dialect="egyptian",
    diacritics_enabled=True,
    batch_size=16
)

evaluator = create_arabic_evaluator(
    model=config.model_path,
    tokenizer=config.tokenizer_path,
    dialect=config.dialect,
    diacritics_enabled=config.diacritics_enabled
)
```

## Performance Optimization

### Memory Management

```python
# Efficient evaluation for large datasets
def evaluate_large_dataset(evaluator, dataset, batch_size=32):
    results = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        batch_results = evaluator.evaluate(batch)
        results.append(batch_results)
        
        # Clear cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    return results
```

### Speed Optimization

```python
# Fast evaluation mode
evaluator = create_arabic_evaluator(
    model=model,
    tokenizer=tokenizer,
    dialect="msa",
    diacritics_enabled=False  # Disable for speed
)

# Use smaller sample for quick assessment
quick_sample = dataset.select(range(100))
quick_results = evaluator.evaluate(quick_sample)
```

## Usage Examples

### Basic Arabic Evaluation

```python
from src.evaluation import create_arabic_evaluator
from datasets import Dataset

# Create evaluator
evaluator = create_arabic_evaluator(
    model="path/to/arabic/model",
    tokenizer="path/to/tokenizer",
    dialect="msa",
    device="cuda"
)

# Prepare dataset
dataset = Dataset.from_dict({
    "text": [
        "السلام عليكم ورحمة الله وبركاته",
        "أهلاً وسهلاً بكم في موقعنا",
        "نحن نقدم خدمات متميزة"
    ]
})

# Evaluate
results = evaluator.evaluate(dataset)
print(f"Overall Arabic quality: {results['text_fluency']:.3f}")
```

### Advanced Dialect-Specific Evaluation

```python
# Egyptian dialect evaluation
egyptian_evaluator = create_arabic_evaluator(
    model=model,
    dialect="egyptian",
    diacritics_enabled=False
)

egyptian_dataset = Dataset.from_dict({
    "text": [
        "إزيك يا صاحبي؟",
        "عامل إيه النهاردة؟",
        "خلاص كده تمام"
    ]
})

results = egyptian_evaluator.evaluate(egyptian_dataset)
print(f"Dialect detection accuracy: {results['dialect_detection_accuracy']:.3f}")
```

### Cultural Sensitivity Assessment

```python
# Evaluate cultural appropriateness
cultural_dataset = Dataset.from_dict({
    "text": [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "صلى الله عليه وسلم"
    ]
})

results = evaluator.evaluate(cultural_dataset)
print(f"Cultural appropriateness: {results['cultural_appropriateness']:.3f}")
print(f"Religious sensitivity: {results['religious_sensitivity']:.3f}")
```

### Integration with Training Pipeline

```python
from src.training import SFTTrainer
from src.evaluation import create_arabic_evaluator

# Training with Arabic evaluation
trainer = SFTTrainer(config=sft_config)
evaluator = create_arabic_evaluator(
    dialect="msa",
    diacritics_enabled=True
)

# Train model
model = trainer.train()

# Evaluate Arabic capabilities
evaluator.model = model
results = evaluator.evaluate(eval_dataset)

print(f"Arabic fluency: {results['text_fluency']:.3f}")
print(f"Grammar correctness: {results['grammar_correctness']:.3f}")
```

## Factory Function

### `create_arabic_evaluator(**kwargs)`

Convenience factory function for creating Arabic evaluators.

```python
# Various creation methods
evaluator = create_arabic_evaluator(
    model="microsoft/DialoGPT-medium",
    tokenizer="microsoft/DialoGPT-medium",
    dialect="gulf",
    diacritics_enabled=True,
    device="cuda"
)

# With custom parameters
evaluator = create_arabic_evaluator(
    model=loaded_model,
    tokenizer=loaded_tokenizer,
    dialect="levantine"
)
```

## Best Practices

### Dialect Selection

```python
# Choose appropriate dialect for your use case
if target_audience == "formal_documents":
    dialect = "msa"
elif target_audience == "egyptian_users":
    dialect = "egyptian"
elif target_audience == "gulf_region":
    dialect = "gulf"
else:
    dialect = "msa"  # Default to MSA

evaluator = create_arabic_evaluator(dialect=dialect)
```

### Evaluation Strategy

```python
# Comprehensive evaluation approach
def comprehensive_arabic_evaluation(model, dataset):
    results = {}
    
    # Evaluate different dialects
    dialects = ["msa", "egyptian", "gulf"]
    
    for dialect in dialects:
        evaluator = create_arabic_evaluator(
            model=model,
            dialect=dialect
        )
        
        dialect_results = evaluator.evaluate(dataset)
        results[f"{dialect}_evaluation"] = dialect_results
    
    return results
```

### Resource Management

```python
# Efficient resource usage
with torch.no_grad():
    evaluator = create_arabic_evaluator(
        model=model,
        device="cuda"
    )
    
    results = evaluator.evaluate(dataset)
    
    # Clean up
    del evaluator
    torch.cuda.empty_cache()
```

## Error Handling

The evaluator includes comprehensive error handling:

```python
try:
    evaluator = create_arabic_evaluator(
        model="invalid/path",
        dialect="unsupported_dialect"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Safe evaluation with error handling
def safe_arabic_evaluation(evaluator, dataset):
    try:
        return evaluator.evaluate(dataset)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Reduce batch size and retry
            torch.cuda.empty_cache()
            return evaluator.evaluate(dataset.select(range(len(dataset)//2)))
        else:
            raise e
```

## Related Documentation

- <mcfile name="base_evaluator.md" path="docs/api/evaluation/base_evaluator.md"></mcfile> - Base evaluation functionality
- <mcfile name="sft_evaluator.md" path="docs/api/evaluation/sft_evaluator.md"></mcfile> - SFT-specific evaluation
- <mcfile name="preference_evaluator.md" path="docs/api/evaluation/preference_evaluator.md"></mcfile> - Preference optimization evaluation
- <mcfile name="index.md" path="docs/api/evaluation/index.md"></mcfile> - Evaluation system overview
- <mcfile name="fine-tuning-guide.md" path="docs/fine-tuning-guide.md"></mcfile> - Complete fine-tuning guide