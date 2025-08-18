# Evaluation System API Documentation

The evaluation system provides comprehensive metrics and tools for assessing Arabic Qwen model performance across different training methods and tasks.

## Overview

The evaluation system is designed to provide:
- **Standardized metrics** for different training methods (SFT, DPO, KTO, IPO, CPO)
- **Arabic-specific evaluations** considering language nuances
- **Custom evaluator framework** for domain-specific assessments
- **Automated evaluation pipelines** for consistent testing
- **Performance benchmarking** against baseline models

## System Architecture

```
src/evaluation/
├── __init__.py
├── base_evaluator.py      # Base evaluation framework
├── metrics.py             # Core evaluation metrics
├── sft_evaluator.py       # SFT-specific evaluation
├── preference_evaluator.py # Preference optimization evaluation
├── arabic_evaluator.py    # Arabic language-specific evaluation
└── benchmarks.py          # Benchmark datasets and procedures
```

## Core Components

### 1. Base Evaluator (`base_evaluator.py`)

Provides the foundation for all evaluation tasks:

```python
from src.evaluation.base_evaluator import BaseEvaluator

class BaseEvaluator:
    """Base class for all model evaluators."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def evaluate(self, dataset, metrics=None):
        """Main evaluation method."""
        pass
    
    def compute_metrics(self, predictions, references):
        """Compute evaluation metrics."""
        pass
```

### 2. Metrics (`metrics.py`)

Implements various evaluation metrics:

```python
from src.evaluation.metrics import (
    compute_perplexity,
    compute_bleu_score,
    compute_rouge_scores,
    compute_bertscore,
    compute_preference_accuracy,
    compute_arabic_specific_metrics
)

# Language modeling metrics
perplexity = compute_perplexity(model, dataset)

# Generation quality metrics
bleu = compute_bleu_score(predictions, references)
rouge = compute_rouge_scores(predictions, references)
bertscore = compute_bertscore(predictions, references)

# Preference optimization metrics
preference_acc = compute_preference_accuracy(chosen_scores, rejected_scores)

# Arabic-specific metrics
arabic_metrics = compute_arabic_specific_metrics(predictions, references)
```

### 3. Training Method Evaluators

#### SFT Evaluator (`sft_evaluator.py`)
```python
from src.evaluation.sft_evaluator import SFTEvaluator

evaluator = SFTEvaluator(model, tokenizer, config)
results = evaluator.evaluate(test_dataset)
```

#### Preference Evaluator (`preference_evaluator.py`)
```python
from src.evaluation.preference_evaluator import PreferenceEvaluator

evaluator = PreferenceEvaluator(model, tokenizer, config, method="dpo")
results = evaluator.evaluate(preference_dataset)
```

#### Arabic Evaluator (`arabic_evaluator.py`)
```python
from src.evaluation.arabic_evaluator import ArabicEvaluator

evaluator = ArabicEvaluator(
    model=model,
    tokenizer=tokenizer,
    dialect="msa",
    diacritics_enabled=True
)
results = evaluator.evaluate(arabic_dataset)
```

## Evaluation Metrics

### Language Modeling Metrics

#### Perplexity
- **Purpose**: Measures how well the model predicts text
- **Range**: Lower is better (1.0 is perfect)
- **Usage**: Primary metric for language modeling evaluation

```python
perplexity = compute_perplexity(
    model=model,
    dataset=test_dataset,
    batch_size=8,
    max_length=512
)
print(f"Perplexity: {perplexity:.2f}")
```

#### Cross-Entropy Loss
- **Purpose**: Direct loss measurement
- **Range**: Lower is better
- **Usage**: Training monitoring and model comparison

### Text Generation Metrics

#### BLEU Score
- **Purpose**: Measures n-gram overlap with reference text
- **Range**: 0-100 (higher is better)
- **Usage**: Translation and generation quality assessment

```python
bleu_scores = compute_bleu_score(
    predictions=generated_texts,
    references=reference_texts,
    max_order=4  # BLEU-4
)
print(f"BLEU-4: {bleu_scores['bleu']:.2f}")
```

#### ROUGE Scores
- **Purpose**: Measures recall-oriented overlap
- **Variants**: ROUGE-1, ROUGE-2, ROUGE-L
- **Usage**: Summarization and generation evaluation

```python
rouge_scores = compute_rouge_scores(
    predictions=generated_texts,
    references=reference_texts
)
print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.3f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.3f}")
```

#### BERTScore
- **Purpose**: Semantic similarity using contextual embeddings
- **Range**: 0-1 (higher is better)
- **Usage**: Semantic quality assessment

```python
bert_scores = compute_bertscore(
    predictions=generated_texts,
    references=reference_texts,
    model_type="bert-base-multilingual-cased"
)
print(f"BERTScore F1: {bert_scores['f1']:.3f}")
```

### Preference Optimization Metrics

#### Preference Accuracy
- **Purpose**: How often model prefers chosen over rejected responses
- **Range**: 0-1 (higher is better)
- **Usage**: DPO, IPO, CPO evaluation

```python
preference_accuracy = compute_preference_accuracy(
    chosen_scores=chosen_logprobs,
    rejected_scores=rejected_logprobs
)
print(f"Preference Accuracy: {preference_accuracy:.2%}")
```

#### KTO Metrics
- **Purpose**: Desirable vs undesirable response accuracy
- **Components**: Desirable accuracy, undesirable accuracy
- **Usage**: KTO evaluation

```python
kto_metrics = compute_kto_metrics(
    desirable_scores=desirable_logprobs,
    undesirable_scores=undesirable_logprobs
)
print(f"Desirable Accuracy: {kto_metrics['desirable_accuracy']:.2%}")
print(f"Undesirable Accuracy: {kto_metrics['undesirable_accuracy']:.2%}")
```

### Arabic-Specific Metrics

#### Diacritization Accuracy
- **Purpose**: Measures correct Arabic diacritization
- **Usage**: Arabic text quality assessment

```python
diac_accuracy = compute_diacritization_accuracy(
    predictions=generated_arabic,
    references=reference_arabic
)
print(f"Diacritization Accuracy: {diac_accuracy:.2%}")
```

#### Dialect Detection Accuracy
- **Purpose**: Measures correct Arabic dialect identification
- **Usage**: Multi-dialect model evaluation

```python
dialect_accuracy = compute_dialect_accuracy(
    predictions=predicted_dialects,
    references=true_dialects
)
print(f"Dialect Accuracy: {dialect_accuracy:.2%}")
```

#### Arabic Grammar Correctness
- **Purpose**: Evaluates grammatical correctness in Arabic
- **Usage**: Language quality assessment

```python
grammar_score = compute_arabic_grammar_score(
    texts=generated_arabic_texts
)
print(f"Grammar Score: {grammar_score:.3f}")
```

## Evaluation Workflows

### Standard Evaluation Pipeline

```python
from src.evaluation import create_evaluator

# Create evaluator based on training method
evaluator = create_evaluator(
    method="sft",  # or "dpo", "kto", "ipo", "cpo"
    model=model,
    tokenizer=tokenizer,
    config=eval_config
)

# Run comprehensive evaluation
results = evaluator.evaluate(
    dataset=test_dataset,
    metrics=[
        "perplexity",
        "bleu",
        "rouge",
        "bertscore",
        "arabic_specific"
    ]
)

# Print results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Custom Evaluation

```python
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.metrics import compute_custom_metric

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, dataset, metrics=None):
        results = super().evaluate(dataset, metrics)
        
        # Add custom evaluation
        custom_score = self.compute_custom_metric(dataset)
        results["custom_metric"] = custom_score
        
        return results
    
    def compute_custom_metric(self, dataset):
        # Implement custom evaluation logic
        pass

# Use custom evaluator
custom_evaluator = CustomEvaluator(model, tokenizer, config)
results = custom_evaluator.evaluate(test_dataset)
```

### Batch Evaluation

```python
from src.evaluation import batch_evaluate

# Evaluate multiple models
models = {
    "baseline": baseline_model,
    "sft": sft_model,
    "dpo": dpo_model,
    "kto": kto_model
}

results = batch_evaluate(
    models=models,
    dataset=test_dataset,
    metrics=["perplexity", "bleu", "rouge"],
    output_dir="./evaluation_results"
)

# Compare results
for model_name, model_results in results.items():
    print(f"\n{model_name.upper()} Results:")
    for metric, value in model_results.items():
        print(f"  {metric}: {value}")
```

## Configuration

### Evaluation Configuration

```python
from src.config import EvaluationConfig

eval_config = EvaluationConfig(
    # Basic settings
    batch_size=8,
    max_length=512,
    device="cuda",
    
    # Metrics to compute
    metrics=[
        "perplexity",
        "bleu",
        "rouge",
        "bertscore",
        "preference_accuracy",  # For preference methods
        "arabic_specific"
    ],
    
    # Generation settings
    generation_config={
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1
    },
    
    # Arabic-specific settings
    arabic_config={
        "check_diacritization": True,
        "detect_dialect": True,
        "grammar_check": True,
        "cultural_appropriateness": True
    },
    
    # Output settings
    save_predictions=True,
    save_detailed_results=True,
    output_dir="./evaluation_results"
)
```

### Method-Specific Configurations

#### SFT Evaluation Config
```python
sft_eval_config = EvaluationConfig(
    metrics=["perplexity", "bleu", "rouge", "bertscore"],
    generation_config={
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    }
)
```

#### DPO Evaluation Config
```python
dpo_eval_config = EvaluationConfig(
    metrics=["preference_accuracy", "bleu", "rouge"],
    preference_config={
        "compare_with_reference": True,
        "reference_model_path": "./checkpoints/sft_baseline"
    }
)
```

#### KTO Evaluation Config
```python
kto_eval_config = EvaluationConfig(
    metrics=["kto_accuracy", "desirable_accuracy", "undesirable_accuracy"],
    kto_config={
        "threshold": 0.5,
        "confidence_interval": 0.95
    }
)
```

## Benchmark Datasets

### Arabic Language Benchmarks

```python
from src.evaluation.benchmarks import (
    load_arabic_benchmark,
    ArabicGLUE,
    ArabicSuperGLUE,
    ARCD,
    XNLI_Arabic
)

# Load standard Arabic benchmarks
arabic_glue = load_arabic_benchmark("arabic_glue")
arcd = load_arabic_benchmark("arcd")
xnli_ar = load_arabic_benchmark("xnli_arabic")

# Evaluate on benchmarks
benchmark_results = {}
for name, dataset in [("ArabicGLUE", arabic_glue), ("ARCD", arcd)]:
    results = evaluator.evaluate(dataset)
    benchmark_results[name] = results
    print(f"{name} Results: {results}")
```

### Custom Benchmark Creation

```python
from src.evaluation.benchmarks import create_custom_benchmark

# Create domain-specific benchmark
custom_benchmark = create_custom_benchmark(
    name="arabic_medical",
    data_path="./data/medical_qa_arabic.jsonl",
    task_type="question_answering",
    metrics=["exact_match", "f1", "bleu"],
    language="arabic"
)

# Register benchmark
register_benchmark("arabic_medical", custom_benchmark)

# Use in evaluation
results = evaluator.evaluate(custom_benchmark)
```

## Advanced Features

### Multi-GPU Evaluation

```python
from src.evaluation import DistributedEvaluator

# Setup distributed evaluation
dist_evaluator = DistributedEvaluator(
    model=model,
    tokenizer=tokenizer,
    config=eval_config,
    world_size=4  # Number of GPUs
)

# Run distributed evaluation
results = dist_evaluator.evaluate(large_dataset)
```

### Streaming Evaluation

```python
from src.evaluation import StreamingEvaluator

# For very large datasets
streaming_evaluator = StreamingEvaluator(
    model=model,
    tokenizer=tokenizer,
    config=eval_config
)

# Evaluate in streaming fashion
results = streaming_evaluator.evaluate_stream(
    dataset_stream=large_dataset_stream,
    chunk_size=1000
)
```

### Real-time Evaluation

```python
from src.evaluation import RealTimeEvaluator

# Setup real-time evaluation
rt_evaluator = RealTimeEvaluator(
    model=model,
    tokenizer=tokenizer,
    config=eval_config
)

# Start evaluation server
rt_evaluator.start_server(port=8080)

# Evaluate single examples
result = rt_evaluator.evaluate_single(
    prompt="ما هو الذكاء الاصطناعي؟",
    reference="الذكاء الاصطناعي هو..."
)
```

## Integration with Training

### Training-time Evaluation

```python
from src.evaluation import TrainingEvaluator
from transformers import TrainerCallback

class EvaluationCallback(TrainerCallback):
    def __init__(self, evaluator, eval_dataset):
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        # Run comprehensive evaluation
        results = self.evaluator.evaluate(self.eval_dataset)
        
        # Log results
        for metric, value in results.items():
            state.log_history[-1][f"eval_{metric}"] = value

# Use in training
evaluator = create_evaluator("sft", model, tokenizer, eval_config)
callback = EvaluationCallback(evaluator, eval_dataset)

trainer = SFTTrainer(config)
trainer.add_callback(callback)
trainer.train()
```

### Model Selection

```python
from src.evaluation import ModelSelector

# Evaluate multiple checkpoints
checkpoints = [
    "./checkpoints/epoch_1",
    "./checkpoints/epoch_2",
    "./checkpoints/epoch_3"
]

selector = ModelSelector(
    checkpoints=checkpoints,
    eval_dataset=validation_dataset,
    metrics=["perplexity", "bleu"],
    selection_metric="bleu",
    higher_is_better=True
)

best_checkpoint = selector.select_best_model()
print(f"Best model: {best_checkpoint}")
```

## Reporting and Visualization

### Evaluation Reports

```python
from src.evaluation.reporting import generate_evaluation_report

# Generate comprehensive report
report = generate_evaluation_report(
    results=evaluation_results,
    model_name="Arabic Qwen SFT",
    dataset_name="Arabic QA Dataset",
    output_path="./reports/evaluation_report.html"
)

print(f"Report saved to: {report.output_path}")
```

### Metric Visualization

```python
from src.evaluation.visualization import (
    plot_metric_comparison,
    plot_training_curves,
    plot_preference_distribution
)

# Compare multiple models
plot_metric_comparison(
    results={
        "Baseline": baseline_results,
        "SFT": sft_results,
        "DPO": dpo_results
    },
    metrics=["perplexity", "bleu", "rouge"],
    save_path="./plots/model_comparison.png"
)

# Plot training evaluation curves
plot_training_curves(
    training_logs=trainer.state.log_history,
    metrics=["eval_loss", "eval_perplexity"],
    save_path="./plots/training_curves.png"
)
```

## Best Practices

### 1. Evaluation Strategy
- Use multiple complementary metrics
- Include both automatic and human evaluation
- Evaluate on diverse test sets
- Consider domain-specific metrics

### 2. Arabic-Specific Considerations
- Test with different Arabic dialects
- Evaluate diacritization quality
- Check cultural appropriateness
- Assess grammatical correctness

### 3. Performance Optimization
- Use batch evaluation for efficiency
- Cache model outputs when possible
- Use appropriate hardware for large evaluations
- Consider distributed evaluation for large datasets

### 4. Reproducibility
- Set random seeds for consistent results
- Document evaluation configurations
- Save detailed evaluation logs
- Version control evaluation scripts

### 5. Continuous Evaluation
- Integrate evaluation into CI/CD pipelines
- Monitor model performance over time
- Set up automated alerts for performance degradation
- Regular benchmark evaluations

## Troubleshooting

### Common Issues

#### Memory Issues
```python
# Reduce batch size
eval_config.batch_size = 4

# Use gradient checkpointing
eval_config.gradient_checkpointing = True

# Clear cache between evaluations
torch.cuda.empty_cache()
```

#### Slow Evaluation
```python
# Use mixed precision
eval_config.fp16 = True

# Reduce sequence length
eval_config.max_length = 256

# Use sampling instead of full evaluation
eval_config.sample_size = 1000
```

#### Inconsistent Results
```python
# Set random seeds
eval_config.seed = 42

# Use deterministic algorithms
eval_config.deterministic = True

# Multiple evaluation runs
results = []
for i in range(5):
    result = evaluator.evaluate(dataset, seed=42+i)
    results.append(result)

# Report mean and std
mean_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
std_results = {k: np.std([r[k] for r in results]) for k in results[0].keys()}
```

This comprehensive evaluation system provides robust tools for assessing Arabic Qwen model performance across various training methods and tasks.