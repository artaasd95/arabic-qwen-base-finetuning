# Model Merging Guide

This guide covers the advanced model merging capabilities in the Arabic Qwen Base Fine-tuning project, allowing you to combine multiple fine-tuned models to create more powerful and versatile Arabic language models.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Merging Strategies](#merging-strategies)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## 🎯 Overview

Model merging is a technique that combines the weights of multiple fine-tuned models to create a single model that inherits the capabilities of all source models. This is particularly useful for Arabic language models where you might have:

- Different models specialized for different Arabic dialects
- Models fine-tuned on different tasks (conversation, summarization, translation)
- Models with different training approaches (SFT, DPO, SimPO)

### Benefits of Model Merging

- **Multi-capability Models**: Combine task-specific models into one versatile model
- **Dialect Coverage**: Merge dialect-specific models for broader Arabic coverage
- **Performance Enhancement**: Leverage strengths of different training approaches
- **Resource Efficiency**: Deploy one merged model instead of multiple specialized models

## 🔄 Merging Strategies

### 1. Weighted Merging
Combines models using weighted averages of their parameters.

**Best for**: Balancing capabilities from multiple models
**Use case**: Merging models with similar architectures but different specializations

```python
config = {
    "strategy": "weighted",
    "models": [
        {"path": "./model1", "weight": 0.6},
        {"path": "./model2", "weight": 0.4}
    ]
}
```

### 2. Sequential Merging
Applies model weights in sequence, useful for incremental improvements.

**Best for**: Building upon a base model with specialized enhancements
**Use case**: Adding domain-specific knowledge to a general model

```python
config = {
    "strategy": "sequential",
    "models": [
        {"path": "./base_model", "weight": 1.0},
        {"path": "./domain_model", "weight": 0.3}
    ]
}
```

### 3. Task Arithmetic
Uses mathematical operations on model weights to add or subtract capabilities.

**Best for**: Fine-grained control over model capabilities
**Use case**: Adding specific skills while removing unwanted behaviors

```python
config = {
    "strategy": "task_arithmetic",
    "base_model": "./base_model",
    "models": [
        {"path": "./arabic_chat_model", "weight": 1.0, "operation": "add"},
        {"path": "./formal_model", "weight": -0.5, "operation": "subtract"}
    ]
}
```

### 4. SLERP (Spherical Linear Interpolation)
Performs spherical interpolation between model weights.

**Best for**: Smooth transitions between model capabilities
**Use case**: Merging models with similar training but different fine-tuning stages

```python
config = {
    "strategy": "slerp",
    "models": [
        {"path": "./model1", "weight": 0.7},
        {"path": "./model2", "weight": 0.3}
    ],
    "slerp_factor": 0.5
}
```

## ⚙️ Configuration

### Basic Configuration Structure

```yaml
# merge_config.yaml
merge_config:
  strategy: "weighted"  # weighted, sequential, task_arithmetic, slerp
  output_path: "./merged_model"
  
  # Model specifications
  models:
    - path: "./arabic_chat_model"
      weight: 0.6
      name: "chat_specialist"
    - path: "./arabic_formal_model"
      weight: 0.4
      name: "formal_specialist"
  
  # Advanced options
  options:
    normalize_weights: true
    preserve_tokenizer: true
    merge_embeddings: true
    merge_lm_head: true
    
  # PEFT-specific options
  peft_config:
    merge_adapters: true
    adapter_names: ["default"]
    
  # Validation options
  validation:
    run_tests: true
    test_prompts:
      - "مرحبا، كيف حالك؟"
      - "اشرح لي مفهوم الذكاء الاصطناعي"
```

### Strategy-Specific Options

#### Weighted Merging Options
```yaml
weighted_options:
  normalize_weights: true  # Ensure weights sum to 1.0
  weight_decay: 0.0       # Apply weight decay during merging
```

#### Task Arithmetic Options
```yaml
task_arithmetic_options:
  base_model: "./base_model"
  scaling_factor: 1.0
  clamp_weights: true
  clamp_range: [-2.0, 2.0]
```

#### SLERP Options
```yaml
slerp_options:
  interpolation_factor: 0.5
  normalize_before_slerp: true
  epsilon: 1e-8
```

## 💻 Usage Examples

### Example 1: Merging Arabic Dialect Models

```python
from src.training.model_merger import ModelMerger, MergeConfig

# Configuration for merging dialect-specific models
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./models/egyptian_arabic", "weight": 0.3, "name": "egyptian"},
        {"path": "./models/gulf_arabic", "weight": 0.3, "name": "gulf"},
        {"path": "./models/levantine_arabic", "weight": 0.2, "name": "levantine"},
        {"path": "./models/maghrebi_arabic", "weight": 0.2, "name": "maghrebi"}
    ],
    output_path="./models/multi_dialect_arabic",
    options={
        "normalize_weights": True,
        "preserve_tokenizer": True
    }
)

# Perform the merge
merger = ModelMerger(config)
merged_model = merger.merge()

print(f"Successfully merged {len(config.models)} dialect models")
print(f"Merged model saved to: {config.output_path}")
```

### Example 2: Task Arithmetic for Capability Control

```python
# Remove formal language bias while enhancing conversational ability
config = MergeConfig(
    strategy="task_arithmetic",
    base_model="./models/arabic_base",
    models=[
        {
            "path": "./models/arabic_conversation", 
            "weight": 1.2, 
            "operation": "add",
            "name": "conversation_boost"
        },
        {
            "path": "./models/arabic_formal", 
            "weight": -0.3, 
            "operation": "subtract",
            "name": "formality_reduction"
        }
    ],
    output_path="./models/casual_arabic_chat"
)

merger = ModelMerger(config)
merged_model = merger.merge()
```

### Example 3: Sequential Enhancement

```python
# Build upon a base model with specialized capabilities
config = MergeConfig(
    strategy="sequential",
    models=[
        {"path": "./models/qwen_arabic_base", "weight": 1.0, "name": "foundation"},
        {"path": "./models/arabic_instruction_tuned", "weight": 0.4, "name": "instruction"},
        {"path": "./models/arabic_preference_aligned", "weight": 0.2, "name": "alignment"}
    ],
    output_path="./models/enhanced_arabic_assistant"
)

merger = ModelMerger(config)
merged_model = merger.merge()
```

### Example 4: PEFT Model Merging

```python
# Merge LoRA adapters from different training runs
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./adapters/arabic_chat_lora", "weight": 0.6},
        {"path": "./adapters/arabic_qa_lora", "weight": 0.4}
    ],
    output_path="./adapters/merged_arabic_lora",
    peft_config={
        "merge_adapters": True,
        "adapter_names": ["default"],
        "base_model_path": "./models/qwen2.5-3b"
    }
)

merger = ModelMerger(config)
merged_adapter = merger.merge()
```

## 📋 Best Practices

### 1. Model Compatibility
- **Architecture Matching**: Ensure all models share the same base architecture
- **Tokenizer Consistency**: Use models with compatible tokenizers
- **Parameter Alignment**: Verify parameter shapes match across models

### 2. Weight Selection
- **Start Conservative**: Begin with equal weights and adjust based on performance
- **Validate Incrementally**: Test intermediate merges before final combination
- **Consider Model Quality**: Give higher weights to better-performing models

### 3. Strategy Selection Guide

| Scenario | Recommended Strategy | Reasoning |
|----------|---------------------|-----------|
| Similar quality models | Weighted | Balanced combination of capabilities |
| Base + specialized models | Sequential | Preserve base while adding specialization |
| Removing unwanted behavior | Task Arithmetic | Precise control over capabilities |
| Smooth capability transition | SLERP | Natural interpolation between models |

### 4. Testing and Validation
- **Comprehensive Testing**: Test merged models on diverse Arabic tasks
- **Dialect Coverage**: Ensure merged models handle multiple Arabic dialects
- **Performance Benchmarks**: Compare against individual source models
- **Safety Checks**: Verify merged models don't exhibit harmful behaviors

### 5. Resource Management
- **Memory Planning**: Model merging requires loading multiple models simultaneously
- **Storage Optimization**: Clean up intermediate files after successful merges
- **Backup Strategy**: Keep copies of source models before merging

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors
**Problem**: Out of memory when loading multiple models
**Solution**: 
- Use gradient checkpointing
- Load models sequentially instead of simultaneously
- Reduce batch size during validation

```python
config.options.update({
    "low_memory_mode": True,
    "sequential_loading": True,
    "gradient_checkpointing": True
})
```

#### 2. Parameter Shape Mismatches
**Problem**: Models have incompatible parameter shapes
**Solution**:
- Verify models use the same base architecture
- Check tokenizer vocabulary sizes
- Ensure consistent model configurations

#### 3. Poor Merged Model Performance
**Problem**: Merged model performs worse than individual models
**Solution**:
- Adjust weight ratios
- Try different merging strategies
- Validate individual model quality first

#### 4. PEFT Merging Issues
**Problem**: LoRA adapters fail to merge properly
**Solution**:
- Ensure adapters are compatible with base model
- Check adapter configurations match
- Verify adapter names are correct

### Performance Optimization

#### 1. Efficient Merging
```python
# Use optimized merging for large models
config.options.update({
    "use_safetensors": True,
    "torch_dtype": "float16",
    "device_map": "auto"
})
```

#### 2. Validation Optimization
```python
# Quick validation setup
config.validation.update({
    "quick_test": True,
    "sample_size": 100,
    "test_batch_size": 4
})
```

## 📊 Evaluation Metrics

After merging models, evaluate performance using:

1. **Perplexity**: Measure language modeling capability
2. **BLEU/ROUGE**: Evaluate generation quality
3. **Dialect Classification**: Test multi-dialect understanding
4. **Task-Specific Metrics**: Assess performance on target tasks

```python
# Example evaluation script
from src.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(merged_model, tokenizer)
results = evaluator.evaluate_arabic_capabilities([
    "perplexity",
    "dialect_classification", 
    "conversation_quality",
    "instruction_following"
])

print(f"Merged model evaluation results: {results}")
```

This comprehensive guide should help you effectively merge Arabic language models to create more capable and versatile systems for your specific use cases.