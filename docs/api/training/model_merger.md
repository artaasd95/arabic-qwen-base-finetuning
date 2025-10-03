# ModelMerger API Documentation

## Overview

The `ModelMerger` class provides advanced model merging capabilities for combining multiple fine-tuned Arabic language models into a single, more capable model. It supports various merging strategies and handles both full models and PEFT adapters.

## Class Definition

```python
from src.training.model_merger import ModelMerger, MergeConfig

merger = ModelMerger(config: MergeConfig)
```

## Key Features

- **Multiple Strategies**: Weighted, Sequential, Task Arithmetic, and SLERP merging
- **PEFT Support**: Merge LoRA adapters and other PEFT methods
- **Arabic Optimization**: Specialized handling for Arabic language models
- **Validation**: Built-in testing and validation of merged models
- **Memory Efficient**: Optimized for large model merging

## Configuration

### MergeConfig Parameters

```python
from src.training.model_merger import MergeConfig

config = MergeConfig(
    # Merging strategy
    strategy="weighted",  # "weighted", "sequential", "task_arithmetic", "slerp"
    
    # Model specifications
    models=[
        {
            "path": "./models/arabic_chat",
            "weight": 0.6,
            "name": "chat_specialist"
        },
        {
            "path": "./models/arabic_formal",
            "weight": 0.4,
            "name": "formal_specialist"
        }
    ],
    
    # Output configuration
    output_path="./models/merged_arabic",
    
    # Advanced options
    options={
        "normalize_weights": True,
        "preserve_tokenizer": True,
        "merge_embeddings": True,
        "merge_lm_head": True,
        "torch_dtype": "float16",
        "device_map": "auto"
    },
    
    # PEFT-specific options
    peft_config={
        "merge_adapters": True,
        "adapter_names": ["default"],
        "base_model_path": "./models/qwen2.5-3b"
    },
    
    # Validation options
    validation={
        "run_tests": True,
        "test_prompts": [
            "مرحبا، كيف حالك؟",
            "اشرح لي مفهوم الذكاء الاصطناعي"
        ],
        "quick_test": False,
        "sample_size": 100
    }
)
```

## Merging Strategies

### 1. Weighted Merging

Combines models using weighted averages of their parameters.

```python
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./model1", "weight": 0.7, "name": "primary"},
        {"path": "./model2", "weight": 0.3, "name": "secondary"}
    ],
    options={
        "normalize_weights": True,  # Ensure weights sum to 1.0
        "weight_decay": 0.0        # Apply weight decay during merging
    }
)
```

### 2. Sequential Merging

Applies model weights in sequence, useful for incremental improvements.

```python
config = MergeConfig(
    strategy="sequential",
    models=[
        {"path": "./base_model", "weight": 1.0, "name": "foundation"},
        {"path": "./domain_model", "weight": 0.3, "name": "specialization"}
    ]
)
```

### 3. Task Arithmetic

Uses mathematical operations on model weights to add or subtract capabilities.

```python
config = MergeConfig(
    strategy="task_arithmetic",
    base_model="./base_model",
    models=[
        {
            "path": "./arabic_chat_model", 
            "weight": 1.0, 
            "operation": "add",
            "name": "chat_enhancement"
        },
        {
            "path": "./formal_model", 
            "weight": -0.5, 
            "operation": "subtract",
            "name": "formality_reduction"
        }
    ],
    options={
        "scaling_factor": 1.0,
        "clamp_weights": True,
        "clamp_range": [-2.0, 2.0]
    }
)
```

### 4. SLERP (Spherical Linear Interpolation)

Performs spherical interpolation between model weights.

```python
config = MergeConfig(
    strategy="slerp",
    models=[
        {"path": "./model1", "weight": 0.7, "name": "primary"},
        {"path": "./model2", "weight": 0.3, "name": "secondary"}
    ],
    options={
        "interpolation_factor": 0.5,
        "normalize_before_slerp": True,
        "epsilon": 1e-8
    }
)
```

## Methods

### Core Methods

#### `__init__(config: MergeConfig)`
Initialize the model merger with configuration.

**Parameters:**
- `config`: MergeConfig object containing all merging parameters

#### `merge()`
Execute the model merging process.

**Returns:**
- Merged model object

**Example:**
```python
merger = ModelMerger(config)
merged_model = merger.merge()
print(f"Successfully merged {len(config.models)} models")
```

#### `validate_merge(merged_model)`
Validate the merged model with test prompts.

**Parameters:**
- `merged_model`: The merged model to validate

**Returns:**
- Validation results and metrics

### Utility Methods

#### `load_models()`
Load all source models specified in the configuration.

#### `check_compatibility()`
Verify that all models are compatible for merging.

#### `save_merged_model(merged_model, output_path)`
Save the merged model to the specified path.

**Parameters:**
- `merged_model`: The merged model to save
- `output_path`: Directory to save the model

## Usage Examples

### Basic Model Merging

```python
from src.training.model_merger import ModelMerger, MergeConfig

# Configure merging
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./models/arabic_chat", "weight": 0.6},
        {"path": "./models/arabic_qa", "weight": 0.4}
    ],
    output_path="./models/merged_arabic"
)

# Perform merge
merger = ModelMerger(config)
merged_model = merger.merge()

print("Merging completed successfully!")
```

### Dialect Model Merging

```python
# Merge multiple Arabic dialect models
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
    },
    validation={
        "run_tests": True,
        "test_prompts": [
            "إزيك؟",      # Egyptian
            "شلونك؟",      # Gulf
            "كيفك؟",       # Levantine
            "كيداير؟"      # Maghrebi
        ]
    }
)

merger = ModelMerger(config)
multi_dialect_model = merger.merge()
```

### Task Arithmetic for Capability Control

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
    output_path="./models/casual_arabic_chat",
    validation={
        "run_tests": True,
        "test_prompts": [
            "مرحبا صديقي، كيف الأحوال؟",
            "حدثني عن يومك"
        ]
    }
)

merger = ModelMerger(config)
casual_model = merger.merge()
```

### PEFT Adapter Merging

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
    },
    options={
        "preserve_base_model": True,
        "merge_method": "linear"
    }
)

merger = ModelMerger(config)
merged_adapter = merger.merge()
```

### Advanced Configuration

```python
# Advanced merging with custom options
config = MergeConfig(
    strategy="slerp",
    models=[
        {"path": "./models/arabic_instruct", "weight": 0.7},
        {"path": "./models/arabic_creative", "weight": 0.3}
    ],
    output_path="./models/balanced_arabic",
    
    # Advanced options
    options={
        "torch_dtype": "float16",
        "device_map": "auto",
        "low_memory_mode": True,
        "sequential_loading": True,
        "gradient_checkpointing": True,
        "use_safetensors": True,
        
        # SLERP-specific
        "interpolation_factor": 0.5,
        "normalize_before_slerp": True,
        "epsilon": 1e-8
    },
    
    # Comprehensive validation
    validation={
        "run_tests": True,
        "test_prompts": [
            "اكتب قصة قصيرة عن الصداقة",
            "اشرح خطوات حل مسألة رياضية",
            "ما هو رأيك في التكنولوجيا الحديثة؟"
        ],
        "quick_test": False,
        "sample_size": 200,
        "test_batch_size": 4,
        "metrics": ["perplexity", "coherence", "fluency"]
    }
)

merger = ModelMerger(config)
balanced_model = merger.merge()
```

## Performance Optimization

### Memory Optimization

```python
# Memory-efficient merging for large models
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./models/large_arabic_1", "weight": 0.5},
        {"path": "./models/large_arabic_2", "weight": 0.5}
    ],
    output_path="./models/merged_large_arabic",
    
    options={
        "low_memory_mode": True,
        "sequential_loading": True,
        "torch_dtype": "float16",
        "device_map": "cpu",  # Use CPU for large models
        "offload_folder": "./temp_offload"
    }
)
```

### Speed Optimization

```python
# Speed-optimized merging
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./models/arabic_1", "weight": 0.5},
        {"path": "./models/arabic_2", "weight": 0.5}
    ],
    output_path="./models/merged_arabic",
    
    options={
        "use_safetensors": True,
        "torch_compile": True,
        "parallel_loading": True,
        "cache_models": True
    },
    
    validation={
        "quick_test": True,
        "sample_size": 50
    }
)
```

## Error Handling

### Compatibility Checking

```python
class ModelCompatibilityError(Exception):
    pass

def check_model_compatibility(models):
    """Check if models are compatible for merging."""
    base_config = None
    
    for model_info in models:
        model_path = model_info["path"]
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise ModelCompatibilityError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        if base_config is None:
            base_config = config
        else:
            # Check critical parameters
            if config.get("vocab_size") != base_config.get("vocab_size"):
                raise ModelCompatibilityError("Vocabulary size mismatch")
            
            if config.get("hidden_size") != base_config.get("hidden_size"):
                raise ModelCompatibilityError("Hidden size mismatch")
```

### Memory Management

```python
def merge_with_memory_management(config):
    """Merge models with automatic memory management."""
    import gc
    import torch
    
    merger = ModelMerger(config)
    
    try:
        # Clear cache before merging
        torch.cuda.empty_cache()
        gc.collect()
        
        # Perform merge
        merged_model = merger.merge()
        
        return merged_model
        
    except torch.cuda.OutOfMemoryError:
        # Fallback to CPU merging
        print("GPU memory insufficient, falling back to CPU")
        config.options["device_map"] = "cpu"
        config.options["low_memory_mode"] = True
        
        merger = ModelMerger(config)
        return merger.merge()
        
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
```

## Integration Examples

### With Training Pipeline

```python
from src.training.sft_trainer import SFTTrainer
from src.training.dpo_trainer import DPOTrainer
from src.training.model_merger import ModelMerger, MergeConfig

# Train base models
sft_trainer = SFTTrainer(sft_config)
sft_model = sft_trainer.train(sft_dataset)

dpo_trainer = DPOTrainer(dpo_config)
dpo_model = dpo_trainer.train(preference_dataset)

# Merge trained models
merge_config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": sft_trainer.config.output_dir, "weight": 0.7},
        {"path": dpo_trainer.config.output_dir, "weight": 0.3}
    ],
    output_path="./models/final_arabic_model"
)

merger = ModelMerger(merge_config)
final_model = merger.merge()
```

### With Evaluation

```python
from src.evaluation.model_evaluator import ModelEvaluator

# Merge models
merger = ModelMerger(config)
merged_model = merger.merge()

# Evaluate merged model
evaluator = ModelEvaluator(merged_model, tokenizer)
results = evaluator.evaluate_arabic_capabilities([
    "perplexity",
    "dialect_classification",
    "conversation_quality",
    "instruction_following"
])

print(f"Merged model evaluation: {results}")
```

## Monitoring and Logging

```python
import logging
from src.training.model_merger import ModelMerger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Merge with detailed logging
config = MergeConfig(
    strategy="weighted",
    models=[
        {"path": "./model1", "weight": 0.6},
        {"path": "./model2", "weight": 0.4}
    ],
    output_path="./merged_model",
    options={
        "verbose": True,
        "log_level": "INFO"
    }
)

merger = ModelMerger(config)
merged_model = merger.merge()

logger.info("Model merging completed successfully")
```

This comprehensive API documentation provides all the information needed to effectively use the ModelMerger for combining Arabic language models.