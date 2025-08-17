# Arabic Qwen Base Model Fine-tuning Guide

This comprehensive guide covers the complete process of fine-tuning Qwen base models for Arabic language tasks. The guide is based on extensive research and testing with RTX 3060 12GB hardware.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Training Pipelines](#training-pipelines)
3. [Optimization Techniques](#optimization-techniques)
4. [Implementation Workflows](#implementation-workflows)
5. [Best Practices](#best-practices)

## 🎯 Overview

### Recommended Base Models for RTX 3060 12GB (No Reasoning Models)

| Model | Size | VRAM (FP16) | VRAM (4-bit) | Arabic Strength | Best Use Case |
|-------|------|-------------|--------------|-----------------|---------------|
| **Qwen2.5-3B** | 3B | ~6GB | ~3-4GB | Excellent MSA support | General instruction-following |
| **Qwen3-1.7B** | 1.7B | ~3.4GB | ~2GB | Enhanced reasoning | Creative writing & logic |
| **Qwen2.5-7B** | 7B | ~14GB | ~4-5GB | Top-tier performance | High-quality generation (quantized) |

### Essential Arabic Datasets

| Category | Dataset | Size | Use Case | Reliability |
|----------|---------|------|----------|-------------|
| **Instruction** | InstAr-500k | 500k | General instruction tuning | High (peer-reviewed) |
| | CIDAR | 10k | Culturally relevant instructions | High (human-reviewed) |
| | Arabic-OpenHermes-2.5 | 982k | Chat/conversation | High (community) |
| **Preference** | Arabic-preference-data-RLHF | 11.5k | RLHF/reward modeling | High (academic) |
| | argilla-dpo-mix-7k-arabic | 7.5k | Direct Preference Optimization | Medium (community) |
| **Domain-Specific** | ArabicQA_2.1M | 2.14M | QA systems | High (large-scale) |
| | Arabic MMLU | 14k-29k | Knowledge evaluation | High (OALL benchmark) |

## 🚀 Training Pipelines for Base Models

### Pipeline 1: Supervised Fine-Tuning (SFT) + Preference Optimization
**Best for: General Arabic conversation and instruction following**

1. **Stage 1 - SFT**: Train base model on instruction-response pairs
2. **Stage 2 - Preference Optimization**: Choose from multiple methods:
   - **DPO (Direct Preference Optimization)**: Direct alignment without reward model
   - **KTO (Kahneman-Tversky Optimization)**: Binary preference optimization
   - **DNO (Distributional Preference Optimization)**: Distribution-aware preference learning
   - **IPO (Identity Preference Optimization)**: Regularized preference optimization
   - **CPO (Conservative Preference Optimization)**: Safe preference alignment

**Goal**: Train base model on Arabic instruction-response pairs  
**Best For**: RTX 3060 12GB, foundation for preference optimization  
**Models**: Qwen2.5-3B, Qwen3-1.7B  
**Datasets**: InstAr-500k, CIDAR, Arabic-OpenHermes-2.5  

#### Stage 1: Supervised Fine-Tuning Implementation

```python
# Step 1: Load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: Configure LoRA for base model fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Step 3: Prepare dataset
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/InstAr-500k")

# Step 4: Format for SFT
def format_prompt(example):
    return {"text": f"### التعليمات:\n{example['instruction']}\n\n### الإجابة:\n{example['output']}"}

dataset = dataset.map(format_prompt)

# Step 5: Training configuration for base models
training_args = TrainingArguments(
    output_dir="./arabic-sft-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=1000,
    logging_steps=50,
    warmup_steps=500,
)

# Step 6: Train
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./arabic-sft-final")
```

### Stage 2: Preference Optimization Methods

**Goal**: Align base model with human preferences using various optimization techniques  
**Best For**: RTX 3060 12GB, improving response quality and safety  
**Models**: SFT checkpoint from Stage 1  
**Datasets**: Arabic-preference-data-RLHF, argilla-dpo-mix-7k-arabic  

#### Method 1: Direct Preference Optimization (DPO)
**Best for**: Direct alignment without reward model, memory efficient

```python
# DPO Training Script
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Load SFT model
model = AutoModelForCausalLM.from_pretrained("./arabic-sft-final")
tokenizer = AutoTokenizer.from_pretrained("./arabic-sft-final")

# Load preference dataset
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/Arabic-preference-data-RLHF")

# Format for DPO
def format_dpo(example):
    return {
        "prompt": example["question"],
        "chosen": example["chosen_response"],
        "rejected": example["rejected_response"]
    }

dataset = dataset.map(format_dpo)

# DPO Training Arguments
dpo_args = TrainingArguments(
    output_dir="./arabic-dpo-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-7,
    num_train_epochs=1,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    warmup_steps=100,
)

# Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create reference model automatically
    args=dpo_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    beta=0.1,  # KL divergence coefficient
)

# Train with DPO
dpo_trainer.train()
dpo_trainer.save_model("./arabic-dpo-final")
```

#### Method 2: Kahneman-Tversky Optimization (KTO)
**Best for**: Binary preference optimization, robust to preference noise

```python
# KTO Training Script
from trl import KTOTrainer

# KTO Configuration
kto_args = TrainingArguments(
    output_dir="./arabic-kto-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-6,
    num_train_epochs=1,
    fp16=True,
    save_steps=500,
    logging_steps=10,
)

# Initialize KTO Trainer
kto_trainer = KTOTrainer(
    model=model,
    ref_model=None,
    args=kto_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

kto_trainer.train()
kto_trainer.save_model("./arabic-kto-final")
```

#### Method 3: Identity Preference Optimization (IPO)
**Best for**: Regularized preference learning, stable training

```python
# IPO Training Script
from trl import DPOTrainer

# IPO is a variant of DPO with different loss function
ipo_args = TrainingArguments(
    output_dir="./arabic-ipo-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-6,
    num_train_epochs=1,
    fp16=True,
    save_steps=500,
    logging_steps=10,
)

# IPO Trainer (using DPO with IPO loss)
ipo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=ipo_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    beta=0.1,
    loss_type="ipo",  # Use IPO loss instead of DPO
)

ipo_trainer.train()
ipo_trainer.save_model("./arabic-ipo-final")
```

#### Method 4: Conservative Preference Optimization (CPO)
**Best for**: Safe preference alignment, preventing harmful outputs

```python
# CPO Training Script
from trl import CPOTrainer

# CPO Configuration
cpo_args = TrainingArguments(
    output_dir="./arabic-cpo-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-6,
    num_train_epochs=1,
    fp16=True,
    save_steps=500,
    logging_steps=10,
)

# Initialize CPO Trainer
cpo_trainer = CPOTrainer(
    model=model,
    ref_model=None,
    args=cpo_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    beta=0.1,
    simpo_gamma=0.5,  # CPO regularization parameter
)

cpo_trainer.train()
cpo_trainer.save_model("./arabic-cpo-final")
```

### Pipeline 2: Domain-Specific Base Model Fine-tuning
**Best for: Specialized Arabic domains (legal, medical, technical)**

1. **Stage 1 - SFT**: Train base model on domain-specific instruction data
2. **Stage 2 - Domain Preference Optimization**: Apply preference methods on domain data
3. **Stage 3 - Continued Training**: Additional fine-tuning on domain-specific datasets

#### Domain-Specific Implementation

```python
# Domain-specific fine-tuning for base models
from datasets import load_dataset, concatenate_datasets

# Load domain-specific Arabic datasets
domain_datasets = [
    load_dataset("FreedomIntelligence/InstAr-500k"),
    load_dataset("OALL/CIDAR"),
    # Add other domain-specific datasets as needed
]

# Format for domain training
def format_domain(example):
    return {"text": f"### التعليمات:\n{example['instruction']}\n\n### الإجابة:\n{example['output']}"}

# Combine and format datasets
combined_domain = concatenate_datasets([ds.map(format_domain) for ds in domain_datasets])

# Domain-specific training configuration
domain_training_args = TrainingArguments(
    output_dir="./arabic-domain-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    save_steps=1000,
    logging_steps=50,
    warmup_steps=500,
)

# Train domain model
domain_trainer = Trainer(
    model=model,
    args=domain_training_args,
    train_dataset=combined_domain["train"],
    tokenizer=tokenizer,
)

domain_trainer.train()
domain_trainer.save_model("./arabic-domain-final")
```

### Pipeline 3: Multi-Stage Preference Optimization
**Best for: Maximum alignment and safety**

1. **Stage 1 - SFT**: Base instruction following on general data
2. **Stage 2 - Primary Preference**: Apply DPO or KTO for general alignment
3. **Stage 3 - Secondary Preference**: Apply additional preference method (IPO/CPO) for refinement

#### Multi-Stage Implementation Example

```python
# Multi-stage preference optimization pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, KTOTrainer

# Stage 1: Load SFT model
base_model = AutoModelForCausalLM.from_pretrained("./arabic-sft-final")
tokenizer = AutoTokenizer.from_pretrained("./arabic-sft-final")

# Stage 2: Primary preference optimization (DPO)
print("Starting Stage 2: DPO Training...")
dpo_trainer = DPOTrainer(
    model=base_model,
    ref_model=None,
    args=dpo_args,
    train_dataset=preference_dataset["train"],
    tokenizer=tokenizer,
    beta=0.1,
)
dpo_trainer.train()
dpo_model = dpo_trainer.save_model("./arabic-dpo-stage2")

# Stage 3: Secondary preference optimization (IPO for refinement)
print("Starting Stage 3: IPO Refinement...")
refined_model = AutoModelForCausalLM.from_pretrained("./arabic-dpo-stage2")
ipo_trainer = DPOTrainer(
    model=refined_model,
    ref_model=None,
    args=ipo_args,
    train_dataset=preference_dataset["train"],
    tokenizer=tokenizer,
    beta=0.05,  # Lower beta for refinement
    loss_type="ipo",
)
ipo_trainer.train()
ipo_trainer.save_model("./arabic-multistage-final")

print("Multi-stage preference optimization complete!")
```

### Choosing the Right Preference Method

| Method | Best For | Memory Usage | Training Speed | Stability |
|--------|----------|--------------|----------------|-----------|
| **DPO** | General alignment | Low | Fast | High |
| **KTO** | Binary preferences | Low | Fast | Very High |
| **IPO** | Stable training | Medium | Medium | Very High |
| **CPO** | Safety-critical | Medium | Medium | High |
| **Multi-stage** | Best quality | High | Slow | Medium |

## ⚡ Optimization Techniques

### 1. Quantization

```python
# 4-bit Quantization for larger models
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 2. LoRA/QLoRA

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Shows <1% trainable parameters
```

### 3. Memory Optimization

```python
# Mixed Precision Training
training_args = TrainingArguments(
    fp16=True,  # Mixed precision
    # or bf16=True if GPU supports bfloat16
)

# Gradient Accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Actual batch size
    gradient_accumulation_steps=8,  # Virtual batch size = 16
)

# Gradient Checkpointing
model.gradient_checkpointing_enable()
```

## 🔄 Implementation Workflows

### Workflow 1: End-to-End Arabic Chatbot

1. **Start**: Qwen2.5-3B base model
2. **SFT**: Fine-tune on InstAr-500k (3 epochs)
3. **DPO**: Preference optimization with Arabic-preference-data-RLHF
4. **Optimization**: Apply LoRA + FP16
5. **Result**: 3B parameter Arabic chatbot with strong instruction-following

### Workflow 2: Resource-Efficient Arabic Model

1. **Start**: Qwen3-1.7B
2. **SFT**: Fine-tune on CIDAR (culturally relevant data)
3. **DPO**: Lightweight preference tuning
4. **Optimization**: Full 4-bit quantization
5. **Result**: Ultra-efficient Arabic model (<2GB VRAM)

### Workflow 3: Arabic QA Specialist

1. **Start**: Qwen2.5-7B (4-bit quantized)
2. **SFT**: Fine-tune on ArabicQA_2.1M
3. **Evaluation**: Test on Arabic MMLU
4. **Optimization**: QLoRA + gradient checkpointing
5. **Result**: Specialized Arabic QA model

## 📝 Best Practices

### Model Selection
- **Start Small**: Begin with Qwen2.5-3B or Qwen3-1.7B for full fine-tuning
- **Quantize for Scale**: Use 4-bit quantization to run 7B+ models
- **Consider Use Case**: Choose models based on your specific Arabic language needs

### Training Strategy
- **Pipeline Matters**: SFT → DPO produces best results
- **LoRA is Essential**: For preference optimization and large models
- **Dataset Quality**: Use curated datasets (CIDAR, InstAr-500k) over raw corpora

### Hardware Optimization
- **Mixed Precision**: Reduces VRAM usage by ~50%
- **Gradient Accumulation**: Achieve larger effective batch sizes
- **Memory Management**: Use gradient checkpointing for larger models

### Evaluation
- **Benchmark Religiously**: Use Arabic MMLU and AlGhafa for evaluation
- **Cultural Relevance**: Test with CIDAR for culturally appropriate responses
- **Domain Testing**: Validate on task-specific datasets

## 🎯 Recommendations by Goal

### For General Arabic Chatbots
- **Model**: Qwen2.5-3B
- **Datasets**: InstAr-500k + CIDAR
- **Pipeline**: SFT → DPO
- **Optimization**: LoRA + FP16

### For Arabic QA Systems
- **Model**: Qwen2.5-7B (4-bit)
- **Datasets**: ArabicQA_2.1M + Arabic MMLU
- **Pipeline**: Domain SFT
- **Optimization**: LoRA + gradient accumulation

### For Resource-Constrained Environments
- **Model**: Qwen3-1.7B
- **Datasets**: CIDAR + Arabic-OpenHermes-2.5 (subset)
- **Pipeline**: SFT only
- **Optimization**: Full 4-bit quantization

---

**Final Recommendation**: Implement Pipeline 1 (SFT + DPO) with Qwen2.5-3B for the best balance of performance and hardware compatibility on RTX 3060. This provides a strong Arabic model for showcases while leaving room for experimentation.

## 📚 Next Steps

1. Review [Model Selection Guide](./model-selection.md) to choose your base model
2. Check [Hardware Requirements](./hardware-requirements.md) for system compatibility
3. Follow [Implementation Examples](./implementation-examples.md) for practical code
4. Consult [Troubleshooting Guide](./troubleshooting.md) for common issues