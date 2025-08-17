# Arabic Qwen Base Model Fine-tuning Guide

This comprehensive guide covers the complete process of fine-tuning Qwen base models for Arabic language tasks. The guide is based on extensive research and testing with RTX 3060 12GB hardware.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Training Pipelines](#training-pipelines)
3. [Optimization Techniques](#optimization-techniques)
4. [Implementation Workflows](#implementation-workflows)
5. [Best Practices](#best-practices)

## 🎯 Overview

### Recommended Models for RTX 3060 12GB

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

## 🚀 Training Pipelines

### Pipeline 1: Supervised Fine-Tuning (SFT) for Arabic Instruction Following

**Goal**: Create a general-purpose Arabic chatbot  
**Best For**: RTX 3060 12GB, showcase demos  
**Models**: Qwen2.5-3B, Qwen3-1.7B  
**Datasets**: InstAr-500k, CIDAR, Arabic-OpenHermes-2.5  

#### Basic Implementation

```python
# Step 1: Load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch

model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: Prepare dataset
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/InstAr-500k")

# Step 3: Format for SFT
def format_prompt(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

dataset = dataset.map(format_prompt)

# Step 4: Training configuration
training_args = TrainingArguments(
    output_dir="./arabic-sft-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    logging_steps=50,
)

# Step 5: Train
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    # Add appropriate data collator
)

trainer.train()
```

### Pipeline 2: Preference Optimization (DPO/RLHF)

**Goal**: Improve response quality using human preferences  
**Best For**: RTX 3060 12GB, preference tuning experiments  
**Models**: Qwen3-1.7B, Qwen2.5-3B  
**Datasets**: Arabic-preference-data-RLHF, argilla-dpo-mix-7k-arabic  

#### DPO Implementation

```python
# Step 1: Load SFT model (from Pipeline 1)
model_name = "./arabic-sft-model"  # Your SFT model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Load preference dataset
dataset = load_dataset("FreedomIntelligence/Arabic-preference-data-RLHF")

# Step 3: Format for DPO
def format_dpo(example):
    return {
        "prompt": example["question"],
        "chosen": example["chosen_response"],
        "rejected": example["rejected_response"]
    }

dataset = dataset.map(format_dpo)

# Step 4: Configure DPO training
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./arabic-dpo-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    fp16=True,
    beta=0.1,  # DPO temperature
)

# Step 5: Train with DPO
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,  # Use same model for reference
    args=dpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

### Pipeline 3: Domain-Specific Fine-Tuning (Arabic QA)

**Goal**: Specialize in Arabic question answering  
**Best For**: RTX 3060 12GB, QA showcases  
**Models**: Qwen2.5-7B (4-bit), Qwen3-4B (4-bit)  
**Datasets**: ArabicQA_2.1M, Arabic MMLU  

#### QA Specialization Implementation

```python
# Step 1: Load quantized model
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Step 2: Load QA dataset
dataset = load_dataset("riotu-lab/ArabicQA_2.1M")

# Step 3: Format for QA
def format_qa(example):
    return {"text": f"السؤال: {example['question']}\nالإجابة: {example['answer']}"}

dataset = dataset.map(format_qa)

# Step 4: Configure LoRA
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Step 5: Train
training_args = TrainingArguments(
    output_dir="./arabic-qa-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=2,
    fp16=True,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

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