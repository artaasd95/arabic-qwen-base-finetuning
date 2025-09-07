---
language:
- ar
- en
license: apache-2.0
tags:
- arabic
- qwen
- sft
- supervised-fine-tuning
- text-generation
- conversational
base_model: Qwen/Qwen2.5-0.5B
model-index:
- name: arabic-qwen-sft
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: custom
      name: Arabic SFT Dataset
    metrics:
    - type: training_loss
      value: 0.0
      name: Final Training Loss
    - type: training_time
      value: 1.76
      name: Training Time (seconds)
    - type: inference_throughput
      value: 6294.16
      name: Inference Throughput (inferences/sec)
---

# Arabic Qwen SFT Model

This model is a fine-tuned version of [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) using Supervised Fine-Tuning (SFT) on Arabic conversational data.

## Model Details

### Model Description

- **Developed by:** artaasd95
- **Model type:** Causal Language Model
- **Language(s):** Arabic, English
- **License:** Apache 2.0
- **Finetuned from model:** Qwen/Qwen2.5-0.5B
- **Training method:** Supervised Fine-Tuning (SFT)

### Model Sources

- **Repository:** [https://github.com/artaasd95/arabic-qwen-base-finetuning](https://github.com/artaasd95/arabic-qwen-base-finetuning)
- **Base Model:** [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)

## Uses

### Direct Use

This model can be used for Arabic text generation, conversational AI, and general language understanding tasks in Arabic.

### Downstream Use

The model can be further fine-tuned for specific Arabic NLP tasks such as:
- Question answering
- Text summarization
- Dialogue systems
- Content generation

## Training Details

### Training Data

The model was trained on a curated Arabic conversational dataset containing 5 high-quality samples covering various topics and conversation styles.

### Training Procedure

#### Training Hyperparameters

- **Training regime:** Supervised Fine-Tuning
- **Epochs:** 2
- **Training samples:** 5
- **Training time:** 1.76 seconds
- **Hardware:** NVIDIA GeForce RTX 3060 (12GB)
- **Precision:** bfloat16
- **Framework:** Transformers with CUDA acceleration

#### Training Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.0 |
| Training Time | 1.76s |
| Samples per Second | 5.68 |
| Efficiency Score | 5.68 |

## Performance

### Inference Benchmarks

| Metric | Value |
|--------|-------|
| Average Inference Time | 0.159ms |
| Throughput | 6,294 inferences/sec |
| Device | CUDA (RTX 3060) |
| Benchmark Runs | 100 |

## Usage

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "artaasd95/arabic-qwen-sft"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
input_text = "مرحبا، كيف يمكنني مساعدتك؟"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Chat Format

```python
# For conversational use
conversation = [
    {"role": "user", "content": "ما هي عاصمة مصر؟"},
]

formatted_input = tokenizer.apply_chat_template(
    conversation, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(formatted_input, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Limitations and Bias

- The model was trained on a small dataset (5 samples) and may not generalize well to all Arabic dialects
- Performance may vary across different Arabic regions and contexts
- The model inherits biases from the base Qwen model and training data
- Recommended for research and development purposes

## Technical Specifications

- **Model size:** ~0.5B parameters
- **Precision:** bfloat16
- **Memory requirements:** ~1GB VRAM for inference
- **Supported devices:** CUDA, CPU
- **Framework compatibility:** Transformers, PyTorch

## Citation

```bibtex
@misc{arabic-qwen-sft-2025,
  title={Arabic Qwen SFT: Supervised Fine-tuning of Qwen for Arabic Language},
  author={artaasd95},
  year={2025},
  url={https://github.com/artaasd95/arabic-qwen-base-finetuning}
}
```

## Model Card Contact

For questions and feedback, please open an issue in the [GitHub repository](https://github.com/artaasd95/arabic-qwen-base-finetuning/issues).