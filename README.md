# Arabic Qwen Base Model Fine-tuning

🚀 **Complete guide for fine-tuning Arabic Qwen base models using SFT + Preference Optimization**

This repository provides comprehensive resources for fine-tuning Qwen base models for Arabic language tasks using modern preference optimization techniques (DPO, KTO, IPO, CPO).

## 🎯 Overview

This project focuses on **base model fine-tuning** rather than reasoning models, providing a two-stage training pipeline:

1. **Supervised Fine-tuning (SFT)** - Adapt base models to Arabic tasks
2. **Preference Optimization** - Align models with human preferences using multiple methods

### Supported Models

- **Qwen2.5-1.5B** - Efficient for RTX 3060 8GB
- **Qwen2.5-3B** - Recommended for RTX 3060 12GB
- **Qwen2.5-7B** - For RTX 4070+ GPUs

### Preference Optimization Methods

| Method | Best For | Training Speed | Memory Usage |
|--------|----------|----------------|---------------|
| **DPO** | General preference alignment | Moderate | Standard |
| **KTO** | Binary preference data | Fast | Low |
| **IPO** | Identity-preserving optimization | Moderate | Standard |
| **CPO** | Contrastive learning | Moderate | Standard |

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/arabic-qwen-base-finetuning.git
cd arabic-qwen-base-finetuning

# Install dependencies
pip install torch transformers datasets accelerate
pip install trl peft bitsandbytes
```

### 2. Basic Training Pipeline

```python
from arabic_qwen_trainer import ArabicQwenPipeline

# Initialize pipeline
pipeline = ArabicQwenPipeline(
    model_name="Qwen/Qwen2.5-3B",
    output_dir="./experiments/qwen3b_arabic"
)

# Stage 1: SFT
pipeline.run_sft_stage(
    dataset_name="FreedomIntelligence/InstAr-500k",
    max_steps=1000
)

# Stage 2: Preference Optimization
pipeline.run_preference_stage(
    dataset_name="FreedomIntelligence/Arabic-preference-data-RLHF",
    method="dpo",  # or "kto", "ipo", "cpo"
    max_steps=500
)
```

### 3. Model Inference

```python
from arabic_qwen_inference import ArabicQwenInference

# Load trained model
model = ArabicQwenInference("./experiments/qwen3b_arabic/dpo_model")

# Generate response
response = model.chat("اكتب قصة قصيرة عن الصداقة")
print(response)
```

## 📚 Documentation

### Core Guides

- **[Model Selection](./docs/model-selection.md)** - Choose the right base model for your hardware
- **[Fine-tuning Guide](./docs/fine-tuning-guide.md)** - Complete training methodology
- **[Dataset Preparation](./docs/dataset-preparation.md)** - Prepare Arabic datasets for training
- **[Implementation Examples](./docs/implementation-examples.md)** - Production-ready code examples

### Technical Resources

- **[Hardware Requirements](./docs/hardware-requirements.md)** - GPU and system requirements
- **[Troubleshooting Guide](./docs/troubleshooting.md)** - Common issues and solutions

## 🎯 Training Recommendations

### For RTX 3060 12GB (Recommended)

```python
# Optimal configuration
config = {
    "model": "Qwen2.5-3B",
    "sft_dataset": "InstAr-500k",
    "preference_method": "dpo",
    "batch_size": 4,
    "gradient_accumulation": 4,
    "max_length": 512
}
```

### Training Pipeline Options

1. **Quick Setup** (2-3 hours)
   - SFT: 1k steps on InstAr-500k
   - DPO: 500 steps on preference data

2. **Production Quality** (6-8 hours)
   - SFT: 3k steps on mixed datasets
   - Preference: 1k steps with chosen method

3. **Research Grade** (12+ hours)
   - Full SFT training
   - Multiple preference methods comparison

## 📊 Performance Expectations

| Model | Hardware | Training Time | Arabic Quality | Use Case |
|-------|----------|---------------|----------------|----------|
| Qwen2.5-1.5B | RTX 3060 8GB | 2-4 hours | Good | Prototyping |
| Qwen2.5-3B | RTX 3060 12GB | 4-6 hours | Excellent | Production |
| Qwen2.5-7B | RTX 4070+ | 6-10 hours | Outstanding | Research |

## 🔧 Advanced Features

### Multi-Method Training

```python
# Compare different preference methods
methods = ["dpo", "kto", "ipo", "cpo"]

for method in methods:
    pipeline.run_preference_stage(
        dataset_name="preference_dataset",
        method=method,
        output_suffix=f"_{method}"
    )
```

### Custom Dataset Integration

```python
# Use your own Arabic dataset
custom_dataset = load_dataset("your_dataset")
formatted_dataset = format_arabic_instructions(custom_dataset)

pipeline.run_sft_stage(
    dataset=formatted_dataset,
    max_steps=2000
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- New Arabic datasets
- Additional preference optimization methods
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent base models
- **FreedomIntelligence** for Arabic datasets
- **HuggingFace** for the transformers ecosystem
- **TRL Team** for preference optimization implementations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/arabic-qwen-base-finetuning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/arabic-qwen-base-finetuning/discussions)
- **Documentation**: [Full Documentation](./docs/README.md)

---

**Ready to fine-tune your Arabic Qwen model?** Start with our [Quick Start Guide](./docs/fine-tuning-guide.md)! 🚀