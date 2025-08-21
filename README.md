# Arabic Qwen Base Fine-tuning 🇸🇦

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for fine-tuning Qwen models on Arabic datasets using various optimization methods including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), Kahneman-Tversky Optimization (KTO), Identity Preference Optimization (IPO), and Conservative Preference Optimization (CPO).

## 🌟 Features

- **Multiple Training Methods**: Support for SFT, DPO, KTO, IPO, and CPO
- **Arabic Language Support**: Specialized handling for Arabic text processing
- **Modular Architecture**: Clean, extensible codebase with clear separation of concerns
- **Comprehensive Evaluation**: Built-in evaluation metrics and benchmarks
- **Configuration Management**: YAML-based configuration with environment variable support
- **Memory Optimization**: Support for quantization, LoRA, and gradient checkpointing
- **Monitoring Integration**: Built-in support for Weights & Biases, TensorBoard, and MLflow
- **Production Ready**: Docker support, CI/CD pipelines, and deployment scripts

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Training Methods](#-training-methods)
- [Data Preparation](#-data-preparation)
- [Evaluation](#-evaluation)
- [Advanced Usage](#-advanced-usage)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU training)
- Git

### Install from Source

```bash
# Clone the repository
git clone https://github.com/artaasd95/arabic-qwen-base-finetuning.git
cd arabic-qwen-base-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev,testing]"
```

### Install from PyPI

```bash
pip install arabic-qwen-base-finetuning
```

### Docker Installation

```bash
# Build the Docker image
docker build -t arabic-qwen-finetuning .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace arabic-qwen-finetuning
```

## ⚡ Quick Start

### 1. Environment Setup

Create a `.env` file based on the example:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 2. Prepare Your Data

```python
from src.data.data_loader import SFTDataLoader
from src.config.sft_config import SFTConfig

# Load configuration
config = SFTConfig.load_from_env()

# Initialize data loader
data_loader = SFTDataLoader(config)
dataset = data_loader.load_dataset()
```

### 3. Train a Model

#### Supervised Fine-Tuning (SFT)

```bash
# Using command line
arabic-qwen-train --config config/sft_config.yaml

# Using Python
python -m src.scripts.train --method sft --config config/sft_config.yaml
```

#### Direct Preference Optimization (DPO)

```bash
arabic-qwen-train --config config/dpo_config.yaml
```

### 4. Evaluate the Model

```bash
arabic-qwen-evaluate --model-path ./checkpoints/sft_model --config config/eval_config.yaml
```

### 5. Run Inference

```python
from src.models.inference import QwenInference

# Initialize inference
inference = QwenInference(model_path="./checkpoints/sft_model")

# Generate response
response = inference.generate(
    prompt="ما هي عاصمة المملكة العربية السعودية؟",
    max_length=100
)
print(response)
```

## ⚙️ Configuration

The framework uses YAML configuration files with environment variable support:

```yaml
# config/sft_config.yaml
model:
  name: "Qwen/Qwen2-7B"
  max_length: 2048
  use_flash_attention: true

training:
  batch_size: 4
  learning_rate: 5e-5
  num_epochs: 3
  gradient_accumulation_steps: 4

data:
  dataset_name: "arabic_instructions"
  train_split: "train"
  validation_split: "validation"
  max_samples: 10000

optimization:
  use_lora: true
  lora_rank: 16
  use_quantization: true
  quantization_bits: 4
```

### Environment Variables

```bash
# Model settings
MODEL_NAME=Qwen/Qwen2-7B
MAX_LENGTH=2048

# Training settings
BATCH_SIZE=4
LEARNING_RATE=5e-5
NUM_EPOCHS=3

# Data settings
DATASET_NAME=arabic_instructions
MAX_SAMPLES=10000

# Optimization
USE_LORA=true
LORA_RANK=16
USE_QUANTIZATION=true

# Monitoring
WANDB_PROJECT=arabic-qwen-finetuning
WANDB_ENTITY=your-team
```

## 🎯 Training Methods

### Supervised Fine-Tuning (SFT)

Standard instruction-following fine-tuning:

```python
from src.training.sft_trainer import SFTTrainer
from src.config.sft_config import SFTConfig

config = SFTConfig.load("config/sft_config.yaml")
trainer = SFTTrainer(config)
trainer.train()
```

### Direct Preference Optimization (DPO)

Train models to prefer certain responses over others:

```python
from src.training.dpo_trainer import DPOTrainer
from src.config.dpo_config import DPOConfig

config = DPOConfig.load("config/dpo_config.yaml")
trainer = DPOTrainer(config)
trainer.train()
```

### Kahneman-Tversky Optimization (KTO)

Optimize based on human preference patterns:

```python
from src.training.kto_trainer import KTOTrainer
from src.config.kto_config import KTOConfig

config = KTOConfig.load("config/kto_config.yaml")
trainer = KTOTrainer(config)
trainer.train()
```

## 📊 Data Preparation

### SFT Data Format

```json
{
  "instruction": "اكتب قصة قصيرة عن الصداقة",
  "input": "",
  "output": "كان هناك صديقان يدعيان أحمد وعلي..."
}
```

### Preference Data Format

```json
{
  "prompt": "ما هو أفضل طعام في العالم؟",
  "chosen": "الطعام المفضل يختلف من شخص لآخر حسب الثقافة والذوق الشخصي.",
  "rejected": "البيتزا هي أفضل طعام في العالم بلا شك."
}
```

### Data Processing Pipeline

```python
from src.data.data_processor import ArabicDataProcessor

processor = ArabicDataProcessor()

# Clean and normalize Arabic text
cleaned_text = processor.clean_text(raw_text)

# Tokenize with Arabic support
tokens = processor.tokenize(cleaned_text)

# Apply data augmentation
augmented_data = processor.augment_data(dataset)
```

## 📈 Evaluation

### Built-in Metrics

- **Perplexity**: Language modeling performance
- **BLEU Score**: Translation quality
- **ROUGE Score**: Summarization quality
- **BERTScore**: Semantic similarity
- **Arabic-specific metrics**: Diacritization accuracy, morphological analysis

### Custom Evaluation

```python
from src.evaluation.evaluator import SFTEvaluator

evaluator = SFTEvaluator(model_path="./checkpoints/sft_model")
results = evaluator.evaluate(test_dataset)

print(f"Perplexity: {results['perplexity']:.2f}")
print(f"BLEU Score: {results['bleu']:.2f}")
print(f"ROUGE-L: {results['rouge_l']:.2f}")
```

## 🔧 Advanced Usage

### Memory Optimization

```python
# Enable gradient checkpointing
config.training.gradient_checkpointing = True

# Use LoRA for parameter-efficient fine-tuning
config.optimization.use_lora = True
config.optimization.lora_rank = 16

# Enable quantization
config.optimization.use_quantization = True
config.optimization.quantization_bits = 4
```

### Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 -m src.scripts.train --config config/sft_config.yaml

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 -m src.scripts.train
```

### Custom Training Loop

```python
from src.training.base_trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def compute_loss(self, model, inputs):
        # Custom loss computation
        outputs = model(**inputs)
        loss = self.custom_loss_function(outputs, inputs)
        return loss
    
    def training_step(self, model, inputs):
        # Custom training step
        model.train()
        loss = self.compute_loss(model, inputs)
        return loss
```

### Monitoring and Logging

```python
# Weights & Biases integration
import wandb

wandb.init(
    project="arabic-qwen-finetuning",
    config=config.to_dict()
)

# Custom metrics logging
trainer.log_metrics({
    "custom_metric": value,
    "epoch": epoch,
    "step": step
})
```

## 📚 API Reference

### Core Classes

- **`BaseConfig`**: Base configuration class
- **`SFTConfig`**: SFT-specific configuration
- **`DPOConfig`**: DPO-specific configuration
- **`BaseDataLoader`**: Base data loading functionality
- **`BaseTrainer`**: Base training functionality
- **`BaseEvaluator`**: Base evaluation functionality

### Utility Functions

- **`set_seed()`**: Set random seeds for reproducibility
- **`get_device()`**: Get optimal device for training
- **`format_time()`**: Format time duration
- **`clean_arabic_text()`**: Clean and normalize Arabic text

### Command Line Tools

- **`arabic-qwen-train`**: Training script
- **`arabic-qwen-evaluate`**: Evaluation script
- **`arabic-qwen-infer`**: Inference script
- **`arabic-qwen-data`**: Data processing utilities
- **`arabic-qwen-config`**: Configuration management

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --module config

# Run with coverage
python run_tests.py --coverage

# Run in parallel
python run_tests.py --parallel
```

## 🐳 Docker Support

### Development Environment

```dockerfile
# Dockerfile.dev
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["bash"]
```

### Production Deployment

```dockerfile
# Dockerfile.prod
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🚀 Deployment

### Model Serving

```python
# src/api/main.py
from fastapi import FastAPI
from src.models.inference import QwenInference

app = FastAPI()
inference = QwenInference(model_path="./checkpoints/best_model")

@app.post("/generate")
async def generate_text(prompt: str, max_length: int = 100):
    response = inference.generate(prompt, max_length)
    return {"response": response}
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arabic-qwen-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arabic-qwen-api
  template:
    metadata:
      labels:
        app: arabic-qwen-api
    spec:
      containers:
      - name: api
        image: arabic-qwen-finetuning:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/artaasd95/arabic-qwen-base-finetuning.git
cd arabic-qwen-base-finetuning

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python run_tests.py
```

### Code Style

We use Black, isort, and flake8 for code formatting:

```bash
# Format code
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base models
- [Hugging Face](https://huggingface.co/) for the transformers library
- [TRL](https://github.com/huggingface/trl) for preference optimization implementations
- The open source community for datasets and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/artaasd95/arabic-qwen-base-finetuning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/artaasd95/arabic-qwen-base-finetuning/discussions)

## 🗺️ Roadmap

- [ ] Support for more Arabic dialects
- [ ] Integration with more evaluation benchmarks
- [ ] Web-based training interface
- [ ] Model compression and optimization
- [ ] Multi-modal capabilities
- [ ] Real-time inference optimization

---

**Made with ❤️ for the open source community**