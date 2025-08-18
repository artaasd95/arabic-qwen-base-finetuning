# API Documentation

This section provides comprehensive documentation for all modules, classes, and functions in the Arabic Qwen Base Fine-tuning framework.

## Overview

The framework is organized into several key modules:

- **[Configuration](config/index.md)**: Configuration management and validation
- **[Data Loading](data_loader/index.md)**: Data loading and preprocessing for different training methods
- **[Training](training/index.md)**: Training implementations for SFT and preference optimization methods
- **[Evaluation](evaluation/index.md)**: Evaluation metrics and custom evaluators
- **[Utilities](utils/index.md)**: Common utilities and helper functions

## Quick Navigation

### Core Components

| Component | Description | Key Classes |
|-----------|-------------|-------------|
| [BaseTrainer](training/base_trainer.md) | Abstract base class for all trainers | `BaseTrainer` |
| [SFTTrainer](training/sft_trainer.md) | Supervised fine-tuning implementation | `SFTTrainer` |
| [DPOTrainer](training/dpo_trainer.md) | Direct Preference Optimization | `DPOTrainer` |
| [BaseLoader](data_loader/base_loader.md) | Abstract base class for data loaders | `BaseDataLoader` |
| [BaseEvaluator](evaluation/base_evaluator.md) | Abstract base class for evaluators | `BaseEvaluator` |

### Configuration

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [BaseConfig](config/base_config.md) | Base configuration class | `BaseConfig` |
| [SFTConfig](config/sft_config.md) | SFT-specific configuration | `SFTConfig` |
| [DPOConfig](config/dpo_config.md) | DPO-specific configuration | `DPOConfig` |
| [EnvConfig](config/env_config.md) | Environment variable configuration | `EnvConfig` |

### Data Loading

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [SFTLoader](data_loader/sft_loader.md) | SFT data loading and preprocessing | `SFTDataLoader` |
| [DPOLoader](data_loader/dpo_loader.md) | DPO data loading and preprocessing | `DPODataLoader` |
| [PreferenceLoader](data_loader/preference_loader.md) | Preference data loading | `PreferenceDataLoader` |

### Training Methods

| Method | Description | Documentation |
|--------|-------------|---------------|
| **SFT** | Supervised Fine-Tuning | [SFTTrainer](training/sft_trainer.md) |
| **DPO** | Direct Preference Optimization | [DPOTrainer](training/dpo_trainer.md) |
| **KTO** | Kahneman-Tversky Optimization | [KTOTrainer](training/kto_trainer.md) |
| **IPO** | Identity Preference Optimization | [IPOTrainer](training/ipo_trainer.md) |
| **CPO** | Conservative Preference Optimization | [CPOTrainer](training/cpo_trainer.md) |

### Evaluation

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [SFTEvaluator](evaluation/sft_evaluator.md) | SFT model evaluation | `SFTEvaluator` |
| [PreferenceEvaluator](evaluation/preference_evaluator.md) | Preference model evaluation | `PreferenceEvaluator` |

### Utilities

| Module | Description | Key Functions |
|--------|-------------|---------------|
| [Common](utils/common.md) | Common utility functions | `setup_logging`, `get_device` |
| [DataUtils](utils/data_utils.md) | Data processing utilities | `preprocess_arabic_text`, `validate_dataset` |
| [ModelUtils](utils/model_utils.md) | Model utilities | `load_model`, `save_checkpoint` |

## Usage Patterns

### Basic Training Workflow

```python
from src.config.sft_config import SFTConfig
from src.data_loader.sft_loader import SFTDataLoader
from src.training.sft_trainer import SFTTrainer

# Load configuration
config = SFTConfig.from_yaml("config/sft_config.yaml")

# Initialize data loader
data_loader = SFTDataLoader(config)
train_dataset = data_loader.load_dataset("data/train.jsonl")

# Initialize trainer
trainer = SFTTrainer(config)

# Train model
trainer.train(train_dataset)
```

### Evaluation Workflow

```python
from src.config.base_config import BaseConfig
from src.evaluation.sft_evaluator import SFTEvaluator

# Load configuration
config = BaseConfig.from_yaml("config/eval_config.yaml")

# Initialize evaluator
evaluator = SFTEvaluator(config)

# Evaluate model
results = evaluator.evaluate(model, test_dataset)
```

## Architecture Overview

```
src/
├── config/           # Configuration management
│   ├── base_config.py
│   ├── sft_config.py
│   ├── dpo_config.py
│   └── ...
├── data_loader/      # Data loading and preprocessing
│   ├── base_loader.py
│   ├── sft_loader.py
│   ├── dpo_loader.py
│   └── ...
├── training/         # Training implementations
│   ├── base_trainer.py
│   ├── sft_trainer.py
│   ├── dpo_trainer.py
│   └── ...
├── evaluation/       # Evaluation metrics and evaluators
│   ├── base_evaluator.py
│   ├── sft_evaluator.py
│   └── ...
└── utils/           # Utility functions
    ├── common.py
    ├── data_utils.py
    └── model_utils.py
```

## Design Principles

1. **Modularity**: Each component is self-contained and can be used independently
2. **Extensibility**: Easy to add new training methods, data loaders, and evaluators
3. **Configuration-driven**: All behavior controlled through YAML configuration files
4. **Type Safety**: Comprehensive type hints throughout the codebase
5. **Testing**: Extensive test coverage for all components

## Getting Started

1. **Installation**: Follow the [installation guide](../README.md#installation)
2. **Configuration**: Set up your [configuration files](config/index.md)
3. **Data Preparation**: Prepare your data using the [data loading guide](data_loader/index.md)
4. **Training**: Start training with the [training guide](training/index.md)
5. **Evaluation**: Evaluate your models using the [evaluation guide](evaluation/index.md)

## Contributing

When adding new components:

1. Follow the existing patterns and inheritance hierarchy
2. Add comprehensive docstrings and type hints
3. Include unit tests for all new functionality
4. Update this documentation

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.