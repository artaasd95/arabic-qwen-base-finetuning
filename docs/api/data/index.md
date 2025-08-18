# Data Loading System Documentation

The data loading system in the Arabic Qwen Base Fine-tuning framework provides comprehensive support for different training methodologies, data formats, and preprocessing pipelines. This system is designed to handle Arabic text efficiently while supporting various fine-tuning approaches.

## Overview

The data loading system consists of several key components:

- **Base Data Loader**: Foundation class for all data loading operations
- **Method-Specific Loaders**: Specialized loaders for SFT, DPO, KTO, IPO, and CPO
- **Data Processors**: Text preprocessing and tokenization utilities
- **Format Handlers**: Support for various data formats (JSONL, CSV, Parquet)
- **Arabic Text Processing**: Specialized handling for Arabic language features

## Architecture

```
src/data/
├── __init__.py
├── base_loader.py          # Base data loading functionality
├── sft_loader.py           # Supervised Fine-Tuning data loader
├── preference_loader.py    # Base preference optimization loader
├── dpo_loader.py          # Direct Preference Optimization loader
├── kto_loader.py          # Kahneman-Tversky Optimization loader
├── ipo_loader.py          # Identity Preference Optimization loader
├── cpo_loader.py          # Contrastive Preference Optimization loader
├── processors/
│   ├── __init__.py
│   ├── text_processor.py   # Text preprocessing utilities
│   ├── arabic_processor.py # Arabic-specific processing
│   └── tokenizer_utils.py  # Tokenization utilities
└── utils/
    ├── __init__.py
    ├── format_utils.py     # Data format utilities
    ├── validation.py       # Data validation
    └── augmentation.py     # Data augmentation
```

## Quick Navigation

### Core Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| [Base Loader](base_loader.md) | Foundation data loading class | Core functionality, caching, validation |
| [SFT Loader](sft_loader.md) | Supervised fine-tuning data | Instruction-response pairs, conversation format |
| [Preference Loader](preference_loader.md) | Base preference optimization | Pairwise preferences, ranking data |
| [DPO Loader](dpo_loader.md) | Direct Preference Optimization | Chosen/rejected pairs, length handling |
| [KTO Loader](kto_loader.md) | Kahneman-Tversky Optimization | Binary feedback, desirable/undesirable |
| [IPO Loader](ipo_loader.md) | Identity Preference Optimization | Length bias mitigation, identity mapping |
| [CPO Loader](cpo_loader.md) | Contrastive Preference Optimization | Multi-negative sampling, contrastive learning |

### Processing Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| [Text Processor](processors/text_processor.md) | General text preprocessing | Cleaning, normalization, formatting |
| [Arabic Processor](processors/arabic_processor.md) | Arabic-specific processing | Diacritics, script handling, RTL support |
| [Tokenizer Utils](processors/tokenizer_utils.md) | Tokenization utilities | Model-specific tokenizers, special tokens |

### Utilities

| Component | Description | Documentation |
|-----------|-------------|---------------|
| [Format Utils](utils/format_utils.md) | Data format handling | JSONL, CSV, Parquet support |
| [Validation](utils/validation.md) | Data validation | Schema validation, quality checks |
| [Augmentation](utils/augmentation.md) | Data augmentation | Paraphrasing, back-translation |

## Supported Data Formats

### Input Formats

- **JSONL**: Primary format for all training methods
- **CSV**: Tabular data with proper column mapping
- **Parquet**: Efficient columnar format for large datasets
- **HuggingFace Datasets**: Direct integration with HF datasets

### Training Method Formats

#### Supervised Fine-Tuning (SFT)
```jsonl
{"instruction": "ما هي عاصمة فرنسا؟", "response": "عاصمة فرنسا هي باريس."}
{"messages": [{"role": "user", "content": "اشرح الذكاء الاصطناعي"}, {"role": "assistant", "content": "الذكاء الاصطناعي هو..."}]}
```

#### Preference Optimization (DPO/IPO/CPO)
```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، المدينة الجميلة.", "rejected": "باريس."}
```

#### Binary Feedback (KTO)
```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "response": "عاصمة فرنسا هي باريس.", "label": "desirable"}
```

## Quick Start Examples

### Basic SFT Data Loading

```python
from src.data.sft_loader import SFTDataLoader
from src.config.sft_config import SFTConfig

# Load configuration
config = SFTConfig.from_yaml("config/sft_config.yaml")

# Create data loader
loader = SFTDataLoader(config.data)

# Load training data
train_dataset = loader.load_dataset("train")
val_dataset = loader.load_dataset("validation")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
```

### Basic DPO Data Loading

```python
from src.data.dpo_loader import DPODataLoader
from src.config.dpo_config import DPOConfig

# Load configuration
config = DPOConfig.from_yaml("config/dpo_config.yaml")

# Create data loader
loader = DPODataLoader(config.data)

# Load preference data
train_dataset = loader.load_dataset("train")

# Access preference pairs
for sample in train_dataset:
    print(f"Prompt: {sample['prompt']}")
    print(f"Chosen: {sample['chosen']}")
    print(f"Rejected: {sample['rejected']}")
    break
```

### Custom Data Processing

```python
from src.data.processors.arabic_processor import ArabicTextProcessor
from src.data.processors.text_processor import TextProcessor

# Create processors
arabic_processor = ArabicTextProcessor()
text_processor = TextProcessor()

# Process Arabic text
text = "مرحباً بك في عالم الذكاء الاصطناعي!"
processed = arabic_processor.process(text)
print(f"Processed: {processed}")

# Apply general text processing
cleaned = text_processor.clean_text(processed)
print(f"Cleaned: {cleaned}")
```

## Data Validation

The system includes comprehensive data validation:

```python
from src.data.utils.validation import DataValidator

# Create validator
validator = DataValidator()

# Validate SFT data
sft_errors = validator.validate_sft_data("data/sft_train.jsonl")
if sft_errors:
    print(f"Found {len(sft_errors)} validation errors")
    for error in sft_errors[:5]:  # Show first 5 errors
        print(f"Line {error['line']}: {error['message']}")

# Validate preference data
dpo_errors = validator.validate_preference_data("data/dpo_train.jsonl")
print(f"DPO validation: {len(dpo_errors)} errors found")
```

## Performance Optimization

### Caching

```python
# Enable caching for faster subsequent loads
loader = SFTDataLoader(config.data, enable_cache=True)
train_dataset = loader.load_dataset("train")  # First load: processes data
train_dataset = loader.load_dataset("train")  # Second load: uses cache
```

### Parallel Processing

```python
# Configure parallel processing
config.data.preprocessing_num_workers = 8
config.data.dataloader_num_workers = 4

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")
```

### Memory Optimization

```python
# Use streaming for large datasets
config.data.streaming = True
config.data.buffer_size = 1000

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train", streaming=True)
```

## Arabic Language Support

The system provides specialized support for Arabic text:

### Text Normalization
- Diacritic handling (removal/preservation)
- Character normalization (different Arabic scripts)
- RTL (Right-to-Left) text support
- Arabic numeral conversion

### Tokenization
- Arabic-aware tokenizers
- Subword tokenization for Arabic
- Special token handling
- Vocabulary optimization

### Quality Checks
- Arabic text detection
- Script mixing validation
- Encoding verification
- Language identification

## Data Augmentation

Built-in data augmentation capabilities:

```python
from src.data.utils.augmentation import DataAugmenter

# Create augmenter
augmenter = DataAugmenter()

# Augment SFT data
original_data = [{"instruction": "ما هي عاصمة فرنسا؟", "response": "باريس"}]
augmented_data = augmenter.augment_sft_data(original_data, methods=["paraphrase", "back_translate"])

print(f"Original: {len(original_data)} samples")
print(f"Augmented: {len(augmented_data)} samples")
```

## Error Handling

Robust error handling throughout the system:

```python
try:
    loader = SFTDataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except DataLoadingError as e:
    print(f"Data loading failed: {e}")
    print(f"Suggestions: {e.suggestions}")
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Invalid samples: {e.invalid_samples}")
```

## Best Practices

### 1. Data Preparation
- Validate data before training
- Use consistent formatting
- Handle encoding properly
- Remove duplicates

### 2. Performance
- Enable caching for repeated loads
- Use parallel processing
- Consider streaming for large datasets
- Monitor memory usage

### 3. Quality Control
- Implement data validation
- Use quality metrics
- Monitor data distribution
- Handle edge cases

### 4. Arabic Text
- Normalize text consistently
- Handle diacritics appropriately
- Consider script variations
- Validate language detection

## Troubleshooting

### Common Issues

#### Encoding Problems
```python
# Solution: Specify encoding explicitly
config.data.encoding = "utf-8"
config.data.handle_encoding_errors = "replace"
```

#### Memory Issues
```python
# Solution: Use streaming and reduce batch size
config.data.streaming = True
config.data.max_samples_in_memory = 1000
```

#### Slow Loading
```python
# Solution: Enable caching and parallel processing
config.data.enable_cache = True
config.data.preprocessing_num_workers = 8
```

## See Also

- [Configuration Documentation](../config/index.md)
- [Training Documentation](../training/index.md)
- [Evaluation Documentation](../evaluation/index.md)
- [Utilities Documentation](../utils/index.md)