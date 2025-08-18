# SFT Data Loader Documentation

The `SFTDataLoader` class handles data loading for Supervised Fine-Tuning (SFT) in the Arabic Qwen Base Fine-tuning framework. It supports various instruction-response formats and conversation templates optimized for Arabic language models.

## Class Overview

```python
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset
from src.data.base_loader import BaseDataLoader
from src.config.sft_config import SFTDataConfig

class SFTDataLoader(BaseDataLoader):
    """Data loader for Supervised Fine-Tuning (SFT)."""
```

## Location

**File**: `src/data/sft_loader.py`

## SFT Overview

Supervised Fine-Tuning (SFT) trains language models on instruction-response pairs to improve their ability to follow instructions and generate appropriate responses. The SFT data loader handles various formats commonly used in instruction tuning.

### Supported Formats

1. **Instruction-Response Pairs**: Simple question-answer format
2. **Conversation Format**: Multi-turn conversations with roles
3. **Chat Templates**: Structured chat conversations
4. **System Messages**: Instructions with system prompts

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: SFTDataConfig,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    **kwargs
):
    """
    Initialize SFT data loader.
    
    Args:
        config: SFT data configuration
        tokenizer: Tokenizer for text processing
        enable_cache: Enable data caching
        **kwargs: Additional arguments
    """
```

### Core Methods

#### `load_dataset()`

```python
def load_dataset(self, split: str) -> Dataset:
    """
    Load SFT dataset for specified split.
    
    Args:
        split: Data split ('train', 'validation', 'test')
        
    Returns:
        SFT dataset
        
    Raises:
        DataLoadingError: If loading fails
        ValidationError: If validation fails
    """
```

#### `process_sample()`

```python
def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single SFT sample.
    
    Args:
        sample: Raw data sample
        
    Returns:
        Processed sample with standardized format
    """
```

#### `validate_sample()`

```python
def validate_sample(self, sample: Dict[str, Any]) -> bool:
    """
    Validate a single SFT sample.
    
    Args:
        sample: Data sample to validate
        
    Returns:
        True if valid, False otherwise
    """
```

## Data Formats

### 1. Instruction-Response Format

Simple instruction-response pairs:

```jsonl
{"instruction": "ما هي عاصمة فرنسا؟", "response": "عاصمة فرنسا هي باريس."}
{"instruction": "اشرح مفهوم الذكاء الاصطناعي", "response": "الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً مثل التعلم والاستدلال وحل المشكلات."}
{"instruction": "اكتب قصيدة قصيرة عن الطبيعة", "response": "في الحديقة الخضراء\nتغرد الطيور بفرح\nوالأزهار تتفتح\nتحت أشعة الشمس الذهبية"}
```

### 2. Input-Output Format

Alternative column names:

```jsonl
{"input": "ترجم هذه الجملة إلى الإنجليزية: مرحباً بك", "output": "Welcome"}
{"input": "ما هو الجذر التربيعي لـ 16؟", "output": "الجذر التربيعي لـ 16 هو 4."}
```

### 3. Conversation Format

Multi-turn conversations:

```jsonl
{
  "messages": [
    {"role": "user", "content": "ما هي فوائد التمرين؟"},
    {"role": "assistant", "content": "التمرين له فوائد عديدة للصحة الجسدية والنفسية، منها تقوية العضلات وتحسين الدورة الدموية وتقليل التوتر."}
  ]
}
{
  "messages": [
    {"role": "user", "content": "كيف أتعلم البرمجة؟"},
    {"role": "assistant", "content": "يمكنك البدء بتعلم لغة برمجة بسيطة مثل Python، ثم ممارسة حل المشاكل البرمجية."},
    {"role": "user", "content": "ما هي أفضل المصادر؟"},
    {"role": "assistant", "content": "أنصح بالمواقع التعليمية مثل Codecademy و freeCodeCamp، بالإضافة إلى الكتب والدورات المجانية."}
  ]
}
```

### 4. System Message Format

With system instructions:

```jsonl
{
  "system": "أنت مساعد ذكي ومفيد. أجب على الأسئلة بوضوح ودقة.",
  "instruction": "ما هي أهمية التعليم؟",
  "response": "التعليم أساس التقدم والتطور في المجتمعات. يساعد على تنمية المهارات والمعرفة، ويفتح آفاقاً جديدة للأفراد والمجتمعات."
}
```

### 5. Chat Template Format

Structured chat format:

```jsonl
{
  "conversations": [
    {"from": "human", "value": "اشرح لي قانون نيوتن الأول"},
    {"from": "gpt", "value": "قانون نيوتن الأول ينص على أن الجسم يبقى في حالة سكون أو حركة منتظمة في خط مستقيم ما لم تؤثر عليه قوة خارجية."}
  ]
}
```

## Configuration

### SFTDataConfig

```python
@dataclass
class SFTDataConfig:
    """Configuration for SFT data loading."""
    
    # File paths
    train_file: str = "data/sft_train.jsonl"
    validation_file: Optional[str] = "data/sft_validation.jsonl"
    test_file: Optional[str] = None
    
    # Column names
    instruction_column: str = "instruction"
    response_column: str = "response"
    input_column: str = "input"
    output_column: str = "output"
    messages_column: str = "messages"
    conversations_column: str = "conversations"
    system_column: Optional[str] = "system"
    
    # Processing parameters
    conversation_template: str = "default"
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"
    max_length: int = 2048
    
    # Data processing
    preprocessing_num_workers: int = 4
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    
    # Quality control
    min_instruction_length: int = 5
    min_response_length: int = 10
    max_instruction_length: int = 1000
    max_response_length: int = 2000
    filter_duplicates: bool = True
    filter_empty: bool = True
```

## Usage Examples

### Basic SFT Data Loading

```python
from src.data.sft_loader import SFTDataLoader
from src.config.sft_config import SFTConfig

# Load configuration
config = SFTConfig.from_yaml("config/sft_config.yaml")

# Create data loader
loader = SFTDataLoader(config.data)

# Load datasets
train_dataset = loader.load_dataset("train")
val_dataset = loader.load_dataset("validation")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Inspect a sample
sample = train_dataset[0]
print(f"Sample: {sample}")
```

### Custom Column Names

```python
# Configure custom column names
config.data.instruction_column = "question"
config.data.response_column = "answer"
config.data.system_column = "system_prompt"

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")
```

### Conversation Format Loading

```python
# Load conversation format data
config.data.messages_column = "messages"
config.data.conversation_template = "chat"

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Access conversation data
for sample in train_dataset:
    print(f"Messages: {sample['messages']}")
    break
```

### Data Filtering

```python
# Configure data filtering
config.data.min_instruction_length = 10
config.data.min_response_length = 20
config.data.max_instruction_length = 500
config.data.max_response_length = 1000
config.data.filter_duplicates = True

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")
```

## Data Processing Pipeline

### 1. Format Detection

The loader automatically detects the data format:

```python
def _detect_format(self, sample: Dict[str, Any]) -> str:
    """
    Detect the format of a data sample.
    
    Returns:
        Format type: 'instruction_response', 'input_output', 
                    'messages', 'conversations'
    """
    if self.config.messages_column in sample:
        return "messages"
    elif self.config.conversations_column in sample:
        return "conversations"
    elif self.config.instruction_column in sample:
        return "instruction_response"
    elif self.config.input_column in sample:
        return "input_output"
    else:
        raise ValueError(f"Unknown data format: {sample.keys()}")
```

### 2. Format Conversion

All formats are converted to a standard format:

```python
def _convert_to_standard_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert sample to standard format.
    
    Returns:
        Standardized sample with 'instruction', 'response', 'system'
    """
    format_type = self._detect_format(sample)
    
    if format_type == "instruction_response":
        return {
            "instruction": sample[self.config.instruction_column],
            "response": sample[self.config.response_column],
            "system": sample.get(self.config.system_column)
        }
    elif format_type == "messages":
        return self._convert_messages_format(sample)
    # ... other format conversions
```

### 3. Text Processing

Applies Arabic-specific text processing:

```python
def _process_text(self, text: str) -> str:
    """
    Process Arabic text.
    
    Args:
        text: Input text
        
    Returns:
        Processed text
    """
    # Apply Arabic text processing
    from src.data.processors.arabic_processor import ArabicTextProcessor
    
    processor = ArabicTextProcessor()
    return processor.process(text)
```

## Conversation Templates

### Default Template

```python
DEFAULT_TEMPLATE = """
{system}

### التعليمات:
{instruction}

### الاستجابة:
{response}
"""
```

### Chat Template

```python
CHAT_TEMPLATE = """
{system}

<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
"""
```

### Custom Template

```python
# Define custom template
custom_template = """
النظام: {system}
المستخدم: {instruction}
المساعد: {response}
"""

# Register template
loader.register_template("custom", custom_template)
config.data.conversation_template = "custom"
```

## Data Validation

### Sample Validation

```python
def validate_sample(self, sample: Dict[str, Any]) -> bool:
    """
    Validate SFT sample.
    
    Checks:
    - Required fields present
    - Text length constraints
    - Content quality
    - Arabic text detection
    """
    try:
        # Check required fields
        format_type = self._detect_format(sample)
        
        # Convert to standard format
        std_sample = self._convert_to_standard_format(sample)
        
        # Validate text lengths
        instruction = std_sample["instruction"]
        response = std_sample["response"]
        
        if len(instruction) < self.config.min_instruction_length:
            return False
        if len(response) < self.config.min_response_length:
            return False
        if len(instruction) > self.config.max_instruction_length:
            return False
        if len(response) > self.config.max_response_length:
            return False
        
        # Check for empty content
        if not instruction.strip() or not response.strip():
            return False
        
        return True
        
    except Exception:
        return False
```

### Quality Metrics

```python
def get_quality_metrics(self, dataset: Dataset) -> Dict[str, Any]:
    """
    Calculate quality metrics for dataset.
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "total_samples": len(dataset),
        "avg_instruction_length": 0,
        "avg_response_length": 0,
        "arabic_ratio": 0,
        "duplicate_ratio": 0
    }
    
    instruction_lengths = []
    response_lengths = []
    arabic_count = 0
    
    for sample in dataset:
        instruction_lengths.append(len(sample["instruction"]))
        response_lengths.append(len(sample["response"]))
        
        # Check if Arabic
        if self._is_arabic_text(sample["instruction"]):
            arabic_count += 1
    
    metrics["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
    metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)
    metrics["arabic_ratio"] = arabic_count / len(dataset)
    
    return metrics
```

## Performance Optimization

### Streaming for Large Datasets

```python
# Enable streaming for large datasets
config.data.streaming = True
config.data.buffer_size = 1000

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train", streaming=True)

# Process in batches
for batch in train_dataset.iter(batch_size=32):
    # Process batch
    pass
```

### Parallel Processing

```python
# Configure parallel processing
config.data.preprocessing_num_workers = 8

loader = SFTDataLoader(config.data)
train_dataset = loader.load_dataset("train")
```

### Memory Management

```python
# Limit samples in memory
config.data.max_samples_in_memory = 5000
config.data.lazy_loading = True

loader = SFTDataLoader(config.data)
```

## Error Handling

### Common Errors

```python
try:
    loader = SFTDataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except DataLoadingError as e:
    if "column not found" in str(e):
        print("Check column names in configuration")
        print(f"Available columns: {e.available_columns}")
    elif "format detection" in str(e):
        print("Unable to detect data format")
        print("Ensure data has required columns")
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Invalid samples: {len(e.invalid_samples)}")
    
    # Show sample validation errors
    for idx in e.invalid_samples[:5]:
        sample = loader.get_raw_sample(idx)
        print(f"Sample {idx}: {sample}")
```

## Best Practices

### 1. Data Preparation
- Use consistent column names across datasets
- Validate data format before training
- Include system messages for better instruction following
- Balance instruction types and complexity

### 2. Quality Control
- Set appropriate length constraints
- Filter duplicates and empty samples
- Monitor Arabic text ratio
- Validate conversation structure

### 3. Performance
- Enable caching for repeated experiments
- Use parallel processing for large datasets
- Consider streaming for memory constraints
- Monitor loading times and memory usage

### 4. Arabic Language
- Ensure proper encoding (UTF-8)
- Handle diacritics consistently
- Consider Arabic text direction
- Validate Arabic content quality

## Troubleshooting

### Column Not Found
```python
# Check available columns
raw_sample = loader._load_raw_data(config.data.train_file)[0]
print(f"Available columns: {list(raw_sample.keys())}")

# Update configuration
config.data.instruction_column = "question"  # Correct column name
```

### Format Detection Issues
```python
# Manually specify format
config.data.force_format = "instruction_response"
loader = SFTDataLoader(config.data)
```

### Memory Issues
```python
# Reduce memory usage
config.data.streaming = True
config.data.max_samples_in_memory = 1000
config.data.preprocessing_num_workers = 2
```

### Slow Loading
```python
# Optimize loading
config.data.enable_cache = True
config.data.preprocessing_num_workers = 8
config.data.lazy_loading = True
```

## See Also

- [Base Data Loader](base_loader.md)
- [Preference Data Loader](preference_loader.md)
- [Arabic Text Processor](processors/arabic_processor.md)
- [SFT Configuration](../config/sft_config.md)
- [SFT Training](../training/sft_trainer.md)