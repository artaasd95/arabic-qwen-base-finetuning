# DPO Data Loader Documentation

The `DPODataLoader` class handles data loading for Direct Preference Optimization (DPO) training in the Arabic Qwen Base Fine-tuning framework. It extends the `PreferenceDataLoader` to provide DPO-specific data processing and validation.

## Class Overview

```python
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset
from src.data.preference_loader import PreferenceDataLoader
from src.config.dpo_config import DPODataConfig

class DPODataLoader(PreferenceDataLoader):
    """Data loader for Direct Preference Optimization (DPO)."""
```

## Location

**File**: `src/data/dpo_loader.py`

## DPO Overview

Direct Preference Optimization (DPO) is a method for training language models to align with human preferences without requiring a separate reward model. It directly optimizes the policy using preference data.

### Key Features

- **Direct optimization**: No reward model required
- **Stable training**: More stable than RLHF approaches
- **Pairwise preferences**: Uses chosen/rejected response pairs
- **Reference model**: Requires a reference model for KL divergence

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: DPODataConfig,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    **kwargs
):
    """
    Initialize DPO data loader.
    
    Args:
        config: DPO data configuration
        tokenizer: Tokenizer for text processing
        enable_cache: Enable data caching
        **kwargs: Additional arguments
    """
    super().__init__(config, tokenizer, enable_cache, **kwargs)
    self.config = config
    self.dpo_specific_config = getattr(config, 'dpo_specific', None)
```

### Core Methods

#### `load_dataset()`

```python
def load_dataset(self, split: str = "train") -> Dataset:
    """
    Load DPO dataset for specified split.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        
    Returns:
        Loaded and processed DPO dataset
    """
    # Load raw data
    raw_data = self._load_raw_data(split)
    
    # Process for DPO
    processed_data = []
    for i, sample in enumerate(raw_data):
        try:
            # Validate sample
            if not self.validate_sample(sample):
                continue
            
            # Process sample
            processed_sample = self.process_sample(sample)
            processed_data.append(processed_sample)
            
        except Exception as e:
            self.logger.warning(f"Error processing sample {i}: {e}")
            continue
    
    # Apply DPO-specific filtering
    filtered_data = self._apply_dpo_filtering(processed_data)
    
    # Create dataset
    dataset = self._create_dataset(filtered_data)
    
    self.logger.info(f"Loaded {len(dataset)} DPO samples for {split} split")
    return dataset
```

#### `_process_preference_sample()`

```python
def _process_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process DPO-specific preference sample.
    
    Args:
        sample: Raw preference sample
        
    Returns:
        Processed DPO sample
    """
    processed = sample.copy()
    
    # Ensure standard format
    if "prompt" not in processed:
        processed["prompt"] = self._extract_prompt(sample)
    
    if "chosen" not in processed:
        processed["chosen"] = self._extract_chosen_response(sample)
    
    if "rejected" not in processed:
        processed["rejected"] = self._extract_rejected_response(sample)
    
    # Add DPO-specific fields
    processed["method"] = "dpo"
    processed["preference_type"] = "pairwise"
    
    # Process system message if present
    if "system" in sample or self.config.system_column in sample:
        processed["system"] = sample.get("system", sample.get(self.config.system_column, ""))
    
    # Add conversation context if available
    if "conversation" in sample:
        processed["conversation"] = self._process_conversation_context(sample["conversation"])
    
    # Calculate preference strength
    processed["preference_strength"] = self._calculate_dpo_preference_strength(
        processed["chosen"], processed["rejected"]
    )
    
    # Add quality metrics
    processed["chosen_quality"] = self._calculate_response_quality(processed["chosen"])
    processed["rejected_quality"] = self._calculate_response_quality(processed["rejected"])
    
    return processed
```

#### `_validate_preference_format()`

```python
def _validate_preference_format(self, sample: Dict[str, Any]) -> bool:
    """
    Validate DPO-specific preference format.
    
    Args:
        sample: Sample to validate
        
    Returns:
        True if valid DPO format
    """
    try:
        # Check required fields
        required_fields = ["prompt", "chosen", "rejected"]
        if not all(field in sample for field in required_fields):
            return False
        
        # Validate field types
        if not all(isinstance(sample[field], str) for field in required_fields):
            return False
        
        # Validate content
        prompt = sample["prompt"].strip()
        chosen = sample["chosen"].strip()
        rejected = sample["rejected"].strip()
        
        # Check minimum lengths
        if len(prompt) < self.config.min_prompt_length:
            return False
        if len(chosen) < self.config.min_response_length:
            return False
        if len(rejected) < self.config.min_response_length:
            return False
        
        # Check maximum lengths
        if len(prompt) > self.config.max_prompt_length:
            return False
        if len(chosen) > self.config.max_response_length:
            return False
        if len(rejected) > self.config.max_response_length:
            return False
        
        # Validate preference quality
        if not self._validate_dpo_preference_quality(chosen, rejected):
            return False
        
        return True
        
    except Exception:
        return False
```

### DPO-Specific Processing

#### `_apply_dpo_filtering()`

```python
def _apply_dpo_filtering(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply DPO-specific filtering to the dataset.
    
    Args:
        data: List of processed samples
        
    Returns:
        Filtered dataset
    """
    filtered_data = []
    
    for sample in data:
        # Filter by preference strength
        if self._should_filter_by_preference_strength(sample):
            continue
        
        # Filter by response quality difference
        if self._should_filter_by_quality_difference(sample):
            continue
        
        # Filter by length ratio
        if self._should_filter_by_length_ratio(sample):
            continue
        
        # Filter by similarity
        if self._should_filter_by_similarity(sample):
            continue
        
        filtered_data.append(sample)
    
    self.logger.info(f"Filtered {len(data) - len(filtered_data)} samples, kept {len(filtered_data)}")
    return filtered_data
```

#### `_calculate_dpo_preference_strength()`

```python
def _calculate_dpo_preference_strength(self, chosen: str, rejected: str) -> float:
    """
    Calculate preference strength for DPO training.
    
    Args:
        chosen: Preferred response
        rejected: Non-preferred response
        
    Returns:
        Preference strength score (0.0 to 1.0)
    """
    # Calculate various quality metrics
    chosen_quality = self._calculate_response_quality(chosen)
    rejected_quality = self._calculate_response_quality(rejected)
    
    # Length-normalized quality difference
    quality_diff = chosen_quality - rejected_quality
    
    # Informativeness difference
    chosen_info = self._calculate_informativeness(chosen)
    rejected_info = self._calculate_informativeness(rejected)
    info_diff = chosen_info - rejected_info
    
    # Coherence difference
    chosen_coherence = self._calculate_coherence(chosen)
    rejected_coherence = self._calculate_coherence(rejected)
    coherence_diff = chosen_coherence - rejected_coherence
    
    # Combine metrics
    preference_strength = (
        quality_diff * 0.4 +
        info_diff * 0.3 +
        coherence_diff * 0.3
    )
    
    # Normalize to [0, 1]
    return max(0.0, min(1.0, (preference_strength + 1.0) / 2.0))
```

#### `_validate_dpo_preference_quality()`

```python
def _validate_dpo_preference_quality(self, chosen: str, rejected: str) -> bool:
    """
    Validate that chosen response is actually better than rejected.
    
    Args:
        chosen: Preferred response
        rejected: Non-preferred response
        
    Returns:
        True if preference is valid
    """
    # Calculate quality scores
    chosen_quality = self._calculate_response_quality(chosen)
    rejected_quality = self._calculate_response_quality(rejected)
    
    # Ensure chosen is better (with some tolerance)
    min_quality_diff = getattr(self.config, 'min_quality_difference', 0.1)
    quality_diff = chosen_quality - rejected_quality
    
    if quality_diff < min_quality_diff:
        return False
    
    # Check diversity
    similarity = self._calculate_similarity(chosen, rejected)
    max_similarity = getattr(self.config, 'max_similarity_threshold', 0.9)
    
    if similarity > max_similarity:
        return False
    
    return True
```

### Conversation Processing

#### `_process_conversation_context()`

```python
def _process_conversation_context(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Process conversation context for DPO training.
    
    Args:
        conversation: List of conversation turns
        
    Returns:
        Processed conversation context
    """
    processed_conversation = []
    
    for turn in conversation:
        processed_turn = {
            "role": turn.get("role", "user"),
            "content": turn.get("content", "").strip()
        }
        
        # Validate turn
        if processed_turn["content"] and processed_turn["role"] in ["user", "assistant", "system"]:
            processed_conversation.append(processed_turn)
    
    return processed_conversation
```

#### `_extract_prompt_from_conversation()`

```python
def _extract_prompt_from_conversation(self, conversation: List[Dict[str, str]]) -> str:
    """
    Extract prompt from conversation context.
    
    Args:
        conversation: Conversation turns
        
    Returns:
        Extracted prompt
    """
    # Find the last user message
    for turn in reversed(conversation):
        if turn.get("role") == "user":
            return turn.get("content", "")
    
    # If no user message found, concatenate all non-assistant messages
    prompt_parts = []
    for turn in conversation:
        if turn.get("role") != "assistant":
            content = turn.get("content", "").strip()
            if content:
                prompt_parts.append(content)
    
    return " ".join(prompt_parts)
```

## Data Formats

### 1. Standard DPO Format

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد ومركزها السياسي والثقافي والاقتصادي.", "rejected": "باريس."}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال علمي متعدد التخصصات يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً مثل التعلم والاستدلال وحل المشكلات والإدراك.", "rejected": "الذكاء الاصطناعي شيء معقد ومفيد."}
```

### 2. DPO with System Messages

```jsonl
{
  "system": "أنت مساعد ذكي ومفيد يجيب على الأسئلة باللغة العربية.",
  "prompt": "ما هي فوائد القراءة؟",
  "chosen": "القراءة لها فوائد عديدة منها: تحسين المفردات والمعرفة، تطوير مهارات التفكير النقدي، تقليل التوتر، تحسين التركيز والذاكرة، وتوسيع الآفاق الثقافية.",
  "rejected": "القراءة مفيدة للعقل والمعرفة."
}
```

### 3. DPO with Conversation Context

```jsonl
{
  "conversation": [
    {"role": "system", "content": "أنت مساعد ذكي متخصص في الطب."},
    {"role": "user", "content": "ما هي أعراض نزلات البرد؟"},
    {"role": "assistant", "content": "أعراض نزلات البرد تشمل العطس والسعال."},
    {"role": "user", "content": "كيف يمكن علاجها؟"}
  ],
  "chosen": "يمكن علاج نزلات البرد من خلال: الراحة الكافية، شرب السوائل الدافئة، استخدام مرطب الهواء، تناول فيتامين C، والغرغرة بالماء المالح. في الحالات الشديدة، يُنصح بمراجعة الطبيب.",
  "rejected": "اشرب الماء واسترح."
}
```

### 4. DPO with Metadata

```jsonl
{
  "prompt": "اشرح قانون الجاذبية",
  "chosen": "قانون الجاذبية العام لنيوتن ينص على أن كل جسم في الكون يجذب كل جسم آخر بقوة تتناسب طردياً مع حاصل ضرب كتلتيهما وعكسياً مع مربع المسافة بينهما.",
  "rejected": "الجاذبية هي القوة التي تجذب الأشياء للأرض.",
  "metadata": {
    "domain": "physics",
    "difficulty": "intermediate",
    "language": "arabic",
    "annotator_id": "annotator_123",
    "preference_strength": 0.8
  }
}
```

## Configuration

### DPO Data Configuration

```python
@dataclass
class DPODataConfig(BasePreferenceConfig):
    """Configuration for DPO data loading."""
    
    # DPO-specific parameters
    min_quality_difference: float = 0.1
    max_similarity_threshold: float = 0.9
    preference_strength_threshold: float = 0.2
    
    # Length constraints
    max_length_ratio: float = 3.0
    min_length_difference: int = 10
    
    # Quality filtering
    filter_low_preference_strength: bool = True
    filter_similar_responses: bool = True
    filter_extreme_length_ratios: bool = True
    
    # Conversation processing
    max_conversation_turns: int = 10
    include_conversation_context: bool = True
    conversation_template: str = "default"
    
    # Arabic-specific
    normalize_arabic_text: bool = True
    remove_diacritics: bool = False
    handle_mixed_script: bool = True
```

## Usage Examples

### Basic DPO Data Loading

```python
from src.data.dpo_loader import DPODataLoader
from src.config.dpo_config import DPOConfig
from transformers import AutoTokenizer

# Load configuration
config = DPOConfig.from_yaml("config/dpo_config.yaml")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

# Create DPO data loader
loader = DPODataLoader(
    config=config.data,
    tokenizer=tokenizer,
    enable_cache=True
)

# Load datasets
train_dataset = loader.load_dataset("train")
val_dataset = loader.load_dataset("validation")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Inspect sample
sample = train_dataset[0]
print(f"Prompt: {sample['prompt']}")
print(f"Chosen: {sample['chosen']}")
print(f"Rejected: {sample['rejected']}")
print(f"Preference strength: {sample['preference_strength']:.3f}")
```

### Custom DPO Configuration

```python
from src.data.dpo_loader import DPODataLoader
from src.config.dpo_config import DPODataConfig

# Create custom configuration
data_config = DPODataConfig(
    train_file="data/dpo_train.jsonl",
    validation_file="data/dpo_val.jsonl",
    max_length=2048,
    max_prompt_length=1024,
    max_response_length=1024,
    min_quality_difference=0.2,
    max_similarity_threshold=0.8,
    filter_low_preference_strength=True,
    preference_strength_threshold=0.3,
    preprocessing_num_workers=8
)

# Create loader with custom config
loader = DPODataLoader(config=data_config)

# Load and analyze data
train_dataset = loader.load_dataset("train")
stats = loader.get_preference_statistics(train_dataset)

print("=== DPO Dataset Statistics ===")
print(f"Total samples: {stats['total_samples']}")
print(f"Average preference strength: {stats.get('avg_preference_strength', 'N/A')}")
print(f"Quality difference stats: {stats.get('quality_difference_stats', 'N/A')}")
```

### Data Quality Analysis

```python
# Analyze DPO data quality
loader = DPODataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Get detailed statistics
stats = loader.get_preference_statistics(train_dataset)

print("=== DPO Data Quality Analysis ===")
print(f"Dataset size: {stats['total_samples']}")

# Length analysis
print("\n--- Length Analysis ---")
print(f"Avg prompt length: {stats['avg_prompt_length']:.1f}")
print(f"Avg chosen length: {stats['avg_chosen_length']:.1f}")
print(f"Avg rejected length: {stats['avg_rejected_length']:.1f}")

# Preference strength analysis
preference_strengths = [sample['preference_strength'] for sample in train_dataset]
avg_strength = sum(preference_strengths) / len(preference_strengths)
min_strength = min(preference_strengths)
max_strength = max(preference_strengths)

print("\n--- Preference Strength Analysis ---")
print(f"Average preference strength: {avg_strength:.3f}")
print(f"Min preference strength: {min_strength:.3f}")
print(f"Max preference strength: {max_strength:.3f}")

# Quality analysis
chosen_qualities = [sample['chosen_quality'] for sample in train_dataset]
rejected_qualities = [sample['rejected_quality'] for sample in train_dataset]

avg_chosen_quality = sum(chosen_qualities) / len(chosen_qualities)
avg_rejected_quality = sum(rejected_qualities) / len(rejected_qualities)

print("\n--- Quality Analysis ---")
print(f"Average chosen quality: {avg_chosen_quality:.3f}")
print(f"Average rejected quality: {avg_rejected_quality:.3f}")
print(f"Quality difference: {avg_chosen_quality - avg_rejected_quality:.3f}")

# Check for potential issues
weak_preferences = sum(1 for s in preference_strengths if s < 0.3)
print(f"\n--- Potential Issues ---")
print(f"Samples with weak preferences (<0.3): {weak_preferences} ({weak_preferences/len(preference_strengths)*100:.1f}%)")

if avg_strength < 0.4:
    print("⚠️  Warning: Low average preference strength")
if avg_chosen_quality - avg_rejected_quality < 0.2:
    print("⚠️  Warning: Small quality difference between chosen and rejected")
```

### Streaming Large Datasets

```python
from torch.utils.data import DataLoader

# Configure for streaming
data_config = DPODataConfig(
    train_file="data/large_dpo_train.jsonl",
    streaming=True,
    buffer_size=10000,
    preprocessing_num_workers=8
)

loader = DPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")

# Create data loader for training
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Process in batches
for batch_idx, batch in enumerate(train_dataloader):
    # Process batch
    prompts = batch['prompt']
    chosen = batch['chosen']
    rejected = batch['rejected']
    
    print(f"Batch {batch_idx}: {len(prompts)} samples")
    
    if batch_idx >= 5:  # Process first 5 batches
        break
```

## Performance Optimization

### Caching

```python
# Enable aggressive caching for repeated experiments
loader = DPODataLoader(
    config=config.data,
    enable_cache=True,
    cache_dir=".cache/dpo_data",
    cache_compression=True
)

# First load (slow)
train_dataset = loader.load_dataset("train")

# Subsequent loads (fast)
train_dataset = loader.load_dataset("train")  # Loaded from cache
```

### Parallel Processing

```python
# Configure parallel processing
data_config = DPODataConfig(
    train_file="data/dpo_train.jsonl",
    preprocessing_num_workers=16,  # Use more workers
    batch_size=1000,  # Process in larger batches
    enable_multiprocessing=True
)

loader = DPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")
```

### Memory Optimization

```python
# Configure for memory efficiency
data_config = DPODataConfig(
    train_file="data/dpo_train.jsonl",
    max_samples_in_memory=50000,  # Limit memory usage
    streaming=True,
    lazy_loading=True,
    garbage_collection_frequency=1000
)

loader = DPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")
```

## Error Handling

### DPO-Specific Errors

```python
class DPODataError(Exception):
    """Raised when DPO data processing fails."""
    pass

class DPOValidationError(Exception):
    """Raised when DPO data validation fails."""
    pass

# Error handling example
try:
    loader = DPODataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except DPODataError as e:
    print(f"DPO data error: {e}")
    # Handle DPO-specific errors
except DPOValidationError as e:
    print(f"DPO validation error: {e}")
    # Handle validation errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Best Practices

### 1. Data Quality
- Ensure high-quality human preferences
- Validate preference consistency
- Monitor preference strength distribution
- Filter out ambiguous preferences

### 2. Preprocessing
- Normalize Arabic text consistently
- Handle mixed-script content properly
- Balance response lengths
- Remove near-duplicate pairs

### 3. Validation
- Implement comprehensive validation
- Check for annotation biases
- Monitor quality metrics
- Validate preference strength

### 4. Performance
- Use caching for repeated experiments
- Enable parallel processing for large datasets
- Consider streaming for very large datasets
- Monitor memory usage

### 5. Arabic Language Considerations
- Handle Arabic text normalization
- Consider dialectal variations
- Validate Arabic-specific quality metrics
- Handle right-to-left text properly

## Troubleshooting

### Common Issues

1. **Low preference strength**
   - Check annotation quality
   - Increase quality difference threshold
   - Review preference criteria

2. **High similarity between responses**
   - Lower similarity threshold
   - Improve response diversity
   - Check for duplicate data

3. **Memory issues with large datasets**
   - Enable streaming mode
   - Reduce batch size
   - Use lazy loading

4. **Slow data loading**
   - Enable caching
   - Increase number of workers
   - Use parallel processing

## See Also

- [Preference Data Loader](preference_loader.md)
- [DPO Configuration](../config/dpo_config.md)
- [DPO Training](../training/dpo_trainer.md)
- [Base Data Loader](base_loader.md)
- [Data Validation](utils/validation.md)