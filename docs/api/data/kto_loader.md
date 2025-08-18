# KTO Data Loader Documentation

The `KTODataLoader` class handles data loading for Kahneman-Tversky Optimization (KTO) training in the Arabic Qwen Base Fine-tuning framework. It extends the `PreferenceDataLoader` to provide KTO-specific data processing and validation.

## Class Overview

```python
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset
from src.data.preference_loader import PreferenceDataLoader
from src.config.kto_config import KTODataConfig

class KTODataLoader(PreferenceDataLoader):
    """Data loader for Kahneman-Tversky Optimization (KTO)."""
```

## Location

**File**: `src/data/kto_loader.py`

## KTO Overview

Kahneman-Tversky Optimization (KTO) is a preference learning method inspired by prospect theory from behavioral economics. Unlike DPO which requires pairwise comparisons, KTO can work with binary feedback (good/bad) for individual responses.

### Key Features

- **Binary feedback**: Works with simple good/bad labels
- **Prospect theory**: Based on human decision-making psychology
- **Loss aversion**: Models human tendency to weigh losses more heavily than gains
- **Reference point**: Uses a reference model for comparison
- **Flexible data**: Can handle both binary and pairwise data

### KTO vs DPO

| Aspect | KTO | DPO |
|--------|-----|-----|
| Data requirement | Binary labels or pairs | Pairwise comparisons |
| Annotation cost | Lower | Higher |
| Theoretical basis | Prospect theory | Bradley-Terry model |
| Loss function | KTO loss with prospect theory | DPO loss with log-sigmoid |
| Data efficiency | Higher | Lower |

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: KTODataConfig,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    **kwargs
):
    """
    Initialize KTO data loader.
    
    Args:
        config: KTO data configuration
        tokenizer: Tokenizer for text processing
        enable_cache: Enable data caching
        **kwargs: Additional arguments
    """
    super().__init__(config, tokenizer, enable_cache, **kwargs)
    self.config = config
    self.kto_specific_config = getattr(config, 'kto_specific', None)
```

### Core Methods

#### `load_dataset()`

```python
def load_dataset(self, split: str = "train") -> Dataset:
    """
    Load KTO dataset for specified split.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        
    Returns:
        Loaded and processed KTO dataset
    """
    # Load raw data
    raw_data = self._load_raw_data(split)
    
    # Process for KTO
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
    
    # Apply KTO-specific filtering
    filtered_data = self._apply_kto_filtering(processed_data)
    
    # Balance positive and negative samples if needed
    if self.config.balance_labels:
        filtered_data = self._balance_kto_labels(filtered_data)
    
    # Create dataset
    dataset = self._create_dataset(filtered_data)
    
    self.logger.info(f"Loaded {len(dataset)} KTO samples for {split} split")
    return dataset
```

#### `_process_preference_sample()`

```python
def _process_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process KTO-specific preference sample.
    
    Args:
        sample: Raw preference sample
        
    Returns:
        Processed KTO sample
    """
    processed = sample.copy()
    
    # Detect and convert format
    format_type = self._detect_kto_format(sample)
    
    if format_type == "binary":
        processed = self._process_binary_format(sample)
    elif format_type == "pairwise":
        processed = self._process_pairwise_format(sample)
    elif format_type == "ranking":
        processed = self._process_ranking_format(sample)
    else:
        raise ValueError(f"Unsupported KTO format: {format_type}")
    
    # Add KTO-specific fields
    processed["method"] = "kto"
    processed["format_type"] = format_type
    
    # Calculate utility score for KTO
    processed["utility_score"] = self._calculate_kto_utility(processed)
    
    # Add quality metrics
    processed["response_quality"] = self._calculate_response_quality(processed["response"])
    
    return processed
```

#### `_detect_kto_format()`

```python
def _detect_kto_format(self, sample: Dict[str, Any]) -> str:
    """
    Detect the format of KTO data.
    
    Args:
        sample: Input sample
        
    Returns:
        Format type: 'binary', 'pairwise', 'ranking'
    """
    # Binary format: prompt + response + label
    if "response" in sample and "label" in sample:
        return "binary"
    
    # Pairwise format: prompt + chosen + rejected
    if "chosen" in sample and "rejected" in sample:
        return "pairwise"
    
    # Ranking format: prompt + responses + rankings
    if "responses" in sample and "rankings" in sample:
        return "ranking"
    
    # Alternative binary formats
    if "completion" in sample and any(key in sample for key in ["score", "rating", "preference"]):
        return "binary"
    
    raise ValueError(f"Cannot detect KTO format from sample keys: {sample.keys()}")
```

### Format Processing

#### `_process_binary_format()`

```python
def _process_binary_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process binary feedback format for KTO.
    
    Args:
        sample: Binary format sample
        
    Returns:
        Processed KTO sample
    """
    processed = {
        "prompt": sample.get("prompt", sample.get(self.config.prompt_column, "")),
        "response": sample.get("response", sample.get("completion", "")),
        "label": self._normalize_kto_label(sample),
        "preference_type": "binary"
    }
    
    # Add system message if present
    if "system" in sample or self.config.system_column in sample:
        processed["system"] = sample.get("system", sample.get(self.config.system_column, ""))
    
    # Add conversation context if available
    if "conversation" in sample:
        processed["conversation"] = self._process_conversation_context(sample["conversation"])
    
    return processed
```

#### `_process_pairwise_format()`

```python
def _process_pairwise_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process pairwise format for KTO by converting to binary samples.
    
    Args:
        sample: Pairwise format sample
        
    Returns:
        List of binary KTO samples
    """
    prompt = sample.get("prompt", "")
    chosen = sample.get("chosen", "")
    rejected = sample.get("rejected", "")
    system = sample.get("system", "")
    
    # Create positive sample (chosen response)
    positive_sample = {
        "prompt": prompt,
        "response": chosen,
        "label": "good",
        "preference_type": "pairwise_positive"
    }
    
    # Create negative sample (rejected response)
    negative_sample = {
        "prompt": prompt,
        "response": rejected,
        "label": "bad",
        "preference_type": "pairwise_negative"
    }
    
    # Add system message if present
    if system:
        positive_sample["system"] = system
        negative_sample["system"] = system
    
    return [positive_sample, negative_sample]
```

#### `_normalize_kto_label()`

```python
def _normalize_kto_label(self, sample: Dict[str, Any]) -> str:
    """
    Normalize KTO label to standard format.
    
    Args:
        sample: Input sample with label
        
    Returns:
        Normalized label: 'good' or 'bad'
    """
    # Get label from various possible fields
    label = sample.get("label", sample.get("score", sample.get("rating", sample.get("preference"))))
    
    if label is None:
        raise ValueError("No label found in sample")
    
    # Convert to string and normalize
    label_str = str(label).lower().strip()
    
    # Map various label formats to binary
    positive_labels = {
        "good", "positive", "1", "true", "yes", "accept", "preferred", 
        "high", "excellent", "great", "جيد", "ممتاز", "مقبول"
    }
    
    negative_labels = {
        "bad", "negative", "0", "false", "no", "reject", "not_preferred",
        "low", "poor", "terrible", "سيء", "ضعيف", "مرفوض"
    }
    
    # Numeric labels
    try:
        numeric_label = float(label_str)
        if numeric_label >= 0.5:  # Threshold for positive
            return "good"
        else:
            return "bad"
    except ValueError:
        pass
    
    # String labels
    if label_str in positive_labels:
        return "good"
    elif label_str in negative_labels:
        return "bad"
    else:
        raise ValueError(f"Unknown label format: {label}")
```

### KTO-Specific Processing

#### `_calculate_kto_utility()`

```python
def _calculate_kto_utility(self, sample: Dict[str, Any]) -> float:
    """
    Calculate utility score for KTO training.
    
    Args:
        sample: Processed KTO sample
        
    Returns:
        Utility score for prospect theory
    """
    response = sample["response"]
    label = sample["label"]
    
    # Base quality score
    quality_score = self._calculate_response_quality(response)
    
    # Apply prospect theory transformation
    if label == "good":
        # Gains: concave utility function
        utility = quality_score ** self.config.kto_specific.alpha
    else:
        # Losses: convex utility function with loss aversion
        utility = -self.config.kto_specific.lambda_param * ((1 - quality_score) ** self.config.kto_specific.alpha)
    
    return utility
```

#### `_apply_kto_filtering()`

```python
def _apply_kto_filtering(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply KTO-specific filtering to the dataset.
    
    Args:
        data: List of processed samples
        
    Returns:
        Filtered dataset
    """
    filtered_data = []
    
    for sample in data:
        # Filter by response quality
        if self._should_filter_by_quality(sample):
            continue
        
        # Filter by utility score
        if self._should_filter_by_utility(sample):
            continue
        
        # Filter by length
        if self._should_filter_by_length(sample):
            continue
        
        filtered_data.append(sample)
    
    self.logger.info(f"Filtered {len(data) - len(filtered_data)} samples, kept {len(filtered_data)}")
    return filtered_data
```

#### `_balance_kto_labels()`

```python
def _balance_kto_labels(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Balance positive and negative labels in KTO dataset.
    
    Args:
        data: List of processed samples
        
    Returns:
        Balanced dataset
    """
    positive_samples = [s for s in data if s["label"] == "good"]
    negative_samples = [s for s in data if s["label"] == "bad"]
    
    # Calculate target size
    min_size = min(len(positive_samples), len(negative_samples))
    target_size = int(min_size * (1 + self.config.label_imbalance_tolerance))
    
    # Sample to target size
    import random
    random.seed(self.config.random_seed)
    
    if len(positive_samples) > target_size:
        positive_samples = random.sample(positive_samples, target_size)
    
    if len(negative_samples) > target_size:
        negative_samples = random.sample(negative_samples, target_size)
    
    balanced_data = positive_samples + negative_samples
    random.shuffle(balanced_data)
    
    self.logger.info(
        f"Balanced dataset: {len(positive_samples)} positive, "
        f"{len(negative_samples)} negative samples"
    )
    
    return balanced_data
```

#### `_validate_preference_format()`

```python
def _validate_preference_format(self, sample: Dict[str, Any]) -> bool:
    """
    Validate KTO-specific preference format.
    
    Args:
        sample: Sample to validate
        
    Returns:
        True if valid KTO format
    """
    try:
        format_type = self._detect_kto_format(sample)
        
        if format_type == "binary":
            return self._validate_binary_format(sample)
        elif format_type == "pairwise":
            return self._validate_pairwise_format(sample)
        elif format_type == "ranking":
            return self._validate_ranking_format(sample)
        else:
            return False
            
    except Exception:
        return False
```

#### `_validate_binary_format()`

```python
def _validate_binary_format(self, sample: Dict[str, Any]) -> bool:
    """
    Validate binary format for KTO.
    
    Args:
        sample: Binary format sample
        
    Returns:
        True if valid
    """
    # Check required fields
    prompt = sample.get("prompt", "")
    response = sample.get("response", sample.get("completion", ""))
    
    if not prompt or not response:
        return False
    
    # Validate label
    try:
        label = self._normalize_kto_label(sample)
        if label not in ["good", "bad"]:
            return False
    except ValueError:
        return False
    
    # Length validation
    if len(prompt.strip()) < self.config.min_prompt_length:
        return False
    if len(response.strip()) < self.config.min_response_length:
        return False
    
    if len(prompt) > self.config.max_prompt_length:
        return False
    if len(response) > self.config.max_response_length:
        return False
    
    return True
```

## Data Formats

### 1. Binary Feedback Format

Standard binary format with good/bad labels:

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "response": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد.", "label": "good"}
{"prompt": "ما هي عاصمة فرنسا؟", "response": "لا أعرف.", "label": "bad"}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "response": "الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة ذكية.", "label": "good"}
```

### 2. Numeric Score Format

Using numeric scores (converted to binary):

```jsonl
{"prompt": "ما هي فوائد التمرين؟", "response": "التمرين مفيد للصحة الجسدية والنفسية.", "score": 0.8}
{"prompt": "ما هي فوائد التمرين؟", "response": "التمرين جيد.", "score": 0.3}
{"prompt": "اشرح قانون الجاذبية", "response": "قانون الجاذبية ينص على أن الأجسام تتجاذب.", "rating": 4}
```

### 3. Alternative Label Formats

Various label formats supported:

```jsonl
{"prompt": "ما هو الاحتباس الحراري؟", "response": "الاحتباس الحراري ظاهرة طبيعية مهمة.", "preference": "positive"}
{"prompt": "كيف يعمل الإنترنت؟", "response": "الإنترنت معقد.", "preference": "negative"}
{"prompt": "ما هي أهمية التعليم؟", "response": "التعليم أساس التقدم والتطور.", "label": "ممتاز"}
{"prompt": "اشرح الديمقراطية", "response": "الديمقراطية شيء جيد.", "label": "ضعيف"}
```

### 4. Pairwise Format (Converted to Binary)

Pairwise data converted to binary samples:

```jsonl
{"prompt": "ما هي أسباب تغير المناخ؟", "chosen": "تغير المناخ ينتج عن عوامل طبيعية وبشرية متعددة منها انبعاثات الغازات الدفيئة.", "rejected": "تغير المناخ بسبب التلوث."}
```

Converted to:
```jsonl
{"prompt": "ما هي أسباب تغير المناخ؟", "response": "تغير المناخ ينتج عن عوامل طبيعية وبشرية متعددة منها انبعاثات الغازات الدفيئة.", "label": "good", "preference_type": "pairwise_positive"}
{"prompt": "ما هي أسباب تغير المناخ؟", "response": "تغير المناخ بسبب التلوث.", "label": "bad", "preference_type": "pairwise_negative"}
```

### 5. Conversation Format

With conversation context:

```jsonl
{
  "conversation": [
    {"role": "system", "content": "أنت مساعد ذكي ومفيد."},
    {"role": "user", "content": "ما هي فوائد القراءة؟"},
    {"role": "assistant", "content": "القراءة تحسن المعرفة والمفردات."}
  ],
  "response": "القراءة لها فوائد عديدة منها تحسين المفردات وتطوير التفكير النقدي وتقليل التوتر.",
  "label": "good"
}
```

## Configuration

### KTO Data Configuration

```python
@dataclass
class KTODataConfig(BasePreferenceConfig):
    """Configuration for KTO data loading."""
    
    # Label handling
    label_column: str = "label"
    response_column: str = "response"
    completion_column: str = "completion"
    score_column: str = "score"
    rating_column: str = "rating"
    
    # Label balancing
    balance_labels: bool = True
    label_imbalance_tolerance: float = 0.1
    min_samples_per_label: int = 100
    
    # Quality filtering
    min_utility_threshold: float = -1.0
    max_utility_threshold: float = 1.0
    filter_low_utility: bool = True
    
    # KTO-specific parameters
    kto_specific: KTOSpecificConfig = field(default_factory=KTOSpecificConfig)
    
    # Conversion settings
    convert_pairwise_to_binary: bool = True
    pairwise_sampling_strategy: str = "all"  # 'all', 'balanced', 'random'
    
    # Arabic-specific
    arabic_label_mapping: Dict[str, str] = field(default_factory=lambda: {
        "جيد": "good", "ممتاز": "good", "مقبول": "good",
        "سيء": "bad", "ضعيف": "bad", "مرفوض": "bad"
    })
```

## Usage Examples

### Basic KTO Data Loading

```python
from src.data.kto_loader import KTODataLoader
from src.config.kto_config import KTOConfig
from transformers import AutoTokenizer

# Load configuration
config = KTOConfig.from_yaml("config/kto_config.yaml")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

# Create KTO data loader
loader = KTODataLoader(
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
print(f"Response: {sample['response']}")
print(f"Label: {sample['label']}")
print(f"Utility score: {sample['utility_score']:.3f}")
```

### Loading Binary Feedback Data

```python
from src.data.kto_loader import KTODataLoader
from src.config.kto_config import KTODataConfig

# Configure for binary feedback
data_config = KTODataConfig(
    train_file="data/kto_binary_train.jsonl",
    validation_file="data/kto_binary_val.jsonl",
    label_column="feedback",
    response_column="completion",
    balance_labels=True,
    label_imbalance_tolerance=0.05
)

loader = KTODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")

# Check label distribution
labels = [sample['label'] for sample in train_dataset]
positive_count = sum(1 for label in labels if label == "good")
negative_count = sum(1 for label in labels if label == "bad")

print(f"Label distribution:")
print(f"  Positive (good): {positive_count} ({positive_count/len(labels)*100:.1f}%)")
print(f"  Negative (bad): {negative_count} ({negative_count/len(labels)*100:.1f}%)")
```

### Converting Pairwise to Binary

```python
# Configure to convert pairwise data to binary
data_config = KTODataConfig(
    train_file="data/dpo_train.jsonl",  # Pairwise data
    convert_pairwise_to_binary=True,
    pairwise_sampling_strategy="balanced",
    balance_labels=True
)

loader = KTODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")

# Each pairwise sample becomes two binary samples
print(f"Converted dataset size: {len(train_dataset)}")

# Check conversion results
for i, sample in enumerate(train_dataset[:4]):
    print(f"Sample {i}:")
    print(f"  Prompt: {sample['prompt'][:50]}...")
    print(f"  Response: {sample['response'][:50]}...")
    print(f"  Label: {sample['label']}")
    print(f"  Type: {sample.get('preference_type', 'binary')}")
    print()
```

### Custom Label Processing

```python
class CustomKTODataLoader(KTODataLoader):
    """Custom KTO loader with specialized label processing."""
    
    def _normalize_kto_label(self, sample: Dict[str, Any]) -> str:
        """Custom label normalization."""
        # Handle custom rating scale (1-5)
        if "rating" in sample:
            rating = float(sample["rating"])
            return "good" if rating >= 4 else "bad"
        
        # Handle Arabic labels
        if "تقييم" in sample:
            arabic_label = sample["تقييم"]
            arabic_mapping = {
                "ممتاز": "good", "جيد جداً": "good", "جيد": "good",
                "مقبول": "bad", "ضعيف": "bad", "سيء": "bad"
            }
            return arabic_mapping.get(arabic_label, "bad")
        
        # Fall back to default processing
        return super()._normalize_kto_label(sample)
    
    def _calculate_kto_utility(self, sample: Dict[str, Any]) -> float:
        """Custom utility calculation."""
        response = sample["response"]
        label = sample["label"]
        
        # Custom quality metrics
        length_score = min(len(response) / 100, 1.0)
        arabic_score = self._calculate_arabic_quality(response)
        informativeness = self._calculate_informativeness(response)
        
        base_quality = (length_score + arabic_score + informativeness) / 3
        
        # Apply prospect theory with custom parameters
        if label == "good":
            return base_quality ** 0.88  # Concave for gains
        else:
            return -2.25 * ((1 - base_quality) ** 0.88)  # Loss aversion
    
    def _calculate_arabic_quality(self, text: str) -> float:
        """Calculate Arabic-specific quality score."""
        # Simple heuristics for Arabic text quality
        score = 0.0
        
        # Check for proper Arabic characters
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        if arabic_chars > 0:
            score += 0.3
        
        # Check for diacritics (indicates formal Arabic)
        diacritics = sum(1 for c in text if '\u064B' <= c <= '\u065F')
        if diacritics > 0:
            score += 0.2
        
        # Check for proper sentence structure
        sentences = text.split('.')
        if len(sentences) > 1:
            score += 0.3
        
        # Check for question/answer patterns
        if any(word in text for word in ['ما', 'كيف', 'لماذا', 'أين', 'متى']):
            score += 0.2
        
        return min(score, 1.0)

# Use custom loader
custom_loader = CustomKTODataLoader(config=data_config)
train_dataset = custom_loader.load_dataset("train")
```

### Data Quality Analysis

```python
# Analyze KTO data quality
loader = KTODataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Label distribution
labels = [sample['label'] for sample in train_dataset]
positive_count = sum(1 for label in labels if label == "good")
negative_count = sum(1 for label in labels if label == "bad")

print("=== KTO Data Analysis ===")
print(f"Total samples: {len(train_dataset)}")
print(f"Positive samples: {positive_count} ({positive_count/len(labels)*100:.1f}%)")
print(f"Negative samples: {negative_count} ({negative_count/len(labels)*100:.1f}%)")

# Utility score analysis
utility_scores = [sample['utility_score'] for sample in train_dataset]
positive_utilities = [sample['utility_score'] for sample in train_dataset if sample['label'] == 'good']
negative_utilities = [sample['utility_score'] for sample in train_dataset if sample['label'] == 'bad']

print("\n=== Utility Score Analysis ===")
print(f"Overall utility range: [{min(utility_scores):.3f}, {max(utility_scores):.3f}]")
print(f"Average positive utility: {sum(positive_utilities)/len(positive_utilities):.3f}")
print(f"Average negative utility: {sum(negative_utilities)/len(negative_utilities):.3f}")

# Quality analysis
response_qualities = [sample['response_quality'] for sample in train_dataset]
print(f"\n=== Quality Analysis ===")
print(f"Average response quality: {sum(response_qualities)/len(response_qualities):.3f}")
print(f"Quality range: [{min(response_qualities):.3f}, {max(response_qualities):.3f}]")

# Check for potential issues
if abs(positive_count - negative_count) / len(labels) > 0.2:
    print("\n⚠️  Warning: Significant label imbalance")

if sum(positive_utilities) / len(positive_utilities) <= 0:
    print("\n⚠️  Warning: Positive samples have non-positive utility")

if sum(negative_utilities) / len(negative_utilities) >= 0:
    print("\n⚠️  Warning: Negative samples have non-negative utility")
```

### Streaming Large Datasets

```python
# Configure for streaming
data_config = KTODataConfig(
    train_file="data/large_kto_train.jsonl",
    streaming=True,
    buffer_size=10000,
    balance_labels=True,
    preprocessing_num_workers=8
)

loader = KTODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")

# Process in batches
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch_idx, batch in enumerate(train_dataloader):
    prompts = batch['prompt']
    responses = batch['response']
    labels = batch['label']
    utilities = batch['utility_score']
    
    print(f"Batch {batch_idx}: {len(prompts)} samples")
    print(f"  Positive samples: {sum(1 for l in labels if l == 'good')}")
    print(f"  Average utility: {sum(utilities)/len(utilities):.3f}")
    
    if batch_idx >= 5:  # Process first 5 batches
        break
```

## Performance Optimization

### Caching

```python
# Enable caching for repeated experiments
loader = KTODataLoader(
    config=config.data,
    enable_cache=True,
    cache_dir=".cache/kto_data",
    cache_compression=True
)

# First load (processes and caches)
train_dataset = loader.load_dataset("train")

# Subsequent loads (from cache)
train_dataset = loader.load_dataset("train")  # Much faster
```

### Memory Optimization

```python
# Configure for memory efficiency
data_config = KTODataConfig(
    train_file="data/kto_train.jsonl",
    max_samples_in_memory=30000,
    streaming=True,
    lazy_loading=True,
    balance_labels=True,
    label_imbalance_tolerance=0.1
)

loader = KTODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")
```

## Error Handling

### KTO-Specific Errors

```python
class KTODataError(Exception):
    """Raised when KTO data processing fails."""
    pass

class KTOLabelError(Exception):
    """Raised when KTO label processing fails."""
    pass

# Error handling example
try:
    loader = KTODataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except KTODataError as e:
    print(f"KTO data error: {e}")
except KTOLabelError as e:
    print(f"KTO label error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Data Quality
- Ensure consistent labeling criteria
- Balance positive and negative samples
- Validate label quality and consistency
- Monitor utility score distributions

### 2. Label Processing
- Use clear labeling guidelines
- Handle multiple label formats consistently
- Consider cultural and linguistic factors for Arabic
- Validate label normalization

### 3. Prospect Theory Application
- Tune KTO parameters (alpha, lambda) carefully
- Monitor utility score distributions
- Ensure positive samples have positive utility
- Consider domain-specific utility functions

### 4. Performance
- Use caching for repeated experiments
- Enable streaming for large datasets
- Balance labels efficiently
- Monitor memory usage

### 5. Arabic Language Considerations
- Handle Arabic label mappings
- Consider dialectal variations in quality assessment
- Validate Arabic text processing
- Use appropriate quality metrics for Arabic

## Troubleshooting

### Common Issues

1. **Label imbalance**
   - Enable label balancing
   - Adjust imbalance tolerance
   - Check data collection process

2. **Poor utility scores**
   - Review quality calculation
   - Tune KTO parameters
   - Check label consistency

3. **Memory issues**
   - Enable streaming mode
   - Reduce buffer size
   - Use lazy loading

4. **Slow processing**
   - Enable caching
   - Increase worker count
   - Use parallel processing

## See Also

- [Preference Data Loader](preference_loader.md)
- [KTO Configuration](../config/kto_config.md)
- [KTO Training](../training/kto_trainer.md)
- [DPO Data Loader](dpo_loader.md)
- [Base Data Loader](base_loader.md)