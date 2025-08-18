# IPO Data Loader Documentation

The `IPODataLoader` class handles data loading for Identity Preference Optimization (IPO) training in the Arabic Qwen Base Fine-tuning framework. It extends the `PreferenceDataLoader` to provide IPO-specific data processing and validation.

## Class Overview

```python
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset
from src.data.preference_loader import PreferenceDataLoader
from src.config.ipo_config import IPODataConfig

class IPODataLoader(PreferenceDataLoader):
    """Data loader for Identity Preference Optimization (IPO)."""
```

## Location

**File**: `src/data/ipo_loader.py`

## IPO Overview

Identity Preference Optimization (IPO) is a preference learning method that addresses length bias in preference optimization. It uses identity mapping and length normalization to ensure fair comparison between responses of different lengths.

### Key Features

- **Length bias mitigation**: Addresses unfair preference for longer responses
- **Identity mapping**: Uses identity function instead of log-sigmoid
- **Length normalization**: Multiple normalization strategies available
- **Stable training**: More stable than standard DPO for length-biased data
- **Fair comparison**: Ensures responses are compared fairly regardless of length

### IPO vs DPO

| Aspect | IPO | DPO |
|--------|-----|-----|
| Length bias | Mitigated | Present |
| Loss function | Identity mapping | Log-sigmoid |
| Length normalization | Multiple strategies | None |
| Training stability | Higher | Lower for length-biased data |
| Response fairness | Higher | Lower |

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: IPODataConfig,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    **kwargs
):
    """
    Initialize IPO data loader.
    
    Args:
        config: IPO data configuration
        tokenizer: Tokenizer for text processing
        enable_cache: Enable data caching
        **kwargs: Additional arguments
    """
    super().__init__(config, tokenizer, enable_cache, **kwargs)
    self.config = config
    self.ipo_specific_config = getattr(config, 'ipo_specific', None)
```

### Core Methods

#### `load_dataset()`

```python
def load_dataset(self, split: str = "train") -> Dataset:
    """
    Load IPO dataset for specified split.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        
    Returns:
        Loaded and processed IPO dataset
    """
    # Load raw data
    raw_data = self._load_raw_data(split)
    
    # Process for IPO
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
    
    # Apply IPO-specific filtering
    filtered_data = self._apply_ipo_filtering(processed_data)
    
    # Apply length normalization
    normalized_data = self._apply_length_normalization(filtered_data)
    
    # Create dataset
    dataset = self._create_dataset(normalized_data)
    
    self.logger.info(f"Loaded {len(dataset)} IPO samples for {split} split")
    return dataset
```

#### `_process_preference_sample()`

```python
def _process_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process IPO-specific preference sample.
    
    Args:
        sample: Raw preference sample
        
    Returns:
        Processed IPO sample
    """
    processed = sample.copy()
    
    # Ensure standard format
    if "prompt" not in processed:
        processed["prompt"] = self._extract_prompt(sample)
    
    if "chosen" not in processed:
        processed["chosen"] = self._extract_chosen_response(sample)
    
    if "rejected" not in processed:
        processed["rejected"] = self._extract_rejected_response(sample)
    
    # Add IPO-specific fields
    processed["method"] = "ipo"
    processed["preference_type"] = "pairwise"
    
    # Calculate length metrics
    processed["chosen_length"] = len(processed["chosen"])
    processed["rejected_length"] = len(processed["rejected"])
    processed["length_ratio"] = processed["chosen_length"] / max(processed["rejected_length"], 1)
    processed["length_difference"] = abs(processed["chosen_length"] - processed["rejected_length"])
    
    # Calculate length-normalized preference strength
    processed["preference_strength"] = self._calculate_ipo_preference_strength(
        processed["chosen"], processed["rejected"]
    )
    
    # Add quality metrics
    processed["chosen_quality"] = self._calculate_response_quality(processed["chosen"])
    processed["rejected_quality"] = self._calculate_response_quality(processed["rejected"])
    
    # Calculate length-normalized quality difference
    processed["normalized_quality_diff"] = self._calculate_normalized_quality_difference(
        processed["chosen"], processed["rejected"]
    )
    
    # Add length bias indicators
    processed["length_bias_score"] = self._calculate_length_bias_score(
        processed["chosen"], processed["rejected"]
    )
    
    return processed
```

#### `_validate_preference_format()`

```python
def _validate_preference_format(self, sample: Dict[str, Any]) -> bool:
    """
    Validate IPO-specific preference format.
    
    Args:
        sample: Sample to validate
        
    Returns:
        True if valid IPO format
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
        
        # IPO-specific validation
        if not self._validate_ipo_length_constraints(chosen, rejected):
            return False
        
        return True
        
    except Exception:
        return False
```

### IPO-Specific Processing

#### `_calculate_ipo_preference_strength()`

```python
def _calculate_ipo_preference_strength(self, chosen: str, rejected: str) -> float:
    """
    Calculate length-normalized preference strength for IPO.
    
    Args:
        chosen: Preferred response
        rejected: Non-preferred response
        
    Returns:
        Length-normalized preference strength
    """
    # Calculate base quality scores
    chosen_quality = self._calculate_response_quality(chosen)
    rejected_quality = self._calculate_response_quality(rejected)
    
    # Apply length normalization
    chosen_normalized = self._apply_length_normalization_to_quality(
        chosen_quality, len(chosen)
    )
    rejected_normalized = self._apply_length_normalization_to_quality(
        rejected_quality, len(rejected)
    )
    
    # Calculate normalized preference strength
    preference_strength = chosen_normalized - rejected_normalized
    
    # Apply identity mapping (IPO characteristic)
    return preference_strength  # No sigmoid transformation
```

#### `_apply_length_normalization_to_quality()`

```python
def _apply_length_normalization_to_quality(self, quality: float, length: int) -> float:
    """
    Apply length normalization to quality score.
    
    Args:
        quality: Base quality score
        length: Response length
        
    Returns:
        Length-normalized quality score
    """
    normalization_type = self.config.ipo_specific.length_normalization
    
    if normalization_type == "none":
        return quality
    elif normalization_type == "sqrt":
        return quality / (length ** 0.5)
    elif normalization_type == "log":
        return quality / max(1.0, math.log(length + 1))
    elif normalization_type == "linear":
        return quality / max(1.0, length / 100.0)
    elif normalization_type == "adaptive":
        return self._adaptive_length_normalization(quality, length)
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")
```

#### `_adaptive_length_normalization()`

```python
def _adaptive_length_normalization(self, quality: float, length: int) -> float:
    """
    Apply adaptive length normalization based on dataset statistics.
    
    Args:
        quality: Base quality score
        length: Response length
        
    Returns:
        Adaptively normalized quality score
    """
    # Use dataset statistics for adaptive normalization
    if not hasattr(self, '_length_stats'):
        self._length_stats = self._calculate_length_statistics()
    
    mean_length = self._length_stats['mean']
    std_length = self._length_stats['std']
    
    # Z-score normalization with quality weighting
    z_score = (length - mean_length) / max(std_length, 1.0)
    normalization_factor = 1.0 + (z_score * 0.1)  # Gentle normalization
    
    return quality / max(normalization_factor, 0.1)
```

#### `_calculate_length_bias_score()`

```python
def _calculate_length_bias_score(self, chosen: str, rejected: str) -> float:
    """
    Calculate length bias score for the preference pair.
    
    Args:
        chosen: Preferred response
        rejected: Non-preferred response
        
    Returns:
        Length bias score (higher = more biased)
    """
    chosen_len = len(chosen)
    rejected_len = len(rejected)
    
    # Calculate length ratio
    length_ratio = chosen_len / max(rejected_len, 1)
    
    # Calculate quality difference
    chosen_quality = self._calculate_response_quality(chosen)
    rejected_quality = self._calculate_response_quality(rejected)
    quality_diff = chosen_quality - rejected_quality
    
    # Length bias occurs when longer response is preferred despite lower quality
    if length_ratio > 1.5 and quality_diff < 0.1:
        bias_score = (length_ratio - 1.0) * (0.1 - quality_diff)
    elif length_ratio < 0.67 and quality_diff > -0.1:
        bias_score = (1.0 - length_ratio) * (quality_diff + 0.1)
    else:
        bias_score = 0.0
    
    return min(bias_score, 1.0)
```

#### `_apply_ipo_filtering()`

```python
def _apply_ipo_filtering(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply IPO-specific filtering to the dataset.
    
    Args:
        data: List of processed samples
        
    Returns:
        Filtered dataset
    """
    filtered_data = []
    
    for sample in data:
        # Filter by length bias
        if self._should_filter_by_length_bias(sample):
            continue
        
        # Filter by length ratio
        if self._should_filter_by_extreme_length_ratio(sample):
            continue
        
        # Filter by normalized quality difference
        if self._should_filter_by_normalized_quality(sample):
            continue
        
        filtered_data.append(sample)
    
    self.logger.info(f"IPO filtering: {len(data) - len(filtered_data)} samples removed")
    return filtered_data
```

#### `_should_filter_by_length_bias()`

```python
def _should_filter_by_length_bias(self, sample: Dict[str, Any]) -> bool:
    """
    Check if sample should be filtered due to length bias.
    
    Args:
        sample: Processed sample
        
    Returns:
        True if should be filtered
    """
    if not self.config.filter_length_biased:
        return False
    
    length_bias_score = sample.get("length_bias_score", 0.0)
    threshold = self.config.length_bias_threshold
    
    return length_bias_score > threshold
```

#### `_validate_ipo_length_constraints()`

```python
def _validate_ipo_length_constraints(self, chosen: str, rejected: str) -> bool:
    """
    Validate IPO-specific length constraints.
    
    Args:
        chosen: Preferred response
        rejected: Non-preferred response
        
    Returns:
        True if constraints are satisfied
    """
    chosen_len = len(chosen)
    rejected_len = len(rejected)
    
    # Check maximum length ratio
    max_ratio = self.config.max_length_ratio
    length_ratio = max(chosen_len, rejected_len) / max(min(chosen_len, rejected_len), 1)
    
    if length_ratio > max_ratio:
        return False
    
    # Check minimum length difference
    min_diff = self.config.min_length_difference
    if abs(chosen_len - rejected_len) < min_diff:
        return False
    
    return True
```

### Length Normalization

#### `_apply_length_normalization()`

```python
def _apply_length_normalization(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply length normalization to the entire dataset.
    
    Args:
        data: List of processed samples
        
    Returns:
        Length-normalized dataset
    """
    if self.config.ipo_specific.length_normalization == "none":
        return data
    
    # Calculate dataset-wide length statistics for adaptive normalization
    if self.config.ipo_specific.length_normalization == "adaptive":
        self._length_stats = self._calculate_length_statistics(data)
    
    normalized_data = []
    for sample in data:
        normalized_sample = sample.copy()
        
        # Apply normalization to preference strength
        normalized_sample["normalized_preference_strength"] = self._normalize_preference_strength(
            sample["preference_strength"],
            sample["chosen_length"],
            sample["rejected_length"]
        )
        
        normalized_data.append(normalized_sample)
    
    return normalized_data
```

#### `_calculate_length_statistics()`

```python
def _calculate_length_statistics(self, data: List[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate length statistics for the dataset.
    
    Args:
        data: Dataset samples (if None, uses current dataset)
        
    Returns:
        Dictionary with length statistics
    """
    if data is None:
        # Use cached statistics or calculate from current dataset
        return getattr(self, '_length_stats', {'mean': 100.0, 'std': 50.0})
    
    all_lengths = []
    for sample in data:
        all_lengths.extend([sample["chosen_length"], sample["rejected_length"]])
    
    mean_length = sum(all_lengths) / len(all_lengths)
    variance = sum((x - mean_length) ** 2 for x in all_lengths) / len(all_lengths)
    std_length = variance ** 0.5
    
    return {
        'mean': mean_length,
        'std': std_length,
        'min': min(all_lengths),
        'max': max(all_lengths),
        'median': sorted(all_lengths)[len(all_lengths) // 2]
    }
```

## Data Formats

### 1. Standard IPO Format

Same as DPO format but processed with length normalization:

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد ومركزها السياسي والثقافي والاقتصادي. تقع في شمال وسط فرنسا على نهر السين.", "rejected": "باريس."}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال علمي متعدد التخصصات يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً.", "rejected": "الذكاء الاصطناعي هو مجال علمي متعدد التخصصات يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً مثل التعلم والاستدلال وحل المشكلات والإدراك والتخطيط ومعالجة اللغة الطبيعية والرؤية الحاسوبية والتفاعل مع البيئة."}
```

### 2. IPO with Length Metadata

Including length information for analysis:

```jsonl
{
  "prompt": "ما هي فوائد القراءة؟",
  "chosen": "القراءة لها فوائد عديدة منها تحسين المفردات والمعرفة العامة.",
  "rejected": "القراءة مفيدة.",
  "metadata": {
    "chosen_length": 65,
    "rejected_length": 15,
    "length_ratio": 4.33,
    "length_bias_score": 0.2
  }
}
```

### 3. IPO with Quality Scores

Including quality metrics for length normalization:

```jsonl
{
  "prompt": "اشرح قانون الجاذبية",
  "chosen": "قانون الجاذبية العام لنيوتن ينص على أن كل جسم في الكون يجذب كل جسم آخر بقوة تتناسب طردياً مع حاصل ضرب كتلتيهما وعكسياً مع مربع المسافة بينهما.",
  "rejected": "الجاذبية هي القوة التي تجذب الأشياء للأرض وتحافظ على دوران الكواكب حول الشمس وتؤثر على حركة الأجسام في الفضاء وتحدد شكل المجرات والنجوم.",
  "quality_scores": {
    "chosen_quality": 0.85,
    "rejected_quality": 0.75,
    "chosen_normalized": 0.82,
    "rejected_normalized": 0.78
  }
}
```

### 4. IPO with Conversation Context

With conversation history:

```jsonl
{
  "conversation": [
    {"role": "system", "content": "أنت مساعد ذكي متخصص في العلوم."},
    {"role": "user", "content": "ما هو الاحتباس الحراري؟"},
    {"role": "assistant", "content": "الاحتباس الحراري ظاهرة طبيعية."},
    {"role": "user", "content": "كيف يحدث؟"}
  ],
  "chosen": "يحدث الاحتباس الحراري عندما تحبس غازات الدفيئة في الغلاف الجوي الحرارة المنبعثة من سطح الأرض، مما يؤدي إلى ارتفاع درجة حرارة الكوكب.",
  "rejected": "يحدث الاحتباس الحراري بسبب الغازات في الجو."
}
```

## Configuration

### IPO Data Configuration

```python
@dataclass
class IPODataConfig(BasePreferenceConfig):
    """Configuration for IPO data loading."""
    
    # Length constraints
    max_length_ratio: float = 4.0
    min_length_difference: int = 20
    
    # Length bias filtering
    filter_length_biased: bool = True
    length_bias_threshold: float = 0.5
    
    # Quality normalization
    normalize_quality_by_length: bool = True
    quality_normalization_method: str = "adaptive"
    
    # IPO-specific parameters
    ipo_specific: IPOSpecificConfig = field(default_factory=IPOSpecificConfig)
    
    # Length analysis
    calculate_length_statistics: bool = True
    length_statistics_cache: bool = True
    
    # Filtering thresholds
    min_normalized_quality_diff: float = 0.05
    max_extreme_length_ratio: float = 10.0

@dataclass
class IPOSpecificConfig:
    """IPO-specific configuration."""
    
    # Length normalization
    length_normalization: str = "sqrt"  # 'none', 'sqrt', 'log', 'linear', 'adaptive'
    
    # Identity mapping parameters
    use_identity_mapping: bool = True
    identity_regularization: float = 0.1
    
    # Length bias mitigation
    length_penalty_weight: float = 0.1
    adaptive_length_penalty: bool = True
```

## Usage Examples

### Basic IPO Data Loading

```python
from src.data.ipo_loader import IPODataLoader
from src.config.ipo_config import IPOConfig
from transformers import AutoTokenizer

# Load configuration
config = IPOConfig.from_yaml("config/ipo_config.yaml")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

# Create IPO data loader
loader = IPODataLoader(
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
print(f"Length ratio: {sample['length_ratio']:.2f}")
print(f"Length bias score: {sample['length_bias_score']:.3f}")
print(f"Normalized preference strength: {sample.get('normalized_preference_strength', 'N/A')}")
```

### Custom Length Normalization

```python
from src.data.ipo_loader import IPODataLoader
from src.config.ipo_config import IPODataConfig, IPOSpecificConfig

# Configure custom length normalization
ipo_specific = IPOSpecificConfig(
    length_normalization="adaptive",
    use_identity_mapping=True,
    length_penalty_weight=0.15
)

data_config = IPODataConfig(
    train_file="data/ipo_train.jsonl",
    validation_file="data/ipo_val.jsonl",
    max_length_ratio=3.0,
    filter_length_biased=True,
    length_bias_threshold=0.3,
    ipo_specific=ipo_specific
)

loader = IPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")

# Analyze length normalization effects
length_stats = loader._calculate_length_statistics()
print(f"Length statistics: {length_stats}")

# Check normalization results
for i, sample in enumerate(train_dataset[:3]):
    print(f"\nSample {i}:")
    print(f"  Chosen length: {sample['chosen_length']}")
    print(f"  Rejected length: {sample['rejected_length']}")
    print(f"  Length ratio: {sample['length_ratio']:.2f}")
    print(f"  Original preference: {sample['preference_strength']:.3f}")
    print(f"  Normalized preference: {sample.get('normalized_preference_strength', 'N/A')}")
```

### Length Bias Analysis

```python
# Analyze length bias in the dataset
loader = IPODataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Collect length bias metrics
length_ratios = []
length_bias_scores = []
preference_strengths = []
normalized_preferences = []

for sample in train_dataset:
    length_ratios.append(sample['length_ratio'])
    length_bias_scores.append(sample['length_bias_score'])
    preference_strengths.append(sample['preference_strength'])
    if 'normalized_preference_strength' in sample:
        normalized_preferences.append(sample['normalized_preference_strength'])

print("=== Length Bias Analysis ===")
print(f"Dataset size: {len(train_dataset)}")
print(f"Average length ratio: {sum(length_ratios)/len(length_ratios):.2f}")
print(f"Average length bias score: {sum(length_bias_scores)/len(length_bias_scores):.3f}")
print(f"Max length bias score: {max(length_bias_scores):.3f}")

# Count highly biased samples
high_bias_count = sum(1 for score in length_bias_scores if score > 0.5)
print(f"Highly biased samples (>0.5): {high_bias_count} ({high_bias_count/len(length_bias_scores)*100:.1f}%)")

# Compare original vs normalized preferences
if normalized_preferences:
    orig_avg = sum(preference_strengths) / len(preference_strengths)
    norm_avg = sum(normalized_preferences) / len(normalized_preferences)
    print(f"\nPreference strength comparison:")
    print(f"  Original average: {orig_avg:.3f}")
    print(f"  Normalized average: {norm_avg:.3f}")
    print(f"  Difference: {abs(orig_avg - norm_avg):.3f}")

# Analyze correlation between length ratio and preference
import statistics
correlation = statistics.correlation(length_ratios, preference_strengths)
print(f"\nLength-preference correlation: {correlation:.3f}")
if abs(correlation) > 0.3:
    print("⚠️  Warning: Strong correlation between length and preference detected")
```

### Custom IPO Loader with Advanced Normalization

```python
class AdvancedIPODataLoader(IPODataLoader):
    """Advanced IPO loader with custom normalization strategies."""
    
    def _apply_length_normalization_to_quality(self, quality: float, length: int) -> float:
        """Advanced length normalization with multiple strategies."""
        normalization_type = self.config.ipo_specific.length_normalization
        
        if normalization_type == "advanced_adaptive":
            return self._advanced_adaptive_normalization(quality, length)
        elif normalization_type == "percentile_based":
            return self._percentile_based_normalization(quality, length)
        else:
            return super()._apply_length_normalization_to_quality(quality, length)
    
    def _advanced_adaptive_normalization(self, quality: float, length: int) -> float:
        """Advanced adaptive normalization using dataset percentiles."""
        if not hasattr(self, '_length_percentiles'):
            self._length_percentiles = self._calculate_length_percentiles()
        
        # Determine length category
        p25, p50, p75 = self._length_percentiles['p25'], self._length_percentiles['p50'], self._length_percentiles['p75']
        
        if length <= p25:
            # Short responses: minimal normalization
            factor = 1.0
        elif length <= p50:
            # Medium responses: moderate normalization
            factor = 1.0 + (length - p25) / (p50 - p25) * 0.1
        elif length <= p75:
            # Long responses: stronger normalization
            factor = 1.1 + (length - p50) / (p75 - p50) * 0.2
        else:
            # Very long responses: strong normalization
            factor = 1.3 + (length - p75) / max(p75, 1) * 0.3
        
        return quality / factor
    
    def _percentile_based_normalization(self, quality: float, length: int) -> float:
        """Normalization based on length percentile ranking."""
        if not hasattr(self, '_length_distribution'):
            self._length_distribution = self._build_length_distribution()
        
        # Find percentile rank of current length
        percentile_rank = self._get_percentile_rank(length, self._length_distribution)
        
        # Apply normalization based on percentile
        if percentile_rank > 80:  # Top 20% longest
            normalization_factor = 1.0 + (percentile_rank - 80) / 20 * 0.5
        elif percentile_rank < 20:  # Bottom 20% shortest
            normalization_factor = 1.0 - (20 - percentile_rank) / 20 * 0.2
        else:
            normalization_factor = 1.0
        
        return quality / normalization_factor
    
    def _calculate_length_percentiles(self) -> Dict[str, float]:
        """Calculate length percentiles for the dataset."""
        # This would be implemented to calculate actual percentiles
        # from the dataset during initialization
        return {'p25': 50, 'p50': 100, 'p75': 200}
    
    def _build_length_distribution(self) -> List[int]:
        """Build sorted length distribution for percentile calculations."""
        # This would be implemented to build actual distribution
        return list(range(10, 500, 5))
    
    def _get_percentile_rank(self, length: int, distribution: List[int]) -> float:
        """Get percentile rank of length in distribution."""
        import bisect
        rank = bisect.bisect_left(distribution, length)
        return (rank / len(distribution)) * 100

# Use advanced loader
advanced_loader = AdvancedIPODataLoader(config=data_config)
train_dataset = advanced_loader.load_dataset("train")
```

### Comparative Analysis: IPO vs DPO

```python
# Compare IPO and DPO processing on the same data
from src.data.dpo_loader import DPODataLoader
from src.data.ipo_loader import IPODataLoader

# Load same data with both loaders
dpo_loader = DPODataLoader(config.data)
ipo_loader = IPODataLoader(config.data)

dpo_dataset = dpo_loader.load_dataset("train")
ipo_dataset = ipo_loader.load_dataset("train")

print("=== DPO vs IPO Comparison ===")
print(f"DPO dataset size: {len(dpo_dataset)}")
print(f"IPO dataset size: {len(ipo_dataset)}")

# Compare preference strengths
dpo_preferences = [sample['preference_strength'] for sample in dpo_dataset[:1000]]
ipo_preferences = [sample['preference_strength'] for sample in ipo_dataset[:1000]]
ipo_normalized = [sample.get('normalized_preference_strength', sample['preference_strength']) for sample in ipo_dataset[:1000]]

print(f"\nPreference strength comparison (first 1000 samples):")
print(f"DPO average: {sum(dpo_preferences)/len(dpo_preferences):.3f}")
print(f"IPO average: {sum(ipo_preferences)/len(ipo_preferences):.3f}")
print(f"IPO normalized average: {sum(ipo_normalized)/len(ipo_normalized):.3f}")

# Analyze length bias reduction
ipo_bias_scores = [sample['length_bias_score'] for sample in ipo_dataset[:1000]]
avg_bias = sum(ipo_bias_scores) / len(ipo_bias_scores)
print(f"\nAverage length bias score: {avg_bias:.3f}")

high_bias_count = sum(1 for score in ipo_bias_scores if score > 0.5)
print(f"High bias samples: {high_bias_count} ({high_bias_count/len(ipo_bias_scores)*100:.1f}%)")
```

## Performance Optimization

### Caching Length Statistics

```python
# Enable caching for length statistics
data_config = IPODataConfig(
    train_file="data/ipo_train.jsonl",
    calculate_length_statistics=True,
    length_statistics_cache=True,
    ipo_specific=IPOSpecificConfig(
        length_normalization="adaptive"
    )
)

loader = IPODataLoader(config=data_config)

# First load calculates and caches statistics
train_dataset = loader.load_dataset("train")

# Subsequent loads use cached statistics
train_dataset = loader.load_dataset("train")  # Faster
```

### Parallel Length Processing

```python
# Configure parallel processing for length calculations
data_config = IPODataConfig(
    train_file="data/large_ipo_train.jsonl",
    preprocessing_num_workers=16,
    batch_size=2000,
    enable_multiprocessing=True,
    ipo_specific=IPOSpecificConfig(
        length_normalization="sqrt"
    )
)

loader = IPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")
```

## Error Handling

### IPO-Specific Errors

```python
class IPODataError(Exception):
    """Raised when IPO data processing fails."""
    pass

class IPOLengthError(Exception):
    """Raised when length processing fails."""
    pass

# Error handling example
try:
    loader = IPODataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except IPODataError as e:
    print(f"IPO data error: {e}")
except IPOLengthError as e:
    print(f"IPO length error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Length Bias Mitigation
- Choose appropriate normalization strategy
- Monitor length bias scores
- Filter highly biased samples
- Validate normalization effectiveness

### 2. Data Quality
- Ensure diverse response lengths
- Validate preference consistency
- Monitor quality-length correlations
- Check for annotation artifacts

### 3. Normalization Strategy
- Start with sqrt normalization
- Use adaptive for diverse datasets
- Monitor normalization effects
- Validate on held-out data

### 4. Performance
- Cache length statistics
- Use parallel processing
- Enable streaming for large datasets
- Monitor memory usage

### 5. Arabic Language Considerations
- Account for Arabic text characteristics
- Consider script mixing effects
- Validate length calculations for Arabic
- Handle diacritics appropriately

## Troubleshooting

### Common Issues

1. **High length bias scores**
   - Review annotation guidelines
   - Increase filtering threshold
   - Use stronger normalization

2. **Poor normalization results**
   - Try different normalization strategies
   - Check length distribution
   - Validate quality metrics

3. **Memory issues with large datasets**
   - Enable streaming mode
   - Cache length statistics
   - Use parallel processing

4. **Slow length processing**
   - Enable caching
   - Use parallel workers
   - Optimize normalization calculations

## See Also

- [Preference Data Loader](preference_loader.md)
- [IPO Configuration](../config/ipo_config.md)
- [IPO Training](../training/ipo_trainer.md)
- [DPO Data Loader](dpo_loader.md)
- [Base Data Loader](base_loader.md)