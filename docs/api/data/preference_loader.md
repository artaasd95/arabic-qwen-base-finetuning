# Preference Data Loader Documentation

The `PreferenceDataLoader` class serves as the base class for all preference optimization data loaders (DPO, KTO, IPO, CPO) in the Arabic Qwen Base Fine-tuning framework. It provides common functionality for handling preference data and pairwise comparisons.

## Class Overview

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from torch.utils.data import Dataset
from src.data.base_loader import BaseDataLoader

class PreferenceDataLoader(BaseDataLoader, ABC):
    """Base class for preference optimization data loaders."""
```

## Location

**File**: `src/data/preference_loader.py`

## Preference Learning Overview

Preference learning trains models to align with human preferences by learning from comparative data. Instead of single correct answers, the model learns to distinguish between preferred and non-preferred responses.

### Key Concepts

- **Pairwise Preferences**: Comparing two responses to determine which is better
- **Ranking Data**: Ordering multiple responses by preference
- **Binary Feedback**: Simple good/bad labels for responses
- **Multi-aspect Preferences**: Preferences across different dimensions

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: Any,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    **kwargs
):
    """
    Initialize preference data loader.
    
    Args:
        config: Preference data configuration
        tokenizer: Tokenizer for text processing
        enable_cache: Enable data caching
        **kwargs: Additional arguments
    """
```

### Abstract Methods

Subclasses must implement these methods:

```python
@abstractmethod
def _process_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process preference-specific sample format."""
    pass

@abstractmethod
def _validate_preference_format(self, sample: Dict[str, Any]) -> bool:
    """Validate preference-specific format."""
    pass
```

## Core Methods

### Data Processing

#### `process_sample()`

```python
def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a preference data sample.
    
    Args:
        sample: Raw preference sample
        
    Returns:
        Processed sample with standardized format
    """
    # Convert to standard format
    std_sample = self._convert_to_standard_format(sample)
    
    # Apply preference-specific processing
    processed_sample = self._process_preference_sample(std_sample)
    
    # Add metadata
    processed_sample["sample_type"] = "preference"
    processed_sample["preference_method"] = self.get_preference_method()
    
    return processed_sample
```

#### `validate_sample()`

```python
def validate_sample(self, sample: Dict[str, Any]) -> bool:
    """
    Validate a preference data sample.
    
    Args:
        sample: Sample to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Basic validation
        if not self._validate_basic_format(sample):
            return False
        
        # Preference-specific validation
        if not self._validate_preference_format(sample):
            return False
        
        # Content validation
        if not self._validate_content_quality(sample):
            return False
        
        return True
        
    except Exception:
        return False
```

### Format Handling

#### `_convert_to_standard_format()`

```python
def _convert_to_standard_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert sample to standard preference format.
    
    Standard format:
    {
        "prompt": str,
        "chosen": str,
        "rejected": str,
        "system": Optional[str],
        "metadata": Dict[str, Any]
    }
    """
    format_type = self._detect_preference_format(sample)
    
    if format_type == "pairwise":
        return self._convert_pairwise_format(sample)
    elif format_type == "ranking":
        return self._convert_ranking_format(sample)
    elif format_type == "binary":
        return self._convert_binary_format(sample)
    else:
        raise ValueError(f"Unknown preference format: {format_type}")
```

#### `_detect_preference_format()`

```python
def _detect_preference_format(self, sample: Dict[str, Any]) -> str:
    """
    Detect the format of preference data.
    
    Returns:
        Format type: 'pairwise', 'ranking', 'binary'
    """
    if "chosen" in sample and "rejected" in sample:
        return "pairwise"
    elif "responses" in sample and "rankings" in sample:
        return "ranking"
    elif "response" in sample and "label" in sample:
        return "binary"
    else:
        raise ValueError(f"Cannot detect preference format: {sample.keys()}")
```

### Content Validation

#### `_validate_content_quality()`

```python
def _validate_content_quality(self, sample: Dict[str, Any]) -> bool:
    """
    Validate content quality of preference sample.
    
    Checks:
    - Text length constraints
    - Content diversity
    - Language detection
    - Quality metrics
    """
    std_sample = self._convert_to_standard_format(sample)
    
    prompt = std_sample["prompt"]
    chosen = std_sample["chosen"]
    rejected = std_sample["rejected"]
    
    # Length validation
    if not self._validate_lengths(prompt, chosen, rejected):
        return False
    
    # Content diversity
    if not self._validate_diversity(chosen, rejected):
        return False
    
    # Language validation
    if not self._validate_language(prompt, chosen, rejected):
        return False
    
    return True
```

#### `_validate_diversity()`

```python
def _validate_diversity(self, chosen: str, rejected: str) -> bool:
    """
    Validate diversity between chosen and rejected responses.
    
    Args:
        chosen: Preferred response
        rejected: Non-preferred response
        
    Returns:
        True if sufficiently diverse
    """
    # Calculate similarity
    similarity = self._calculate_similarity(chosen, rejected)
    
    # Ensure responses are different enough
    min_diversity = getattr(self.config, 'min_diversity_threshold', 0.1)
    return similarity < (1.0 - min_diversity)
```

### Utility Methods

#### `get_preference_statistics()`

```python
def get_preference_statistics(self, dataset: Dataset) -> Dict[str, Any]:
    """
    Calculate statistics for preference dataset.
    
    Returns:
        Dictionary with preference statistics
    """
    stats = {
        "total_samples": len(dataset),
        "avg_prompt_length": 0,
        "avg_chosen_length": 0,
        "avg_rejected_length": 0,
        "length_ratio_stats": {},
        "diversity_stats": {},
        "quality_metrics": {}
    }
    
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    length_ratios = []
    diversity_scores = []
    
    for sample in dataset:
        prompt_lengths.append(len(sample["prompt"]))
        chosen_lengths.append(len(sample["chosen"]))
        rejected_lengths.append(len(sample["rejected"]))
        
        # Length ratio (chosen vs rejected)
        ratio = len(sample["chosen"]) / max(len(sample["rejected"]), 1)
        length_ratios.append(ratio)
        
        # Diversity score
        diversity = 1.0 - self._calculate_similarity(
            sample["chosen"], sample["rejected"]
        )
        diversity_scores.append(diversity)
    
    # Calculate averages
    stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
    stats["avg_chosen_length"] = sum(chosen_lengths) / len(chosen_lengths)
    stats["avg_rejected_length"] = sum(rejected_lengths) / len(rejected_lengths)
    
    # Length ratio statistics
    stats["length_ratio_stats"] = {
        "mean": sum(length_ratios) / len(length_ratios),
        "min": min(length_ratios),
        "max": max(length_ratios),
        "std": self._calculate_std(length_ratios)
    }
    
    # Diversity statistics
    stats["diversity_stats"] = {
        "mean": sum(diversity_scores) / len(diversity_scores),
        "min": min(diversity_scores),
        "max": max(diversity_scores),
        "std": self._calculate_std(diversity_scores)
    }
    
    return stats
```

## Data Formats

### 1. Pairwise Format

Standard pairwise preference format:

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد ومركزها السياسي والثقافي.", "rejected": "باريس."}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً مثل التعلم والاستدلال وحل المشكلات.", "rejected": "الذكاء الاصطناعي شيء معقد."}
```

### 2. Ranking Format

Multiple responses with rankings:

```jsonl
{
  "prompt": "ما هي فوائد التمرين؟",
  "responses": [
    "التمرين مفيد للصحة.",
    "التمرين له فوائد عديدة للصحة الجسدية والنفسية، منها تقوية العضلات وتحسين الدورة الدموية.",
    "التمرين يساعد على تقوية العضلات وتحسين اللياقة البدنية وتقليل التوتر وتحسين المزاج."
  ],
  "rankings": [3, 1, 2]
}
```

### 3. Binary Feedback Format

Single response with binary label:

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "response": "عاصمة فرنسا هي باريس.", "label": "good"}
{"prompt": "ما هي عاصمة فرنسا؟", "response": "لا أعرف.", "label": "bad"}
```

### 4. Multi-aspect Format

Preferences across multiple dimensions:

```jsonl
{
  "prompt": "اشرح الاحتباس الحراري",
  "chosen": "الاحتباس الحراري هو ظاهرة طبيعية تحدث عندما تحبس الغازات في الغلاف الجوي الحرارة من الشمس...",
  "rejected": "الاحتباس الحراري سيء للبيئة.",
  "aspects": {
    "accuracy": {"chosen": 5, "rejected": 2},
    "completeness": {"chosen": 4, "rejected": 1},
    "clarity": {"chosen": 4, "rejected": 3}
  }
}
```

## Configuration

### Base Preference Configuration

```python
@dataclass
class BasePreferenceConfig:
    """Base configuration for preference data loading."""
    
    # File paths
    train_file: str
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Column names
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    system_column: Optional[str] = "system"
    
    # Alternative column names
    response_column: str = "response"
    label_column: str = "label"
    responses_column: str = "responses"
    rankings_column: str = "rankings"
    
    # Processing parameters
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    truncation_mode: str = "keep_end"
    
    # Quality control
    min_prompt_length: int = 5
    min_response_length: int = 10
    max_length_ratio: float = 5.0
    min_diversity_threshold: float = 0.1
    
    # Data processing
    preprocessing_num_workers: int = 4
    max_samples: Optional[int] = None
    validation_split_percentage: float = 0.1
    
    # Filtering
    filter_duplicates: bool = True
    filter_low_quality: bool = True
    filter_similar_pairs: bool = True
    similarity_threshold: float = 0.9
```

## Usage Examples

### Basic Preference Data Loading

```python
from src.data.dpo_loader import DPODataLoader
from src.config.dpo_config import DPOConfig

# Load configuration
config = DPOConfig.from_yaml("config/dpo_config.yaml")

# Create preference data loader
loader = DPODataLoader(config.data)

# Load preference dataset
train_dataset = loader.load_dataset("train")

# Get preference statistics
stats = loader.get_preference_statistics(train_dataset)
print(f"Dataset statistics: {stats}")

# Inspect samples
for i, sample in enumerate(train_dataset):
    print(f"Sample {i}:")
    print(f"  Prompt: {sample['prompt']}")
    print(f"  Chosen: {sample['chosen']}")
    print(f"  Rejected: {sample['rejected']}")
    if i >= 2:  # Show first 3 samples
        break
```

### Custom Preference Loader

```python
from src.data.preference_loader import PreferenceDataLoader
from torch.utils.data import Dataset

class CustomPreferenceLoader(PreferenceDataLoader):
    """Custom preference data loader."""
    
    def _process_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Custom preference processing."""
        # Add custom processing logic
        processed = sample.copy()
        
        # Add preference strength score
        processed["preference_strength"] = self._calculate_preference_strength(
            sample["chosen"], sample["rejected"]
        )
        
        # Add quality scores
        processed["chosen_quality"] = self._calculate_quality_score(sample["chosen"])
        processed["rejected_quality"] = self._calculate_quality_score(sample["rejected"])
        
        return processed
    
    def _validate_preference_format(self, sample: Dict[str, Any]) -> bool:
        """Custom preference validation."""
        # Check required fields
        required_fields = ["prompt", "chosen", "rejected"]
        if not all(field in sample for field in required_fields):
            return False
        
        # Custom validation logic
        chosen_quality = self._calculate_quality_score(sample["chosen"])
        rejected_quality = self._calculate_quality_score(sample["rejected"])
        
        # Ensure chosen is actually better
        return chosen_quality > rejected_quality
    
    def _calculate_preference_strength(self, chosen: str, rejected: str) -> float:
        """Calculate preference strength between responses."""
        # Implement preference strength calculation
        chosen_score = self._calculate_quality_score(chosen)
        rejected_score = self._calculate_quality_score(rejected)
        return chosen_score - rejected_score
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for text."""
        # Implement quality scoring logic
        # Consider factors like length, complexity, informativeness
        score = 0.0
        
        # Length factor
        length_score = min(len(text) / 100, 1.0)
        score += length_score * 0.3
        
        # Informativeness factor (simple heuristic)
        info_score = len(set(text.split())) / max(len(text.split()), 1)
        score += info_score * 0.7
        
        return score
```

### Data Quality Analysis

```python
# Analyze data quality
loader = DPODataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Get comprehensive statistics
stats = loader.get_preference_statistics(train_dataset)

print("=== Preference Dataset Analysis ===")
print(f"Total samples: {stats['total_samples']}")
print(f"Average prompt length: {stats['avg_prompt_length']:.1f}")
print(f"Average chosen length: {stats['avg_chosen_length']:.1f}")
print(f"Average rejected length: {stats['avg_rejected_length']:.1f}")

print("\n=== Length Ratio Analysis ===")
ratio_stats = stats['length_ratio_stats']
print(f"Mean ratio (chosen/rejected): {ratio_stats['mean']:.2f}")
print(f"Min ratio: {ratio_stats['min']:.2f}")
print(f"Max ratio: {ratio_stats['max']:.2f}")
print(f"Std deviation: {ratio_stats['std']:.2f}")

print("\n=== Diversity Analysis ===")
diversity_stats = stats['diversity_stats']
print(f"Mean diversity: {diversity_stats['mean']:.2f}")
print(f"Min diversity: {diversity_stats['min']:.2f}")
print(f"Max diversity: {diversity_stats['max']:.2f}")

# Check for potential issues
if ratio_stats['mean'] > 2.0:
    print("\n⚠️  Warning: Chosen responses are significantly longer than rejected")
if diversity_stats['mean'] < 0.3:
    print("\n⚠️  Warning: Low diversity between chosen and rejected responses")
```

## Data Preprocessing

### Length Balancing

```python
def balance_response_lengths(self, dataset: Dataset) -> Dataset:
    """
    Balance lengths between chosen and rejected responses.
    
    Args:
        dataset: Input dataset
        
    Returns:
        Balanced dataset
    """
    balanced_samples = []
    
    for sample in dataset:
        chosen_len = len(sample["chosen"])
        rejected_len = len(sample["rejected"])
        
        # Skip if length ratio is too extreme
        ratio = max(chosen_len, rejected_len) / max(min(chosen_len, rejected_len), 1)
        if ratio > self.config.max_length_ratio:
            continue
        
        balanced_samples.append(sample)
    
    return balanced_samples
```

### Duplicate Removal

```python
def remove_duplicates(self, dataset: Dataset) -> Dataset:
    """
    Remove duplicate preference pairs.
    
    Args:
        dataset: Input dataset
        
    Returns:
        Deduplicated dataset
    """
    seen_pairs = set()
    unique_samples = []
    
    for sample in dataset:
        # Create hash of prompt + chosen + rejected
        pair_hash = hash((
            sample["prompt"],
            sample["chosen"],
            sample["rejected"]
        ))
        
        if pair_hash not in seen_pairs:
            seen_pairs.add(pair_hash)
            unique_samples.append(sample)
    
    return unique_samples
```

## Error Handling

### Preference-Specific Errors

```python
class PreferenceDataError(Exception):
    """Raised when preference data is invalid."""
    
    def __init__(self, message: str, sample_index: int = None, sample_data: Dict = None):
        super().__init__(message)
        self.sample_index = sample_index
        self.sample_data = sample_data

class PreferenceValidationError(Exception):
    """Raised when preference validation fails."""
    
    def __init__(self, message: str, validation_errors: List[Dict] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or []
```

### Error Handling Example

```python
try:
    loader = DPODataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except PreferenceDataError as e:
    print(f"Preference data error: {e}")
    if e.sample_index is not None:
        print(f"Error in sample {e.sample_index}: {e.sample_data}")
except PreferenceValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Number of validation errors: {len(e.validation_errors)}")
    
    # Show first few validation errors
    for error in e.validation_errors[:5]:
        print(f"  Sample {error['index']}: {error['message']}")
```

## Best Practices

### 1. Data Quality
- Ensure high-quality human annotations
- Validate preference consistency
- Balance response lengths
- Remove near-duplicate pairs

### 2. Diversity
- Maintain sufficient diversity between chosen/rejected
- Include various types of preferences
- Balance different preference strengths
- Consider multi-aspect preferences

### 3. Validation
- Implement comprehensive validation
- Monitor data quality metrics
- Check for annotation biases
- Validate preference consistency

### 4. Performance
- Use caching for repeated loads
- Enable parallel processing
- Consider streaming for large datasets
- Monitor memory usage

## See Also

- [DPO Data Loader](dpo_loader.md)
- [KTO Data Loader](kto_loader.md)
- [IPO Data Loader](ipo_loader.md)
- [CPO Data Loader](cpo_loader.md)
- [Base Data Loader](base_loader.md)
- [Data Validation](utils/validation.md)