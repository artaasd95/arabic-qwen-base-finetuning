# CPO Data Loader Documentation

The `CPODataLoader` class handles data loading for Contrastive Preference Optimization (CPO) training in the Arabic Qwen Base Fine-tuning framework. It extends the `PreferenceDataLoader` to provide CPO-specific data processing and validation.

## Class Overview

```python
from typing import Dict, List, Optional, Union, Any, Tuple
from torch.utils.data import Dataset
from src.data.preference_loader import PreferenceDataLoader
from src.config.cpo_config import CPODataConfig

class CPODataLoader(PreferenceDataLoader):
    """Data loader for Contrastive Preference Optimization (CPO)."""
```

## Location

**File**: `src/data/cpo_loader.py`

## CPO Overview

Contrastive Preference Optimization (CPO) is a preference learning method that uses contrastive learning principles to improve preference modeling. It leverages multiple negative examples and sophisticated sampling strategies to enhance training effectiveness.

### Key Features

- **Multi-negative sampling**: Uses multiple negative examples per positive
- **Contrastive learning**: Applies contrastive loss for better representation
- **Hard negative mining**: Identifies challenging negative examples
- **Temperature scaling**: Controls contrastive learning dynamics
- **Batch contrastive**: Leverages batch-level contrastive learning
- **Adaptive sampling**: Dynamic negative sampling strategies

### CPO vs DPO

| Aspect | CPO | DPO |
|--------|-----|-----|
| Negative examples | Multiple per positive | One per positive |
| Learning approach | Contrastive | Pairwise |
| Sampling strategy | Adaptive/Hard mining | Random |
| Batch utilization | Cross-sample contrastive | Independent pairs |
| Training stability | Higher with proper tuning | Standard |
| Data efficiency | Higher | Standard |

## Class Structure

### Initialization

```python
def __init__(
    self,
    config: CPODataConfig,
    tokenizer: Optional[Any] = None,
    enable_cache: bool = True,
    **kwargs
):
    """
    Initialize CPO data loader.
    
    Args:
        config: CPO data configuration
        tokenizer: Tokenizer for text processing
        enable_cache: Enable data caching
        **kwargs: Additional arguments
    """
    super().__init__(config, tokenizer, enable_cache, **kwargs)
    self.config = config
    self.cpo_specific_config = getattr(config, 'cpo_specific', None)
    
    # Initialize negative sampling components
    self.negative_sampler = self._initialize_negative_sampler()
    self.hard_negative_miner = self._initialize_hard_negative_miner()
    
    # Contrastive learning components
    self.temperature = self.cpo_specific_config.temperature
    self.contrastive_batch_size = self.cpo_specific_config.contrastive_batch_size
```

### Core Methods

#### `load_dataset()`

```python
def load_dataset(self, split: str = "train") -> Dataset:
    """
    Load CPO dataset for specified split.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        
    Returns:
        Loaded and processed CPO dataset
    """
    # Load raw data
    raw_data = self._load_raw_data(split)
    
    # Process for CPO
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
    
    # Apply CPO-specific processing
    cpo_data = self._apply_cpo_processing(processed_data)
    
    # Generate negative samples
    augmented_data = self._generate_negative_samples(cpo_data)
    
    # Apply contrastive grouping
    contrastive_data = self._create_contrastive_groups(augmented_data)
    
    # Create dataset
    dataset = self._create_dataset(contrastive_data)
    
    self.logger.info(f"Loaded {len(dataset)} CPO samples for {split} split")
    return dataset
```

#### `_process_preference_sample()`

```python
def _process_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process CPO-specific preference sample.
    
    Args:
        sample: Raw preference sample
        
    Returns:
        Processed CPO sample
    """
    processed = sample.copy()
    
    # Ensure standard format
    if "prompt" not in processed:
        processed["prompt"] = self._extract_prompt(sample)
    
    if "chosen" not in processed:
        processed["chosen"] = self._extract_chosen_response(sample)
    
    if "rejected" not in processed:
        processed["rejected"] = self._extract_rejected_response(sample)
    
    # Add CPO-specific fields
    processed["method"] = "cpo"
    processed["preference_type"] = "contrastive"
    
    # Extract multiple negatives if available
    processed["negatives"] = self._extract_multiple_negatives(sample)
    
    # Calculate contrastive features
    processed["positive_embedding"] = self._calculate_response_embedding(processed["chosen"])
    processed["negative_embeddings"] = [
        self._calculate_response_embedding(neg) for neg in processed["negatives"]
    ]
    
    # Calculate contrastive scores
    processed["contrastive_scores"] = self._calculate_contrastive_scores(
        processed["positive_embedding"],
        processed["negative_embeddings"]
    )
    
    # Add quality metrics
    processed["chosen_quality"] = self._calculate_response_quality(processed["chosen"])
    processed["negative_qualities"] = [
        self._calculate_response_quality(neg) for neg in processed["negatives"]
    ]
    
    # Calculate diversity metrics
    processed["negative_diversity"] = self._calculate_negative_diversity(processed["negatives"])
    processed["hard_negative_score"] = self._calculate_hard_negative_score(
        processed["chosen"], processed["negatives"]
    )
    
    return processed
```

#### `_validate_preference_format()`

```python
def _validate_preference_format(self, sample: Dict[str, Any]) -> bool:
    """
    Validate CPO-specific preference format.
    
    Args:
        sample: Sample to validate
        
    Returns:
        True if valid CPO format
    """
    try:
        # Check required fields
        required_fields = ["prompt", "chosen"]
        if not all(field in sample for field in required_fields):
            return False
        
        # Check for negatives (either 'rejected' or 'negatives')
        has_rejected = "rejected" in sample
        has_negatives = "negatives" in sample
        
        if not (has_rejected or has_negatives):
            return False
        
        # Validate field types
        if not isinstance(sample["prompt"], str):
            return False
        if not isinstance(sample["chosen"], str):
            return False
        
        # Validate negatives
        negatives = self._extract_multiple_negatives(sample)
        if len(negatives) < self.config.min_negatives_per_sample:
            return False
        if len(negatives) > self.config.max_negatives_per_sample:
            return False
        
        # Validate content lengths
        prompt = sample["prompt"].strip()
        chosen = sample["chosen"].strip()
        
        if len(prompt) < self.config.min_prompt_length:
            return False
        if len(chosen) < self.config.min_response_length:
            return False
        
        # Validate all negatives
        for negative in negatives:
            if not isinstance(negative, str):
                return False
            if len(negative.strip()) < self.config.min_response_length:
                return False
        
        # CPO-specific validation
        if not self._validate_cpo_contrastive_requirements(chosen, negatives):
            return False
        
        return True
        
    except Exception:
        return False
```

### CPO-Specific Processing

#### `_extract_multiple_negatives()`

```python
def _extract_multiple_negatives(self, sample: Dict[str, Any]) -> List[str]:
    """
    Extract multiple negative examples from sample.
    
    Args:
        sample: Raw sample
        
    Returns:
        List of negative examples
    """
    negatives = []
    
    # Check for explicit negatives list
    if "negatives" in sample:
        if isinstance(sample["negatives"], list):
            negatives.extend(sample["negatives"])
        else:
            negatives.append(sample["negatives"])
    
    # Check for single rejected response
    if "rejected" in sample:
        negatives.append(sample["rejected"])
    
    # Check for multiple rejected responses
    if "rejected_responses" in sample:
        if isinstance(sample["rejected_responses"], list):
            negatives.extend(sample["rejected_responses"])
    
    # Check for alternative response formats
    for key in ["bad_responses", "negative_responses", "alternatives"]:
        if key in sample:
            if isinstance(sample[key], list):
                negatives.extend(sample[key])
            else:
                negatives.append(sample[key])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_negatives = []
    for neg in negatives:
        if isinstance(neg, str) and neg.strip() not in seen:
            unique_negatives.append(neg.strip())
            seen.add(neg.strip())
    
    return unique_negatives
```

#### `_calculate_response_embedding()`

```python
def _calculate_response_embedding(self, response: str) -> List[float]:
    """
    Calculate response embedding for contrastive learning.
    
    Args:
        response: Response text
        
    Returns:
        Response embedding vector
    """
    # Simple embedding calculation (in practice, use proper embeddings)
    # This is a placeholder - real implementation would use sentence transformers
    
    # Tokenize and get basic features
    tokens = response.split()
    
    # Calculate basic features
    features = [
        len(response),  # Length
        len(tokens),    # Token count
        len(set(tokens)),  # Unique tokens
        response.count('؟'),  # Question marks
        response.count('!'),  # Exclamation marks
        response.count('.'),  # Periods
        sum(1 for c in response if c.isdigit()),  # Digit count
        sum(1 for c in response if c.isupper()),  # Uppercase count
    ]
    
    # Normalize features
    max_val = max(features) if features else 1
    normalized_features = [f / max_val for f in features]
    
    # Pad to fixed size
    embedding_size = 128
    while len(normalized_features) < embedding_size:
        normalized_features.append(0.0)
    
    return normalized_features[:embedding_size]
```

#### `_calculate_contrastive_scores()`

```python
def _calculate_contrastive_scores(
    self, 
    positive_embedding: List[float], 
    negative_embeddings: List[List[float]]
) -> List[float]:
    """
    Calculate contrastive scores between positive and negative embeddings.
    
    Args:
        positive_embedding: Positive response embedding
        negative_embeddings: List of negative response embeddings
        
    Returns:
        List of contrastive scores
    """
    import math
    
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    scores = []
    for neg_embedding in negative_embeddings:
        # Calculate similarity
        similarity = cosine_similarity(positive_embedding, neg_embedding)
        
        # Apply temperature scaling
        score = similarity / self.temperature
        scores.append(score)
    
    return scores
```

#### `_calculate_hard_negative_score()`

```python
def _calculate_hard_negative_score(self, chosen: str, negatives: List[str]) -> float:
    """
    Calculate hard negative score for the sample.
    
    Args:
        chosen: Chosen response
        negatives: List of negative responses
        
    Returns:
        Hard negative score (higher = harder negatives)
    """
    chosen_quality = self._calculate_response_quality(chosen)
    negative_qualities = [self._calculate_response_quality(neg) for neg in negatives]
    
    # Calculate quality gaps
    quality_gaps = [chosen_quality - neg_quality for neg_quality in negative_qualities]
    
    # Hard negatives have smaller quality gaps
    min_gap = min(quality_gaps) if quality_gaps else 0.0
    avg_gap = sum(quality_gaps) / len(quality_gaps) if quality_gaps else 0.0
    
    # Calculate hardness score (inverse of quality gap)
    hardness_score = 1.0 / (avg_gap + 0.1)  # Add small epsilon
    
    # Normalize to [0, 1]
    normalized_score = min(hardness_score / 10.0, 1.0)
    
    return normalized_score
```

#### `_calculate_negative_diversity()`

```python
def _calculate_negative_diversity(self, negatives: List[str]) -> float:
    """
    Calculate diversity among negative examples.
    
    Args:
        negatives: List of negative responses
        
    Returns:
        Diversity score (higher = more diverse)
    """
    if len(negatives) < 2:
        return 0.0
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(negatives)):
        for j in range(i + 1, len(negatives)):
            # Simple similarity based on shared words
            words_i = set(negatives[i].lower().split())
            words_j = set(negatives[j].lower().split())
            
            if len(words_i) == 0 and len(words_j) == 0:
                similarity = 1.0
            elif len(words_i) == 0 or len(words_j) == 0:
                similarity = 0.0
            else:
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
    
    # Diversity is inverse of average similarity
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    diversity = 1.0 - avg_similarity
    
    return diversity
```

#### `_apply_cpo_processing()`

```python
def _apply_cpo_processing(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply CPO-specific processing to the dataset.
    
    Args:
        data: List of processed samples
        
    Returns:
        CPO-processed dataset
    """
    processed_data = []
    
    for sample in data:
        # Ensure minimum number of negatives
        if len(sample["negatives"]) < self.config.min_negatives_per_sample:
            # Generate additional negatives if needed
            additional_negatives = self._generate_additional_negatives(
                sample["prompt"], 
                sample["chosen"], 
                sample["negatives"]
            )
            sample["negatives"].extend(additional_negatives)
        
        # Limit maximum number of negatives
        if len(sample["negatives"]) > self.config.max_negatives_per_sample:
            # Select best negatives
            sample["negatives"] = self._select_best_negatives(
                sample["chosen"], 
                sample["negatives"]
            )
        
        # Calculate CPO-specific metrics
        sample["cpo_score"] = self._calculate_cpo_score(sample)
        sample["contrastive_difficulty"] = self._calculate_contrastive_difficulty(sample)
        
        processed_data.append(sample)
    
    return processed_data
```

#### `_generate_negative_samples()`

```python
def _generate_negative_samples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate additional negative samples using various strategies.
    
    Args:
        data: List of CPO-processed samples
        
    Returns:
        Dataset with additional negative samples
    """
    if not self.config.enable_negative_generation:
        return data
    
    augmented_data = []
    
    for sample in data:
        augmented_sample = sample.copy()
        
        # Generate hard negatives
        if self.config.enable_hard_negative_mining:
            hard_negatives = self._mine_hard_negatives(
                sample["prompt"], 
                sample["chosen"], 
                data  # Use other samples as source
            )
            augmented_sample["negatives"].extend(hard_negatives)
        
        # Generate synthetic negatives
        if self.config.enable_synthetic_negatives:
            synthetic_negatives = self._generate_synthetic_negatives(
                sample["prompt"], 
                sample["chosen"]
            )
            augmented_sample["negatives"].extend(synthetic_negatives)
        
        # Remove duplicates and limit count
        augmented_sample["negatives"] = self._deduplicate_and_limit_negatives(
            augmented_sample["negatives"]
        )
        
        augmented_data.append(augmented_sample)
    
    return augmented_data
```

#### `_mine_hard_negatives()`

```python
def _mine_hard_negatives(
    self, 
    prompt: str, 
    chosen: str, 
    all_data: List[Dict[str, Any]]
) -> List[str]:
    """
    Mine hard negatives from other samples in the dataset.
    
    Args:
        prompt: Current prompt
        chosen: Current chosen response
        all_data: All samples in dataset
        
    Returns:
        List of hard negative examples
    """
    hard_negatives = []
    chosen_quality = self._calculate_response_quality(chosen)
    
    # Find responses from other samples that are similar but lower quality
    for other_sample in all_data:
        if other_sample["prompt"] == prompt:
            continue  # Skip same prompt
        
        # Check chosen response from other sample
        other_chosen = other_sample["chosen"]
        other_quality = self._calculate_response_quality(other_chosen)
        
        # Hard negative: similar quality but different context
        quality_diff = abs(chosen_quality - other_quality)
        if quality_diff < 0.2 and other_quality < chosen_quality:
            hard_negatives.append(other_chosen)
        
        # Also check negatives from other samples
        for other_negative in other_sample.get("negatives", []):
            other_neg_quality = self._calculate_response_quality(other_negative)
            if other_neg_quality > chosen_quality * 0.8:  # High-quality negative
                hard_negatives.append(other_negative)
    
    # Limit number of hard negatives
    max_hard_negatives = self.config.max_hard_negatives_per_sample
    return hard_negatives[:max_hard_negatives]
```

#### `_create_contrastive_groups()`

```python
def _create_contrastive_groups(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create contrastive groups for batch-level contrastive learning.
    
    Args:
        data: List of augmented samples
        
    Returns:
        Dataset with contrastive group information
    """
    if not self.config.enable_batch_contrastive:
        return data
    
    # Group samples by similarity for contrastive learning
    grouped_data = []
    group_size = self.contrastive_batch_size
    
    for i in range(0, len(data), group_size):
        group = data[i:i + group_size]
        
        # Add group information to each sample
        for j, sample in enumerate(group):
            sample_with_group = sample.copy()
            sample_with_group["contrastive_group_id"] = i // group_size
            sample_with_group["group_position"] = j
            sample_with_group["group_size"] = len(group)
            
            # Add other samples in group as additional negatives
            group_negatives = []
            for other_sample in group:
                if other_sample != sample:
                    group_negatives.append(other_sample["chosen"])
                    group_negatives.extend(other_sample["negatives"][:2])  # Limit
            
            sample_with_group["group_negatives"] = group_negatives
            grouped_data.append(sample_with_group)
    
    return grouped_data
```

### Negative Sampling Strategies

#### `_initialize_negative_sampler()`

```python
def _initialize_negative_sampler(self):
    """
    Initialize negative sampling component.
    
    Returns:
        Negative sampler instance
    """
    sampling_strategy = self.cpo_specific_config.negative_sampling_strategy
    
    if sampling_strategy == "random":
        return RandomNegativeSampler()
    elif sampling_strategy == "hard":
        return HardNegativeSampler()
    elif sampling_strategy == "adaptive":
        return AdaptiveNegativeSampler()
    elif sampling_strategy == "mixed":
        return MixedNegativeSampler()
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
```

#### `_select_best_negatives()`

```python
def _select_best_negatives(self, chosen: str, negatives: List[str]) -> List[str]:
    """
    Select best negatives based on CPO criteria.
    
    Args:
        chosen: Chosen response
        negatives: List of candidate negatives
        
    Returns:
        Selected best negatives
    """
    if len(negatives) <= self.config.max_negatives_per_sample:
        return negatives
    
    # Score negatives based on multiple criteria
    scored_negatives = []
    chosen_quality = self._calculate_response_quality(chosen)
    
    for negative in negatives:
        neg_quality = self._calculate_response_quality(negative)
        
        # Calculate selection score
        quality_score = 1.0 - abs(chosen_quality - neg_quality)  # Prefer similar quality
        diversity_score = self._calculate_diversity_score(chosen, negative)
        hardness_score = self._calculate_individual_hardness_score(chosen, negative)
        
        # Weighted combination
        total_score = (
            0.4 * quality_score + 
            0.3 * diversity_score + 
            0.3 * hardness_score
        )
        
        scored_negatives.append((negative, total_score))
    
    # Sort by score and select top negatives
    scored_negatives.sort(key=lambda x: x[1], reverse=True)
    selected_negatives = [neg for neg, score in scored_negatives[:self.config.max_negatives_per_sample]]
    
    return selected_negatives
```

## Data Formats

### 1. Standard CPO Format

Basic format with multiple negatives:

```jsonl
{"prompt": "ما هي عاصمة فرنسا؟", "chosen": "عاصمة فرنسا هي باريس، وهي أكبر مدينة في البلاد ومركزها السياسي والثقافي والاقتصادي.", "negatives": ["باريس.", "لا أعرف.", "مدينة في أوروبا."]}
{"prompt": "اشرح مفهوم الذكاء الاصطناعي", "chosen": "الذكاء الاصطناعي هو مجال علمي متعدد التخصصات يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً.", "negatives": ["الذكاء الاصطناعي هو الكمبيوتر.", "لا أعرف ما هو الذكاء الاصطناعي.", "شيء معقد جداً."]}
```

### 2. CPO with Quality Scores

Including quality scores for better negative selection:

```jsonl
{
  "prompt": "ما هي فوائد القراءة؟",
  "chosen": "القراءة لها فوائد عديدة منها تحسين المفردات والمعرفة العامة وتطوير مهارات التفكير النقدي.",
  "negatives": [
    "القراءة مفيدة للعقل.",
    "لا أحب القراءة.",
    "الكتب مملة."
  ],
  "quality_scores": {
    "chosen": 0.85,
    "negatives": [0.6, 0.2, 0.1]
  }
}
```

### 3. CPO with Hard Negatives

Explicitly marked hard negatives:

```jsonl
{
  "prompt": "اشرح قانون الجاذبية",
  "chosen": "قانون الجاذبية العام لنيوتن ينص على أن كل جسم في الكون يجذب كل جسم آخر بقوة تتناسب طردياً مع حاصل ضرب كتلتيهما وعكسياً مع مربع المسافة بينهما.",
  "negatives": [
    "الجاذبية هي القوة التي تجذب الأشياء للأرض.",
    "قانون نيوتن يقول أن الأشياء تسقط."
  ],
  "hard_negatives": [
    "قانون الجاذبية ينص على أن القوة تتناسب مع الكتلة والمسافة.",
    "الجاذبية قوة تعتمد على الكتلة ولكن لا علاقة لها بالمسافة."
  ]
}
```

### 4. CPO with Contrastive Groups

Grouped samples for batch contrastive learning:

```jsonl
{
  "prompt": "ما هو الاحتباس الحراري؟",
  "chosen": "الاحتباس الحراري ظاهرة طبيعية تحدث عندما تحبس غازات الدفيئة الحرارة في الغلاف الجوي.",
  "negatives": ["الاحتباس الحراري سيء.", "لا أعرف."],
  "contrastive_group": "climate_science",
  "group_id": 1
}
{
  "prompt": "كيف يؤثر الاحتباس الحراري على البيئة؟",
  "chosen": "الاحتباس الحراري يؤدي إلى ارتفاع درجات الحرارة وذوبان الأنهار الجليدية وتغير أنماط الطقس.",
  "negatives": ["يجعل الطقس حاراً.", "لا يؤثر كثيراً."],
  "contrastive_group": "climate_science",
  "group_id": 1
}
```

### 5. CPO with Metadata

Rich metadata for advanced processing:

```jsonl
{
  "prompt": "اشرح مفهوم البرمجة الكائنية",
  "chosen": "البرمجة الكائنية هي نموذج برمجي يعتمد على مفهوم الكائنات التي تحتوي على بيانات وطرق للتعامل مع هذه البيانات.",
  "negatives": [
    "البرمجة الكائنية نوع من البرمجة.",
    "لا أفهم البرمجة.",
    "الكائنات في البرمجة معقدة."
  ],
  "metadata": {
    "domain": "programming",
    "difficulty": "intermediate",
    "negative_types": ["vague", "irrelevant", "partial"],
    "contrastive_difficulty": 0.7,
    "diversity_score": 0.8
  }
}
```

## Configuration

### CPO Data Configuration

```python
@dataclass
class CPODataConfig(BasePreferenceConfig):
    """Configuration for CPO data loading."""
    
    # Negative sampling
    min_negatives_per_sample: int = 2
    max_negatives_per_sample: int = 8
    enable_negative_generation: bool = True
    
    # Hard negative mining
    enable_hard_negative_mining: bool = True
    max_hard_negatives_per_sample: int = 3
    hard_negative_similarity_threshold: float = 0.7
    
    # Synthetic negatives
    enable_synthetic_negatives: bool = False
    synthetic_negative_strategies: List[str] = field(default_factory=lambda: ["corruption", "truncation"])
    
    # Contrastive learning
    enable_batch_contrastive: bool = True
    contrastive_batch_size: int = 16
    temperature: float = 0.1
    
    # Quality filtering
    min_negative_quality: float = 0.1
    max_negative_quality: float = 0.9
    quality_diversity_weight: float = 0.5
    
    # CPO-specific parameters
    cpo_specific: CPOSpecificConfig = field(default_factory=CPOSpecificConfig)

@dataclass
class CPOSpecificConfig:
    """CPO-specific configuration."""
    
    # Negative sampling strategy
    negative_sampling_strategy: str = "adaptive"  # 'random', 'hard', 'adaptive', 'mixed'
    
    # Contrastive learning
    contrastive_loss_weight: float = 1.0
    margin: float = 0.5
    
    # Hard negative mining
    hard_negative_ratio: float = 0.3
    mining_strategy: str = "similarity"  # 'similarity', 'quality', 'mixed'
    
    # Batch processing
    enable_cross_batch_negatives: bool = True
    max_cross_batch_negatives: int = 5
```

## Usage Examples

### Basic CPO Data Loading

```python
from src.data.cpo_loader import CPODataLoader
from src.config.cpo_config import CPOConfig
from transformers import AutoTokenizer

# Load configuration
config = CPOConfig.from_yaml("config/cpo_config.yaml")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

# Create CPO data loader
loader = CPODataLoader(
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
print(f"Number of negatives: {len(sample['negatives'])}")
print(f"Negatives: {sample['negatives']}")
print(f"Hard negative score: {sample.get('hard_negative_score', 'N/A')}")
print(f"Negative diversity: {sample.get('negative_diversity', 'N/A')}")
```

### Custom Negative Sampling

```python
from src.data.cpo_loader import CPODataLoader
from src.config.cpo_config import CPODataConfig, CPOSpecificConfig

# Configure custom negative sampling
cpo_specific = CPOSpecificConfig(
    negative_sampling_strategy="mixed",
    hard_negative_ratio=0.5,
    mining_strategy="quality",
    contrastive_loss_weight=1.5
)

data_config = CPODataConfig(
    train_file="data/cpo_train.jsonl",
    validation_file="data/cpo_val.jsonl",
    min_negatives_per_sample=3,
    max_negatives_per_sample=6,
    enable_hard_negative_mining=True,
    max_hard_negatives_per_sample=2,
    cpo_specific=cpo_specific
)

loader = CPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")

# Analyze negative sampling results
negative_counts = [len(sample['negatives']) for sample in train_dataset]
hard_negative_scores = [sample.get('hard_negative_score', 0) for sample in train_dataset]

print(f"Average negatives per sample: {sum(negative_counts)/len(negative_counts):.2f}")
print(f"Average hard negative score: {sum(hard_negative_scores)/len(hard_negative_scores):.3f}")

# Check negative quality distribution
for i, sample in enumerate(train_dataset[:3]):
    print(f"\nSample {i}:")
    print(f"  Chosen quality: {sample.get('chosen_quality', 'N/A')}")
    print(f"  Negative qualities: {sample.get('negative_qualities', 'N/A')}")
    print(f"  Contrastive scores: {sample.get('contrastive_scores', 'N/A')}")
```

### Contrastive Learning Analysis

```python
# Analyze contrastive learning setup
loader = CPODataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Collect contrastive metrics
contrastive_scores = []
diversity_scores = []
group_sizes = []

for sample in train_dataset:
    if 'contrastive_scores' in sample:
        contrastive_scores.extend(sample['contrastive_scores'])
    if 'negative_diversity' in sample:
        diversity_scores.append(sample['negative_diversity'])
    if 'group_size' in sample:
        group_sizes.append(sample['group_size'])

print("=== Contrastive Learning Analysis ===")
print(f"Dataset size: {len(train_dataset)}")
print(f"Total contrastive pairs: {len(contrastive_scores)}")
print(f"Average contrastive score: {sum(contrastive_scores)/len(contrastive_scores):.3f}")
print(f"Average negative diversity: {sum(diversity_scores)/len(diversity_scores):.3f}")

if group_sizes:
    print(f"Average group size: {sum(group_sizes)/len(group_sizes):.1f}")
    print(f"Number of contrastive groups: {len(set(group_sizes))}")

# Analyze score distribution
low_scores = sum(1 for score in contrastive_scores if score < -0.5)
high_scores = sum(1 for score in contrastive_scores if score > 0.5)
print(f"\nScore distribution:")
print(f"  Low scores (<-0.5): {low_scores} ({low_scores/len(contrastive_scores)*100:.1f}%)")
print(f"  High scores (>0.5): {high_scores} ({high_scores/len(contrastive_scores)*100:.1f}%)")
```

### Hard Negative Mining Analysis

```python
# Analyze hard negative mining effectiveness
loader = CPODataLoader(config.data)
train_dataset = loader.load_dataset("train")

# Collect hard negative metrics
hard_negative_counts = []
hard_negative_scores = []
quality_differences = []

for sample in train_dataset:
    negatives = sample.get('negatives', [])
    hard_negatives = [neg for neg in negatives if 'hard_negative' in str(neg).lower()]
    
    hard_negative_counts.append(len(hard_negatives))
    
    if 'hard_negative_score' in sample:
        hard_negative_scores.append(sample['hard_negative_score'])
    
    # Calculate quality differences
    chosen_quality = sample.get('chosen_quality', 0)
    negative_qualities = sample.get('negative_qualities', [])
    
    for neg_quality in negative_qualities:
        quality_differences.append(chosen_quality - neg_quality)

print("=== Hard Negative Mining Analysis ===")
print(f"Average hard negatives per sample: {sum(hard_negative_counts)/len(hard_negative_counts):.2f}")
print(f"Average hard negative score: {sum(hard_negative_scores)/len(hard_negative_scores):.3f}")
print(f"Average quality difference: {sum(quality_differences)/len(quality_differences):.3f}")

# Analyze quality gap distribution
small_gaps = sum(1 for diff in quality_differences if 0 < diff < 0.2)
medium_gaps = sum(1 for diff in quality_differences if 0.2 <= diff < 0.5)
large_gaps = sum(1 for diff in quality_differences if diff >= 0.5)

print(f"\nQuality gap distribution:")
print(f"  Small gaps (0-0.2): {small_gaps} ({small_gaps/len(quality_differences)*100:.1f}%)")
print(f"  Medium gaps (0.2-0.5): {medium_gaps} ({medium_gaps/len(quality_differences)*100:.1f}%)")
print(f"  Large gaps (0.5+): {large_gaps} ({large_gaps/len(quality_differences)*100:.1f}%)")

# Check for effective hard negatives (small quality gaps)
effective_hard_negatives = sum(1 for diff in quality_differences if 0 < diff < 0.3)
print(f"\nEffective hard negatives: {effective_hard_negatives} ({effective_hard_negatives/len(quality_differences)*100:.1f}%)")
```

### Custom CPO Loader with Advanced Features

```python
class AdvancedCPODataLoader(CPODataLoader):
    """Advanced CPO loader with custom features."""
    
    def _generate_synthetic_negatives(self, prompt: str, chosen: str) -> List[str]:
        """Generate synthetic negatives using various corruption strategies."""
        synthetic_negatives = []
        
        # Strategy 1: Truncation
        words = chosen.split()
        if len(words) > 3:
            truncated = ' '.join(words[:len(words)//2])
            synthetic_negatives.append(truncated)
        
        # Strategy 2: Word shuffling
        if len(words) > 2:
            import random
            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            shuffled = ' '.join(shuffled_words)
            synthetic_negatives.append(shuffled)
        
        # Strategy 3: Negation injection
        negation_words = ['لا', 'ليس', 'غير']
        for neg_word in negation_words:
            if neg_word not in chosen:
                negated = f"{neg_word} {chosen}"
                synthetic_negatives.append(negated)
                break
        
        # Strategy 4: Generic response
        generic_responses = [
            "لا أعرف الإجابة على هذا السؤال.",
            "هذا سؤال صعب.",
            "أحتاج لمزيد من المعلومات."
        ]
        synthetic_negatives.extend(generic_responses[:1])
        
        return synthetic_negatives[:3]  # Limit synthetic negatives
    
    def _calculate_advanced_contrastive_scores(
        self, 
        positive_embedding: List[float], 
        negative_embeddings: List[List[float]]
    ) -> Dict[str, List[float]]:
        """Calculate advanced contrastive scores with multiple metrics."""
        scores = {
            'cosine_similarity': [],
            'euclidean_distance': [],
            'manhattan_distance': [],
            'contrastive_score': []
        }
        
        for neg_embedding in negative_embeddings:
            # Cosine similarity
            cos_sim = self._cosine_similarity(positive_embedding, neg_embedding)
            scores['cosine_similarity'].append(cos_sim)
            
            # Euclidean distance
            euclidean = self._euclidean_distance(positive_embedding, neg_embedding)
            scores['euclidean_distance'].append(euclidean)
            
            # Manhattan distance
            manhattan = self._manhattan_distance(positive_embedding, neg_embedding)
            scores['manhattan_distance'].append(manhattan)
            
            # Combined contrastive score
            contrastive = (cos_sim - euclidean/10 + manhattan/100) / self.temperature
            scores['contrastive_score'].append(contrastive)
        
        return scores
    
    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    
    def _manhattan_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Manhattan distance between two vectors."""
        return sum(abs(x - y) for x, y in zip(a, b))
    
    def _adaptive_negative_selection(self, chosen: str, candidates: List[str]) -> List[str]:
        """Adaptively select negatives based on training progress."""
        # This would adapt based on model performance
        # For now, implement a simple adaptive strategy
        
        chosen_quality = self._calculate_response_quality(chosen)
        
        # Early training: easier negatives
        # Later training: harder negatives
        training_progress = getattr(self, 'training_progress', 0.0)  # 0.0 to 1.0
        
        selected_negatives = []
        for candidate in candidates:
            candidate_quality = self._calculate_response_quality(candidate)
            quality_gap = chosen_quality - candidate_quality
            
            # Adaptive threshold based on training progress
            min_gap = 0.1 + (0.4 * (1 - training_progress))  # Start easy, get harder
            max_gap = 0.8 - (0.3 * training_progress)  # Avoid too easy negatives later
            
            if min_gap <= quality_gap <= max_gap:
                selected_negatives.append(candidate)
        
        return selected_negatives

# Use advanced loader
advanced_loader = AdvancedCPODataLoader(config=data_config)
train_dataset = advanced_loader.load_dataset("train")
```

## Performance Optimization

### Caching Embeddings

```python
# Enable embedding caching for faster processing
data_config = CPODataConfig(
    train_file="data/cpo_train.jsonl",
    enable_cache=True,
    cache_embeddings=True,
    embedding_cache_size=10000,
    cpo_specific=CPOSpecificConfig(
        negative_sampling_strategy="adaptive"
    )
)

loader = CPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")
```

### Parallel Negative Generation

```python
# Configure parallel processing for negative generation
data_config = CPODataConfig(
    train_file="data/large_cpo_train.jsonl",
    preprocessing_num_workers=16,
    batch_size=1000,
    enable_multiprocessing=True,
    enable_negative_generation=True,
    max_negatives_per_sample=5,
    cpo_specific=CPOSpecificConfig(
        negative_sampling_strategy="mixed"
    )
)

loader = CPODataLoader(config=data_config)
train_dataset = loader.load_dataset("train")
```

## Error Handling

### CPO-Specific Errors

```python
class CPODataError(Exception):
    """Raised when CPO data processing fails."""
    pass

class CPONegativeError(Exception):
    """Raised when negative sampling fails."""
    pass

class CPOContrastiveError(Exception):
    """Raised when contrastive processing fails."""
    pass

# Error handling example
try:
    loader = CPODataLoader(config.data)
    train_dataset = loader.load_dataset("train")
except CPODataError as e:
    print(f"CPO data error: {e}")
except CPONegativeError as e:
    print(f"CPO negative sampling error: {e}")
except CPOContrastiveError as e:
    print(f"CPO contrastive error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Negative Sampling
- Use diverse negative sampling strategies
- Balance hard and easy negatives
- Monitor negative quality distribution
- Avoid overly similar negatives

### 2. Contrastive Learning
- Tune temperature parameter carefully
- Use appropriate batch sizes
- Monitor contrastive score distribution
- Balance positive and negative examples

### 3. Data Quality
- Ensure high-quality positive examples
- Validate negative diversity
- Check for annotation consistency
- Monitor hard negative effectiveness

### 4. Performance
- Cache embeddings and computations
- Use parallel processing
- Enable streaming for large datasets
- Monitor memory usage

### 5. Arabic Language Considerations
- Account for Arabic text characteristics
- Handle diacritics appropriately
- Consider cultural context in negatives
- Validate Arabic-specific quality metrics

## Troubleshooting

### Common Issues

1. **Poor negative quality**
   - Review negative generation strategies
   - Increase quality thresholds
   - Use hard negative mining

2. **Low contrastive scores**
   - Adjust temperature parameter
   - Check embedding quality
   - Validate negative diversity

3. **Memory issues with large datasets**
   - Enable streaming mode
   - Cache embeddings efficiently
   - Use parallel processing

4. **Slow negative generation**
   - Enable caching
   - Use parallel workers
   - Optimize sampling algorithms

## See Also

- [Preference Data Loader](preference_loader.md)
- [CPO Configuration](../config/cpo_config.md)
- [CPO Training](../training/cpo_trainer.md)
- [DPO Data Loader](dpo_loader.md)
- [Base Data Loader](base_loader.md)