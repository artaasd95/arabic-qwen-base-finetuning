# Arabic Dialect Utilities API Documentation

## Overview

The Arabic dialect utilities provide comprehensive support for Arabic dialect detection, text augmentation, and dataset processing. These utilities are integrated into the data processing pipeline and support various Arabic dialects and text normalization techniques.

## Core Classes

### ArabicDialectDetector

Detects Arabic dialects from text input with high accuracy.

```python
from src.data.arabic_dialect_utils import ArabicDialectDetector

detector = ArabicDialectDetector()
```

#### Methods

##### `detect_dialect(text: str) -> Dict[str, Any]`

Detect the dialect of Arabic text.

**Parameters:**
- `text` (str): Arabic text to analyze

**Returns:**
- Dictionary containing:
  - `dialect`: Detected dialect code
  - `confidence`: Confidence score (0.0-1.0)
  - `probabilities`: Probability distribution across all dialects
  - `is_arabic`: Boolean indicating if text is Arabic

**Example:**
```python
detector = ArabicDialectDetector()

# Egyptian Arabic
result = detector.detect_dialect("إزيك يا صاحبي؟")
print(result)
# {
#     'dialect': 'EGY',
#     'confidence': 0.95,
#     'probabilities': {'EGY': 0.95, 'LEV': 0.03, 'GLF': 0.02},
#     'is_arabic': True
# }

# Gulf Arabic
result = detector.detect_dialect("شلونك حبيبي؟")
print(result)
# {
#     'dialect': 'GLF',
#     'confidence': 0.88,
#     'probabilities': {'GLF': 0.88, 'EGY': 0.07, 'LEV': 0.05},
#     'is_arabic': True
# }
```

##### `is_arabic_text(text: str) -> bool`

Check if the given text is Arabic.

**Parameters:**
- `text` (str): Text to check

**Returns:**
- Boolean indicating if text contains Arabic characters

**Example:**
```python
print(detector.is_arabic_text("مرحبا"))  # True
print(detector.is_arabic_text("Hello"))   # False
print(detector.is_arabic_text("مرحبا Hello"))  # True (mixed)
```

##### `get_supported_dialects() -> List[str]`

Get list of supported Arabic dialects.

**Returns:**
- List of dialect codes

**Example:**
```python
dialects = detector.get_supported_dialects()
print(dialects)
# ['MSA', 'EGY', 'LEV', 'GLF', 'MAG', 'IRQ', 'SUD']
```

### ArabicTextAugmenter

Provides text augmentation capabilities for Arabic text, including dialect conversion and normalization.

```python
from src.data.arabic_dialect_utils import ArabicTextAugmenter

augmenter = ArabicTextAugmenter()
```

#### Methods

##### `augment_text(text: str, target_dialect: str = None, **kwargs) -> List[str]`

Generate augmented versions of Arabic text.

**Parameters:**
- `text` (str): Original Arabic text
- `target_dialect` (str, optional): Target dialect for conversion
- `**kwargs`: Additional augmentation options

**Returns:**
- List of augmented text variations

**Example:**
```python
augmenter = ArabicTextAugmenter()

# Basic augmentation
original = "كيف حالك اليوم؟"
augmented = augmenter.augment_text(original)
print(augmented)
# [
#     "كيف حالك اليوم؟",           # Original
#     "كيف الحال اليوم؟",          # Synonym variation
#     "إزيك النهارده؟",            # Egyptian dialect
#     "شلونك اليوم؟",              # Gulf dialect
#     "كيفك اليوم؟"                # Levantine dialect
# ]

# Dialect-specific augmentation
egy_versions = augmenter.augment_text(original, target_dialect="EGY")
print(egy_versions)
# [
#     "إزيك النهارده؟",
#     "إيه أخبارك النهارده؟",
#     "عامل إيه النهارده؟"
# ]
```

##### `normalize_text(text: str, level: str = "basic") -> str`

Normalize Arabic text by removing diacritics, standardizing characters, etc.

**Parameters:**
- `text` (str): Arabic text to normalize
- `level` (str): Normalization level ("basic", "aggressive", "conservative")

**Returns:**
- Normalized Arabic text

**Example:**
```python
# Basic normalization
text = "مَرْحَباً بِكُمْ فِي المَوْقِعِ"
normalized = augmenter.normalize_text(text, level="basic")
print(normalized)
# "مرحبا بكم في الموقع"

# Aggressive normalization
normalized = augmenter.normalize_text(text, level="aggressive")
print(normalized)
# "مرحبا بكم في الموقع"  # Also handles character variants
```

##### `convert_dialect(text: str, source_dialect: str, target_dialect: str) -> str`

Convert text from one Arabic dialect to another.

**Parameters:**
- `text` (str): Source text
- `source_dialect` (str): Source dialect code
- `target_dialect` (str): Target dialect code

**Returns:**
- Converted text in target dialect

**Example:**
```python
# Convert Egyptian to Gulf
egy_text = "إزيك يا صاحبي؟"
gulf_text = augmenter.convert_dialect(egy_text, "EGY", "GLF")
print(gulf_text)
# "شلونك يا صديقي؟"

# Convert Levantine to MSA
lev_text = "كيفك اليوم؟"
msa_text = augmenter.convert_dialect(lev_text, "LEV", "MSA")
print(msa_text)
# "كيف حالك اليوم؟"
```

### ArabicDatasetProcessor

Processes datasets with Arabic dialect handling capabilities.

```python
from src.data.arabic_dialect_utils import ArabicDatasetProcessor

processor = ArabicDatasetProcessor()
```

#### Methods

##### `process_dataset(dataset, config: Dict[str, Any]) -> Dataset`

Process a dataset with Arabic dialect handling.

**Parameters:**
- `dataset`: Input dataset (HuggingFace Dataset or similar)
- `config`: Processing configuration

**Returns:**
- Processed dataset with dialect information and augmentations

**Example:**
```python
from datasets import Dataset

# Sample dataset
data = {
    "text": [
        "مرحبا كيف حالك؟",
        "إزيك يا صاحبي؟",
        "شلونك حبيبي؟"
    ],
    "label": ["greeting", "greeting", "greeting"]
}
dataset = Dataset.from_dict(data)

# Processing configuration
config = {
    "detect_dialects": True,
    "augment_data": True,
    "normalize_text": True,
    "target_dialects": ["MSA", "EGY", "GLF"],
    "augmentation_factor": 2
}

# Process dataset
processed = processor.process_dataset(dataset, config)
print(processed)
# Dataset with additional columns:
# - original_text
# - dialect_info
# - augmented_texts
# - normalized_text
```

##### `add_dialect_labels(dataset, text_column: str = "text") -> Dataset`

Add dialect detection labels to dataset.

**Parameters:**
- `dataset`: Input dataset
- `text_column`: Name of the text column

**Returns:**
- Dataset with dialect labels added

**Example:**
```python
labeled_dataset = processor.add_dialect_labels(dataset, "text")
print(labeled_dataset[0])
# {
#     "text": "إزيك يا صاحبي؟",
#     "label": "greeting",
#     "dialect": "EGY",
#     "dialect_confidence": 0.95,
#     "is_arabic": True
# }
```

##### `augment_dataset(dataset, config: Dict[str, Any]) -> Dataset`

Augment dataset with dialect variations.

**Parameters:**
- `dataset`: Input dataset
- `config`: Augmentation configuration

**Returns:**
- Augmented dataset

**Example:**
```python
aug_config = {
    "augmentation_factor": 3,
    "target_dialects": ["MSA", "EGY", "LEV"],
    "preserve_original": True,
    "balance_dialects": True
}

augmented = processor.augment_dataset(dataset, aug_config)
print(f"Original size: {len(dataset)}")
print(f"Augmented size: {len(augmented)}")
# Original size: 3
# Augmented size: 12  # 3 original + 9 augmented
```

## Configuration Classes

### DialectDetectionConfig

Configuration for dialect detection.

```python
from src.data.arabic_dialect_utils import DialectDetectionConfig

config = DialectDetectionConfig(
    model_name="arabic-dialect-classifier",
    confidence_threshold=0.7,
    supported_dialects=["MSA", "EGY", "LEV", "GLF", "MAG"],
    fallback_dialect="MSA",
    batch_size=32,
    device="auto"
)
```

### AugmentationConfig

Configuration for text augmentation.

```python
from src.data.arabic_dialect_utils import AugmentationConfig

config = AugmentationConfig(
    augmentation_methods=[
        "dialect_conversion",
        "synonym_replacement",
        "text_normalization"
    ],
    target_dialects=["MSA", "EGY", "LEV", "GLF"],
    augmentation_factor=2,
    preserve_original=True,
    balance_dialects=True,
    normalization_level="basic",
    max_augmentations_per_sample=5
)
```

## Utility Functions

### `get_dialect_info(dialect_code: str) -> Dict[str, Any]`

Get detailed information about a specific dialect.

**Parameters:**
- `dialect_code` (str): Dialect code (e.g., "EGY", "LEV")

**Returns:**
- Dictionary with dialect information

**Example:**
```python
from src.data.arabic_dialect_utils import get_dialect_info

info = get_dialect_info("EGY")
print(info)
# {
#     "code": "EGY",
#     "name": "Egyptian Arabic",
#     "native_name": "العربية المصرية",
#     "region": "Egypt",
#     "speakers": "100M+",
#     "characteristics": ["يا", "إيه", "ازاي"],
#     "sample_phrases": {
#         "hello": "إزيك",
#         "how_are_you": "إزيك",
#         "thank_you": "شكرا"
#     }
# }
```

### `validate_arabic_text(text: str) -> Dict[str, Any]`

Validate and analyze Arabic text quality.

**Parameters:**
- `text` (str): Text to validate

**Returns:**
- Validation results

**Example:**
```python
from src.data.arabic_dialect_utils import validate_arabic_text

result = validate_arabic_text("مرحبا كيف حالك؟")
print(result)
# {
#     "is_valid": True,
#     "is_arabic": True,
#     "has_diacritics": False,
#     "character_count": 15,
#     "word_count": 3,
#     "quality_score": 0.95,
#     "issues": []
# }
```

### `convert_text_encoding(text: str, target_encoding: str = "utf-8") -> str`

Convert Arabic text encoding.

**Parameters:**
- `text` (str): Input text
- `target_encoding` (str): Target encoding

**Returns:**
- Text in target encoding

## Integration Examples

### With Data Loading Pipeline

```python
from src.data.data_utils import ArabicDataProcessor
from src.data.arabic_dialect_utils import ArabicDatasetProcessor

# Initialize processors
data_processor = ArabicDataProcessor()
dialect_processor = ArabicDatasetProcessor()

# Load and process dataset
dataset = data_processor.load_dataset("arabic_conversations")

# Add dialect processing
dialect_config = {
    "detect_dialects": True,
    "augment_data": True,
    "target_dialects": ["MSA", "EGY", "LEV", "GLF"],
    "augmentation_factor": 2
}

processed_dataset = dialect_processor.process_dataset(dataset, dialect_config)

# Use in training
train_dataset = processed_dataset.train_test_split(test_size=0.2)
```

### With Training Pipeline

```python
from src.training.sft_trainer import SFTTrainer
from src.data.arabic_dialect_utils import ArabicDatasetProcessor

# Process dataset with dialect augmentation
processor = ArabicDatasetProcessor()
config = {
    "detect_dialects": True,
    "augment_data": True,
    "balance_dialects": True,
    "target_dialects": ["MSA", "EGY", "LEV", "GLF"]
}

augmented_dataset = processor.process_dataset(original_dataset, config)

# Train with augmented data
trainer = SFTTrainer(training_config)
model = trainer.train(augmented_dataset)
```

### Custom Dialect Processing

```python
from src.data.arabic_dialect_utils import (
    ArabicDialectDetector, 
    ArabicTextAugmenter,
    ArabicDatasetProcessor
)

class CustomArabicProcessor:
    def __init__(self):
        self.detector = ArabicDialectDetector()
        self.augmenter = ArabicTextAugmenter()
        self.processor = ArabicDatasetProcessor()
    
    def process_conversation_data(self, conversations):
        """Process conversation data with dialect handling."""
        processed = []
        
        for conv in conversations:
            # Detect dialect
            dialect_info = self.detector.detect_dialect(conv["text"])
            
            # Augment if needed
            if dialect_info["confidence"] > 0.8:
                augmented = self.augmenter.augment_text(
                    conv["text"], 
                    target_dialect="MSA"
                )
                
                for aug_text in augmented:
                    processed.append({
                        "text": aug_text,
                        "original_text": conv["text"],
                        "dialect": dialect_info["dialect"],
                        "confidence": dialect_info["confidence"],
                        "label": conv["label"]
                    })
            else:
                # Keep original if low confidence
                processed.append({
                    "text": conv["text"],
                    "original_text": conv["text"],
                    "dialect": "UNKNOWN",
                    "confidence": dialect_info["confidence"],
                    "label": conv["label"]
                })
        
        return processed
```

## Performance Optimization

### Batch Processing

```python
# Process multiple texts efficiently
detector = ArabicDialectDetector()
texts = ["مرحبا", "إزيك", "شلونك", "كيفك"]

# Batch detection
results = detector.detect_dialect_batch(texts, batch_size=32)
for text, result in zip(texts, results):
    print(f"{text}: {result['dialect']} ({result['confidence']:.2f})")
```

### Memory Optimization

```python
# Memory-efficient dataset processing
processor = ArabicDatasetProcessor()

config = {
    "detect_dialects": True,
    "augment_data": True,
    "batch_size": 1000,  # Process in batches
    "cache_results": True,  # Cache dialect detection
    "streaming": True  # Use streaming for large datasets
}

# Process large dataset efficiently
large_dataset = processor.process_dataset(huge_dataset, config)
```

### Caching

```python
from functools import lru_cache

class CachedDialectDetector(ArabicDialectDetector):
    @lru_cache(maxsize=10000)
    def detect_dialect_cached(self, text: str):
        """Cached dialect detection for repeated texts."""
        return self.detect_dialect(text)

# Use cached detector for better performance
detector = CachedDialectDetector()
```

## Error Handling

### Robust Text Processing

```python
def safe_dialect_detection(text: str) -> Dict[str, Any]:
    """Safely detect dialect with error handling."""
    try:
        detector = ArabicDialectDetector()
        
        # Validate input
        if not text or not isinstance(text, str):
            return {
                "dialect": "UNKNOWN",
                "confidence": 0.0,
                "error": "Invalid input text"
            }
        
        # Check if Arabic
        if not detector.is_arabic_text(text):
            return {
                "dialect": "NON_ARABIC",
                "confidence": 1.0,
                "is_arabic": False
            }
        
        # Detect dialect
        result = detector.detect_dialect(text)
        return result
        
    except Exception as e:
        return {
            "dialect": "ERROR",
            "confidence": 0.0,
            "error": str(e)
        }
```

### Dataset Validation

```python
def validate_arabic_dataset(dataset) -> Dict[str, Any]:
    """Validate Arabic dataset quality."""
    issues = []
    stats = {
        "total_samples": len(dataset),
        "arabic_samples": 0,
        "dialect_distribution": {},
        "quality_issues": []
    }
    
    detector = ArabicDialectDetector()
    
    for i, sample in enumerate(dataset):
        text = sample.get("text", "")
        
        # Check if Arabic
        if detector.is_arabic_text(text):
            stats["arabic_samples"] += 1
            
            # Detect dialect
            result = detector.detect_dialect(text)
            dialect = result["dialect"]
            
            if dialect not in stats["dialect_distribution"]:
                stats["dialect_distribution"][dialect] = 0
            stats["dialect_distribution"][dialect] += 1
            
            # Check quality
            if result["confidence"] < 0.5:
                stats["quality_issues"].append({
                    "index": i,
                    "issue": "Low confidence dialect detection",
                    "confidence": result["confidence"]
                })
        else:
            stats["quality_issues"].append({
                "index": i,
                "issue": "Non-Arabic text detected",
                "text": text[:50]
            })
    
    return stats
```

This comprehensive API documentation covers all aspects of the Arabic dialect utilities, providing developers with the information needed to effectively use these tools in their Arabic language processing pipelines.