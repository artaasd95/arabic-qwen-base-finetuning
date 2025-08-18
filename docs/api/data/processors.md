# Data Processors Documentation

The data processors module provides specialized components for processing and transforming data in the Arabic Qwen Base Fine-tuning framework. These processors handle Arabic text normalization, data augmentation, quality filtering, and format conversion.

## Overview

Data processors are modular components that can be chained together to create sophisticated data processing pipelines. They are designed to work with different training methods and handle Arabic language-specific requirements.

## Location

**File**: `src/data/processors.py`

## Core Components

### Base Processor

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class BaseProcessor(ABC):
    """Base class for all data processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.processed_count = 0
        self.error_count = 0
    
    @abstractmethod
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample."""
        pass
    
    def process_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples."""
        processed_samples = []
        for sample in samples:
            try:
                processed_sample = self.process(sample)
                processed_samples.append(processed_sample)
                self.processed_count += 1
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Error processing sample: {e}")
                continue
        return processed_samples
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": self.processed_count / max(self.processed_count + self.error_count, 1)
        }
```

## Text Processing

### Arabic Text Normalizer

```python
import re
from typing import Dict, Any

class ArabicTextNormalizer(BaseProcessor):
    """Normalizes Arabic text for consistent processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Normalization options
        self.remove_diacritics = config.get('remove_diacritics', True)
        self.normalize_alef = config.get('normalize_alef', True)
        self.normalize_teh = config.get('normalize_teh', True)
        self.normalize_yeh = config.get('normalize_yeh', True)
        self.remove_tatweel = config.get('remove_tatweel', True)
        self.normalize_spaces = config.get('normalize_spaces', True)
        
        # Arabic character mappings
        self.alef_variants = ['أ', 'إ', 'آ', 'ا']
        self.teh_variants = ['ة', 'ت']
        self.yeh_variants = ['ي', 'ى']
        
        # Diacritics pattern
        self.diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        
        # Tatweel character
        self.tatweel = 'ـ'
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Arabic text in the sample."""
        processed_sample = sample.copy()
        
        # Process text fields
        text_fields = ['prompt', 'chosen', 'rejected', 'response', 'instruction', 'input', 'output']
        
        for field in text_fields:
            if field in processed_sample and isinstance(processed_sample[field], str):
                processed_sample[field] = self._normalize_text(processed_sample[field])
        
        # Process list fields (like negatives)
        list_fields = ['negatives', 'alternatives', 'rejected_responses']
        for field in list_fields:
            if field in processed_sample and isinstance(processed_sample[field], list):
                processed_sample[field] = [
                    self._normalize_text(text) if isinstance(text, str) else text
                    for text in processed_sample[field]
                ]
        
        return processed_sample
    
    def _normalize_text(self, text: str) -> str:
        """Apply Arabic text normalization."""
        if not text:
            return text
        
        normalized = text
        
        # Remove diacritics
        if self.remove_diacritics:
            normalized = self.diacritics_pattern.sub('', normalized)
        
        # Normalize Alef variants
        if self.normalize_alef:
            for variant in self.alef_variants[1:]:  # Skip the first one (target)
                normalized = normalized.replace(variant, self.alef_variants[0])
        
        # Normalize Teh Marbuta
        if self.normalize_teh:
            normalized = normalized.replace(self.teh_variants[0], self.teh_variants[1])
        
        # Normalize Yeh variants
        if self.normalize_yeh:
            normalized = normalized.replace(self.yeh_variants[1], self.yeh_variants[0])
        
        # Remove Tatweel
        if self.remove_tatweel:
            normalized = normalized.replace(self.tatweel, '')
        
        # Normalize spaces
        if self.normalize_spaces:
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
```

### Text Quality Filter

```python
class TextQualityFilter(BaseProcessor):
    """Filters samples based on text quality metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Quality thresholds
        self.min_length = config.get('min_length', 10)
        self.max_length = config.get('max_length', 2000)
        self.min_words = config.get('min_words', 3)
        self.max_repetition_ratio = config.get('max_repetition_ratio', 0.5)
        self.min_arabic_ratio = config.get('min_arabic_ratio', 0.7)
        self.max_special_char_ratio = config.get('max_special_char_ratio', 0.2)
        
        # Arabic character range
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        self.special_char_pattern = re.compile(r'[^\w\s\u0600-\u06FF]')
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sample based on quality metrics."""
        # Check all text fields
        text_fields = ['prompt', 'chosen', 'rejected', 'response']
        
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                if not self._is_quality_text(sample[field]):
                    # Mark sample for filtering
                    sample['_filter_reason'] = f'Low quality {field}'
                    sample['_should_filter'] = True
                    return sample
        
        # Check negatives if present
        if 'negatives' in sample:
            for i, negative in enumerate(sample['negatives']):
                if isinstance(negative, str) and not self._is_quality_text(negative):
                    # Remove low-quality negative
                    sample['negatives'].pop(i)
        
        sample['_should_filter'] = False
        return sample
    
    def _is_quality_text(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Length checks
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            return False
        
        # Repetition check
        if self._calculate_repetition_ratio(text) > self.max_repetition_ratio:
            return False
        
        # Arabic content check
        if self._calculate_arabic_ratio(text) < self.min_arabic_ratio:
            return False
        
        # Special character check
        if self._calculate_special_char_ratio(text) > self.max_special_char_ratio:
            return False
        
        return True
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated words."""
        words = text.lower().split()
        if len(words) <= 1:
            return 0.0
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        return repeated_words / len(words)
    
    def _calculate_arabic_ratio(self, text: str) -> float:
        """Calculate ratio of Arabic characters."""
        if not text:
            return 0.0
        
        arabic_chars = len(self.arabic_pattern.findall(text))
        total_chars = len([c for c in text if not c.isspace()])
        
        return arabic_chars / max(total_chars, 1)
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters."""
        if not text:
            return 0.0
        
        special_chars = len(self.special_char_pattern.findall(text))
        total_chars = len(text)
        
        return special_chars / max(total_chars, 1)
```

## Data Augmentation

### Arabic Data Augmenter

```python
import random
from typing import List

class ArabicDataAugmenter(BaseProcessor):
    """Augments Arabic text data using various techniques."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Augmentation options
        self.enable_synonym_replacement = config.get('enable_synonym_replacement', True)
        self.enable_back_translation = config.get('enable_back_translation', False)
        self.enable_paraphrasing = config.get('enable_paraphrasing', True)
        self.enable_noise_injection = config.get('enable_noise_injection', False)
        
        # Augmentation probabilities
        self.synonym_prob = config.get('synonym_prob', 0.1)
        self.paraphrase_prob = config.get('paraphrase_prob', 0.05)
        self.noise_prob = config.get('noise_prob', 0.02)
        
        # Arabic synonyms dictionary (simplified)
        self.synonyms = {
            'كبير': ['ضخم', 'عظيم', 'هائل'],
            'صغير': ['ضئيل', 'قليل', 'محدود'],
            'جميل': ['رائع', 'حسن', 'بديع'],
            'سيء': ['رديء', 'قبيح', 'مؤذي'],
            'سريع': ['عاجل', 'مستعجل', 'فوري'],
            'بطيء': ['متأني', 'هادئ', 'متمهل']
        }
        
        # Paraphrasing templates
        self.paraphrase_templates = [
            lambda x: f"بمعنى آخر، {x}",
            lambda x: f"يمكن القول أن {x}",
            lambda x: f"من الممكن أن نقول {x}",
            lambda x: f"بعبارة أخرى، {x}"
        ]
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the sample with additional variations."""
        augmented_sample = sample.copy()
        
        # Create augmented versions
        augmented_versions = []
        
        # Original sample
        augmented_versions.append(sample)
        
        # Synonym replacement
        if self.enable_synonym_replacement:
            synonym_version = self._apply_synonym_replacement(sample)
            if synonym_version != sample:
                augmented_versions.append(synonym_version)
        
        # Paraphrasing
        if self.enable_paraphrasing:
            paraphrase_version = self._apply_paraphrasing(sample)
            if paraphrase_version != sample:
                augmented_versions.append(paraphrase_version)
        
        # Noise injection
        if self.enable_noise_injection:
            noise_version = self._apply_noise_injection(sample)
            if noise_version != sample:
                augmented_versions.append(noise_version)
        
        # Add augmented versions to sample
        if len(augmented_versions) > 1:
            augmented_sample['_augmented_versions'] = augmented_versions[1:]  # Exclude original
            augmented_sample['_augmentation_count'] = len(augmented_versions) - 1
        
        return augmented_sample
    
    def _apply_synonym_replacement(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Replace words with synonyms."""
        augmented = sample.copy()
        
        text_fields = ['prompt', 'chosen', 'rejected', 'response']
        for field in text_fields:
            if field in augmented and isinstance(augmented[field], str):
                augmented[field] = self._replace_synonyms(augmented[field])
        
        return augmented
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with their synonyms."""
        words = text.split()
        augmented_words = []
        
        for word in words:
            if word in self.synonyms and random.random() < self.synonym_prob:
                synonym = random.choice(self.synonyms[word])
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def _apply_paraphrasing(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply paraphrasing templates."""
        augmented = sample.copy()
        
        # Apply to response fields
        response_fields = ['chosen', 'rejected', 'response']
        for field in response_fields:
            if field in augmented and isinstance(augmented[field], str):
                if random.random() < self.paraphrase_prob:
                    template = random.choice(self.paraphrase_templates)
                    augmented[field] = template(augmented[field])
        
        return augmented
    
    def _apply_noise_injection(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Inject small amounts of noise."""
        augmented = sample.copy()
        
        text_fields = ['prompt', 'chosen', 'rejected', 'response']
        for field in text_fields:
            if field in augmented and isinstance(augmented[field], str):
                augmented[field] = self._inject_noise(augmented[field])
        
        return augmented
    
    def _inject_noise(self, text: str) -> str:
        """Inject character-level noise."""
        chars = list(text)
        noisy_chars = []
        
        for char in chars:
            if random.random() < self.noise_prob:
                # Small chance to duplicate or skip character
                if random.random() < 0.5:
                    noisy_chars.append(char)  # Duplicate
                    noisy_chars.append(char)
                # else: skip character
            else:
                noisy_chars.append(char)
        
        return ''.join(noisy_chars)
```

## Format Conversion

### Format Converter

```python
class FormatConverter(BaseProcessor):
    """Converts between different data formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.target_format = config.get('target_format', 'standard')
        self.source_format = config.get('source_format', 'auto')
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sample to target format."""
        # Detect source format if auto
        if self.source_format == 'auto':
            source_format = self._detect_format(sample)
        else:
            source_format = self.source_format
        
        # Convert to target format
        if source_format == self.target_format:
            return sample
        
        converter_method = f'_convert_{source_format}_to_{self.target_format}'
        if hasattr(self, converter_method):
            return getattr(self, converter_method)(sample)
        else:
            raise ValueError(f"Conversion from {source_format} to {self.target_format} not supported")
    
    def _detect_format(self, sample: Dict[str, Any]) -> str:
        """Detect the format of the sample."""
        # Conversation format
        if 'conversation' in sample or 'messages' in sample:
            return 'conversation'
        
        # Instruction format
        if 'instruction' in sample and 'input' in sample and 'output' in sample:
            return 'instruction'
        
        # Preference format
        if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
            return 'preference'
        
        # Chat format
        if 'prompt' in sample and 'response' in sample:
            return 'chat'
        
        return 'unknown'
    
    def _convert_conversation_to_standard(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert conversation format to standard format."""
        conversation = sample.get('conversation', sample.get('messages', []))
        
        # Extract prompt and response
        prompt_parts = []
        response = ""
        
        for message in conversation:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role in ['user', 'human']:
                prompt_parts.append(content)
            elif role in ['assistant', 'ai', 'bot']:
                response = content
                break
        
        return {
            'prompt': ' '.join(prompt_parts),
            'response': response,
            '_original_format': 'conversation',
            **{k: v for k, v in sample.items() if k not in ['conversation', 'messages']}
        }
    
    def _convert_instruction_to_standard(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert instruction format to standard format."""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        
        # Combine instruction and input as prompt
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        
        return {
            'prompt': prompt,
            'response': output_text,
            '_original_format': 'instruction',
            **{k: v for k, v in sample.items() if k not in ['instruction', 'input', 'output']}
        }
    
    def _convert_preference_to_standard(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert preference format to standard format."""
        # Already in preference format, just ensure consistency
        return {
            'prompt': sample.get('prompt', ''),
            'chosen': sample.get('chosen', ''),
            'rejected': sample.get('rejected', ''),
            '_original_format': 'preference',
            **{k: v for k, v in sample.items() if k not in ['prompt', 'chosen', 'rejected']}
        }
    
    def _convert_chat_to_standard(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chat format to standard format."""
        return {
            'prompt': sample.get('prompt', ''),
            'response': sample.get('response', ''),
            '_original_format': 'chat',
            **{k: v for k, v in sample.items() if k not in ['prompt', 'response']}
        }
```

## Validation

### Data Validator

```python
class DataValidator(BaseProcessor):
    """Validates data samples for consistency and completeness."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.required_fields = config.get('required_fields', ['prompt'])
        self.optional_fields = config.get('optional_fields', [])
        self.validation_rules = config.get('validation_rules', {})
        
        # Validation statistics
        self.validation_errors = []
        self.field_errors = {}
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the sample."""
        validated_sample = sample.copy()
        validation_errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in sample:
                validation_errors.append(f"Missing required field: {field}")
            elif not sample[field] or (isinstance(sample[field], str) and not sample[field].strip()):
                validation_errors.append(f"Empty required field: {field}")
        
        # Apply validation rules
        for field, rules in self.validation_rules.items():
            if field in sample:
                field_errors = self._validate_field(sample[field], rules, field)
                validation_errors.extend(field_errors)
        
        # Add validation results to sample
        validated_sample['_validation_errors'] = validation_errors
        validated_sample['_is_valid'] = len(validation_errors) == 0
        
        # Update statistics
        self.validation_errors.extend(validation_errors)
        for error in validation_errors:
            field = error.split(':')[0] if ':' in error else 'general'
            self.field_errors[field] = self.field_errors.get(field, 0) + 1
        
        return validated_sample
    
    def _validate_field(self, value: Any, rules: Dict[str, Any], field_name: str) -> List[str]:
        """Validate a single field against rules."""
        errors = []
        
        # Type validation
        if 'type' in rules:
            expected_type = rules['type']
            if expected_type == 'string' and not isinstance(value, str):
                errors.append(f"{field_name}: Expected string, got {type(value).__name__}")
            elif expected_type == 'list' and not isinstance(value, list):
                errors.append(f"{field_name}: Expected list, got {type(value).__name__}")
            elif expected_type == 'dict' and not isinstance(value, dict):
                errors.append(f"{field_name}: Expected dict, got {type(value).__name__}")
        
        # Length validation for strings
        if isinstance(value, str):
            if 'min_length' in rules and len(value) < rules['min_length']:
                errors.append(f"{field_name}: Length {len(value)} below minimum {rules['min_length']}")
            if 'max_length' in rules and len(value) > rules['max_length']:
                errors.append(f"{field_name}: Length {len(value)} above maximum {rules['max_length']}")
        
        # Pattern validation
        if isinstance(value, str) and 'pattern' in rules:
            pattern = re.compile(rules['pattern'])
            if not pattern.search(value):
                errors.append(f"{field_name}: Does not match required pattern")
        
        # Custom validation functions
        if 'custom_validator' in rules:
            validator = rules['custom_validator']
            if callable(validator):
                try:
                    is_valid, error_msg = validator(value)
                    if not is_valid:
                        errors.append(f"{field_name}: {error_msg}")
                except Exception as e:
                    errors.append(f"{field_name}: Validation error - {str(e)}")
        
        return errors
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        total_errors = len(self.validation_errors)
        
        return {
            'total_errors': total_errors,
            'error_rate': total_errors / max(self.processed_count, 1),
            'field_errors': self.field_errors,
            'most_common_errors': self._get_most_common_errors(),
            'validation_summary': self._get_validation_summary()
        }
    
    def _get_most_common_errors(self) -> List[tuple]:
        """Get most common validation errors."""
        error_counts = {}
        for error in self.validation_errors:
            error_type = error.split(':')[0] if ':' in error else error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        return {
            'samples_processed': self.processed_count,
            'samples_with_errors': len(set(self.validation_errors)),
            'error_free_rate': (self.processed_count - len(set(self.validation_errors))) / max(self.processed_count, 1),
            'fields_with_errors': len(self.field_errors)
        }
```

## Pipeline Management

### Processing Pipeline

```python
class ProcessingPipeline:
    """Manages a pipeline of data processors."""
    
    def __init__(self, processors: List[BaseProcessor]):
        self.processors = processors
        self.pipeline_stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'processor_stats': {}
        }
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample through the pipeline."""
        current_sample = sample
        
        for processor in self.processors:
            try:
                current_sample = processor.process(current_sample)
                
                # Check if sample should be filtered
                if current_sample.get('_should_filter', False):
                    self.pipeline_stats['total_filtered'] += 1
                    return None
                
            except Exception as e:
                self.logger.error(f"Error in processor {processor.name}: {e}")
                return None
        
        self.pipeline_stats['total_processed'] += 1
        return current_sample
    
    def process_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples through the pipeline."""
        processed_samples = []
        
        for sample in samples:
            processed_sample = self.process_sample(sample)
            if processed_sample is not None:
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self.pipeline_stats.copy()
        
        # Add individual processor stats
        for processor in self.processors:
            stats['processor_stats'][processor.name] = processor.get_stats()
        
        # Calculate overall metrics
        stats['filter_rate'] = stats['total_filtered'] / max(stats['total_processed'] + stats['total_filtered'], 1)
        stats['success_rate'] = stats['total_processed'] / max(stats['total_processed'] + stats['total_filtered'], 1)
        
        return stats
```

## Usage Examples

### Basic Processing Pipeline

```python
from src.data.processors import (
    ArabicTextNormalizer,
    TextQualityFilter,
    DataValidator,
    ProcessingPipeline
)

# Configure processors
normalizer_config = {
    'remove_diacritics': True,
    'normalize_alef': True,
    'normalize_spaces': True
}

filter_config = {
    'min_length': 20,
    'max_length': 1000,
    'min_arabic_ratio': 0.8,
    'max_repetition_ratio': 0.3
}

validator_config = {
    'required_fields': ['prompt', 'chosen'],
    'validation_rules': {
        'prompt': {'type': 'string', 'min_length': 10},
        'chosen': {'type': 'string', 'min_length': 5}
    }
}

# Create processors
normalizer = ArabicTextNormalizer(normalizer_config)
quality_filter = TextQualityFilter(filter_config)
validator = DataValidator(validator_config)

# Create pipeline
pipeline = ProcessingPipeline([normalizer, quality_filter, validator])

# Process samples
raw_samples = [
    {
        'prompt': 'ما هي عاصمة فرنسا؟',
        'chosen': 'عاصمة فرنسا هي باريس.',
        'rejected': 'لا أعرف.'
    },
    {
        'prompt': 'اشرح الذكاء الاصطناعي',
        'chosen': 'الذكاء الاصطناعي مجال علمي...',
        'rejected': 'شيء معقد'
    }
]

processed_samples = pipeline.process_batch(raw_samples)

print(f"Processed {len(processed_samples)} samples")
print(f"Pipeline stats: {pipeline.get_pipeline_stats()}")
```

### Arabic Text Processing

```python
# Specialized Arabic processing
arabic_normalizer = ArabicTextNormalizer({
    'remove_diacritics': True,
    'normalize_alef': True,
    'normalize_teh': True,
    'normalize_yeh': True,
    'remove_tatweel': True
})

# Sample with Arabic text
sample = {
    'prompt': 'مَا هِيَ عَاصِمَةُ فَرَنْسَا؟',  # With diacritics
    'chosen': 'عَاصِمَةُ فَرَنْسَا هِيَ بَارِيس.',
    'rejected': 'لاَ أَعْرِف.'
}

normalized_sample = arabic_normalizer.process(sample)

print("Original prompt:", sample['prompt'])
print("Normalized prompt:", normalized_sample['prompt'])
print("Original chosen:", sample['chosen'])
print("Normalized chosen:", normalized_sample['chosen'])
```

### Data Augmentation

```python
from src.data.processors import ArabicDataAugmenter

# Configure augmentation
augmenter = ArabicDataAugmenter({
    'enable_synonym_replacement': True,
    'enable_paraphrasing': True,
    'synonym_prob': 0.2,
    'paraphrase_prob': 0.1
})

# Sample for augmentation
sample = {
    'prompt': 'ما هو الشيء الكبير في السماء؟',
    'chosen': 'الشمس هي النجم الكبير في السماء.',
    'rejected': 'لا أعرف.'
}

augmented_sample = augmenter.process(sample)

print("Original sample:", sample)
print("Augmented versions:", augmented_sample.get('_augmented_versions', []))
print("Augmentation count:", augmented_sample.get('_augmentation_count', 0))
```

### Format Conversion

```python
from src.data.processors import FormatConverter

# Convert conversation format to standard
converter = FormatConverter({
    'target_format': 'standard',
    'source_format': 'auto'
})

# Conversation format sample
conversation_sample = {
    'conversation': [
        {'role': 'user', 'content': 'ما هي عاصمة مصر؟'},
        {'role': 'assistant', 'content': 'عاصمة مصر هي القاهرة.'}
    ]
}

standard_sample = converter.process(conversation_sample)

print("Converted sample:", standard_sample)
print("Original format:", standard_sample.get('_original_format'))
```

### Quality Filtering

```python
# Configure quality filter
quality_filter = TextQualityFilter({
    'min_length': 15,
    'max_length': 500,
    'min_words': 3,
    'max_repetition_ratio': 0.4,
    'min_arabic_ratio': 0.7
})

# Test samples
test_samples = [
    {
        'prompt': 'سؤال قصير؟',  # Too short
        'chosen': 'جواب قصير.'
    },
    {
        'prompt': 'ما هي فوائد القراءة للإنسان؟',  # Good quality
        'chosen': 'القراءة تفيد الإنسان في تطوير معرفته وتحسين مهاراته اللغوية.'
    },
    {
        'prompt': 'What is the capital of France?',  # Low Arabic ratio
        'chosen': 'The capital of France is Paris.'
    }
]

for i, sample in enumerate(test_samples):
    filtered_sample = quality_filter.process(sample)
    should_filter = filtered_sample.get('_should_filter', False)
    filter_reason = filtered_sample.get('_filter_reason', 'N/A')
    
    print(f"Sample {i}: Filter={should_filter}, Reason={filter_reason}")
```

### Data Validation

```python
# Configure validator
validator = DataValidator({
    'required_fields': ['prompt', 'chosen', 'rejected'],
    'validation_rules': {
        'prompt': {
            'type': 'string',
            'min_length': 10,
            'max_length': 200
        },
        'chosen': {
            'type': 'string',
            'min_length': 5,
            'pattern': r'[\u0600-\u06FF]'  # Must contain Arabic
        },
        'rejected': {
            'type': 'string',
            'min_length': 3
        }
    }
})

# Test samples
test_samples = [
    {
        'prompt': 'ما هي عاصمة فرنسا؟',
        'chosen': 'باريس هي عاصمة فرنسا.',
        'rejected': 'لا أعرف.'
    },
    {
        'prompt': 'قصير',  # Too short
        'chosen': 'جواب',  # Too short
        # Missing 'rejected' field
    },
    {
        'prompt': 'ما هو اللون المفضل لديك؟',
        'chosen': 'Blue is my favorite color.',  # No Arabic
        'rejected': 'لا يوجد لون مفضل.'
    }
]

for i, sample in enumerate(test_samples):
    validated_sample = validator.process(sample)
    is_valid = validated_sample.get('_is_valid', False)
    errors = validated_sample.get('_validation_errors', [])
    
    print(f"Sample {i}: Valid={is_valid}")
    if errors:
        print(f"  Errors: {errors}")

# Get validation report
report = validator.get_validation_report()
print("\nValidation Report:")
print(f"Total errors: {report['total_errors']}")
print(f"Error rate: {report['error_rate']:.2%}")
print(f"Most common errors: {report['most_common_errors']}")
```

### Complete Processing Pipeline

```python
# Create comprehensive processing pipeline
def create_arabic_processing_pipeline():
    """Create a complete Arabic text processing pipeline."""
    
    # Step 1: Format conversion
    format_converter = FormatConverter({
        'target_format': 'standard',
        'source_format': 'auto'
    })
    
    # Step 2: Arabic text normalization
    text_normalizer = ArabicTextNormalizer({
        'remove_diacritics': True,
        'normalize_alef': True,
        'normalize_teh': True,
        'normalize_yeh': True,
        'remove_tatweel': True,
        'normalize_spaces': True
    })
    
    # Step 3: Quality filtering
    quality_filter = TextQualityFilter({
        'min_length': 20,
        'max_length': 1000,
        'min_words': 4,
        'max_repetition_ratio': 0.3,
        'min_arabic_ratio': 0.8,
        'max_special_char_ratio': 0.15
    })
    
    # Step 4: Data validation
    validator = DataValidator({
        'required_fields': ['prompt', 'chosen'],
        'validation_rules': {
            'prompt': {
                'type': 'string',
                'min_length': 15,
                'max_length': 300
            },
            'chosen': {
                'type': 'string',
                'min_length': 10,
                'pattern': r'[\u0600-\u06FF]'
            }
        }
    })
    
    # Step 5: Data augmentation (optional)
    augmenter = ArabicDataAugmenter({
        'enable_synonym_replacement': True,
        'enable_paraphrasing': False,  # Disable for production
        'synonym_prob': 0.05
    })
    
    return ProcessingPipeline([
        format_converter,
        text_normalizer,
        quality_filter,
        validator,
        # augmenter  # Uncomment for augmentation
    ])

# Use the pipeline
pipeline = create_arabic_processing_pipeline()

# Process a batch of samples
raw_data = [
    # Various format samples...
]

processed_data = pipeline.process_batch(raw_data)
stats = pipeline.get_pipeline_stats()

print(f"Processing complete:")
print(f"  Input samples: {len(raw_data)}")
print(f"  Output samples: {len(processed_data)}")
print(f"  Filter rate: {stats['filter_rate']:.2%}")
print(f"  Success rate: {stats['success_rate']:.2%}")

# Detailed processor stats
for processor_name, processor_stats in stats['processor_stats'].items():
    print(f"\n{processor_name}:")
    print(f"  Processed: {processor_stats['processed_count']}")
    print(f"  Errors: {processor_stats['error_count']}")
    print(f"  Success rate: {processor_stats['success_rate']:.2%}")
```

## Best Practices

### 1. Pipeline Design
- Order processors logically (normalize → filter → validate)
- Use early filtering to improve performance
- Monitor processor statistics
- Handle errors gracefully

### 2. Arabic Text Processing
- Always normalize Arabic text consistently
- Consider diacritics handling carefully
- Validate Arabic content ratio
- Handle mixed-script text appropriately

### 3. Quality Control
- Set appropriate quality thresholds
- Monitor filtering rates
- Validate data consistency
- Track processing statistics

### 4. Performance
- Use batch processing when possible
- Cache expensive computations
- Profile processor performance
- Optimize bottleneck processors

### 5. Extensibility
- Design processors as modular components
- Use configuration for flexibility
- Implement proper error handling
- Document processor behavior

## See Also

- [Base Data Loader](base_loader.md)
- [SFT Data Loader](sft_loader.md)
- [Preference Data Loader](preference_loader.md)
- [Data Loading Index](index.md)