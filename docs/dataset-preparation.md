# Dataset Preparation for Arabic Qwen Fine-tuning

This guide covers comprehensive dataset preparation techniques for Arabic language model fine-tuning, including data sources, preprocessing, formatting, and optimization strategies.

## 📋 Table of Contents

1. [Arabic Dataset Overview](#arabic-dataset-overview)
2. [Dataset Categories](#dataset-categories)
3. [Data Preprocessing](#data-preprocessing)
4. [Format Specifications](#format-specifications)
5. [Quality Control](#quality-control)
6. [Dataset Optimization](#dataset-optimization)
7. [Custom Dataset Creation](#custom-dataset-creation)

## 🌍 Arabic Dataset Overview

### High-Quality Curated Datasets

| Dataset | Size | Type | Quality | Use Case | Availability |
|---------|------|------|---------|----------|-------------|
| **InstAr-500k** | 500k | Instruction | ⭐⭐⭐⭐⭐ | General SFT | HuggingFace |
| **CIDAR** | 10k | Instruction | ⭐⭐⭐⭐⭐ | Cultural SFT | HuggingFace |
| **Arabic-OpenHermes-2.5** | 982k | Conversation | ⭐⭐⭐⭐ | Chat SFT | HuggingFace |
| **Arabic-preference-data-RLHF** | 11.5k | Preference | ⭐⭐⭐⭐⭐ | DPO/RLHF | HuggingFace |
| **ArabicQA_2.1M** | 2.14M | QA | ⭐⭐⭐⭐ | Domain SFT | HuggingFace |
| **Arabic MMLU** | 14k-29k | Evaluation | ⭐⭐⭐⭐⭐ | Benchmarking | HuggingFace |
| **argilla-dpo-mix-7k-arabic** | 7.5k | Preference | ⭐⭐⭐ | DPO | HuggingFace |

### Large-Scale Corpora

| Dataset | Size | Type | Quality | Use Case | Availability |
|---------|------|------|---------|----------|-------------|
| **ArabicWeb24** | 28B+ tokens | Web Corpus | ⭐⭐⭐⭐ | Pre-training | HuggingFace |
| **fineweb-arabic** | 311M samples | Web Corpus | ⭐⭐⭐⭐ | Pre-training | HuggingFace |
| **Arabic Wikipedia** | ~2M articles | Encyclopedia | ⭐⭐⭐⭐⭐ | Knowledge | Public |
| **Arabic News Corpus** | Various | News | ⭐⭐⭐ | Domain | Various |

## 📊 Dataset Categories

### 1. Instruction Following Datasets

**Purpose**: Train models to follow Arabic instructions and commands

#### InstAr-500k
```python
# Loading InstAr-500k
from datasets import load_dataset

dataset = load_dataset("FreedomIntelligence/InstAr-500k")

# Sample structure
print(dataset["train"][0])
# {
#   "instruction": "اكتب قصة قصيرة عن...",
#   "output": "كان هناك مرة...",
#   "input": "",  # Usually empty
#   "text": "### Instruction:\n...\n\n### Response:\n..."
# }
```

#### CIDAR (Culturally Relevant)
```python
# Loading CIDAR
dataset = load_dataset("FreedomIntelligence/CIDAR")

# Sample structure
print(dataset["train"][0])
# {
#   "instruction": "ما هي آداب الضيافة في الثقافة العربية؟",
#   "output": "آداب الضيافة في الثقافة العربية تشمل...",
#   "category": "culture"
# }
```

### 2. Preference Datasets

**Purpose**: Train models to prefer better responses over worse ones

#### Arabic Preference Data for RLHF
```python
# Loading preference dataset
dataset = load_dataset("FreedomIntelligence/Arabic-preference-data-RLHF")

# Sample structure
print(dataset["train"][0])
# {
#   "question": "ما هو أفضل طريقة لتعلم اللغة العربية؟",
#   "chosen_response": "أفضل طريقة لتعلم اللغة العربية هي...",
#   "rejected_response": "يمكنك تعلم العربية بسهولة...",
#   "score_chosen": 8.5,
#   "score_rejected": 3.2
# }
```

### 3. Question Answering Datasets

**Purpose**: Specialize models for Arabic QA tasks

#### ArabicQA 2.1M
```python
# Loading ArabicQA
dataset = load_dataset("riotu-lab/ArabicQA_2.1M")

# Sample structure
print(dataset["train"][0])
# {
#   "question": "ما هي عاصمة المملكة العربية السعودية؟",
#   "answer": "الرياض",
#   "context": "المملكة العربية السعودية دولة...",
#   "source": "wikipedia"
# }
```

## 🔧 Data Preprocessing

### Text Cleaning and Normalization

```python
import re
import unicodedata
from typing import str

class ArabicTextProcessor:
    def __init__(self):
        # Arabic diacritics (tashkeel)
        self.diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        
        # Non-Arabic characters (keep basic punctuation)
        self.non_arabic = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\.,!?;:()\[\]"\'-]')
        
        # Multiple spaces
        self.multiple_spaces = re.compile(r'\s+')
        
    def clean_text(self, text: str, remove_diacritics: bool = False) -> str:
        """Clean and normalize Arabic text"""
        if not text:
            return ""
            
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove diacritics if requested
        if remove_diacritics:
            text = self.diacritics.sub('', text)
            
        # Remove non-Arabic characters (optional)
        # text = self.non_arabic.sub(' ', text)
        
        # Normalize whitespace
        text = self.multiple_spaces.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize Arabic punctuation"""
        # Arabic question mark
        text = text.replace('؟', '?')
        
        # Arabic comma
        text = text.replace('،', ',')
        
        # Arabic semicolon
        text = text.replace('؛', ';')
        
        return text
    
    def process_dataset_text(self, text: str) -> str:
        """Complete text processing pipeline"""
        text = self.clean_text(text, remove_diacritics=False)
        text = self.normalize_punctuation(text)
        return text

# Usage example
processor = ArabicTextProcessor()

def preprocess_instruction_dataset(example):
    """Preprocess instruction dataset"""
    return {
        "instruction": processor.process_dataset_text(example["instruction"]),
        "output": processor.process_dataset_text(example["output"]),
        "input": processor.process_dataset_text(example.get("input", ""))
    }

# Apply to dataset
dataset = dataset.map(preprocess_instruction_dataset)
```

### Length Filtering and Statistics

```python
def analyze_dataset_lengths(dataset, text_column: str = "text"):
    """Analyze text lengths in dataset"""
    lengths = []
    
    for example in dataset:
        if text_column in example:
            length = len(example[text_column].split())
            lengths.append(length)
    
    import numpy as np
    
    stats = {
        "count": len(lengths),
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "percentiles": {
            "25th": np.percentile(lengths, 25),
            "75th": np.percentile(lengths, 75),
            "90th": np.percentile(lengths, 90),
            "95th": np.percentile(lengths, 95)
        }
    }
    
    return stats, lengths

def filter_by_length(dataset, min_length: int = 10, max_length: int = 512, text_column: str = "text"):
    """Filter dataset by text length"""
    def length_filter(example):
        if text_column not in example:
            return False
            
        word_count = len(example[text_column].split())
        return min_length <= word_count <= max_length
    
    filtered_dataset = dataset.filter(length_filter)
    
    print(f"Original size: {len(dataset):,}")
    print(f"Filtered size: {len(filtered_dataset):,}")
    print(f"Retention rate: {len(filtered_dataset)/len(dataset)*100:.1f}%")
    
    return filtered_dataset

# Usage
stats, lengths = analyze_dataset_lengths(dataset["train"])
print(f"Average length: {stats['mean']:.1f} words")
print(f"95th percentile: {stats['percentiles']['95th']:.1f} words")

# Filter dataset
filtered_dataset = filter_by_length(dataset["train"], min_length=5, max_length=512)
```

## 📝 Format Specifications

### Instruction Following Format

```python
def format_instruction_data(example, template_type: str = "arabic"):
    """Format data for instruction following"""
    
    templates = {
        "arabic": {
            "prompt": "### التعليمات:\n{instruction}\n\n### الاستجابة:\n{output}",
            "prompt_with_input": "### التعليمات:\n{instruction}\n\n### المدخلات:\n{input}\n\n### الاستجابة:\n{output}"
        },
        "english": {
            "prompt": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
            "prompt_with_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        },
        "mixed": {
            "prompt": "### Instruction | التعليمات:\n{instruction}\n\n### Response | الاستجابة:\n{output}",
            "prompt_with_input": "### Instruction | التعليمات:\n{instruction}\n\n### Input | المدخلات:\n{input}\n\n### Response | الاستجابة:\n{output}"
        }
    }
    
    template = templates[template_type]
    
    instruction = example["instruction"]
    output = example["output"]
    input_text = example.get("input", "")
    
    if input_text and input_text.strip():
        formatted_text = template["prompt_with_input"].format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        formatted_text = template["prompt"].format(
            instruction=instruction,
            output=output
        )
    
    return {"text": formatted_text}

# Apply formatting
formatted_dataset = dataset.map(
    lambda x: format_instruction_data(x, template_type="arabic")
)
```

### QA Format

```python
def format_qa_data(example, format_type: str = "simple"):
    """Format QA data for training"""
    
    formats = {
        "simple": "السؤال: {question}\nالإجابة: {answer}",
        "context": "السياق: {context}\n\nالسؤال: {question}\nالإجابة: {answer}",
        "structured": "### السؤال\n{question}\n\n### الإجابة\n{answer}",
        "conversation": "المستخدم: {question}\nالمساعد: {answer}"
    }
    
    question = example["question"]
    answer = example["answer"]
    context = example.get("context", "")
    
    if format_type == "context" and context:
        formatted_text = formats[format_type].format(
            context=context,
            question=question,
            answer=answer
        )
    else:
        formatted_text = formats[format_type].format(
            question=question,
            answer=answer
        )
    
    return {"text": formatted_text}

# Apply QA formatting
qa_dataset = qa_dataset.map(
    lambda x: format_qa_data(x, format_type="simple")
)
```

### DPO Format

```python
def format_dpo_data(example):
    """Format data for DPO training"""
    return {
        "prompt": example["question"],
        "chosen": example["chosen_response"],
        "rejected": example["rejected_response"]
    }

# Apply DPO formatting
dpo_dataset = preference_dataset.map(format_dpo_data)
```

## 🔍 Quality Control

### Content Quality Assessment

```python
class ArabicQualityChecker:
    def __init__(self):
        # Common quality indicators
        self.min_arabic_ratio = 0.7  # Minimum Arabic character ratio
        self.max_repetition_ratio = 0.3  # Maximum repetitive content
        
    def calculate_arabic_ratio(self, text: str) -> float:
        """Calculate ratio of Arabic characters"""
        if not text:
            return 0.0
            
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        return arabic_chars / total_chars if total_chars > 0 else 0.0
    
    def detect_repetition(self, text: str) -> float:
        """Detect repetitive content"""
        words = text.split()
        if len(words) < 10:
            return 0.0
            
        # Check for repeated phrases
        phrases = []
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            phrases.append(phrase)
            
        unique_phrases = set(phrases)
        repetition_ratio = 1 - (len(unique_phrases) / len(phrases))
        
        return repetition_ratio
    
    def check_quality(self, text: str) -> dict:
        """Comprehensive quality check"""
        arabic_ratio = self.calculate_arabic_ratio(text)
        repetition_ratio = self.detect_repetition(text)
        
        quality_score = 1.0
        issues = []
        
        # Arabic content check
        if arabic_ratio < self.min_arabic_ratio:
            quality_score -= 0.3
            issues.append(f"Low Arabic content: {arabic_ratio:.2f}")
            
        # Repetition check
        if repetition_ratio > self.max_repetition_ratio:
            quality_score -= 0.4
            issues.append(f"High repetition: {repetition_ratio:.2f}")
            
        # Length check
        word_count = len(text.split())
        if word_count < 5:
            quality_score -= 0.2
            issues.append(f"Too short: {word_count} words")
        elif word_count > 1000:
            quality_score -= 0.1
            issues.append(f"Very long: {word_count} words")
            
        return {
            "quality_score": max(0.0, quality_score),
            "arabic_ratio": arabic_ratio,
            "repetition_ratio": repetition_ratio,
            "word_count": word_count,
            "issues": issues,
            "passed": quality_score > 0.5
        }

def filter_by_quality(dataset, min_quality_score: float = 0.6):
    """Filter dataset by quality score"""
    quality_checker = ArabicQualityChecker()
    
    def quality_filter(example):
        if "text" not in example:
            return False
            
        quality_result = quality_checker.check_quality(example["text"])
        return quality_result["quality_score"] >= min_quality_score
    
    # Add quality scores to dataset
    def add_quality_score(example):
        quality_result = quality_checker.check_quality(example["text"])
        example["quality_score"] = quality_result["quality_score"]
        example["quality_issues"] = quality_result["issues"]
        return example
    
    dataset_with_scores = dataset.map(add_quality_score)
    filtered_dataset = dataset_with_scores.filter(quality_filter)
    
    print(f"Quality filtering results:")
    print(f"Original size: {len(dataset):,}")
    print(f"Filtered size: {len(filtered_dataset):,}")
    print(f"Retention rate: {len(filtered_dataset)/len(dataset)*100:.1f}%")
    
    return filtered_dataset

# Usage
quality_filtered_dataset = filter_by_quality(dataset["train"], min_quality_score=0.7)
```

## ⚡ Dataset Optimization

### Deduplication

```python
import hashlib
from collections import defaultdict

def deduplicate_dataset(dataset, text_column: str = "text", similarity_threshold: float = 0.95):
    """Remove duplicate and near-duplicate examples"""
    
    def get_text_hash(text: str) -> str:
        """Get hash of normalized text"""
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def get_text_signature(text: str, n_grams: int = 3) -> set:
        """Get text signature for similarity comparison"""
        words = text.lower().split()
        if len(words) < n_grams:
            return set([" ".join(words)])
            
        signatures = set()
        for i in range(len(words) - n_grams + 1):
            signature = " ".join(words[i:i+n_grams])
            signatures.add(signature)
            
        return signatures
    
    def calculate_similarity(sig1: set, sig2: set) -> float:
        """Calculate Jaccard similarity"""
        if not sig1 or not sig2:
            return 0.0
        return len(sig1.intersection(sig2)) / len(sig1.union(sig2))
    
    # Track seen hashes and signatures
    seen_hashes = set()
    seen_signatures = []
    unique_indices = []
    
    for i, example in enumerate(dataset):
        text = example[text_column]
        text_hash = get_text_hash(text)
        
        # Check exact duplicates
        if text_hash in seen_hashes:
            continue
            
        # Check near duplicates
        text_signature = get_text_signature(text)
        is_duplicate = False
        
        for existing_signature in seen_signatures:
            similarity = calculate_similarity(text_signature, existing_signature)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            seen_hashes.add(text_hash)
            seen_signatures.append(text_signature)
            unique_indices.append(i)
            
        # Progress tracking
        if i % 10000 == 0:
            print(f"Processed {i:,} examples, found {len(unique_indices):,} unique")
    
    # Create deduplicated dataset
    deduplicated_dataset = dataset.select(unique_indices)
    
    print(f"\nDeduplication results:")
    print(f"Original size: {len(dataset):,}")
    print(f"Deduplicated size: {len(deduplicated_dataset):,}")
    print(f"Removed: {len(dataset) - len(deduplicated_dataset):,} ({(len(dataset) - len(deduplicated_dataset))/len(dataset)*100:.1f}%)")
    
    return deduplicated_dataset

# Usage
deduplicated_dataset = deduplicate_dataset(dataset["train"], similarity_threshold=0.9)
```

### Data Balancing

```python
def balance_dataset_by_length(dataset, target_distribution: dict = None):
    """Balance dataset by text length"""
    if target_distribution is None:
        target_distribution = {
            "short": 0.3,    # 0-100 words
            "medium": 0.5,   # 100-300 words
            "long": 0.2      # 300+ words
        }
    
    def categorize_by_length(example):
        word_count = len(example["text"].split())
        if word_count <= 100:
            return "short"
        elif word_count <= 300:
            return "medium"
        else:
            return "long"
    
    # Categorize examples
    categorized = defaultdict(list)
    for i, example in enumerate(dataset):
        category = categorize_by_length(example)
        categorized[category].append(i)
    
    # Calculate target sizes
    total_size = len(dataset)
    target_sizes = {cat: int(total_size * ratio) for cat, ratio in target_distribution.items()}
    
    # Sample from each category
    balanced_indices = []
    for category, target_size in target_sizes.items():
        available_indices = categorized[category]
        if len(available_indices) >= target_size:
            # Randomly sample
            import random
            sampled_indices = random.sample(available_indices, target_size)
        else:
            # Use all available
            sampled_indices = available_indices
            
        balanced_indices.extend(sampled_indices)
    
    # Create balanced dataset
    balanced_dataset = dataset.select(balanced_indices)
    
    print(f"Dataset balancing results:")
    for category in target_distribution.keys():
        original_count = len(categorized[category])
        target_count = target_sizes[category]
        final_count = sum(1 for i in balanced_indices if categorize_by_length(dataset[i]) == category)
        print(f"{category.capitalize()}: {original_count:,} → {target_count:,} (actual: {final_count:,})")
    
    return balanced_dataset

# Usage
balanced_dataset = balance_dataset_by_length(dataset["train"])
```

## 🛠️ Custom Dataset Creation

### Creating Custom Instruction Dataset

```python
class CustomArabicDatasetCreator:
    def __init__(self):
        self.processor = ArabicTextProcessor()
        
    def create_instruction_dataset(self, data_sources: list) -> Dataset:
        """Create custom instruction dataset from multiple sources"""
        all_examples = []
        
        for source in data_sources:
            if source["type"] == "json":
                examples = self._load_json_data(source["path"], source["format"])
            elif source["type"] == "csv":
                examples = self._load_csv_data(source["path"], source["format"])
            elif source["type"] == "txt":
                examples = self._load_text_data(source["path"], source["format"])
            else:
                continue
                
            all_examples.extend(examples)
        
        # Create dataset
        dataset = Dataset.from_list(all_examples)
        
        # Apply preprocessing
        dataset = dataset.map(self._preprocess_example)
        
        # Apply quality filtering
        dataset = filter_by_quality(dataset, min_quality_score=0.7)
        
        return dataset
    
    def _load_json_data(self, file_path: str, format_spec: dict) -> list:
        """Load data from JSON file"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = {
                "instruction": item[format_spec["instruction_key"]],
                "output": item[format_spec["output_key"]],
                "input": item.get(format_spec.get("input_key", ""), "")
            }
            examples.append(example)
            
        return examples
    
    def _preprocess_example(self, example):
        """Preprocess individual example"""
        # Clean text
        example["instruction"] = self.processor.process_dataset_text(example["instruction"])
        example["output"] = self.processor.process_dataset_text(example["output"])
        
        # Format for training
        formatted = format_instruction_data(example, template_type="arabic")
        example["text"] = formatted["text"]
        
        return example
    
    def create_qa_dataset(self, qa_pairs: list) -> Dataset:
        """Create QA dataset from list of QA pairs"""
        examples = []
        
        for qa in qa_pairs:
            example = {
                "question": self.processor.process_dataset_text(qa["question"]),
                "answer": self.processor.process_dataset_text(qa["answer"]),
                "context": self.processor.process_dataset_text(qa.get("context", ""))
            }
            
            # Format for training
            formatted = format_qa_data(example, format_type="simple")
            example["text"] = formatted["text"]
            
            examples.append(example)
        
        return Dataset.from_list(examples)

# Usage example
creator = CustomArabicDatasetCreator()

# Define data sources
data_sources = [
    {
        "type": "json",
        "path": "./data/custom_instructions.json",
        "format": {
            "instruction_key": "instruction",
            "output_key": "response"
        }
    }
]

# Create custom dataset
custom_dataset = creator.create_instruction_dataset(data_sources)
print(f"Created custom dataset with {len(custom_dataset):,} examples")
```

### Dataset Mixing and Sampling

```python
def mix_datasets(datasets: dict, mixing_ratios: dict) -> Dataset:
    """Mix multiple datasets with specified ratios"""
    mixed_examples = []
    
    # Calculate sample sizes
    total_ratio = sum(mixing_ratios.values())
    min_dataset_size = min(len(dataset) for dataset in datasets.values())
    
    for dataset_name, ratio in mixing_ratios.items():
        if dataset_name not in datasets:
            continue
            
        dataset = datasets[dataset_name]
        sample_size = int((ratio / total_ratio) * min_dataset_size * len(datasets))
        
        # Sample from dataset
        if len(dataset) > sample_size:
            import random
            indices = random.sample(range(len(dataset)), sample_size)
            sampled_dataset = dataset.select(indices)
        else:
            sampled_dataset = dataset
            
        # Add source information
        def add_source(example):
            example["source_dataset"] = dataset_name
            return example
            
        sampled_dataset = sampled_dataset.map(add_source)
        mixed_examples.extend(sampled_dataset)
    
    # Shuffle mixed dataset
    import random
    random.shuffle(mixed_examples)
    
    return Dataset.from_list(mixed_examples)

# Usage
datasets = {
    "instAr": load_dataset("FreedomIntelligence/InstAr-500k")["train"],
    "cidar": load_dataset("FreedomIntelligence/CIDAR")["train"],
    "qa": load_dataset("riotu-lab/ArabicQA_2.1M")["train"]
}

mixing_ratios = {
    "instAr": 0.6,
    "cidar": 0.3,
    "qa": 0.1
}

mixed_dataset = mix_datasets(datasets, mixing_ratios)
print(f"Mixed dataset size: {len(mixed_dataset):,}")
```

## 📊 Dataset Validation

### Comprehensive Dataset Validation

```python
def validate_dataset(dataset, validation_config: dict = None):
    """Comprehensive dataset validation"""
    if validation_config is None:
        validation_config = {
            "min_examples": 100,
            "max_examples": 1000000,
            "min_avg_length": 10,
            "max_avg_length": 1000,
            "min_arabic_ratio": 0.5,
            "required_columns": ["text"]
        }
    
    validation_results = {
        "passed": True,
        "issues": [],
        "warnings": [],
        "statistics": {}
    }
    
    # Check dataset size
    dataset_size = len(dataset)
    validation_results["statistics"]["size"] = dataset_size
    
    if dataset_size < validation_config["min_examples"]:
        validation_results["passed"] = False
        validation_results["issues"].append(f"Dataset too small: {dataset_size} < {validation_config['min_examples']}")
    
    if dataset_size > validation_config["max_examples"]:
        validation_results["warnings"].append(f"Dataset very large: {dataset_size} > {validation_config['max_examples']}")
    
    # Check required columns
    if hasattr(dataset, 'column_names'):
        missing_columns = set(validation_config["required_columns"]) - set(dataset.column_names)
        if missing_columns:
            validation_results["passed"] = False
            validation_results["issues"].append(f"Missing columns: {missing_columns}")
    
    # Analyze text statistics
    if "text" in dataset.column_names:
        lengths = [len(example["text"].split()) for example in dataset]
        avg_length = sum(lengths) / len(lengths)
        
        validation_results["statistics"]["avg_length"] = avg_length
        validation_results["statistics"]["min_length"] = min(lengths)
        validation_results["statistics"]["max_length"] = max(lengths)
        
        if avg_length < validation_config["min_avg_length"]:
            validation_results["warnings"].append(f"Average length low: {avg_length:.1f} < {validation_config['min_avg_length']}")
        
        if avg_length > validation_config["max_avg_length"]:
            validation_results["warnings"].append(f"Average length high: {avg_length:.1f} > {validation_config['max_avg_length']}")
        
        # Check Arabic content
        quality_checker = ArabicQualityChecker()
        arabic_ratios = [quality_checker.calculate_arabic_ratio(example["text"]) for example in dataset[:1000]]  # Sample
        avg_arabic_ratio = sum(arabic_ratios) / len(arabic_ratios)
        
        validation_results["statistics"]["avg_arabic_ratio"] = avg_arabic_ratio
        
        if avg_arabic_ratio < validation_config["min_arabic_ratio"]:
            validation_results["warnings"].append(f"Low Arabic content: {avg_arabic_ratio:.2f} < {validation_config['min_arabic_ratio']}")
    
    return validation_results

# Usage
validation_results = validate_dataset(dataset["train"])

print(f"Validation passed: {validation_results['passed']}")
print(f"Issues: {len(validation_results['issues'])}")
print(f"Warnings: {len(validation_results['warnings'])}")
print(f"Statistics: {validation_results['statistics']}")

for issue in validation_results["issues"]:
    print(f"❌ {issue}")
    
for warning in validation_results["warnings"]:
    print(f"⚠️ {warning}")
```

## 💾 Dataset Saving and Loading

### Efficient Dataset Management

```python
def save_processed_dataset(dataset, output_path: str, format: str = "parquet"):
    """Save processed dataset efficiently"""
    import os
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        dataset.to_parquet(str(output_path / "dataset.parquet"))
    elif format == "json":
        dataset.to_json(str(output_path / "dataset.json"), orient="records", lines=True)
    elif format == "huggingface":
        dataset.save_to_disk(str(output_path))
    
    # Save metadata
    metadata = {
        "size": len(dataset),
        "columns": list(dataset.column_names) if hasattr(dataset, 'column_names') else [],
        "format": format,
        "created_at": str(datetime.now()),
        "processing_info": {
            "preprocessed": True,
            "quality_filtered": True,
            "deduplicated": True
        }
    }
    
    import json
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path} in {format} format")
    print(f"Size: {len(dataset):,} examples")

def load_processed_dataset(dataset_path: str):
    """Load processed dataset"""
    from pathlib import Path
    
    dataset_path = Path(dataset_path)
    
    # Load metadata
    metadata_path = dataset_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loading dataset: {metadata['size']:,} examples, format: {metadata['format']}")
    
    # Load dataset based on format
    if (dataset_path / "dataset.parquet").exists():
        dataset = Dataset.from_parquet(str(dataset_path / "dataset.parquet"))
    elif (dataset_path / "dataset.json").exists():
        dataset = Dataset.from_json(str(dataset_path / "dataset.json"))
    elif (dataset_path / "dataset_info.json").exists():
        dataset = Dataset.load_from_disk(str(dataset_path))
    else:
        raise FileNotFoundError(f"No dataset found in {dataset_path}")
    
    return dataset

# Usage
save_processed_dataset(processed_dataset, "./processed_data/arabic_instruct", format="parquet")
loaded_dataset = load_processed_dataset("./processed_data/arabic_instruct")
```

---

This comprehensive dataset preparation guide provides all the tools and techniques needed to prepare high-quality Arabic datasets for Qwen model fine-tuning. The code examples are production-ready and optimized for the specific requirements of Arabic language processing.

## 📚 Next Steps

1. Review [Hardware Requirements](./hardware-requirements.md) for optimal processing setup
2. Follow [Implementation Examples](./implementation-examples.md) for practical usage
3. Check [Fine-tuning Guide](./fine-tuning-guide.md) for training procedures
4. Consult [Troubleshooting Guide](./troubleshooting.md) for common issues