"""Data Utility Functions

This module contains utility functions for data processing and handling
in the Arabic Qwen fine-tuning project.
"""

import logging
import re
import json
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_dataset_from_path(
    data_path: str,
    data_format: str = "auto",
    split: Optional[str] = None,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """Load dataset from various formats.
    
    Args:
        data_path: Path to dataset
        data_format: Data format (auto, json, csv, parquet, text)
        split: Specific split to load
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Dataset or DatasetDict
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    # Auto-detect format
    if data_format == "auto":
        suffix = data_path.suffix.lower()
        format_mapping = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".parquet": "parquet",
            ".txt": "text"
        }
        data_format = format_mapping.get(suffix, "json")
    
    logger.info(f"Loading dataset from {data_path} (format: {data_format})")
    
    try:
        if data_format == "json":
            dataset = load_dataset("json", data_files=str(data_path), split=split, **kwargs)
        elif data_format == "csv":
            dataset = load_dataset("csv", data_files=str(data_path), split=split, **kwargs)
        elif data_format == "parquet":
            dataset = load_dataset("parquet", data_files=str(data_path), split=split, **kwargs)
        elif data_format == "text":
            dataset = load_dataset("text", data_files=str(data_path), split=split, **kwargs)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        logger.info(f"Dataset loaded successfully: {len(dataset) if isinstance(dataset, Dataset) else sum(len(split) for split in dataset.values())} examples")
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def save_dataset(
    dataset: Union[Dataset, DatasetDict],
    output_path: str,
    format: str = "json",
    **kwargs
):
    """Save dataset to file.
    
    Args:
        dataset: Dataset to save
        output_path: Output path
        format: Output format (json, csv, parquet)
        **kwargs: Additional arguments
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving dataset to {output_path} (format: {format})")
    
    try:
        if format == "json":
            if isinstance(dataset, DatasetDict):
                for split_name, split_dataset in dataset.items():
                    split_path = output_path.parent / f"{output_path.stem}_{split_name}.json"
                    split_dataset.to_json(str(split_path), **kwargs)
            else:
                dataset.to_json(str(output_path), **kwargs)
        elif format == "csv":
            if isinstance(dataset, DatasetDict):
                for split_name, split_dataset in dataset.items():
                    split_path = output_path.parent / f"{output_path.stem}_{split_name}.csv"
                    split_dataset.to_csv(str(split_path), **kwargs)
            else:
                dataset.to_csv(str(output_path), **kwargs)
        elif format == "parquet":
            if isinstance(dataset, DatasetDict):
                for split_name, split_dataset in dataset.items():
                    split_path = output_path.parent / f"{output_path.stem}_{split_name}.parquet"
                    split_dataset.to_parquet(str(split_path), **kwargs)
            else:
                dataset.to_parquet(str(output_path), **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved successfully to {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """Split dataset into train/validation/test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        DatasetDict with splits
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    logger.info(f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # First split: train vs (val + test)
    train_test_split = dataset.train_test_split(
        test_size=val_ratio + test_ratio,
        seed=seed
    )
    
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]
    
    # Second split: val vs test
    if test_ratio > 0:
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_test_split = temp_dataset.train_test_split(
            test_size=1 - val_test_ratio,
            seed=seed
        )
        val_dataset = val_test_split["train"]
        test_dataset = val_test_split["test"]
        
        splits = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
    else:
        splits = {
            "train": train_dataset,
            "validation": temp_dataset
        }
    
    dataset_dict = DatasetDict(splits)
    
    logger.info(f"Dataset split complete: {[(k, len(v)) for k, v in dataset_dict.items()]}")
    return dataset_dict


def filter_dataset_by_length(
    dataset: Dataset,
    text_column: str,
    min_length: int = 10,
    max_length: int = 2048,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> Dataset:
    """Filter dataset by text length.
    
    Args:
        dataset: Dataset to filter
        text_column: Column containing text
        min_length: Minimum length
        max_length: Maximum length
        tokenizer: Tokenizer for token-based filtering
        
    Returns:
        Filtered dataset
    """
    logger.info(f"Filtering dataset by length: {min_length}-{max_length}")
    
    original_size = len(dataset)
    
    def length_filter(example):
        text = example[text_column]
        if tokenizer is not None:
            # Token-based filtering
            tokens = tokenizer.encode(text, add_special_tokens=False)
            length = len(tokens)
        else:
            # Character-based filtering
            length = len(text)
        
        return min_length <= length <= max_length
    
    filtered_dataset = dataset.filter(length_filter)
    filtered_size = len(filtered_dataset)
    
    logger.info(f"Dataset filtered: {original_size} -> {filtered_size} ({100 * filtered_size / original_size:.1f}% retained)")
    return filtered_dataset


def clean_text(
    text: str,
    remove_extra_whitespace: bool = True,
    remove_special_chars: bool = False,
    normalize_arabic: bool = True,
    remove_diacritics: bool = False
) -> str:
    """Clean and normalize text.
    
    Args:
        text: Text to clean
        remove_extra_whitespace: Remove extra whitespace
        remove_special_chars: Remove special characters
        normalize_arabic: Normalize Arabic text
        remove_diacritics: Remove Arabic diacritics
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize Arabic text
    if normalize_arabic:
        # Normalize Arabic letters
        text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alef
        text = re.sub(r'ى', 'ي', text)  # Normalize Yeh
        text = re.sub(r'ة', 'ه', text)  # Normalize Teh Marbuta
    
    # Remove Arabic diacritics
    if remove_diacritics:
        arabic_diacritics = r'[\u064B-\u065F\u0670\u06D6-\u06ED]'
        text = re.sub(arabic_diacritics, '', text)
    
    # Remove special characters (keep Arabic, English, numbers, and basic punctuation)
    if remove_special_chars:
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s.,!?;:()\[\]{}"\'-]', '', text)
    
    return text.strip()


def clean_dataset(
    dataset: Dataset,
    text_columns: List[str],
    **clean_kwargs
) -> Dataset:
    """Clean text in dataset columns.
    
    Args:
        dataset: Dataset to clean
        text_columns: List of text columns to clean
        **clean_kwargs: Arguments for clean_text function
        
    Returns:
        Cleaned dataset
    """
    logger.info(f"Cleaning dataset text columns: {text_columns}")
    
    def clean_example(example):
        for column in text_columns:
            if column in example:
                example[column] = clean_text(example[column], **clean_kwargs)
        return example
    
    cleaned_dataset = dataset.map(clean_example)
    
    logger.info("Dataset cleaning complete")
    return cleaned_dataset


def analyze_dataset(
    dataset: Union[Dataset, DatasetDict],
    text_columns: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> Dict[str, Any]:
    """Analyze dataset statistics.
    
    Args:
        dataset: Dataset to analyze
        text_columns: Text columns to analyze
        tokenizer: Tokenizer for token analysis
        
    Returns:
        Analysis results
    """
    logger.info("Analyzing dataset")
    
    if isinstance(dataset, DatasetDict):
        # Analyze each split
        analysis = {}
        for split_name, split_dataset in dataset.items():
            analysis[split_name] = analyze_dataset(split_dataset, text_columns, tokenizer)
        return analysis
    
    # Basic statistics
    analysis = {
        "num_examples": len(dataset),
        "columns": list(dataset.column_names),
        "features": str(dataset.features)
    }
    
    # Text analysis
    if text_columns is None:
        # Auto-detect text columns
        text_columns = [col for col in dataset.column_names if 'text' in col.lower() or 'content' in col.lower()]
    
    for column in text_columns:
        if column not in dataset.column_names:
            continue
        
        texts = dataset[column]
        
        # Character-level statistics
        char_lengths = [len(text) for text in texts]
        analysis[f"{column}_char_stats"] = {
            "min_length": min(char_lengths),
            "max_length": max(char_lengths),
            "mean_length": np.mean(char_lengths),
            "median_length": np.median(char_lengths),
            "std_length": np.std(char_lengths)
        }
        
        # Token-level statistics (if tokenizer provided)
        if tokenizer is not None:
            token_lengths = []
            for text in texts[:1000]:  # Sample for efficiency
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_lengths.append(len(tokens))
            
            analysis[f"{column}_token_stats"] = {
                "min_length": min(token_lengths),
                "max_length": max(token_lengths),
                "mean_length": np.mean(token_lengths),
                "median_length": np.median(token_lengths),
                "std_length": np.std(token_lengths)
            }
        
        # Language detection (basic)
        arabic_count = sum(1 for text in texts if has_arabic(text))
        analysis[f"{column}_language"] = {
            "arabic_examples": arabic_count,
            "arabic_ratio": arabic_count / len(texts)
        }
    
    logger.info("Dataset analysis complete")
    return analysis


def has_arabic(text: str) -> bool:
    """Check if text contains Arabic characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains Arabic
    """
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    return bool(re.search(arabic_pattern, text))


def sample_dataset(
    dataset: Dataset,
    n_samples: int,
    seed: int = 42,
    stratify_column: Optional[str] = None
) -> Dataset:
    """Sample examples from dataset.
    
    Args:
        dataset: Dataset to sample from
        n_samples: Number of samples
        seed: Random seed
        stratify_column: Column for stratified sampling
        
    Returns:
        Sampled dataset
    """
    if n_samples >= len(dataset):
        logger.warning(f"Requested samples ({n_samples}) >= dataset size ({len(dataset)}), returning full dataset")
        return dataset
    
    logger.info(f"Sampling {n_samples} examples from dataset")
    
    if stratify_column is not None and stratify_column in dataset.column_names:
        # Stratified sampling
        labels = dataset[stratify_column]
        label_counts = Counter(labels)
        
        # Calculate samples per label
        samples_per_label = {}
        total_labels = len(label_counts)
        
        for label, count in label_counts.items():
            ratio = count / len(dataset)
            samples_per_label[label] = max(1, int(n_samples * ratio))
        
        # Adjust if total exceeds n_samples
        total_samples = sum(samples_per_label.values())
        if total_samples > n_samples:
            # Reduce proportionally
            factor = n_samples / total_samples
            for label in samples_per_label:
                samples_per_label[label] = max(1, int(samples_per_label[label] * factor))
        
        # Sample from each label
        sampled_indices = []
        for label, n_label_samples in samples_per_label.items():
            label_indices = [i for i, l in enumerate(labels) if l == label]
            random.seed(seed)
            sampled_label_indices = random.sample(label_indices, min(n_label_samples, len(label_indices)))
            sampled_indices.extend(sampled_label_indices)
        
        sampled_dataset = dataset.select(sampled_indices)
        
    else:
        # Random sampling
        sampled_dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    
    logger.info(f"Sampled {len(sampled_dataset)} examples")
    return sampled_dataset


def create_instruction_dataset(
    examples: List[Dict[str, str]],
    instruction_template: str = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
    instruction_key: str = "instruction",
    response_key: str = "response"
) -> Dataset:
    """Create instruction-following dataset.
    
    Args:
        examples: List of instruction-response pairs
        instruction_template: Template for formatting
        instruction_key: Key for instruction text
        response_key: Key for response text
        
    Returns:
        Formatted dataset
    """
    logger.info(f"Creating instruction dataset with {len(examples)} examples")
    
    formatted_examples = []
    
    for example in examples:
        if instruction_key not in example or response_key not in example:
            logger.warning(f"Skipping example missing required keys: {example.keys()}")
            continue
        
        formatted_text = instruction_template.format(
            instruction=example[instruction_key],
            response=example[response_key]
        )
        
        formatted_examples.append({
            "text": formatted_text,
            "instruction": example[instruction_key],
            "response": example[response_key]
        })
    
    dataset = Dataset.from_list(formatted_examples)
    
    logger.info(f"Instruction dataset created with {len(dataset)} examples")
    return dataset


def create_preference_dataset(
    examples: List[Dict[str, str]],
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected"
) -> Dataset:
    """Create preference optimization dataset.
    
    Args:
        examples: List of preference examples
        prompt_key: Key for prompt text
        chosen_key: Key for chosen response
        rejected_key: Key for rejected response
        
    Returns:
        Preference dataset
    """
    logger.info(f"Creating preference dataset with {len(examples)} examples")
    
    formatted_examples = []
    
    for example in examples:
        required_keys = [prompt_key, chosen_key, rejected_key]
        if not all(key in example for key in required_keys):
            logger.warning(f"Skipping example missing required keys: {example.keys()}")
            continue
        
        formatted_examples.append({
            "prompt": example[prompt_key],
            "chosen": example[chosen_key],
            "rejected": example[rejected_key]
        })
    
    dataset = Dataset.from_list(formatted_examples)
    
    logger.info(f"Preference dataset created with {len(dataset)} examples")
    return dataset


def visualize_dataset_stats(
    analysis: Dict[str, Any],
    output_dir: str,
    figsize: Tuple[int, int] = (12, 8)
):
    """Visualize dataset statistics.
    
    Args:
        analysis: Analysis results from analyze_dataset
        output_dir: Output directory for plots
        figsize: Figure size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating dataset visualizations in {output_dir}")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # If analysis contains multiple splits, create comparison plots
    if any(isinstance(v, dict) and 'num_examples' in v for v in analysis.values()):
        # Multi-split analysis
        splits = {k: v for k, v in analysis.items() if isinstance(v, dict) and 'num_examples' in v}
        
        # Plot 1: Number of examples per split
        fig, ax = plt.subplots(figsize=(8, 6))
        split_names = list(splits.keys())
        split_counts = [splits[split]['num_examples'] for split in split_names]
        
        bars = ax.bar(split_names, split_counts)
        ax.set_title('Number of Examples per Split')
        ax.set_ylabel('Number of Examples')
        
        # Add value labels on bars
        for bar, count in zip(bars, split_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(split_counts)*0.01,
                   f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'split_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Single dataset analysis
        splits = {'dataset': analysis}
    
    # Plot text length distributions for each split
    for split_name, split_analysis in splits.items():
        text_columns = [key.replace('_char_stats', '') for key in split_analysis.keys() if key.endswith('_char_stats')]
        
        if text_columns:
            fig, axes = plt.subplots(len(text_columns), 2, figsize=(figsize[0], figsize[1] * len(text_columns)))
            if len(text_columns) == 1:
                axes = axes.reshape(1, -1)
            
            for i, column in enumerate(text_columns):
                char_stats = split_analysis.get(f'{column}_char_stats', {})
                token_stats = split_analysis.get(f'{column}_token_stats', {})
                
                # Character length distribution (placeholder - would need actual data)
                ax1 = axes[i, 0]
                ax1.set_title(f'{column} - Character Length Distribution ({split_name})')
                ax1.set_xlabel('Character Length')
                ax1.set_ylabel('Frequency')
                
                # Add statistics text
                stats_text = f"Mean: {char_stats.get('mean_length', 0):.1f}\n"
                stats_text += f"Median: {char_stats.get('median_length', 0):.1f}\n"
                stats_text += f"Std: {char_stats.get('std_length', 0):.1f}"
                ax1.text(0.7, 0.7, stats_text, transform=ax1.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Token length distribution (if available)
                ax2 = axes[i, 1]
                if token_stats:
                    ax2.set_title(f'{column} - Token Length Distribution ({split_name})')
                    ax2.set_xlabel('Token Length')
                    ax2.set_ylabel('Frequency')
                    
                    stats_text = f"Mean: {token_stats.get('mean_length', 0):.1f}\n"
                    stats_text += f"Median: {token_stats.get('median_length', 0):.1f}\n"
                    stats_text += f"Std: {token_stats.get('std_length', 0):.1f}"
                    ax2.text(0.7, 0.7, stats_text, transform=ax2.transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                else:
                    ax2.text(0.5, 0.5, 'Token statistics\nnot available', 
                            transform=ax2.transAxes, ha='center', va='center')
                    ax2.set_title(f'{column} - Token Length (Not Available)')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{split_name}_length_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Dataset visualizations saved to {output_dir}")


def export_dataset_report(
    analysis: Dict[str, Any],
    output_path: str
):
    """Export dataset analysis report.
    
    Args:
        analysis: Analysis results
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting dataset report to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Analysis Report\n\n")
        
        def write_analysis(name, data, level=1):
            f.write(f"{'#' * (level + 1)} {name}\n\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        write_analysis(key, value, level + 1)
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            else:
                f.write(f"{data}\n\n")
        
        write_analysis("Dataset Analysis", analysis)
    
    logger.info(f"Dataset report exported to {output_path}")


# Arabic Dialect Integration Functions
def process_arabic_dialects(
    dataset: Dataset,
    text_column: str = "text",
    balance_dialects: bool = True,
    target_samples_per_dialect: int = 1000,
    create_splits: bool = True,
    augment_data: bool = True,
    min_confidence: float = 0.5
) -> Dict[str, Any]:
    """Process dataset with Arabic dialect-specific handling.
    
    Args:
        dataset: Input dataset
        text_column: Column containing text data
        balance_dialects: Whether to balance dialect distribution
        target_samples_per_dialect: Target samples per dialect when balancing
        create_splits: Whether to create dialect-specific splits
        augment_data: Whether to augment with dialect variations
        min_confidence: Minimum confidence for dialect classification
        
    Returns:
        Processing results and statistics
    """
    try:
        from .arabic_dialect_utils import ArabicDialectDatasetProcessor
        
        processor = ArabicDialectDatasetProcessor()
        
        # Analyze original dataset
        analysis = processor.analyze_dataset_dialects(dataset, text_column)
        logger.info(f"Original dataset dialect distribution: {analysis['dialect_distribution']}")
        
        results = {
            "original_analysis": analysis,
            "processed_dataset": dataset
        }
        
        # Balance dialects if requested
        if balance_dialects:
            balanced_dataset = processor.balance_dataset_by_dialect(
                dataset,
                text_column=text_column,
                target_samples_per_dialect=target_samples_per_dialect
            )
            results["processed_dataset"] = balanced_dataset
            results["balanced_analysis"] = processor.analyze_dataset_dialects(
                balanced_dataset, text_column
            )
        
        # Create dialect splits if requested
        if create_splits:
            dialect_splits = processor.create_dialect_specific_splits(
                results["processed_dataset"],
                text_column=text_column,
                min_confidence=min_confidence
            )
            results["dialect_splits"] = dialect_splits
        
        # Augment with dialects if requested
        if augment_data:
            from .arabic_dialect_utils import ArabicDialect
            target_dialects = [
                ArabicDialect.EGYPTIAN,
                ArabicDialect.LEVANTINE,
                ArabicDialect.GULF
            ]
            
            augmented_dataset = processor.augment_dataset_with_dialects(
                results["processed_dataset"],
                text_column=text_column,
                target_dialects=target_dialects,
                augmentation_factor=1
            )
            results["processed_dataset"] = augmented_dataset
            results["augmented_analysis"] = processor.analyze_dataset_dialects(
                augmented_dataset, text_column
            )
        
        return results
        
    except ImportError:
        logger.warning("Arabic dialect utilities not available. Skipping dialect processing.")
        return {"processed_dataset": dataset, "error": "Arabic dialect utilities not available"}


def detect_text_dialect(text: str) -> Tuple[str, float]:
    """Detect Arabic dialect in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (dialect_name, confidence_score)
    """
    try:
        from .arabic_dialect_utils import ArabicDialectDetector
        
        detector = ArabicDialectDetector()
        dialect, confidence = detector.detect_dialect(text)
        return dialect.value, confidence
        
    except ImportError:
        logger.warning("Arabic dialect utilities not available.")
        return "unknown", 0.0


def augment_text_with_dialect(text: str, target_dialect: str) -> str:
    """Augment text to match target Arabic dialect.
    
    Args:
        text: Original text
        target_dialect: Target dialect name
        
    Returns:
        Augmented text
    """
    try:
        from .arabic_dialect_utils import ArabicDialectAugmentor, ArabicDialect
        
        augmentor = ArabicDialectAugmentor()
        dialect_enum = ArabicDialect(target_dialect)
        return augmentor.augment_text(text, dialect_enum)
        
    except (ImportError, ValueError):
        logger.warning(f"Could not augment text with dialect {target_dialect}")
        return text


def create_dialect_balanced_dataset(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    target_samples: int = 1000
) -> Dict[str, Any]:
    """Create a dialect-balanced dataset from input data.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save balanced dataset
        text_column: Column containing text data
        target_samples: Target samples per dialect
        
    Returns:
        Processing results
    """
    logger.info(f"Creating dialect-balanced dataset: {input_path} -> {output_path}")
    
    # Load dataset
    dataset = load_dataset_from_path(input_path)
    
    # Process with dialect handling
    results = process_arabic_dialects(
        dataset,
        text_column=text_column,
        balance_dialects=True,
        target_samples_per_dialect=target_samples,
        create_splits=False,
        augment_data=True
    )
    
    # Save processed dataset
    save_dataset(results["processed_dataset"], output_path)
    
    # Save processing report
    report_path = Path(output_path).parent / "dialect_processing_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        # Convert any non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key == "processed_dataset":
                serializable_results[key] = f"Dataset with {len(value)} samples"
            elif key == "dialect_splits":
                serializable_results[key] = {
                    dialect: len(split) for dialect, split in value.items()
                }
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dialect-balanced dataset created: {output_path}")
    logger.info(f"Processing report saved: {report_path}")
    
    return results