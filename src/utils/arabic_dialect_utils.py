"""Arabic Dialect Utilities

This module provides comprehensive utilities for handling Arabic dialects
in datasets, including dialect detection, normalization, augmentation,
and dataset balancing across different Arabic varieties.
"""

import logging
import re
import json
import random
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ArabicDialect(Enum):
    """Arabic dialect classifications."""
    MSA = "msa"  # Modern Standard Arabic
    EGYPTIAN = "egyptian"
    LEVANTINE = "levantine"
    GULF = "gulf"
    MAGHREBI = "maghrebi"
    IRAQI = "iraqi"
    SUDANESE = "sudanese"
    YEMENI = "yemeni"
    UNKNOWN = "unknown"


@dataclass
class DialectFeatures:
    """Features for dialect identification."""
    keywords: List[str]
    patterns: List[str]
    negation_words: List[str]
    question_words: List[str]
    common_phrases: List[str]


class ArabicDialectDetector:
    """Arabic dialect detection and classification."""
    
    def __init__(self):
        self.dialect_features = self._initialize_dialect_features()
        self.normalization_rules = self._initialize_normalization_rules()
    
    def _initialize_dialect_features(self) -> Dict[ArabicDialect, DialectFeatures]:
        """Initialize dialect-specific features for detection."""
        return {
            ArabicDialect.EGYPTIAN: DialectFeatures(
                keywords=["ازيك", "ايه", "كده", "علشان", "عايز", "عاوز", "بقى", "خلاص", "يلا"],
                patterns=[r"ش\s+", r"مش\s+", r"ده\s+", r"دي\s+"],
                negation_words=["مش", "مبقاش", "مكانش"],
                question_words=["ايه", "فين", "امتى", "ازاي"],
                common_phrases=["ازيك", "اخبارك ايه", "كله تمام", "ربنا يخليك"]
            ),
            ArabicDialect.LEVANTINE: DialectFeatures(
                keywords=["شو", "كيف", "هيك", "هاد", "هاي", "بدي", "بدك", "يعني", "شوي"],
                patterns=[r"ما\s+.*ش", r"هاد\s+", r"هاي\s+", r"شو\s+"],
                negation_words=["ما", "مو", "مش"],
                question_words=["شو", "وين", "كيف", "ليش"],
                common_phrases=["كيفك", "شو اخبارك", "يعطيك العافية", "الله يعطيك العافية"]
            ),
            ArabicDialect.GULF: DialectFeatures(
                keywords=["شلون", "وين", "شنو", "اشلون", "ابي", "ابغى", "زين", "مال", "شكو"],
                patterns=[r"شلون\s+", r"وين\s+", r"شنو\s+", r"مال\s+"],
                negation_words=["ما", "مو", "مب"],
                question_words=["شلون", "وين", "شنو", "ليش"],
                common_phrases=["شلونك", "شنو اخبارك", "الله يعطيك العافية", "زين كذا"]
            ),
            ArabicDialect.MAGHREBI: DialectFeatures(
                keywords=["كيفاش", "فين", "واش", "بغيت", "نتا", "نتي", "هاذا", "هاذي", "بزاف"],
                patterns=[r"كيفاش\s+", r"واش\s+", r"هاذا\s+", r"هاذي\s+"],
                negation_words=["ما", "ماشي", "مكاين"],
                question_words=["كيفاش", "فين", "واش", "علاش"],
                common_phrases=["كيفاش راك", "واش اخبارك", "الله يعطيك الصحة", "بزاف مزيان"]
            ),
            ArabicDialect.IRAQI: DialectFeatures(
                keywords=["شلون", "شنو", "وين", "اريد", "اكو", "ماكو", "هسه", "شوكت"],
                patterns=[r"شلون\s+", r"شنو\s+", r"اكو\s+", r"ماكو\s+"],
                negation_words=["ما", "مو", "ماكو"],
                question_words=["شلون", "شنو", "وين", "شوكت"],
                common_phrases=["شلونك", "شنو اخبارك", "الله يعطيك العافية", "اكو شي"]
            ),
            ArabicDialect.MSA: DialectFeatures(
                keywords=["كيف", "ماذا", "أين", "متى", "لماذا", "أريد", "أحتاج", "يجب"],
                patterns=[r"لا\s+", r"ليس\s+", r"غير\s+", r"إن\s+"],
                negation_words=["لا", "ليس", "لم", "لن"],
                question_words=["كيف", "ماذا", "أين", "متى", "لماذا"],
                common_phrases=["كيف حالك", "ما أخبارك", "أشكرك جزيلا", "مع السلامة"]
            )
        }
    
    def _initialize_normalization_rules(self) -> Dict[str, str]:
        """Initialize text normalization rules."""
        return {
            # Diacritics removal
            r'[\u064B-\u0652\u0670\u0640]': '',
            # Normalize Alef variants
            r'[إأآا]': 'ا',
            # Normalize Teh Marbuta
            r'ة': 'ه',
            # Normalize Yeh variants
            r'[يى]': 'ي',
            # Remove extra spaces
            r'\s+': ' ',
            # Remove punctuation for analysis
            r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]': ''
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Arabic text for better dialect detection."""
        normalized = text.strip()
        for pattern, replacement in self.normalization_rules.items():
            normalized = re.sub(pattern, replacement, normalized)
        return normalized.strip()
    
    def detect_dialect(self, text: str, confidence_threshold: float = 0.3) -> Tuple[ArabicDialect, float]:
        """Detect Arabic dialect in text.
        
        Args:
            text: Arabic text to analyze
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Tuple of (dialect, confidence_score)
        """
        if not self.has_arabic(text):
            return ArabicDialect.UNKNOWN, 0.0
        
        normalized_text = self.normalize_text(text.lower())
        scores = defaultdict(float)
        
        # Score based on dialect-specific features
        for dialect, features in self.dialect_features.items():
            score = 0.0
            total_features = 0
            
            # Check keywords
            for keyword in features.keywords:
                if keyword in normalized_text:
                    score += 2.0
                total_features += 1
            
            # Check patterns
            for pattern in features.patterns:
                if re.search(pattern, normalized_text):
                    score += 1.5
                total_features += 1
            
            # Check negation words
            for neg_word in features.negation_words:
                if neg_word in normalized_text:
                    score += 1.0
                total_features += 1
            
            # Check question words
            for q_word in features.question_words:
                if q_word in normalized_text:
                    score += 1.0
                total_features += 1
            
            # Check common phrases
            for phrase in features.common_phrases:
                if phrase in normalized_text:
                    score += 3.0
                total_features += 1
            
            # Normalize score
            if total_features > 0:
                scores[dialect] = score / total_features
        
        # Find best match
        if not scores:
            return ArabicDialect.UNKNOWN, 0.0
        
        best_dialect = max(scores, key=scores.get)
        confidence = scores[best_dialect]
        
        if confidence < confidence_threshold:
            return ArabicDialect.UNKNOWN, confidence
        
        return best_dialect, confidence
    
    def has_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        return bool(re.search(arabic_pattern, text))


class ArabicDialectAugmentor:
    """Arabic dialect data augmentation."""
    
    def __init__(self):
        self.detector = ArabicDialectDetector()
        self.augmentation_rules = self._initialize_augmentation_rules()
    
    def _initialize_augmentation_rules(self) -> Dict[ArabicDialect, Dict[str, List[str]]]:
        """Initialize dialect-specific augmentation rules."""
        return {
            ArabicDialect.EGYPTIAN: {
                "synonyms": {
                    "كيف": ["ازاي", "ازيك"],
                    "ماذا": ["ايه", "ايه ده"],
                    "أين": ["فين"],
                    "أريد": ["عايز", "عاوز"],
                    "لا": ["مش", "لأ"],
                    "نعم": ["ايوه", "اه"]
                },
                "transformations": [
                    (r"\bكيف\b", "ازاي"),
                    (r"\bماذا\b", "ايه"),
                    (r"\bأين\b", "فين"),
                    (r"\bلا\b", "مش")
                ]
            },
            ArabicDialect.LEVANTINE: {
                "synonyms": {
                    "كيف": ["شو", "كيف"],
                    "ماذا": ["شو", "ايش"],
                    "أين": ["وين"],
                    "أريد": ["بدي", "بدك"],
                    "لا": ["ما", "مو"],
                    "هذا": ["هاد"],
                    "هذه": ["هاي"]
                },
                "transformations": [
                    (r"\bكيف\b", "شو"),
                    (r"\bماذا\b", "شو"),
                    (r"\bأين\b", "وين"),
                    (r"\bهذا\b", "هاد"),
                    (r"\bهذه\b", "هاي")
                ]
            },
            ArabicDialect.GULF: {
                "synonyms": {
                    "كيف": ["شلون"],
                    "ماذا": ["شنو"],
                    "أين": ["وين"],
                    "أريد": ["ابي", "ابغى"],
                    "لا": ["ما", "مو"],
                    "جيد": ["زين", "طيب"]
                },
                "transformations": [
                    (r"\bكيف\b", "شلون"),
                    (r"\bماذا\b", "شنو"),
                    (r"\bأين\b", "وين"),
                    (r"\bأريد\b", "ابي")
                ]
            }
        }
    
    def augment_text(self, text: str, target_dialect: ArabicDialect, 
                    augmentation_ratio: float = 0.3) -> str:
        """Augment text to match target dialect.
        
        Args:
            text: Original text
            target_dialect: Target dialect for augmentation
            augmentation_ratio: Ratio of words to augment
            
        Returns:
            Augmented text
        """
        if target_dialect not in self.augmentation_rules:
            return text
        
        rules = self.augmentation_rules[target_dialect]
        augmented_text = text
        
        # Apply transformations
        for pattern, replacement in rules.get("transformations", []):
            if random.random() < augmentation_ratio:
                augmented_text = re.sub(pattern, replacement, augmented_text)
        
        # Apply synonym replacements
        words = augmented_text.split()
        for i, word in enumerate(words):
            if word in rules.get("synonyms", {}) and random.random() < augmentation_ratio:
                synonyms = rules["synonyms"][word]
                words[i] = random.choice(synonyms)
        
        return " ".join(words)
    
    def generate_dialect_variants(self, text: str, 
                                dialects: List[ArabicDialect] = None) -> Dict[ArabicDialect, str]:
        """Generate variants of text in different dialects.
        
        Args:
            text: Original text
            dialects: List of target dialects
            
        Returns:
            Dictionary mapping dialects to variant texts
        """
        if dialects is None:
            dialects = [ArabicDialect.EGYPTIAN, ArabicDialect.LEVANTINE, ArabicDialect.GULF]
        
        variants = {}
        for dialect in dialects:
            variants[dialect] = self.augment_text(text, dialect)
        
        return variants


class ArabicDialectDatasetProcessor:
    """Comprehensive Arabic dialect dataset processing."""
    
    def __init__(self):
        self.detector = ArabicDialectDetector()
        self.augmentor = ArabicDialectAugmentor()
    
    def analyze_dataset_dialects(self, dataset: Dataset, 
                               text_column: str = "text") -> Dict[str, Any]:
        """Analyze dialect distribution in dataset.
        
        Args:
            dataset: Dataset to analyze
            text_column: Column containing text data
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing dialect distribution in dataset...")
        
        dialect_counts = Counter()
        confidence_scores = []
        dialect_examples = defaultdict(list)
        
        for i, example in enumerate(dataset):
            text = example.get(text_column, "")
            if not text:
                continue
            
            dialect, confidence = self.detector.detect_dialect(text)
            dialect_counts[dialect.value] += 1
            confidence_scores.append(confidence)
            
            # Store examples for each dialect
            if len(dialect_examples[dialect.value]) < 5:
                dialect_examples[dialect.value].append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "confidence": confidence
                })
        
        total_samples = len(dataset)
        analysis = {
            "total_samples": total_samples,
            "dialect_distribution": dict(dialect_counts),
            "dialect_percentages": {
                dialect: (count / total_samples) * 100 
                for dialect, count in dialect_counts.items()
            },
            "average_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
            "confidence_std": np.std(confidence_scores) if confidence_scores else 0.0,
            "dialect_examples": dict(dialect_examples)
        }
        
        logger.info(f"Dialect analysis complete. Found {len(dialect_counts)} dialects.")
        return analysis
    
    def balance_dataset_by_dialect(self, dataset: Dataset, 
                                 text_column: str = "text",
                                 target_samples_per_dialect: int = 1000,
                                 augment_minority: bool = True) -> Dataset:
        """Balance dataset across Arabic dialects.
        
        Args:
            dataset: Input dataset
            text_column: Column containing text data
            target_samples_per_dialect: Target number of samples per dialect
            augment_minority: Whether to augment minority dialects
            
        Returns:
            Balanced dataset
        """
        logger.info("Balancing dataset by Arabic dialects...")
        
        # Group samples by dialect
        dialect_groups = defaultdict(list)
        
        for i, example in enumerate(dataset):
            text = example.get(text_column, "")
            if not text:
                continue
            
            dialect, confidence = self.detector.detect_dialect(text)
            if confidence > 0.3:  # Only include confident classifications
                dialect_groups[dialect.value].append(example)
        
        balanced_samples = []
        
        for dialect, samples in dialect_groups.items():
            current_count = len(samples)
            
            if current_count >= target_samples_per_dialect:
                # Downsample
                selected_samples = random.sample(samples, target_samples_per_dialect)
                balanced_samples.extend(selected_samples)
                logger.info(f"Downsampled {dialect}: {current_count} -> {target_samples_per_dialect}")
            
            elif augment_minority and current_count > 0:
                # Upsample with augmentation
                balanced_samples.extend(samples)
                needed_samples = target_samples_per_dialect - current_count
                
                # Generate augmented samples
                for _ in range(needed_samples):
                    original_sample = random.choice(samples)
                    augmented_text = self.augmentor.augment_text(
                        original_sample[text_column], 
                        ArabicDialect(dialect)
                    )
                    
                    # Create augmented sample
                    augmented_sample = original_sample.copy()
                    augmented_sample[text_column] = augmented_text
                    augmented_sample["augmented"] = True
                    balanced_samples.append(augmented_sample)
                
                logger.info(f"Augmented {dialect}: {current_count} -> {target_samples_per_dialect}")
            
            else:
                # Keep all samples if not augmenting
                balanced_samples.extend(samples)
                logger.info(f"Kept {dialect}: {current_count} samples")
        
        # Shuffle the balanced dataset
        random.shuffle(balanced_samples)
        
        # Convert back to Dataset
        balanced_dataset = Dataset.from_list(balanced_samples)
        
        logger.info(f"Dataset balanced: {len(dataset)} -> {len(balanced_dataset)} samples")
        return balanced_dataset
    
    def create_dialect_specific_splits(self, dataset: Dataset,
                                     text_column: str = "text",
                                     min_confidence: float = 0.5) -> DatasetDict:
        """Create dialect-specific dataset splits.
        
        Args:
            dataset: Input dataset
            text_column: Column containing text data
            min_confidence: Minimum confidence for dialect classification
            
        Returns:
            DatasetDict with dialect-specific splits
        """
        logger.info("Creating dialect-specific dataset splits...")
        
        dialect_datasets = defaultdict(list)
        
        for example in dataset:
            text = example.get(text_column, "")
            if not text:
                continue
            
            dialect, confidence = self.detector.detect_dialect(text)
            
            if confidence >= min_confidence:
                dialect_datasets[dialect.value].append(example)
        
        # Convert to DatasetDict
        dataset_dict = {}
        for dialect, samples in dialect_datasets.items():
            if samples:  # Only include dialects with samples
                dataset_dict[dialect] = Dataset.from_list(samples)
                logger.info(f"Created {dialect} split with {len(samples)} samples")
        
        return DatasetDict(dataset_dict)
    
    def augment_dataset_with_dialects(self, dataset: Dataset,
                                    text_column: str = "text",
                                    target_dialects: List[ArabicDialect] = None,
                                    augmentation_factor: int = 2) -> Dataset:
        """Augment dataset with dialect variations.
        
        Args:
            dataset: Input dataset
            text_column: Column containing text data
            target_dialects: Dialects to generate variations for
            augmentation_factor: Number of variations per original sample
            
        Returns:
            Augmented dataset
        """
        if target_dialects is None:
            target_dialects = [ArabicDialect.EGYPTIAN, ArabicDialect.LEVANTINE, ArabicDialect.GULF]
        
        logger.info(f"Augmenting dataset with {len(target_dialects)} dialect variations...")
        
        augmented_samples = []
        
        for example in dataset:
            text = example.get(text_column, "")
            if not text or not self.detector.has_arabic(text):
                continue
            
            # Add original sample
            augmented_samples.append(example)
            
            # Generate dialect variations
            for dialect in target_dialects:
                for i in range(augmentation_factor):
                    augmented_text = self.augmentor.augment_text(text, dialect)
                    
                    # Create augmented sample
                    augmented_sample = example.copy()
                    augmented_sample[text_column] = augmented_text
                    augmented_sample["source_dialect"] = dialect.value
                    augmented_sample["augmented"] = True
                    augmented_sample["augmentation_id"] = i
                    
                    augmented_samples.append(augmented_sample)
        
        # Convert to Dataset
        augmented_dataset = Dataset.from_list(augmented_samples)
        
        logger.info(f"Dataset augmented: {len(dataset)} -> {len(augmented_dataset)} samples")
        return augmented_dataset


def create_dialect_aware_dataset(input_path: str,
                                output_path: str,
                                processing_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a dialect-aware dataset with comprehensive processing.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save processed dataset
        processing_config: Configuration for processing
        
    Returns:
        Processing results and statistics
    """
    if processing_config is None:
        processing_config = {
            "balance_dialects": True,
            "target_samples_per_dialect": 1000,
            "create_dialect_splits": True,
            "augment_with_dialects": True,
            "augmentation_factor": 1,
            "min_confidence": 0.5,
            "text_column": "text"
        }
    
    logger.info(f"Creating dialect-aware dataset from {input_path}")
    
    # Load dataset
    from .data_utils import load_dataset_from_path, save_dataset
    dataset = load_dataset_from_path(input_path)
    
    # Initialize processor
    processor = ArabicDialectDatasetProcessor()
    
    # Analyze original dataset
    original_analysis = processor.analyze_dataset_dialects(
        dataset, processing_config["text_column"]
    )
    
    results = {
        "original_analysis": original_analysis,
        "processing_config": processing_config
    }
    
    # Balance dialects if requested
    if processing_config.get("balance_dialects", False):
        dataset = processor.balance_dataset_by_dialect(
            dataset,
            text_column=processing_config["text_column"],
            target_samples_per_dialect=processing_config["target_samples_per_dialect"]
        )
        
        balanced_analysis = processor.analyze_dataset_dialects(
            dataset, processing_config["text_column"]
        )
        results["balanced_analysis"] = balanced_analysis
    
    # Create dialect-specific splits if requested
    if processing_config.get("create_dialect_splits", False):
        dialect_splits = processor.create_dialect_specific_splits(
            dataset,
            text_column=processing_config["text_column"],
            min_confidence=processing_config["min_confidence"]
        )
        
        # Save dialect splits
        output_dir = Path(output_path).parent
        for dialect, split_dataset in dialect_splits.items():
            split_path = output_dir / f"dialect_{dialect}.json"
            save_dataset(split_dataset, str(split_path))
        
        results["dialect_splits"] = {
            dialect: len(split_dataset) 
            for dialect, split_dataset in dialect_splits.items()
        }
    
    # Augment with dialects if requested
    if processing_config.get("augment_with_dialects", False):
        target_dialects = [
            ArabicDialect.EGYPTIAN,
            ArabicDialect.LEVANTINE,
            ArabicDialect.GULF
        ]
        
        dataset = processor.augment_dataset_with_dialects(
            dataset,
            text_column=processing_config["text_column"],
            target_dialects=target_dialects,
            augmentation_factor=processing_config["augmentation_factor"]
        )
        
        augmented_analysis = processor.analyze_dataset_dialects(
            dataset, processing_config["text_column"]
        )
        results["augmented_analysis"] = augmented_analysis
    
    # Save final dataset
    save_dataset(dataset, output_path)
    
    # Save processing results
    results_path = Path(output_path).parent / "dialect_processing_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dialect-aware dataset created: {output_path}")
    logger.info(f"Processing results saved: {results_path}")
    
    return results