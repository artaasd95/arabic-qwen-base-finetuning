"""Arabic Evaluator Module

This module contains the evaluator implementation specifically designed for
Arabic language models with specialized metrics and evaluation methods.
"""

import logging
import torch
import numpy as np
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from datasets import Dataset
import json
from pathlib import Path

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ArabicEvaluator(BaseEvaluator):
    """Evaluator for Arabic language models.
    
    This class provides evaluation methods specifically designed for
    Arabic language models with specialized metrics for Arabic text.
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        dialect: str = "msa",
        diacritics_enabled: bool = True
    ):
        """Initialize the Arabic evaluator.
        
        Args:
            model: Model instance or path
            tokenizer: Tokenizer instance or path
            device: Device to run evaluation on
            dialect: Arabic dialect ('msa', 'egyptian', 'gulf', 'levantine', 'maghrebi')
            diacritics_enabled: Whether to evaluate diacritics
        """
        super().__init__(model, tokenizer, device)
        self.dialect = dialect.lower()
        self.diacritics_enabled = diacritics_enabled
        
        # Arabic character ranges
        self.arabic_range = (0x0600, 0x06FF)
        self.arabic_supplement_range = (0x0750, 0x077F)
        self.diacritics_range = (0x064B, 0x065F)
        
        # Common Arabic patterns
        self.arabic_patterns = self._load_arabic_patterns()
        
        logger.info(f"ArabicEvaluator initialized for {self.dialect} dialect")
    
    def _load_arabic_patterns(self) -> Dict[str, List[str]]:
        """Load Arabic language patterns for evaluation."""
        return {
            "question_words": ["ما", "كيف", "لماذا", "أين", "متى", "من", "ماذا", "هل"],
            "common_prefixes": ["ال", "و", "ف", "ب", "ك", "ل"],
            "common_suffixes": ["ة", "ات", "ان", "ين", "ون", "ها", "هم", "هن"],
            "formal_indicators": ["إن", "أن", "كان", "يكون", "سوف", "قد"],
            "dialectal_markers": {
                "egyptian": ["ده", "دي", "إيه", "ازاي", "فين"],
                "gulf": ["شلون", "وين", "شنو", "هذا", "هذي"],
                "levantine": ["شو", "وين", "كيف", "هيك", "هاي"],
                "maghrebi": ["شنو", "فين", "كيفاش", "واش", "هاذ"]
            }
        }
    
    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """Main evaluation method for Arabic models.
        
        Args:
            dataset: Dataset containing Arabic text
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing comprehensive Arabic evaluation metrics
        """
        # Get base evaluation metrics
        base_results = super().compute_basic_metrics(dataset)
        
        # Compute Arabic-specific metrics
        arabic_results = self.compute_arabic_metrics(dataset)
        
        # Combine results
        results = {
            **base_results,
            **arabic_results,
            "dialect": self.dialect,
            "diacritics_enabled": self.diacritics_enabled
        }
        
        return results
    
    def compute_arabic_metrics(self, dataset: Dataset) -> Dict[str, Any]:
        """Compute comprehensive Arabic language metrics.
        
        Args:
            dataset: Dataset with Arabic text
            
        Returns:
            Dictionary with Arabic-specific metrics
        """
        texts = self._extract_texts(dataset)
        
        metrics = {
            # Character-level metrics
            "arabic_character_ratio": self._compute_arabic_character_ratio(texts),
            "diacritics_coverage": self._compute_diacritics_coverage(texts),
            "script_consistency": self._compute_script_consistency(texts),
            
            # Linguistic metrics
            "dialect_detection_accuracy": self._compute_dialect_detection(texts),
            "formality_score": self._compute_formality_score(texts),
            "grammar_correctness": self._compute_grammar_correctness(texts),
            
            # Cultural metrics
            "cultural_appropriateness": self._compute_cultural_appropriateness(texts),
            "religious_sensitivity": self._compute_religious_sensitivity(texts),
            
            # Quality metrics
            "text_fluency": self._compute_text_fluency(texts),
            "coherence_score": self._compute_coherence_score(texts),
            "readability_score": self._compute_readability_score(texts)
        }
        
        return metrics
    
    def _extract_texts(self, dataset: Dataset) -> List[str]:
        """Extract text content from dataset."""
        texts = []
        
        for example in dataset:
            # Handle different dataset formats
            if "text" in example:
                texts.append(example["text"])
            elif "response" in example:
                texts.append(example["response"])
            elif "completion" in example:
                texts.append(example["completion"])
            elif "chosen" in example:
                texts.append(example["chosen"])
            else:
                # Try to find any text field
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 10:
                        texts.append(value)
                        break
        
        return texts
    
    def _compute_arabic_character_ratio(self, texts: List[str]) -> float:
        """Compute ratio of Arabic characters in texts."""
        total_chars = 0
        arabic_chars = 0
        
        for text in texts:
            for char in text:
                if char.isalpha():
                    total_chars += 1
                    if self._is_arabic_char(char):
                        arabic_chars += 1
        
        return arabic_chars / total_chars if total_chars > 0 else 0.0
    
    def _is_arabic_char(self, char: str) -> bool:
        """Check if character is Arabic."""
        code = ord(char)
        return (self.arabic_range[0] <= code <= self.arabic_range[1] or
                self.arabic_supplement_range[0] <= code <= self.arabic_supplement_range[1])
    
    def _compute_diacritics_coverage(self, texts: List[str]) -> float:
        """Compute diacritics coverage in texts."""
        if not self.diacritics_enabled:
            return 0.0
        
        total_words = 0
        diacritized_words = 0
        
        for text in texts:
            words = text.split()
            for word in words:
                if any(self._is_arabic_char(c) for c in word):
                    total_words += 1
                    if any(self.diacritics_range[0] <= ord(c) <= self.diacritics_range[1] for c in word):
                        diacritized_words += 1
        
        return diacritized_words / total_words if total_words > 0 else 0.0
    
    def _compute_script_consistency(self, texts: List[str]) -> float:
        """Compute script consistency (avoiding mixed scripts)."""
        consistent_texts = 0
        
        for text in texts:
            # Check for script mixing
            has_arabic = any(self._is_arabic_char(c) for c in text)
            has_latin = any(c.isalpha() and not self._is_arabic_char(c) for c in text)
            
            # Allow some Latin for technical terms, URLs, etc.
            latin_ratio = sum(1 for c in text if c.isalpha() and not self._is_arabic_char(c)) / max(1, len([c for c in text if c.isalpha()]))
            
            if has_arabic and (not has_latin or latin_ratio < 0.1):
                consistent_texts += 1
            elif not has_arabic and has_latin:
                consistent_texts += 1
        
        return consistent_texts / len(texts) if texts else 0.0
    
    def _compute_dialect_detection(self, texts: List[str]) -> float:
        """Compute dialect detection accuracy."""
        correct_dialect = 0
        
        for text in texts:
            detected_dialect = self._detect_dialect(text)
            if detected_dialect == self.dialect:
                correct_dialect += 1
        
        return correct_dialect / len(texts) if texts else 0.0
    
    def _detect_dialect(self, text: str) -> str:
        """Detect Arabic dialect in text."""
        text_lower = text.lower()
        
        # Count dialect markers
        dialect_scores = {"msa": 0}
        
        for dialect, markers in self.arabic_patterns["dialectal_markers"].items():
            score = sum(1 for marker in markers if marker in text_lower)
            dialect_scores[dialect] = score
        
        # Check for formal indicators (MSA)
        formal_score = sum(1 for indicator in self.arabic_patterns["formal_indicators"] if indicator in text_lower)
        dialect_scores["msa"] += formal_score * 2  # Weight formal indicators more
        
        # Return dialect with highest score
        return max(dialect_scores, key=dialect_scores.get)
    
    def _compute_formality_score(self, texts: List[str]) -> float:
        """Compute formality score of Arabic texts."""
        total_score = 0.0
        
        for text in texts:
            score = 0.0
            words = text.split()
            
            # Check for formal indicators
            formal_count = sum(1 for indicator in self.arabic_patterns["formal_indicators"] if indicator in text)
            score += formal_count / max(1, len(words)) * 0.4
            
            # Check for diacritics (formal Arabic often has diacritics)
            if self.diacritics_enabled:
                diacritic_count = sum(1 for c in text if self.diacritics_range[0] <= ord(c) <= self.diacritics_range[1])
                score += min(diacritic_count / max(1, len(text)), 0.1) * 0.3
            
            # Check for dialectal markers (reduce formality)
            dialectal_count = 0
            for markers in self.arabic_patterns["dialectal_markers"].values():
                dialectal_count += sum(1 for marker in markers if marker in text.lower())
            score -= dialectal_count / max(1, len(words)) * 0.3
            
            # Check sentence structure complexity
            sentences = re.split(r'[.!?]', text)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
            if avg_sentence_length > 10:  # Longer sentences often more formal
                score += 0.2
            
            total_score += max(0.0, min(1.0, score))
        
        return total_score / len(texts) if texts else 0.0
    
    def _compute_grammar_correctness(self, texts: List[str]) -> float:
        """Compute grammar correctness score."""
        # Simplified grammar checking based on patterns
        total_score = 0.0
        
        for text in texts:
            score = 1.0  # Start with perfect score
            
            # Check for common grammar issues
            # 1. Proper use of definite article
            words = text.split()
            for i, word in enumerate(words):
                # Check for double definite articles
                if word.startswith("الال"):
                    score -= 0.1
                
                # Check for proper prefix usage
                if len(word) > 3 and word[:2] in self.arabic_patterns["common_prefixes"]:
                    if word[2:4] == "ال" and word[:2] in ["و", "ف", "ب"]:
                        score -= 0.05  # Redundant prefix
            
            # Check for sentence completeness
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence.split()) < 3:
                    score -= 0.05  # Very short sentences might be incomplete
            
            total_score += max(0.0, score)
        
        return total_score / len(texts) if texts else 0.0
    
    def _compute_cultural_appropriateness(self, texts: List[str]) -> float:
        """Compute cultural appropriateness score."""
        # Check for culturally appropriate content
        appropriate_count = 0
        
        # Define inappropriate patterns (simplified)
        inappropriate_patterns = [
            # Add patterns that might be culturally inappropriate
            # This is a simplified example
        ]
        
        for text in texts:
            is_appropriate = True
            text_lower = text.lower()
            
            # Check for inappropriate content
            for pattern in inappropriate_patterns:
                if pattern in text_lower:
                    is_appropriate = False
                    break
            
            if is_appropriate:
                appropriate_count += 1
        
        return appropriate_count / len(texts) if texts else 1.0
    
    def _compute_religious_sensitivity(self, texts: List[str]) -> float:
        """Compute religious sensitivity score."""
        # Check for appropriate handling of religious content
        sensitive_count = 0
        
        religious_terms = ["الله", "النبي", "القرآن", "الإسلام", "المسجد", "الصلاة"]
        
        for text in texts:
            has_religious_content = any(term in text for term in religious_terms)
            
            if has_religious_content:
                # Check for respectful treatment
                # This is a simplified check
                is_respectful = True  # Assume respectful unless proven otherwise
                
                # Add specific checks for disrespectful language
                # This would need more sophisticated analysis
                
                if is_respectful:
                    sensitive_count += 1
            else:
                # No religious content, so no sensitivity issues
                sensitive_count += 1
        
        return sensitive_count / len(texts) if texts else 1.0
    
    def _compute_text_fluency(self, texts: List[str]) -> float:
        """Compute text fluency score."""
        total_score = 0.0
        
        for text in texts:
            score = 0.0
            words = text.split()
            
            # Check word length distribution
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if 3 <= avg_word_length <= 8:  # Reasonable word length
                    score += 0.3
            
            # Check sentence structure
            sentences = re.split(r'[.!?]', text)
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if 5 <= avg_sentence_length <= 20:  # Reasonable sentence length
                    score += 0.3
            
            # Check for repetition
            unique_words = set(words)
            if words and len(unique_words) / len(words) > 0.7:  # Good vocabulary diversity
                score += 0.2
            
            # Check for proper punctuation
            if re.search(r'[.!?]', text):
                score += 0.2
            
            total_score += score
        
        return total_score / len(texts) if texts else 0.0
    
    def _compute_coherence_score(self, texts: List[str]) -> float:
        """Compute text coherence score."""
        total_score = 0.0
        
        for text in texts:
            score = 0.0
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                # Check for logical flow between sentences
                # This is a simplified approach
                
                # Check for connecting words
                connectors = ["و", "لكن", "أيضا", "كما", "بالإضافة", "علاوة", "لذلك", "إذن"]
                connector_count = sum(1 for connector in connectors if connector in text)
                score += min(connector_count / len(sentences), 0.5)
                
                # Check for topic consistency (simplified)
                # Count repeated key terms across sentences
                all_words = text.split()
                key_words = [word for word in all_words if len(word) > 4 and self._is_arabic_char(word[0])]
                if key_words:
                    word_freq = {}
                    for word in key_words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                    
                    repeated_words = sum(1 for freq in word_freq.values() if freq > 1)
                    score += min(repeated_words / len(sentences), 0.5)
            else:
                score = 0.8  # Single sentence, assume coherent
            
            total_score += score
        
        return total_score / len(texts) if texts else 0.0
    
    def _compute_readability_score(self, texts: List[str]) -> float:
        """Compute readability score for Arabic texts."""
        total_score = 0.0
        
        for text in texts:
            score = 1.0  # Start with perfect readability
            
            words = text.split()
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if words and sentences:
                # Average word length (shorter is more readable)
                avg_word_length = sum(len(word) for word in words) / len(words)
                if avg_word_length > 8:
                    score -= 0.2
                
                # Average sentence length (shorter is more readable)
                avg_sentence_length = len(words) / len(sentences)
                if avg_sentence_length > 15:
                    score -= 0.2
                
                # Vocabulary complexity (simpler is more readable)
                complex_words = sum(1 for word in words if len(word) > 10)
                complexity_ratio = complex_words / len(words)
                score -= complexity_ratio * 0.3
                
                # Punctuation usage (proper punctuation improves readability)
                if not re.search(r'[.!?]', text):
                    score -= 0.2
            
            total_score += max(0.0, score)
        
        return total_score / len(texts) if texts else 0.0
    
    def evaluate_diacritization_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Evaluate diacritization accuracy.
        
        Args:
            predictions: List of predicted diacritized texts
            references: List of reference diacritized texts
            
        Returns:
            Diacritization accuracy score
        """
        if not self.diacritics_enabled:
            return 0.0
        
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            # Remove spaces and normalize
            pred_clean = re.sub(r'\s+', '', pred)
            ref_clean = re.sub(r'\s+', '', ref)
            
            # Compare character by character
            min_len = min(len(pred_clean), len(ref_clean))
            for i in range(min_len):
                total_chars += 1
                if pred_clean[i] == ref_clean[i]:
                    correct_chars += 1
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def evaluate_dialect_consistency(self, texts: List[str]) -> float:
        """Evaluate dialect consistency across texts.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            Dialect consistency score
        """
        if not texts:
            return 0.0
        
        detected_dialects = [self._detect_dialect(text) for text in texts]
        
        # Count most common dialect
        dialect_counts = {}
        for dialect in detected_dialects:
            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
        
        most_common_dialect = max(dialect_counts, key=dialect_counts.get)
        consistency_score = dialect_counts[most_common_dialect] / len(texts)
        
        return consistency_score
    
    def generate_arabic_response(self, prompt: str, **kwargs) -> str:
        """Generate Arabic response for given prompt.
        
        Args:
            prompt: Input prompt in Arabic
            **kwargs: Generation parameters
            
        Returns:
            Generated Arabic response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded for generation")
        
        # Ensure prompt is properly formatted for Arabic
        if not any(self._is_arabic_char(c) for c in prompt):
            logger.warning("Prompt does not contain Arabic characters")
        
        # Generate response
        response = self.generate_text(prompt, **kwargs)
        
        return response
    
    def compare_arabic_models(self, other_evaluator: 'ArabicEvaluator', dataset: Dataset) -> Dict[str, Any]:
        """Compare two Arabic models.
        
        Args:
            other_evaluator: Another ArabicEvaluator instance
            dataset: Evaluation dataset
            
        Returns:
            Comparison results
        """
        # Evaluate both models
        results_1 = self.evaluate(dataset)
        results_2 = other_evaluator.evaluate(dataset)
        
        # Compare key metrics
        comparison = {
            "model_1": {
                "dialect": self.dialect,
                "metrics": results_1
            },
            "model_2": {
                "dialect": other_evaluator.dialect,
                "metrics": results_2
            },
            "comparison": {}
        }
        
        # Compare specific metrics
        arabic_metrics = [
            "arabic_character_ratio",
            "diacritics_coverage",
            "dialect_detection_accuracy",
            "formality_score",
            "grammar_correctness",
            "cultural_appropriateness",
            "text_fluency",
            "coherence_score",
            "readability_score"
        ]
        
        for metric in arabic_metrics:
            if metric in results_1 and metric in results_2:
                comparison["comparison"][metric] = {
                    "model_1": results_1[metric],
                    "model_2": results_2[metric],
                    "difference": results_1[metric] - results_2[metric],
                    "better_model": "model_1" if results_1[metric] > results_2[metric] else "model_2"
                }
        
        return comparison


def create_arabic_evaluator(
    model=None,
    tokenizer=None,
    device=None,
    dialect: str = "msa",
    diacritics_enabled: bool = True,
    **kwargs
) -> ArabicEvaluator:
    """Factory function to create an Arabic evaluator.
    
    Args:
        model: Model instance or path
        tokenizer: Tokenizer instance or path
        device: Device to use
        dialect: Arabic dialect
        diacritics_enabled: Whether to evaluate diacritics
        **kwargs: Additional parameters
        
    Returns:
        ArabicEvaluator instance
    """
    return ArabicEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dialect=dialect,
        diacritics_enabled=diacritics_enabled,
        **kwargs
    )