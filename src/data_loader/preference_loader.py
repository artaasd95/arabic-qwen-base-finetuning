"""Preference Optimization Data Loader Module

This module contains the data loader implementation for preference optimization
methods including KTO, IPO, and CPO training of Arabic Qwen models.
"""

import logging
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizer
from datasets import Dataset

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class PreferenceDataLoader(BaseDataLoader):
    """Data loader for preference optimization methods (KTO, IPO, CPO).
    
    This class handles loading and preprocessing of preference datasets
    for various preference optimization training methods.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        method: str = "kto",
        max_prompt_length: int = 512,
        max_length: int = 1024,
        **kwargs
    ):
        """Initialize the preference data loader.
        
        Args:
            tokenizer: Pre-trained tokenizer
            method: Preference optimization method ("kto", "ipo", "cpo")
            max_prompt_length: Maximum length for prompts
            max_length: Maximum total sequence length
            **kwargs: Additional arguments passed to BaseDataLoader
        """
        super().__init__(tokenizer, **kwargs)
        
        self.method = method.lower()
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length
        
        if self.method not in ["kto", "ipo", "cpo"]:
            raise ValueError(f"Unsupported method: {method}. Supported methods: kto, ipo, cpo")
        
        logger.info(f"Initialized PreferenceDataLoader for {method.upper()} with max_prompt_length={max_prompt_length}, max_length={max_length}")
    
    def get_required_columns(self) -> List[str]:
        """Get the list of required columns for preference optimization.
        
        Returns:
            List of required column names
        """
        if self.method == "kto":
            # KTO requires prompt and completion with desirability label
            return ["prompt", "completion", "label"]
        else:
            # IPO and CPO use the same format as DPO
            return ["prompt", "chosen", "rejected"]
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for preference optimization datasets.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Preprocessed batch with tokenized inputs
        """
        if self.method == "kto":
            return self._preprocess_kto(examples)
        else:
            return self._preprocess_pairwise(examples)
    
    def _preprocess_kto(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for KTO datasets.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Preprocessed batch for KTO training
        """
        batch_size = len(examples[list(examples.keys())[0]])
        
        # Prepare lists for processed data
        prompt_input_ids = []
        prompt_attention_mask = []
        completion_input_ids = []
        completion_attention_mask = []
        completion_labels = []
        kto_labels = []
        
        for i in range(batch_size):
            # Extract prompt, completion, and label
            prompt, completion, label = self._extract_kto_data(examples, i)
            
            if prompt is None or completion is None or label is None:
                logger.warning(f"Skipping example {i} due to missing data")
                continue
            
            # Process prompt
            prompt_tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                padding=False,
                return_tensors=None
            )
            
            # Process full sequence (prompt + completion)
            full_text = prompt + completion
            full_tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Create labels (mask prompt part)
            labels_list = full_tokenized["input_ids"].copy()
            prompt_length = len(prompt_tokenized["input_ids"])
            
            # Mask prompt tokens
            for j in range(min(prompt_length, len(labels_list))):
                labels_list[j] = -100
            
            # Convert label to binary (1 for desirable, 0 for undesirable)
            binary_label = 1 if label in [1, "1", True, "true", "desirable", "good"] else 0
            
            # Add to batch
            prompt_input_ids.append(prompt_tokenized["input_ids"])
            prompt_attention_mask.append(prompt_tokenized["attention_mask"])
            completion_input_ids.append(full_tokenized["input_ids"])
            completion_attention_mask.append(full_tokenized["attention_mask"])
            completion_labels.append(labels_list)
            kto_labels.append(binary_label)
        
        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completion_input_ids": completion_input_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_labels": completion_labels,
            "kto_labels": kto_labels
        }
    
    def _preprocess_pairwise(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for pairwise preference datasets (IPO, CPO).
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Preprocessed batch for pairwise preference training
        """
        batch_size = len(examples[list(examples.keys())[0]])
        
        # Prepare lists for processed data
        prompt_input_ids = []
        prompt_attention_mask = []
        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_labels = []
        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_labels = []
        
        for i in range(batch_size):
            # Extract prompt, chosen, and rejected responses
            prompt, chosen, rejected = self._extract_pairwise_data(examples, i)
            
            if prompt is None or chosen is None or rejected is None:
                logger.warning(f"Skipping example {i} due to missing data")
                continue
            
            # Process prompt
            prompt_tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                padding=False,
                return_tensors=None
            )
            
            # Process chosen response
            chosen_full = prompt + chosen
            chosen_tokenized = self.tokenizer(
                chosen_full,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Process rejected response
            rejected_full = prompt + rejected
            rejected_tokenized = self.tokenizer(
                rejected_full,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Create labels for chosen response (mask prompt part)
            chosen_labels_list = chosen_tokenized["input_ids"].copy()
            prompt_length = len(prompt_tokenized["input_ids"])
            
            # Mask prompt tokens in chosen labels
            for j in range(min(prompt_length, len(chosen_labels_list))):
                chosen_labels_list[j] = -100
            
            # Create labels for rejected response (mask prompt part)
            rejected_labels_list = rejected_tokenized["input_ids"].copy()
            
            # Mask prompt tokens in rejected labels
            for j in range(min(prompt_length, len(rejected_labels_list))):
                rejected_labels_list[j] = -100
            
            # Add to batch
            prompt_input_ids.append(prompt_tokenized["input_ids"])
            prompt_attention_mask.append(prompt_tokenized["attention_mask"])
            
            chosen_input_ids.append(chosen_tokenized["input_ids"])
            chosen_attention_mask.append(chosen_tokenized["attention_mask"])
            chosen_labels.append(chosen_labels_list)
            
            rejected_input_ids.append(rejected_tokenized["input_ids"])
            rejected_attention_mask.append(rejected_tokenized["attention_mask"])
            rejected_labels.append(rejected_labels_list)
        
        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels
        }
    
    def _extract_kto_data(self, examples: Dict[str, List], index: int) -> tuple:
        """Extract prompt, completion, and label for KTO.
        
        Args:
            examples: Batch of examples
            index: Index of the current example
            
        Returns:
            Tuple of (prompt, completion, label)
        """
        prompt = None
        completion = None
        label = None
        
        # Standard KTO format
        if all(key in examples for key in ["prompt", "completion", "label"]):
            prompt = examples["prompt"][index]
            completion = examples["completion"][index]
            label = examples["label"][index]
        
        # Alternative format: input/output with rating
        elif all(key in examples for key in ["input", "output", "rating"]):
            prompt = examples["input"][index]
            completion = examples["output"][index]
            # Convert rating to binary (assuming rating > threshold means desirable)
            rating = examples["rating"][index]
            if isinstance(rating, (int, float)):
                label = 1 if rating >= 3 else 0  # Assuming 5-point scale
            else:
                label = rating
        
        # Alternative format: question/answer with score
        elif all(key in examples for key in ["question", "answer", "score"]):
            prompt = examples["question"][index]
            completion = examples["answer"][index]
            score = examples["score"][index]
            if isinstance(score, (int, float)):
                label = 1 if score >= 0.5 else 0  # Assuming normalized score
            else:
                label = score
        
        return prompt, completion, label
    
    def _extract_pairwise_data(self, examples: Dict[str, List], index: int) -> tuple:
        """Extract prompt, chosen, and rejected for pairwise methods.
        
        Args:
            examples: Batch of examples
            index: Index of the current example
            
        Returns:
            Tuple of (prompt, chosen, rejected)
        """
        # Reuse DPO extraction logic
        from .dpo_loader import DPODataLoader
        dpo_loader = DPODataLoader(self.tokenizer)
        return dpo_loader._extract_preference_data(examples, index)
    
    def get_data_collator(self):
        """Get the appropriate data collator for preference optimization.
        
        Returns:
            Data collator instance
        """
        try:
            if self.method == "kto":
                from trl import KTODataCollatorWithPadding
                return KTODataCollatorWithPadding(
                    tokenizer=self.tokenizer,
                    max_prompt_length=self.max_prompt_length,
                    max_length=self.max_length,
                    pad_to_multiple_of=8,
                    return_tensors="pt"
                )
            else:
                # IPO and CPO use DPO-style data collator
                from trl import DPODataCollatorWithPadding
                return DPODataCollatorWithPadding(
                    tokenizer=self.tokenizer,
                    max_prompt_length=self.max_prompt_length,
                    max_length=self.max_length,
                    pad_to_multiple_of=8,
                    return_tensors="pt"
                )
        except ImportError:
            logger.warning("TRL not available, falling back to standard data collator")
            return super().get_data_collator()
    
    def validate_dataset(self, dataset: Union[Dataset, dict]) -> bool:
        """Validate that the dataset has the required format.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        if not super().validate_dataset(dataset):
            return False
        
        # Method-specific validation
        try:
            # Check if we can extract data from first example
            if hasattr(dataset, 'column_names'):
                test_dataset = dataset
            else:
                test_dataset = dataset["train"]
            
            if len(test_dataset) > 0:
                first_example = test_dataset[0]
                test_examples = {key: [value] for key, value in first_example.items()}
                
                if self.method == "kto":
                    prompt, completion, label = self._extract_kto_data(test_examples, 0)
                    
                    if prompt is None or completion is None or label is None:
                        logger.error(f"Could not extract KTO data from dataset")
                        logger.error(f"Available columns: {test_dataset.column_names}")
                        return False
                    
                    if not prompt.strip() or not completion.strip():
                        logger.error("Prompt or completion is empty")
                        return False
                
                else:
                    prompt, chosen, rejected = self._extract_pairwise_data(test_examples, 0)
                    
                    if prompt is None or chosen is None or rejected is None:
                        logger.error(f"Could not extract pairwise data from dataset")
                        logger.error(f"Available columns: {test_dataset.column_names}")
                        return False
                    
                    if not prompt.strip() or not chosen.strip() or not rejected.strip():
                        logger.error("Prompt, chosen, or rejected response is empty")
                        return False
            
            logger.info(f"{self.method.upper()} dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"{self.method.upper()} dataset validation failed: {e}")
            return False
    
    def get_dataset_statistics(self, dataset: Union[Dataset, dict]) -> Dict:
        """Get statistics about the preference dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = self.get_dataset_info(dataset)
        
        # Add method-specific statistics
        try:
            if hasattr(dataset, 'column_names'):
                analysis_dataset = dataset
            else:
                analysis_dataset = dataset["train"]
            
            # Sample a subset for analysis (max 1000 examples)
            sample_size = min(1000, len(analysis_dataset))
            sample_indices = list(range(0, len(analysis_dataset), len(analysis_dataset) // sample_size))[:sample_size]
            
            if self.method == "kto":
                prompt_lengths = []
                completion_lengths = []
                label_distribution = {0: 0, 1: 0}
                
                for idx in sample_indices:
                    example = analysis_dataset[idx]
                    test_examples = {key: [value] for key, value in example.items()}
                    prompt, completion, label = self._extract_kto_data(test_examples, 0)
                    
                    if prompt and completion and label is not None:
                        prompt_tokens = len(self.tokenizer.encode(prompt))
                        completion_tokens = len(self.tokenizer.encode(completion))
                        
                        prompt_lengths.append(prompt_tokens)
                        completion_lengths.append(completion_tokens)
                        
                        binary_label = 1 if label in [1, "1", True, "true", "desirable", "good"] else 0
                        label_distribution[binary_label] += 1
                
                if prompt_lengths:
                    stats[f"{self.method}_statistics"] = {
                        "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
                        "avg_completion_length": sum(completion_lengths) / len(completion_lengths),
                        "max_prompt_length": max(prompt_lengths),
                        "max_completion_length": max(completion_lengths),
                        "desirable_ratio": label_distribution[1] / (label_distribution[0] + label_distribution[1]),
                        "samples_analyzed": len(prompt_lengths)
                    }
            
            else:
                # Pairwise statistics (similar to DPO)
                prompt_lengths = []
                chosen_lengths = []
                rejected_lengths = []
                length_differences = []
                
                for idx in sample_indices:
                    example = analysis_dataset[idx]
                    test_examples = {key: [value] for key, value in example.items()}
                    prompt, chosen, rejected = self._extract_pairwise_data(test_examples, 0)
                    
                    if prompt and chosen and rejected:
                        prompt_tokens = len(self.tokenizer.encode(prompt))
                        chosen_tokens = len(self.tokenizer.encode(chosen))
                        rejected_tokens = len(self.tokenizer.encode(rejected))
                        
                        prompt_lengths.append(prompt_tokens)
                        chosen_lengths.append(chosen_tokens)
                        rejected_lengths.append(rejected_tokens)
                        length_differences.append(abs(chosen_tokens - rejected_tokens))
                
                if prompt_lengths:
                    stats[f"{self.method}_statistics"] = {
                        "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
                        "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths),
                        "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths),
                        "avg_length_difference": sum(length_differences) / len(length_differences),
                        "max_prompt_length": max(prompt_lengths),
                        "max_chosen_length": max(chosen_lengths),
                        "max_rejected_length": max(rejected_lengths),
                        "samples_analyzed": len(prompt_lengths)
                    }
        
        except Exception as e:
            logger.warning(f"Could not compute {self.method.upper()} statistics: {e}")
        
        return stats