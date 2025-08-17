"""Direct Preference Optimization (DPO) Data Loader Module

This module contains the data loader implementation for Direct Preference Optimization
training of Arabic Qwen models using preference datasets.
"""

import logging
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizer
from datasets import Dataset

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class DPODataLoader(BaseDataLoader):
    """Data loader for Direct Preference Optimization (DPO).
    
    This class handles loading and preprocessing of preference datasets
    for DPO training.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 512,
        max_length: int = 1024,
        **kwargs
    ):
        """Initialize the DPO data loader.
        
        Args:
            tokenizer: Pre-trained tokenizer
            max_prompt_length: Maximum length for prompts
            max_length: Maximum total sequence length
            **kwargs: Additional arguments passed to BaseDataLoader
        """
        super().__init__(tokenizer, **kwargs)
        
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length
        
        logger.info(f"Initialized DPODataLoader with max_prompt_length={max_prompt_length}, max_length={max_length}")
    
    def get_required_columns(self) -> List[str]:
        """Get the list of required columns for DPO.
        
        Returns:
            List of required column names
        """
        return ["prompt", "chosen", "rejected"]
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for DPO datasets.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Preprocessed batch with tokenized inputs for DPO
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
            prompt, chosen, rejected = self._extract_preference_data(examples, i)
            
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
    
    def _extract_preference_data(self, examples: Dict[str, List], index: int) -> tuple:
        """Extract prompt, chosen, and rejected responses from examples.
        
        Args:
            examples: Batch of examples
            index: Index of the current example
            
        Returns:
            Tuple of (prompt, chosen, rejected)
        """
        prompt = None
        chosen = None
        rejected = None
        
        # Standard DPO format
        if all(key in examples for key in ["prompt", "chosen", "rejected"]):
            prompt = examples["prompt"][index]
            chosen = examples["chosen"][index]
            rejected = examples["rejected"][index]
        
        # Alternative format: question/answer_a/answer_b with preference
        elif all(key in examples for key in ["question", "answer_a", "answer_b"]):
            prompt = examples["question"][index]
            answer_a = examples["answer_a"][index]
            answer_b = examples["answer_b"][index]
            
            # Check for preference indicator
            if "preference" in examples:
                pref = examples["preference"][index]
                if pref == "a" or pref == 0:
                    chosen = answer_a
                    rejected = answer_b
                else:
                    chosen = answer_b
                    rejected = answer_a
            elif "chosen" in examples:
                # Explicit chosen field
                chosen_idx = examples["chosen"][index]
                if chosen_idx == "a" or chosen_idx == 0:
                    chosen = answer_a
                    rejected = answer_b
                else:
                    chosen = answer_b
                    rejected = answer_a
        
        # Anthropic HH format
        elif "chosen" in examples and "rejected" in examples:
            chosen_text = examples["chosen"][index]
            rejected_text = examples["rejected"][index]
            
            # Extract prompt from chosen text (assuming it's the common prefix)
            if chosen_text and rejected_text:
                # Find common prefix
                common_prefix = ""
                min_len = min(len(chosen_text), len(rejected_text))
                for i in range(min_len):
                    if chosen_text[i] == rejected_text[i]:
                        common_prefix += chosen_text[i]
                    else:
                        break
                
                # Use common prefix as prompt
                prompt = common_prefix.strip()
                chosen = chosen_text[len(common_prefix):].strip()
                rejected = rejected_text[len(common_prefix):].strip()
        
        # Conversations format
        elif "conversations" in examples:
            conversations = examples["conversations"][index]
            if isinstance(conversations, dict):
                if "chosen" in conversations and "rejected" in conversations:
                    chosen_conv = conversations["chosen"]
                    rejected_conv = conversations["rejected"]
                    
                    # Extract prompt (usually the last human message)
                    if isinstance(chosen_conv, list) and len(chosen_conv) >= 2:
                        prompt = chosen_conv[-2].get("value", "")
                        chosen = chosen_conv[-1].get("value", "")
                    
                    if isinstance(rejected_conv, list) and len(rejected_conv) >= 1:
                        rejected = rejected_conv[-1].get("value", "")
        
        return prompt, chosen, rejected
    
    def get_data_collator(self):
        """Get the appropriate data collator for DPO.
        
        Returns:
            Data collator instance
        """
        try:
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
        """Validate that the dataset has the required format for DPO.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        if not super().validate_dataset(dataset):
            return False
        
        # Additional DPO-specific validation
        try:
            # Check if we can extract preference data from first example
            if hasattr(dataset, 'column_names'):
                test_dataset = dataset
            else:
                test_dataset = dataset["train"]
            
            if len(test_dataset) > 0:
                first_example = test_dataset[0]
                
                # Convert single example to batch format for testing
                test_examples = {key: [value] for key, value in first_example.items()}
                prompt, chosen, rejected = self._extract_preference_data(test_examples, 0)
                
                if prompt is None or chosen is None or rejected is None:
                    logger.error("Could not extract prompt, chosen, and rejected from dataset")
                    logger.error(f"Available columns: {test_dataset.column_names}")
                    logger.error(f"First example keys: {list(first_example.keys())}")
                    return False
                
                if not prompt.strip():
                    logger.error("Prompt is empty")
                    return False
                
                if not chosen.strip() or not rejected.strip():
                    logger.error("Chosen or rejected response is empty")
                    return False
                
                if chosen.strip() == rejected.strip():
                    logger.warning("Chosen and rejected responses are identical")
            
            logger.info("DPO dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"DPO dataset validation failed: {e}")
            return False
    
    def get_dataset_statistics(self, dataset: Union[Dataset, dict]) -> Dict:
        """Get statistics about the DPO dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = self.get_dataset_info(dataset)
        
        # Add DPO-specific statistics
        try:
            if hasattr(dataset, 'column_names'):
                analysis_dataset = dataset
            else:
                analysis_dataset = dataset["train"]
            
            # Sample a subset for analysis (max 1000 examples)
            sample_size = min(1000, len(analysis_dataset))
            sample_indices = list(range(0, len(analysis_dataset), len(analysis_dataset) // sample_size))[:sample_size]
            
            prompt_lengths = []
            chosen_lengths = []
            rejected_lengths = []
            length_differences = []
            
            for idx in sample_indices:
                example = analysis_dataset[idx]
                test_examples = {key: [value] for key, value in example.items()}
                prompt, chosen, rejected = self._extract_preference_data(test_examples, 0)
                
                if prompt and chosen and rejected:
                    prompt_tokens = len(self.tokenizer.encode(prompt))
                    chosen_tokens = len(self.tokenizer.encode(chosen))
                    rejected_tokens = len(self.tokenizer.encode(rejected))
                    
                    prompt_lengths.append(prompt_tokens)
                    chosen_lengths.append(chosen_tokens)
                    rejected_lengths.append(rejected_tokens)
                    length_differences.append(abs(chosen_tokens - rejected_tokens))
            
            if prompt_lengths:
                stats["dpo_statistics"] = {
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
            logger.warning(f"Could not compute DPO statistics: {e}")
        
        return stats