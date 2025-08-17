"""Supervised Fine-Tuning (SFT) Data Loader Module

This module contains the data loader implementation for supervised fine-tuning
of Arabic Qwen models on instruction-following datasets.
"""

import logging
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizer
from datasets import Dataset

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class SFTDataLoader(BaseDataLoader):
    """Data loader for Supervised Fine-Tuning (SFT).
    
    This class handles loading and preprocessing of instruction-following
    datasets for supervised fine-tuning.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        instruction_template: str = "### التعليمات:\n{instruction}\n\n### الإجابة:\n{response}",
        response_template: str = "### الإجابة:",
        packing: bool = False,
        **kwargs
    ):
        """Initialize the SFT data loader.
        
        Args:
            tokenizer: Pre-trained tokenizer
            instruction_template: Template for formatting instruction-response pairs
            response_template: Template for identifying response start
            packing: Whether to pack multiple examples into one sequence
            **kwargs: Additional arguments passed to BaseDataLoader
        """
        super().__init__(tokenizer, **kwargs)
        
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.packing = packing
        
        logger.info(f"Initialized SFTDataLoader with packing={packing}")
    
    def get_required_columns(self) -> List[str]:
        """Get the list of required columns for SFT.
        
        Returns:
            List of required column names
        """
        # Common column names for instruction datasets
        return ["instruction", "response"]  # or ["input", "output"] or ["prompt", "completion"]
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for SFT datasets.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Preprocessed batch with tokenized inputs
        """
        batch_size = len(examples[list(examples.keys())[0]])
        
        # Prepare lists for processed data
        input_ids = []
        attention_mask = []
        labels = []
        
        for i in range(batch_size):
            # Extract instruction and response
            instruction, response = self._extract_instruction_response(examples, i)
            
            if instruction is None or response is None:
                logger.warning(f"Skipping example {i} due to missing instruction or response")
                continue
            
            # Format the conversation
            formatted_text = self.instruction_template.format(
                instruction=instruction.strip(),
                response=response.strip()
            )
            
            # Tokenize the full conversation
            tokenized = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None
            )
            
            # Create labels for training (only response part should be trained)
            input_text = self.instruction_template.format(
                instruction=instruction.strip(),
                response=""
            ).rstrip()
            
            # Tokenize just the input part to find where response starts
            input_tokenized = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None
            )
            
            # Create labels (mask input tokens, keep response tokens)
            label_ids = tokenized["input_ids"].copy()
            input_length = len(input_tokenized["input_ids"])
            
            # Mask the instruction part (set to -100 so it's ignored in loss)
            for j in range(min(input_length, len(label_ids))):
                label_ids[j] = -100
            
            input_ids.append(tokenized["input_ids"])
            attention_mask.append(tokenized["attention_mask"])
            labels.append(label_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _extract_instruction_response(self, examples: Dict[str, List], index: int) -> tuple:
        """Extract instruction and response from examples.
        
        Args:
            examples: Batch of examples
            index: Index of the current example
            
        Returns:
            Tuple of (instruction, response)
        """
        # Try different common column name patterns
        instruction = None
        response = None
        
        # Pattern 1: instruction/response
        if "instruction" in examples and "response" in examples:
            instruction = examples["instruction"][index]
            response = examples["response"][index]
        
        # Pattern 2: input/output
        elif "input" in examples and "output" in examples:
            instruction = examples["input"][index]
            response = examples["output"][index]
        
        # Pattern 3: prompt/completion
        elif "prompt" in examples and "completion" in examples:
            instruction = examples["prompt"][index]
            response = examples["completion"][index]
        
        # Pattern 4: question/answer
        elif "question" in examples and "answer" in examples:
            instruction = examples["question"][index]
            response = examples["answer"][index]
        
        # Pattern 5: text (for single column datasets)
        elif "text" in examples:
            # Try to split on response template
            text = examples["text"][index]
            if self.response_template in text:
                parts = text.split(self.response_template, 1)
                if len(parts) == 2:
                    instruction = parts[0].strip()
                    response = parts[1].strip()
            else:
                # Fallback: treat entire text as response with empty instruction
                instruction = ""
                response = text
        
        # Pattern 6: conversations (for chat datasets)
        elif "conversations" in examples:
            conversations = examples["conversations"][index]
            if isinstance(conversations, list) and len(conversations) >= 2:
                # Assume alternating human/assistant messages
                instruction = conversations[0].get("value", "")
                response = conversations[1].get("value", "")
        
        return instruction, response
    
    def get_data_collator(self):
        """Get the appropriate data collator for SFT.
        
        Returns:
            Data collator instance
        """
        if self.packing:
            # Use packing data collator for efficiency
            try:
                from trl import DataCollatorForCompletionOnlyLM
                
                return DataCollatorForCompletionOnlyLM(
                    tokenizer=self.tokenizer,
                    response_template=self.response_template,
                    mlm=False
                )
            except ImportError:
                logger.warning("TRL not available, falling back to standard data collator")
                return super().get_data_collator()
        else:
            return super().get_data_collator()
    
    def validate_dataset(self, dataset: Union[Dataset, dict]) -> bool:
        """Validate that the dataset has the required format for SFT.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        if not super().validate_dataset(dataset):
            return False
        
        # Additional SFT-specific validation
        try:
            # Check if we can extract instruction/response from first example
            if hasattr(dataset, 'column_names'):
                test_dataset = dataset
            else:
                test_dataset = dataset["train"]
            
            if len(test_dataset) > 0:
                first_example = test_dataset[0]
                
                # Convert single example to batch format for testing
                test_examples = {key: [value] for key, value in first_example.items()}
                instruction, response = self._extract_instruction_response(test_examples, 0)
                
                if instruction is None or response is None:
                    logger.error("Could not extract instruction and response from dataset")
                    logger.error(f"Available columns: {test_dataset.column_names}")
                    logger.error(f"First example keys: {list(first_example.keys())}")
                    return False
                
                if not instruction.strip() and not response.strip():
                    logger.error("Both instruction and response are empty")
                    return False
            
            logger.info("SFT dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"SFT dataset validation failed: {e}")
            return False
    
    def get_dataset_statistics(self, dataset: Union[Dataset, dict]) -> Dict:
        """Get statistics about the SFT dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = self.get_dataset_info(dataset)
        
        # Add SFT-specific statistics
        try:
            if hasattr(dataset, 'column_names'):
                analysis_dataset = dataset
            else:
                analysis_dataset = dataset["train"]
            
            # Sample a subset for analysis (max 1000 examples)
            sample_size = min(1000, len(analysis_dataset))
            sample_indices = list(range(0, len(analysis_dataset), len(analysis_dataset) // sample_size))[:sample_size]
            
            instruction_lengths = []
            response_lengths = []
            total_lengths = []
            
            for idx in sample_indices:
                example = analysis_dataset[idx]
                test_examples = {key: [value] for key, value in example.items()}
                instruction, response = self._extract_instruction_response(test_examples, 0)
                
                if instruction and response:
                    inst_tokens = len(self.tokenizer.encode(instruction))
                    resp_tokens = len(self.tokenizer.encode(response))
                    
                    instruction_lengths.append(inst_tokens)
                    response_lengths.append(resp_tokens)
                    total_lengths.append(inst_tokens + resp_tokens)
            
            if instruction_lengths:
                stats["sft_statistics"] = {
                    "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths),
                    "avg_response_length": sum(response_lengths) / len(response_lengths),
                    "avg_total_length": sum(total_lengths) / len(total_lengths),
                    "max_instruction_length": max(instruction_lengths),
                    "max_response_length": max(response_lengths),
                    "max_total_length": max(total_lengths),
                    "samples_analyzed": len(instruction_lengths)
                }
        
        except Exception as e:
            logger.warning(f"Could not compute SFT statistics: {e}")
        
        return stats