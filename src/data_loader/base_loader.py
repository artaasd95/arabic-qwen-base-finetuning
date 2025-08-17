"""Base Data Loader Module

This module contains the base data loader class and common functionality
for loading and preprocessing datasets for Arabic Qwen fine-tuning.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.
    
    This class provides common functionality for loading and preprocessing
    datasets for different training methods.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        validation_split: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """Initialize the base data loader.
        
        Args:
            tokenizer: Pre-trained tokenizer
            max_seq_length: Maximum sequence length
            dataset_name: Name of the dataset to load
            dataset_config: Dataset configuration name
            dataset_split: Dataset split to use for training
            validation_split: Dataset split to use for validation
            max_samples: Maximum number of samples to use
            shuffle: Whether to shuffle the dataset
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.validation_split = validation_split
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        
        # Set tokenizer padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized {self.__class__.__name__} with max_seq_length={max_seq_length}")
    
    def load_dataset(self) -> Union[Dataset, DatasetDict]:
        """Load dataset from Hugging Face Hub or local path.
        
        Returns:
            Loaded dataset
            
        Raises:
            ValueError: If dataset cannot be loaded
        """
        try:
            if self.dataset_name is None:
                raise ValueError("dataset_name must be provided")
            
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            # Load dataset
            if Path(self.dataset_name).exists():
                # Local dataset
                dataset = load_dataset(
                    "json",
                    data_files=self.dataset_name,
                    split=self.dataset_split
                )
            else:
                # Hugging Face Hub dataset
                load_kwargs = {
                    "path": self.dataset_name,
                    "split": self.dataset_split
                }
                
                if self.dataset_config:
                    load_kwargs["name"] = self.dataset_config
                
                dataset = load_dataset(**load_kwargs)
            
            logger.info(f"Loaded dataset with {len(dataset)} samples")
            
            # Load validation dataset if specified
            validation_dataset = None
            if self.validation_split:
                try:
                    if Path(self.dataset_name).exists():
                        # For local datasets, assume validation is in the same file
                        # or a separate file with _val suffix
                        val_path = str(Path(self.dataset_name).with_suffix('')) + "_val.json"
                        if Path(val_path).exists():
                            validation_dataset = load_dataset(
                                "json",
                                data_files=val_path,
                                split="train"
                            )
                    else:
                        load_kwargs["split"] = self.validation_split
                        validation_dataset = load_dataset(**load_kwargs)
                    
                    if validation_dataset:
                        logger.info(f"Loaded validation dataset with {len(validation_dataset)} samples")
                except Exception as e:
                    logger.warning(f"Could not load validation dataset: {e}")
            
            # Limit samples if specified
            if self.max_samples and len(dataset) > self.max_samples:
                dataset = dataset.select(range(self.max_samples))
                logger.info(f"Limited dataset to {self.max_samples} samples")
            
            # Shuffle if requested
            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)
                if validation_dataset:
                    validation_dataset = validation_dataset.shuffle(seed=self.seed)
            
            if validation_dataset:
                return DatasetDict({
                    "train": dataset,
                    "validation": validation_dataset
                })
            else:
                return dataset
                
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise ValueError(f"Could not load dataset: {e}")
    
    @abstractmethod
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for the dataset.
        
        This method must be implemented by subclasses to define
        how to preprocess the data for their specific training method.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Preprocessed batch of examples
        """
        pass
    
    def tokenize_and_format(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing and tokenization to the dataset.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Tokenized and formatted dataset
        """
        logger.info("Tokenizing and formatting dataset...")
        
        # Apply preprocessing function
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing dataset"
        )
        
        logger.info(f"Processed dataset with {len(processed_dataset)} samples")
        return processed_dataset
    
    def prepare_dataset(self) -> Union[Dataset, DatasetDict]:
        """Load and prepare the complete dataset.
        
        Returns:
            Prepared dataset ready for training
        """
        # Load raw dataset
        raw_dataset = self.load_dataset()
        
        # Process based on dataset type
        if isinstance(raw_dataset, DatasetDict):
            processed_dataset = DatasetDict()
            for split_name, split_dataset in raw_dataset.items():
                processed_dataset[split_name] = self.tokenize_and_format(split_dataset)
        else:
            processed_dataset = self.tokenize_and_format(raw_dataset)
        
        return processed_dataset
    
    def get_data_collator(self):
        """Get the appropriate data collator for this training method.
        
        This method should be overridden by subclasses if they need
        a specific data collator.
        
        Returns:
            Data collator instance
        """
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling
            pad_to_multiple_of=8,  # For efficiency on TPUs
        )
    
    def validate_dataset(self, dataset: Union[Dataset, DatasetDict]) -> bool:
        """Validate that the dataset has the required format.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            if isinstance(dataset, DatasetDict):
                # Check that train split exists
                if "train" not in dataset:
                    logger.error("Dataset must contain a 'train' split")
                    return False
                
                # Validate each split
                for split_name, split_dataset in dataset.items():
                    if not self._validate_split(split_dataset, split_name):
                        return False
            else:
                if not self._validate_split(dataset, "train"):
                    return False
            
            logger.info("Dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def _validate_split(self, dataset: Dataset, split_name: str) -> bool:
        """Validate a single dataset split.
        
        Args:
            dataset: Dataset split to validate
            split_name: Name of the split
            
        Returns:
            True if split is valid, False otherwise
        """
        if len(dataset) == 0:
            logger.error(f"Dataset split '{split_name}' is empty")
            return False
        
        # Check required columns (to be overridden by subclasses)
        required_columns = self.get_required_columns()
        missing_columns = set(required_columns) - set(dataset.column_names)
        
        if missing_columns:
            logger.error(
                f"Dataset split '{split_name}' missing required columns: {missing_columns}"
            )
            return False
        
        return True
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get the list of required columns for this data loader.
        
        Returns:
            List of required column names
        """
        pass
    
    def get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        if isinstance(dataset, DatasetDict):
            info["type"] = "DatasetDict"
            info["splits"] = {}
            
            for split_name, split_dataset in dataset.items():
                info["splits"][split_name] = {
                    "num_samples": len(split_dataset),
                    "columns": split_dataset.column_names,
                    "features": str(split_dataset.features)
                }
        else:
            info["type"] = "Dataset"
            info["num_samples"] = len(dataset)
            info["columns"] = dataset.column_names
            info["features"] = str(dataset.features)
        
        return info