"""Odds Ratio Preference Optimization (ORPO) Trainer Module

This module contains the trainer implementation for Odds Ratio Preference Optimization
training of Arabic Qwen models using preference datasets without requiring a reference model.
"""

import logging
import torch
from typing import Dict, Optional, Union

from transformers import TrainingArguments
from datasets import Dataset

from .base_trainer import BaseTrainer
from ..config import ORPOConfig

logger = logging.getLogger(__name__)


class ORPOTrainer(BaseTrainer):
    """Trainer for Odds Ratio Preference Optimization (ORPO).
    
    This class handles the complete ORPO training pipeline including
    preference data loading, model setup, and ORPO training execution.
    ORPO doesn't require a reference model, making it more memory efficient.
    """
    
    def __init__(self, config: ORPOConfig):
        """Initialize the ORPO trainer.
        
        Args:
            config: ORPO configuration object
        """
        super().__init__(config)
        
        if not isinstance(config, ORPOConfig):
            raise TypeError(f"Expected ORPOConfig, got {type(config)}")
        
        logger.info("Initialized ORPOTrainer")
    
    def _get_training_method(self) -> str:
        """Get the training method name.
        
        Returns:
            Training method name
        """
        return "orpo"
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the preference dataset for ORPO training.
        
        Returns:
            Processed dataset ready for ORPO training
        """
        logger.info(f"Loading ORPO dataset: {self.config.dataset_name}")
        
        from datasets import load_dataset
        
        # Load the dataset
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.dataset_split
        )
        
        # Apply sample limit if specified
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")
        
        # Process the dataset for ORPO format
        dataset = self._process_preference_dataset(dataset)
        
        logger.info(f"Loaded {len(dataset)} training samples")
        return dataset
    
    def _process_preference_dataset(self, dataset: Dataset) -> Dataset:
        """Process the preference dataset for ORPO training.
        
        Args:
            dataset: Raw preference dataset
            
        Returns:
            Processed dataset with proper ORPO format
        """
        def format_orpo_example(example):
            """Format a single example for ORPO training."""
            # ORPO expects: prompt, chosen, rejected
            if 'prompt' in example and 'chosen' in example and 'rejected' in example:
                return {
                    'prompt': example['prompt'],
                    'chosen': example['chosen'],
                    'rejected': example['rejected']
                }
            elif 'messages' in example:
                # Handle conversational format
                messages = example['messages']
                if len(messages) >= 2:
                    prompt = messages[0]['content'] if messages[0]['role'] == 'user' else ""
                    chosen = messages[1]['content'] if messages[1]['role'] == 'assistant' else ""
                    rejected = example.get('rejected', chosen + " [REJECTED]")  # Fallback
                    
                    return {
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected
                    }
            
            # Fallback for other formats
            return {
                'prompt': str(example.get('prompt', example.get('instruction', ''))),
                'chosen': str(example.get('chosen', example.get('output', ''))),
                'rejected': str(example.get('rejected', example.get('output', '') + " [REJECTED]"))
            }
        
        processed_dataset = dataset.map(
            format_orpo_example,
            remove_columns=dataset.column_names,
            desc="Processing ORPO dataset"
        )
        
        return processed_dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup the ORPO trainer with the processed dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        logger.info("Setting up ORPO trainer")
        
        try:
            from trl import ORPOTrainer as TRLORPOTrainer
        except ImportError:
            logger.error("TRL library with ORPO support not found. Please install: pip install trl>=0.7.0")
            raise ImportError("TRL library with ORPO support required")
        
        # Setup training arguments
        training_args = TrainingArguments(**self.config.get_training_args_dict())
        
        # Setup LoRA if enabled
        peft_config = None
        if self.config.use_lora:
            from peft import LoraConfig
            peft_config = LoraConfig(**self.config.get_lora_config_dict())
            logger.info("LoRA configuration enabled for ORPO training")
        
        # Initialize ORPO trainer
        self.trainer = TRLORPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            **self.config.get_orpo_args_dict()
        )
        
        logger.info("ORPO trainer setup completed")
    
    def train(self) -> Dict[str, float]:
        """Execute the ORPO training process.
        
        Returns:
            Training metrics and results
        """
        logger.info("Starting ORPO training")
        
        # Load dataset
        train_dataset = self.load_dataset()
        
        # Load validation dataset if specified
        eval_dataset = None
        if hasattr(self.config, 'validation_split') and self.config.validation_split:
            try:
                from datasets import load_dataset
                eval_dataset = load_dataset(
                    self.config.dataset_name,
                    self.config.dataset_config,
                    split=self.config.validation_split
                )
                eval_dataset = self._process_preference_dataset(eval_dataset)
                logger.info(f"Loaded {len(eval_dataset)} validation samples")
            except Exception as e:
                logger.warning(f"Could not load validation dataset: {e}")
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        train_result = self.trainer.train()
        
        # Save the model
        self.save_model()
        
        # Extract metrics
        metrics = {
            'train_loss': train_result.training_loss,
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
            'train_steps_per_second': train_result.metrics.get('train_steps_per_second', 0),
            'total_flos': train_result.metrics.get('total_flos', 0),
            'train_runtime': train_result.metrics.get('train_runtime', 0),
            'epoch': train_result.metrics.get('epoch', 0),
        }
        
        logger.info("ORPO training completed successfully")
        return metrics
    
    def save_model(self):
        """Save the trained model and tokenizer."""
        logger.info(f"Saving ORPO model to {self.config.output_dir}")
        
        # Save the model
        self.trainer.save_model()
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Model and tokenizer saved successfully")