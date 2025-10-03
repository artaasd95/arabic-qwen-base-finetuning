"""Simple Preference Optimization (SimPO) Trainer Module

This module contains the trainer implementation for Simple Preference Optimization
training of Arabic Qwen models using preference datasets with a simplified approach.
"""

import logging
import torch
from typing import Dict, Optional, Union

from transformers import TrainingArguments
from datasets import Dataset

from .base_trainer import BaseTrainer
from ..config import SimPOConfig

logger = logging.getLogger(__name__)


class SimPOTrainer(BaseTrainer):
    """Trainer for Simple Preference Optimization (SimPO).
    
    This class handles the complete SimPO training pipeline including
    preference data loading, model setup, and SimPO training execution.
    SimPO simplifies preference optimization without requiring a reference model.
    """
    
    def __init__(self, config: SimPOConfig):
        """Initialize the SimPO trainer.
        
        Args:
            config: SimPO configuration object
        """
        super().__init__(config)
        
        if not isinstance(config, SimPOConfig):
            raise TypeError(f"Expected SimPOConfig, got {type(config)}")
        
        logger.info("Initialized SimPOTrainer")
    
    def _get_training_method(self) -> str:
        """Get the training method name.
        
        Returns:
            Training method name
        """
        return "simpo"
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the preference dataset for SimPO training.
        
        Returns:
            Processed dataset ready for SimPO training
        """
        logger.info(f"Loading SimPO dataset: {self.config.dataset_name}")
        
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
        
        # Process the dataset for SimPO format
        dataset = self._process_preference_dataset(dataset)
        
        logger.info(f"Loaded {len(dataset)} training samples")
        return dataset
    
    def _process_preference_dataset(self, dataset: Dataset) -> Dataset:
        """Process the preference dataset for SimPO training.
        
        Args:
            dataset: Raw preference dataset
            
        Returns:
            Processed dataset with proper SimPO format
        """
        def format_simpo_example(example):
            """Format a single example for SimPO training."""
            # SimPO expects: prompt, chosen, rejected
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
            format_simpo_example,
            remove_columns=dataset.column_names,
            desc="Processing SimPO dataset"
        )
        
        return processed_dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup the SimPO trainer with the processed dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        logger.info("Setting up SimPO trainer")
        
        try:
            from trl import DPOTrainer as TRLDPOTrainer
            # Note: SimPO is often implemented as a variant of DPO
            # If a dedicated SimPOTrainer becomes available in TRL, use that instead
        except ImportError:
            logger.error("TRL library not found. Please install: pip install trl>=0.7.0")
            raise ImportError("TRL library required for SimPO training")
        
        # Setup training arguments
        training_args = TrainingArguments(**self.config.get_training_args_dict())
        
        # Setup LoRA if enabled
        peft_config = None
        if self.config.use_lora:
            from peft import LoraConfig
            peft_config = LoraConfig(**self.config.get_lora_config_dict())
            logger.info("LoRA configuration enabled for SimPO training")
        
        # Initialize SimPO trainer (using DPO trainer with SimPO parameters)
        simpo_args = self.config.get_simpo_args_dict()
        
        self.trainer = TRLDPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            beta=simpo_args["beta"],
            max_prompt_length=simpo_args["max_prompt_length"],
            max_length=simpo_args["max_length"],
            loss_type=simpo_args["loss_type"]
        )
        
        logger.info("SimPO trainer setup completed")
    
    def train(self) -> Dict[str, float]:
        """Execute the SimPO training process.
        
        Returns:
            Training metrics and results
        """
        logger.info("Starting SimPO training")
        
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
        
        logger.info("SimPO training completed successfully")
        return metrics
    
    def save_model(self):
        """Save the trained model and tokenizer."""
        logger.info(f"Saving SimPO model to {self.config.output_dir}")
        
        # Save the model
        self.trainer.save_model()
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Model and tokenizer saved successfully")