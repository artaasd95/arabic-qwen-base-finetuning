"""Base Trainer Module

This module contains the base trainer class with common functionality
for all training methods.
"""

import logging
import os
import json
import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from ..data_loader import get_data_loader
from ..utils.logging_utils import setup_logging
from ..utils.model_utils import get_model_memory_footprint, print_trainable_parameters

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer class for all training methods.
    
    This class provides common functionality for model loading, tokenizer setup,
    LoRA configuration, and training utilities.
    """
    
    def __init__(self, config):
        """Initialize the base trainer.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.data_loader = None
        self.trainer = None
        
        # Setup logging
        setup_logging(
            log_level=getattr(config, 'log_level', 'INFO'),
            log_file=getattr(config, 'log_file', None)
        )
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.__class__.__name__}")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, self.config.torch_dtype),
            "device_map": self.config.device_map if hasattr(self.config, 'device_map') else "auto",
        }
        
        # Add quantization config if specified
        if hasattr(self.config, 'load_in_4bit') and self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            logger.info("Enabled 4-bit quantization")
        
        elif hasattr(self.config, 'load_in_8bit') and self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            logger.info("Enabled 8-bit quantization")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Print model memory footprint
        memory_footprint = get_model_memory_footprint(self.model)
        logger.info(f"Model memory footprint: {memory_footprint:.2f} MB")
        
        # Setup LoRA if enabled
        if hasattr(self.config, 'use_lora') and self.config.use_lora:
            self._setup_lora()
        
        # Print trainable parameters
        print_trainable_parameters(self.model)
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(self.config, 'lora_r', 16),
            lora_alpha=getattr(self.config, 'lora_alpha', 32),
            lora_dropout=getattr(self.config, 'lora_dropout', 0.1),
            target_modules=getattr(self.config, 'lora_target_modules', ["q_proj", "v_proj"]),
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"LoRA configuration applied: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    def setup_data_loader(self, method: str):
        """Setup data loader for the specified method.
        
        Args:
            method: Training method (sft, dpo, kto, ipo, cpo)
        """
        logger.info(f"Setting up data loader for {method.upper()}")
        
        # Get method-specific parameters
        data_loader_kwargs = {
            "max_seq_length": getattr(self.config, 'max_seq_length', 512)
        }
        
        # Add method-specific parameters
        if method == "sft":
            data_loader_kwargs.update({
                "instruction_template": getattr(self.config, 'instruction_template', None),
                "response_template": getattr(self.config, 'response_template', None),
                "packing": getattr(self.config, 'packing', False)
            })
        elif method in ["dpo", "kto", "ipo", "cpo"]:
            data_loader_kwargs.update({
                "max_prompt_length": getattr(self.config, 'max_prompt_length', 256),
                "max_length": getattr(self.config, 'max_length', 512)
            })
        
        self.data_loader = get_data_loader(
            method=method,
            tokenizer=self.tokenizer,
            **data_loader_kwargs
        )
    
    def load_dataset(self) -> Union[Dataset, Dict[str, Dataset]]:
        """Load and prepare the dataset.
        
        Returns:
            Prepared dataset(s)
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset
        dataset = self.data_loader.prepare_dataset(
            dataset_name=self.config.dataset_name,
            split=getattr(self.config, 'dataset_split', 'train'),
            max_samples=getattr(self.config, 'max_samples', None),
            shuffle=getattr(self.config, 'shuffle_data', True)
        )
        
        # Validate dataset
        if not self.data_loader.validate_dataset(dataset):
            raise ValueError("Dataset validation failed")
        
        # Print dataset statistics
        stats = self.data_loader.get_dataset_statistics(dataset)
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        
        return dataset
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get training arguments.
        
        Returns:
            TrainingArguments instance
        """
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=getattr(self.config, 'eval_batch_size', self.config.batch_size),
            gradient_accumulation_steps=getattr(self.config, 'gradient_accumulation_steps', 1),
            learning_rate=self.config.learning_rate,
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            warmup_ratio=getattr(self.config, 'warmup_ratio', 0.1),
            lr_scheduler_type=getattr(self.config, 'lr_scheduler_type', 'cosine'),
            logging_steps=getattr(self.config, 'logging_steps', 10),
            save_steps=getattr(self.config, 'save_steps', 500),
            eval_steps=getattr(self.config, 'eval_steps', 500),
            evaluation_strategy=getattr(self.config, 'evaluation_strategy', 'steps'),
            save_strategy=getattr(self.config, 'save_strategy', 'steps'),
            save_total_limit=getattr(self.config, 'save_total_limit', 3),
            load_best_model_at_end=getattr(self.config, 'load_best_model_at_end', True),
            metric_for_best_model=getattr(self.config, 'metric_for_best_model', 'eval_loss'),
            greater_is_better=getattr(self.config, 'greater_is_better', False),
            report_to=getattr(self.config, 'report_to', ['tensorboard']),
            run_name=getattr(self.config, 'run_name', None),
            seed=getattr(self.config, 'seed', 42),
            data_seed=getattr(self.config, 'data_seed', 42),
            fp16=getattr(self.config, 'fp16', False),
            bf16=getattr(self.config, 'bf16', False),
            gradient_checkpointing=getattr(self.config, 'gradient_checkpointing', True),
            dataloader_num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            remove_unused_columns=getattr(self.config, 'remove_unused_columns', False),
            label_names=getattr(self.config, 'label_names', None),
        )
        
        return training_args
    
    def get_callbacks(self):
        """Get training callbacks.
        
        Returns:
            List of callback instances
        """
        callbacks = []
        
        # Early stopping callback
        if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience > 0:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=getattr(self.config, 'early_stopping_threshold', 0.0)
            )
            callbacks.append(early_stopping)
            logger.info(f"Added early stopping with patience: {self.config.early_stopping_patience}")
        
        return callbacks
    
    @abstractmethod
    def create_trainer(self, dataset, training_args, callbacks) -> Trainer:
        """Create the trainer instance.
        
        Args:
            dataset: Training dataset
            training_args: Training arguments
            callbacks: List of callbacks
            
        Returns:
            Trainer instance
        """
        pass
    
    def train(self):
        """Main training method."""
        logger.info("Starting training process")
        
        try:
            # Setup model and tokenizer
            self.setup_model_and_tokenizer()
            
            # Setup data loader
            method = self._get_training_method()
            self.setup_data_loader(method)
            
            # Load dataset
            dataset = self.load_dataset()
            
            # Get training arguments
            training_args = self.get_training_arguments()
            
            # Get callbacks
            callbacks = self.get_callbacks()
            
            # Create trainer
            self.trainer = self.create_trainer(dataset, training_args, callbacks)
            
            # Start training
            logger.info("Beginning model training")
            train_result = self.trainer.train()
            
            # Save final model
            self.save_model()
            
            # Log training results
            logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the trained model.
        
        Args:
            output_dir: Directory to save the model (defaults to config.output_dir)
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_path}")
        
        # Save model and tokenizer
        if hasattr(self.config, 'use_lora') and self.config.use_lora:
            # Save LoRA adapter
            self.model.save_pretrained(output_path)
        else:
            # Save full model
            self.model.save_pretrained(output_path)
        
        self.tokenizer.save_pretrained(output_path)
        
        # Save training configuration
        config_path = output_path / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info("Model saved successfully")
    
    def load_model(self, model_path: str):
        """Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        if hasattr(self.config, 'use_lora') and self.config.use_lora:
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                device_map="auto"
            )
            
            # Load LoRA adapter
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                device_map="auto"
            )
        
        logger.info("Model loaded successfully")
    
    @abstractmethod
    def _get_training_method(self) -> str:
        """Get the training method name.
        
        Returns:
            Training method name
        """
        pass
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        logger.info("Starting model evaluation")
        
        if eval_dataset is not None:
            eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        else:
            eval_result = self.trainer.evaluate()
        
        logger.info(f"Evaluation completed. Results: {eval_result}")
        return eval_result
    
    def predict(self, text: str, max_length: int = 512, **kwargs) -> str:
        """Generate prediction for input text.
        
        Args:
            text: Input text
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=kwargs.get('do_sample', True),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()