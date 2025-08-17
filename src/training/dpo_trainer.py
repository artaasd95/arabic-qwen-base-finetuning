"""Direct Preference Optimization (DPO) Trainer Module

This module contains the trainer implementation for Direct Preference Optimization
training of Arabic Qwen models using preference datasets.
"""

import logging
import torch
from typing import Dict, Optional, Union

from transformers import TrainingArguments
from datasets import Dataset

from .base_trainer import BaseTrainer
from ..config import DPOConfig

logger = logging.getLogger(__name__)


class DPOTrainer(BaseTrainer):
    """Trainer for Direct Preference Optimization (DPO).
    
    This class handles the complete DPO training pipeline including
    preference data loading, model setup, and DPO training execution.
    """
    
    def __init__(self, config: DPOConfig):
        """Initialize the DPO trainer.
        
        Args:
            config: DPO configuration object
        """
        super().__init__(config)
        
        if not isinstance(config, DPOConfig):
            raise TypeError(f"Expected DPOConfig, got {type(config)}")
        
        self.ref_model = None
        logger.info("Initialized DPOTrainer")
    
    def _get_training_method(self) -> str:
        """Get the training method name.
        
        Returns:
            Training method name
        """
        return "dpo"
    
    def setup_reference_model(self):
        """Setup the reference model for DPO."""
        logger.info("Setting up reference model for DPO")
        
        # Load reference model (usually the same as the base model)
        ref_model_name = getattr(self.config, 'ref_model_name', self.config.model_name)
        
        from transformers import AutoModelForCausalLM
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, self.config.torch_dtype),
            "device_map": "auto",
        }
        
        # Add quantization for reference model if specified
        if hasattr(self.config, 'load_in_4bit') and self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name,
            **model_kwargs
        )
        
        # Ensure reference model is in eval mode
        self.ref_model.eval()
        
        # Disable gradients for reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Reference model loaded: {ref_model_name}")
    
    def create_trainer(self, dataset, training_args, callbacks):
        """Create the DPO trainer instance.
        
        Args:
            dataset: Training dataset
            training_args: Training arguments
            callbacks: List of callbacks
            
        Returns:
            DPOTrainer instance
        """
        logger.info("Creating DPO trainer")
        
        try:
            from trl import DPOTrainer as TRLDPOTrainer
        except ImportError:
            raise ImportError(
                "TRL library is required for DPO training. "
                "Install it with: pip install trl"
            )
        
        # Setup reference model
        if not getattr(self.config, 'force_use_ref_model', False):
            self.setup_reference_model()
        
        # Prepare datasets
        if isinstance(dataset, dict):
            train_dataset = dataset.get("train")
            eval_dataset = dataset.get("validation") or dataset.get("test")
        else:
            train_dataset = dataset
            eval_dataset = None
        
        # Apply preprocessing
        if train_dataset is not None:
            train_dataset = train_dataset.map(
                self.data_loader.tokenize_and_format,
                batched=True,
                remove_columns=train_dataset.column_names,
                desc="Tokenizing training data"
            )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.data_loader.tokenize_and_format,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing evaluation data"
            )
        
        # Get DPO-specific arguments
        dpo_args = self.config.get_dpo_args()
        
        # Create DPO trainer
        trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
            **dpo_args
        )
        
        logger.info(f"DPO trainer created with {len(train_dataset) if train_dataset else 0} training samples")
        
        return trainer
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get DPO-specific training arguments.
        
        Returns:
            TrainingArguments instance with DPO-specific settings
        """
        # Get base training arguments
        training_args = super().get_training_arguments()
        
        # DPO-specific modifications
        training_args.remove_unused_columns = getattr(self.config, 'remove_unused_columns', False)
        training_args.gradient_checkpointing = getattr(self.config, 'gradient_checkpointing', True)
        
        # Set appropriate label names for DPO
        training_args.label_names = [
            "chosen_labels", "rejected_labels", 
            "chosen_input_ids", "rejected_input_ids",
            "chosen_attention_mask", "rejected_attention_mask"
        ]
        
        # DPO typically requires smaller learning rates
        if not hasattr(self.config, 'learning_rate_override'):
            training_args.learning_rate = min(training_args.learning_rate, 5e-7)
            logger.info(f"Adjusted learning rate for DPO: {training_args.learning_rate}")
        
        logger.info("DPO training arguments configured")
        
        return training_args
    
    def evaluate_model(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the DPO model.
        
        Args:
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Evaluation metrics including DPO-specific metrics
        """
        eval_result = super().evaluate(eval_dataset)
        
        # Add DPO-specific evaluation metrics
        if "eval_loss" in eval_result:
            # DPO loss is already computed by the trainer
            logger.info(f"DPO evaluation loss: {eval_result['eval_loss']:.4f}")
        
        return eval_result
    
    def compute_preference_accuracy(self, eval_dataset: Dataset) -> float:
        """Compute preference accuracy on evaluation dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Preference accuracy (0-1)
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        logger.info("Computing preference accuracy")
        
        # This would require implementing custom evaluation logic
        # For now, we'll use the trainer's evaluation
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # Extract accuracy if available
        accuracy = eval_result.get("eval_accuracy", 0.0)
        
        logger.info(f"Preference accuracy: {accuracy:.4f}")
        return accuracy
    
    def generate_comparison(self, prompt: str, max_length: int = 512, **kwargs) -> Dict[str, str]:
        """Generate responses from both the trained model and reference model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with responses from both models
        """
        if self.model is None or self.ref_model is None:
            raise ValueError("Models not loaded")
        
        # Generate from trained model
        trained_response = self.predict(
            text=prompt,
            max_length=max_length,
            **kwargs
        )
        
        # Generate from reference model
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        device = next(self.ref_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            ref_outputs = self.ref_model.generate(
                **inputs,
                max_length=max_length,
                do_sample=kwargs.get('do_sample', True),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        ref_response = self.tokenizer.decode(
            ref_outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return {
            "prompt": prompt,
            "trained_model_response": trained_response,
            "reference_model_response": ref_response
        }
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the DPO model with additional metadata.
        
        Args:
            output_dir: Directory to save the model
        """
        super().save_model(output_dir)
        
        # Save DPO-specific metadata
        if output_dir is None:
            output_dir = self.config.output_dir
        
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        
        # Save DPO-specific configs
        dpo_metadata = {
            "training_method": "dpo",
            "beta": getattr(self.config, 'beta', 0.1),
            "loss_type": getattr(self.config, 'loss_type', 'sigmoid'),
            "label_smoothing": getattr(self.config, 'label_smoothing', 0.0),
            "max_prompt_length": getattr(self.config, 'max_prompt_length', 256),
            "max_length": getattr(self.config, 'max_length', 512),
            "ref_model_name": getattr(self.config, 'ref_model_name', self.config.model_name),
        }
        
        metadata_path = output_path / "dpo_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("DPO model and metadata saved successfully")


def create_dpo_trainer(config: DPOConfig) -> DPOTrainer:
    """Factory function to create a DPO trainer.
    
    Args:
        config: DPO configuration
        
    Returns:
        DPOTrainer instance
    """
    return DPOTrainer(config)