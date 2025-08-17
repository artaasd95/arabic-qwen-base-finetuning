"""Preference Optimization Trainer Module

This module contains the trainer implementation for preference optimization
methods including KTO, IPO, and CPO training of Arabic Qwen models.
"""

import logging
import torch
from typing import Dict, Optional, Union

from transformers import TrainingArguments
from datasets import Dataset

from .base_trainer import BaseTrainer
from ..config import KTOConfig, IPOConfig, CPOConfig

logger = logging.getLogger(__name__)


class PreferenceTrainer(BaseTrainer):
    """Trainer for preference optimization methods (KTO, IPO, CPO).
    
    This class handles the complete training pipeline for various
    preference optimization methods.
    """
    
    def __init__(self, config: Union[KTOConfig, IPOConfig, CPOConfig]):
        """Initialize the preference trainer.
        
        Args:
            config: Preference optimization configuration object
        """
        super().__init__(config)
        
        # Determine the method from config type
        if isinstance(config, KTOConfig):
            self.method = "kto"
        elif isinstance(config, IPOConfig):
            self.method = "ipo"
        elif isinstance(config, CPOConfig):
            self.method = "cpo"
        else:
            raise TypeError(f"Expected KTOConfig, IPOConfig, or CPOConfig, got {type(config)}")
        
        self.ref_model = None
        logger.info(f"Initialized PreferenceTrainer for {self.method.upper()}")
    
    def _get_training_method(self) -> str:
        """Get the training method name.
        
        Returns:
            Training method name
        """
        return self.method
    
    def setup_reference_model(self):
        """Setup the reference model for preference optimization."""
        if self.method == "kto":
            # KTO doesn't always require a reference model
            if not getattr(self.config, 'use_ref_model', True):
                logger.info("KTO configured without reference model")
                return
        
        logger.info(f"Setting up reference model for {self.method.upper()}")
        
        # Load reference model
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
        """Create the preference optimization trainer instance.
        
        Args:
            dataset: Training dataset
            training_args: Training arguments
            callbacks: List of callbacks
            
        Returns:
            Preference optimization trainer instance
        """
        logger.info(f"Creating {self.method.upper()} trainer")
        
        try:
            if self.method == "kto":
                from trl import KTOTrainer as TRLTrainer
            elif self.method == "ipo":
                from trl import DPOTrainer as TRLTrainer  # IPO uses DPO trainer with different loss
            elif self.method == "cpo":
                from trl import CPOTrainer as TRLTrainer
            else:
                raise ValueError(f"Unsupported method: {self.method}")
        except ImportError:
            raise ImportError(
                f"TRL library is required for {self.method.upper()} training. "
                "Install it with: pip install trl"
            )
        
        # Setup reference model if needed
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
                desc=f"Tokenizing training data for {self.method.upper()}"
            )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.data_loader.tokenize_and_format,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc=f"Tokenizing evaluation data for {self.method.upper()}"
            )
        
        # Get method-specific arguments
        method_args = self._get_method_specific_args()
        
        # Create trainer based on method
        if self.method == "kto":
            trainer = TRLTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                callbacks=callbacks,
                **method_args
            )
        else:
            # IPO and CPO use similar interface to DPO
            trainer = TRLTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                callbacks=callbacks,
                **method_args
            )
        
        logger.info(f"{self.method.upper()} trainer created with {len(train_dataset) if train_dataset else 0} training samples")
        
        return trainer
    
    def _get_method_specific_args(self) -> Dict:
        """Get method-specific arguments for trainer creation.
        
        Returns:
            Dictionary of method-specific arguments
        """
        if self.method == "kto":
            return self.config.get_kto_args()
        elif self.method == "ipo":
            return self.config.get_ipo_args()
        elif self.method == "cpo":
            return self.config.get_cpo_args()
        else:
            return {}
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get method-specific training arguments.
        
        Returns:
            TrainingArguments instance with method-specific settings
        """
        # Get base training arguments
        training_args = super().get_training_arguments()
        
        # Common preference optimization modifications
        training_args.remove_unused_columns = getattr(self.config, 'remove_unused_columns', False)
        training_args.gradient_checkpointing = getattr(self.config, 'gradient_checkpointing', True)
        
        # Set appropriate label names based on method
        if self.method == "kto":
            training_args.label_names = [
                "completion_labels", "completion_input_ids", 
                "completion_attention_mask", "kto_labels"
            ]
        else:
            training_args.label_names = [
                "chosen_labels", "rejected_labels", 
                "chosen_input_ids", "rejected_input_ids",
                "chosen_attention_mask", "rejected_attention_mask"
            ]
        
        # Adjust learning rate for preference optimization
        if not hasattr(self.config, 'learning_rate_override'):
            if self.method in ["ipo", "cpo"]:
                training_args.learning_rate = min(training_args.learning_rate, 1e-6)
            elif self.method == "kto":
                training_args.learning_rate = min(training_args.learning_rate, 5e-7)
            
            logger.info(f"Adjusted learning rate for {self.method.upper()}: {training_args.learning_rate}")
        
        logger.info(f"{self.method.upper()} training arguments configured")
        
        return training_args
    
    def evaluate_model(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the preference optimization model.
        
        Args:
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Evaluation metrics including method-specific metrics
        """
        eval_result = super().evaluate(eval_dataset)
        
        # Add method-specific evaluation metrics
        if "eval_loss" in eval_result:
            logger.info(f"{self.method.upper()} evaluation loss: {eval_result['eval_loss']:.4f}")
        
        return eval_result
    
    def compute_preference_metrics(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Compute preference-specific metrics.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary of preference metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        logger.info(f"Computing {self.method.upper()} preference metrics")
        
        # This would require implementing custom evaluation logic
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # Extract method-specific metrics
        metrics = {}
        
        if self.method == "kto":
            # KTO-specific metrics
            metrics["kto_loss"] = eval_result.get("eval_loss", 0.0)
            metrics["desirable_accuracy"] = eval_result.get("eval_desirable_accuracy", 0.0)
        else:
            # Pairwise preference metrics
            metrics[f"{self.method}_loss"] = eval_result.get("eval_loss", 0.0)
            metrics["preference_accuracy"] = eval_result.get("eval_accuracy", 0.0)
        
        logger.info(f"{self.method.upper()} metrics: {metrics}")
        return metrics
    
    def generate_with_comparison(self, prompt: str, max_length: int = 512, **kwargs) -> Dict[str, str]:
        """Generate responses and compare with reference model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with responses from both models
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Generate from trained model
        trained_response = self.predict(
            text=prompt,
            max_length=max_length,
            **kwargs
        )
        
        result = {
            "prompt": prompt,
            "trained_model_response": trained_response,
        }
        
        # Generate from reference model if available
        if self.ref_model is not None:
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
            
            result["reference_model_response"] = ref_response
        
        return result
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the preference optimization model with metadata.
        
        Args:
            output_dir: Directory to save the model
        """
        super().save_model(output_dir)
        
        # Save method-specific metadata
        if output_dir is None:
            output_dir = self.config.output_dir
        
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        
        # Save method-specific configs
        metadata = {
            "training_method": self.method,
            "max_prompt_length": getattr(self.config, 'max_prompt_length', 256),
            "max_length": getattr(self.config, 'max_length', 512),
            "ref_model_name": getattr(self.config, 'ref_model_name', self.config.model_name),
        }
        
        # Add method-specific parameters
        if self.method == "kto":
            metadata.update({
                "beta": getattr(self.config, 'beta', 0.1),
                "desirable_weight": getattr(self.config, 'desirable_weight', 1.0),
                "undesirable_weight": getattr(self.config, 'undesirable_weight', 1.0),
            })
        elif self.method == "ipo":
            metadata.update({
                "beta": getattr(self.config, 'beta', 0.1),
                "tau": getattr(self.config, 'tau', 0.1),
                "label_smoothing": getattr(self.config, 'label_smoothing', 0.0),
            })
        elif self.method == "cpo":
            metadata.update({
                "beta": getattr(self.config, 'beta', 0.1),
                "loss_type": getattr(self.config, 'loss_type', 'sigmoid'),
                "label_smoothing": getattr(self.config, 'label_smoothing', 0.0),
                "cpo_alpha": getattr(self.config, 'cpo_alpha', 1.0),
                "simpo_gamma": getattr(self.config, 'simpo_gamma', 1.0),
            })
        
        metadata_path = output_path / f"{self.method}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"{self.method.upper()} model and metadata saved successfully")


def create_preference_trainer(config: Union[KTOConfig, IPOConfig, CPOConfig]) -> PreferenceTrainer:
    """Factory function to create a preference optimization trainer.
    
    Args:
        config: Preference optimization configuration
        
    Returns:
        PreferenceTrainer instance
    """
    return PreferenceTrainer(config)


def create_kto_trainer(config: KTOConfig) -> PreferenceTrainer:
    """Factory function to create a KTO trainer.
    
    Args:
        config: KTO configuration
        
    Returns:
        PreferenceTrainer instance for KTO
    """
    return PreferenceTrainer(config)


def create_ipo_trainer(config: IPOConfig) -> PreferenceTrainer:
    """Factory function to create an IPO trainer.
    
    Args:
        config: IPO configuration
        
    Returns:
        PreferenceTrainer instance for IPO
    """
    return PreferenceTrainer(config)


def create_cpo_trainer(config: CPOConfig) -> PreferenceTrainer:
    """Factory function to create a CPO trainer.
    
    Args:
        config: CPO configuration
        
    Returns:
        PreferenceTrainer instance for CPO
    """
    return PreferenceTrainer(config)