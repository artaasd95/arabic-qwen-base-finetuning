"""Supervised Fine-Tuning (SFT) Trainer Module

This module contains the trainer implementation for supervised fine-tuning
of Arabic Qwen models on instruction-following datasets.
"""

import logging
from typing import Dict, Optional, Union

from transformers import Trainer, TrainingArguments
from datasets import Dataset

from .base_trainer import BaseTrainer
from ..config import SFTConfig

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """Trainer for Supervised Fine-Tuning (SFT).
    
    This class handles the complete SFT training pipeline including
    data loading, model setup, and training execution.
    """
    
    def __init__(self, config: SFTConfig):
        """Initialize the SFT trainer.
        
        Args:
            config: SFT configuration object
        """
        super().__init__(config)
        
        if not isinstance(config, SFTConfig):
            raise TypeError(f"Expected SFTConfig, got {type(config)}")
        
        logger.info("Initialized SFTTrainer")
    
    def _get_training_method(self) -> str:
        """Get the training method name.
        
        Returns:
            Training method name
        """
        return "sft"
    
    def create_trainer(self, dataset, training_args, callbacks) -> Trainer:
        """Create the SFT trainer instance.
        
        Args:
            dataset: Training dataset
            training_args: Training arguments
            callbacks: List of callbacks
            
        Returns:
            Trainer instance for SFT
        """
        logger.info("Creating SFT trainer")
        
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
        
        # Get data collator
        data_collator = self.data_loader.get_data_collator()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        logger.info(f"SFT trainer created with {len(train_dataset) if train_dataset else 0} training samples")
        
        return trainer
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get SFT-specific training arguments.
        
        Returns:
            TrainingArguments instance with SFT-specific settings
        """
        # Get base training arguments
        training_args = super().get_training_arguments()
        
        # SFT-specific modifications
        # Enable gradient checkpointing for memory efficiency
        training_args.gradient_checkpointing = getattr(self.config, 'gradient_checkpointing', True)
        
        # Set appropriate label names for SFT
        training_args.label_names = ["labels"]
        
        # Remove unused columns (important for SFT)
        training_args.remove_unused_columns = getattr(self.config, 'remove_unused_columns', True)
        
        # Set dataloader drop last for consistent batch sizes
        training_args.dataloader_drop_last = getattr(self.config, 'dataloader_drop_last', True)
        
        # Group by length for efficiency (if packing is disabled)
        if not getattr(self.config, 'packing', False):
            training_args.group_by_length = getattr(self.config, 'group_by_length', True)
            training_args.length_column_name = "length"
        
        logger.info("SFT training arguments configured")
        
        return training_args
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the training loss for SFT.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor (and optionally model outputs)
        """
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Calculate loss
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate_model(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the SFT model.
        
        Args:
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Evaluation metrics including perplexity
        """
        eval_result = super().evaluate(eval_dataset)
        
        # Calculate perplexity from eval loss
        if "eval_loss" in eval_result:
            import math
            eval_result["eval_perplexity"] = math.exp(eval_result["eval_loss"])
        
        return eval_result
    
    def generate_response(self, instruction: str, max_length: int = 512, **kwargs) -> str:
        """Generate a response for a given instruction.
        
        Args:
            instruction: Input instruction
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded")
        
        # Format instruction using the template
        instruction_template = getattr(self.config, 'instruction_template', 
                                     "### التعليمات:\n{instruction}\n\n### الإجابة:\n")
        
        # Extract just the instruction part of the template
        if "{instruction}" in instruction_template and "{response}" in instruction_template:
            # Split template to get just the instruction part
            template_parts = instruction_template.split("{response}")
            prompt_template = template_parts[0]
        else:
            prompt_template = instruction_template
        
        formatted_prompt = prompt_template.format(instruction=instruction)
        
        # Generate response
        response = self.predict(
            text=formatted_prompt,
            max_length=max_length,
            **kwargs
        )
        
        return response
    
    def batch_generate(self, instructions: list, max_length: int = 512, **kwargs) -> list:
        """Generate responses for a batch of instructions.
        
        Args:
            instructions: List of input instructions
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for instruction in instructions:
            try:
                response = self.generate_response(
                    instruction=instruction,
                    max_length=max_length,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                logger.warning(f"Failed to generate response for instruction: {e}")
                responses.append("")
        
        return responses
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the SFT model with additional metadata.
        
        Args:
            output_dir: Directory to save the model
        """
        super().save_model(output_dir)
        
        # Save SFT-specific metadata
        if output_dir is None:
            output_dir = self.config.output_dir
        
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        
        # Save instruction template and other SFT-specific configs
        sft_metadata = {
            "training_method": "sft",
            "instruction_template": getattr(self.config, 'instruction_template', None),
            "response_template": getattr(self.config, 'response_template', None),
            "packing": getattr(self.config, 'packing', False),
            "max_seq_length": getattr(self.config, 'max_seq_length', 512),
        }
        
        metadata_path = output_path / "sft_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(sft_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("SFT model and metadata saved successfully")


def create_sft_trainer(config: SFTConfig) -> SFTTrainer:
    """Factory function to create an SFT trainer.
    
    Args:
        config: SFT configuration
        
    Returns:
        SFTTrainer instance
    """
    return SFTTrainer(config)