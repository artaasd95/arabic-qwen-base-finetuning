"""Odds Ratio Preference Optimization (ORPO) Configuration Module

This module contains configuration settings specific to ORPO training
for Arabic Qwen models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from .base_config import BaseConfig


@dataclass
class ORPOConfig(BaseConfig):
    """Configuration class for Odds Ratio Preference Optimization (ORPO).
    
    This class extends BaseConfig with ORPO-specific parameters
    for preference-based fine-tuning without requiring a reference model.
    """
    
    # ORPO-specific Data Configuration
    dataset_name: str = field(default_factory=lambda: os.getenv("ORPO_DATASET_NAME", "HuggingFaceH4/ultrafeedback_binarized"))
    dataset_config: Optional[str] = field(default_factory=lambda: os.getenv("ORPO_DATASET_CONFIG"))
    dataset_split: str = field(default_factory=lambda: os.getenv("ORPO_DATASET_SPLIT", "train_prefs"))
    validation_split: str = field(default_factory=lambda: os.getenv("ORPO_VALIDATION_SPLIT", "test_prefs"))
    
    # ORPO Algorithm Parameters
    lambda_reg: float = field(default_factory=lambda: float(os.getenv("ORPO_LAMBDA_REG", "0.1")))
    odds_ratio_loss_beta: float = field(default_factory=lambda: float(os.getenv("ORPO_ODDS_RATIO_LOSS_BETA", "0.1")))
    
    # Data processing
    max_samples: Optional[int] = field(default_factory=lambda: 
        int(os.getenv("ORPO_MAX_SAMPLES")) if os.getenv("ORPO_MAX_SAMPLES") else None
    )
    max_prompt_length: int = field(default_factory=lambda: int(os.getenv("ORPO_MAX_PROMPT_LENGTH", "1024")))
    max_length: int = field(default_factory=lambda: int(os.getenv("ORPO_MAX_LENGTH", "2048")))
    
    # Training-specific parameters
    remove_unused_columns: bool = field(default_factory=lambda: os.getenv("ORPO_REMOVE_UNUSED_COLUMNS", "false").lower() == "true")
    
    # LoRA Configuration
    use_lora: bool = field(default_factory=lambda: os.getenv("ORPO_USE_LORA", "true").lower() == "true")
    lora_r: int = field(default_factory=lambda: int(os.getenv("ORPO_LORA_R", "16")))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv("ORPO_LORA_ALPHA", "32")))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv("ORPO_LORA_DROPOUT", "0.1")))
    lora_target_modules: List[str] = field(default_factory=lambda: 
        os.getenv("ORPO_LORA_TARGET_MODULES", "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
    )
    
    # Optimization parameters
    gradient_checkpointing: bool = field(default_factory=lambda: os.getenv("ORPO_GRADIENT_CHECKPOINTING", "true").lower() == "true")
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        super().__post_init__()
        
        # Validate ORPO-specific parameters
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive")
        if self.odds_ratio_loss_beta <= 0:
            raise ValueError("odds_ratio_loss_beta must be positive")
        
        # Set method-specific output directory
        if not hasattr(self, '_output_dir_set'):
            self.output_dir = os.path.join(self.output_dir, "orpo")
            self._output_dir_set = True
    
    def get_training_args_dict(self) -> dict:
        """Get training arguments as dictionary for ORPO trainer."""
        return {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.train_batch_size,
            "per_device_eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "eval_strategy": self.eval_strategy,
            "save_strategy": self.save_strategy,
            "save_total_limit": self.save_total_limit,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "remove_unused_columns": self.remove_unused_columns,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
        }
    
    def get_orpo_args_dict(self) -> dict:
        """Get ORPO-specific arguments."""
        return {
            "lambda_reg": self.lambda_reg,
            "odds_ratio_loss_beta": self.odds_ratio_loss_beta,
            "max_prompt_length": self.max_prompt_length,
            "max_length": self.max_length,
        }
    
    def get_lora_config_dict(self) -> dict:
        """Get LoRA configuration dictionary."""
        if not self.use_lora:
            return {}
        
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }