"""Kahneman-Tversky Optimization (KTO) Configuration Module

This module contains configuration settings specific to KTO training
for Arabic Qwen models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from .base_config import BaseConfig


@dataclass
class KTOConfig(BaseConfig):
    """Configuration class for Kahneman-Tversky Optimization (KTO).
    
    This class extends BaseConfig with KTO-specific parameters
    for preference-based fine-tuning using binary feedback.
    """
    
    # KTO-specific Data Configuration
    dataset_name: str = field(default_factory=lambda: os.getenv("KTO_DATASET_NAME", "argilla/ultrafeedback-binarized-preferences-cleaned"))
    dataset_config: Optional[str] = field(default_factory=lambda: os.getenv("KTO_DATASET_CONFIG"))
    dataset_split: str = field(default_factory=lambda: os.getenv("KTO_DATASET_SPLIT", "train"))
    validation_split: str = field(default_factory=lambda: os.getenv("KTO_VALIDATION_SPLIT", "test"))
    
    # KTO Algorithm Parameters
    beta: float = field(default_factory=lambda: float(os.getenv("KTO_BETA", "0.1")))
    desirable_weight: float = field(default_factory=lambda: float(os.getenv("KTO_DESIRABLE_WEIGHT", "1.0")))
    undesirable_weight: float = field(default_factory=lambda: float(os.getenv("KTO_UNDESIRABLE_WEIGHT", "1.0")))
    
    # Reference Model Configuration
    ref_model_name: Optional[str] = field(default_factory=lambda: os.getenv("KTO_REF_MODEL_NAME"))
    ref_model_path: Optional[str] = field(default_factory=lambda: os.getenv("KTO_REF_MODEL_PATH"))
    
    # Data processing
    max_samples: Optional[int] = field(default_factory=lambda: 
        int(os.getenv("KTO_MAX_SAMPLES")) if os.getenv("KTO_MAX_SAMPLES") else None
    )
    max_prompt_length: int = field(default_factory=lambda: int(os.getenv("KTO_MAX_PROMPT_LENGTH", "1024")))
    max_length: int = field(default_factory=lambda: int(os.getenv("KTO_MAX_LENGTH", "2048")))
    
    # Training-specific parameters
    remove_unused_columns: bool = field(default_factory=lambda: os.getenv("KTO_REMOVE_UNUSED_COLUMNS", "false").lower() == "true")
    force_use_ref_model: bool = field(default_factory=lambda: os.getenv("KTO_FORCE_USE_REF_MODEL", "false").lower() == "true")
    
    # LoRA Configuration
    use_lora: bool = field(default_factory=lambda: os.getenv("KTO_USE_LORA", "true").lower() == "true")
    lora_r: int = field(default_factory=lambda: int(os.getenv("KTO_LORA_R", "16")))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv("KTO_LORA_ALPHA", "32")))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv("KTO_LORA_DROPOUT", "0.1")))
    lora_target_modules: List[str] = field(default_factory=lambda: 
        os.getenv("KTO_LORA_TARGET_MODULES", "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
    )
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        super().__post_init__()
        self._validate_kto_config()
    
    def _validate_kto_config(self):
        """Validate KTO-specific configuration parameters."""
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        
        if self.desirable_weight <= 0:
            raise ValueError("desirable_weight must be positive")
        
        if self.undesirable_weight <= 0:
            raise ValueError("undesirable_weight must be positive")
        
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        
        if self.max_length <= self.max_prompt_length:
            raise ValueError("max_length must be greater than max_prompt_length")
        
        if self.use_lora:
            if self.lora_r <= 0:
                raise ValueError("lora_r must be positive when using LoRA")
            
            if self.lora_alpha <= 0:
                raise ValueError("lora_alpha must be positive when using LoRA")
            
            if not (0 <= self.lora_dropout <= 1):
                raise ValueError("lora_dropout must be between 0 and 1")
        
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive if specified")
    
    def get_lora_config(self):
        """Get LoRA configuration dictionary."""
        if not self.use_lora:
            return None
        
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    
    def get_training_args(self):
        """Get training arguments specific to KTO."""
        return {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.train_batch_size,
            "per_device_eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_train_epochs": self.num_train_epochs,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "logging_dir": self.logging_dir,
            "evaluation_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_pin_memory": False,
            "report_to": self.report_to,
            "seed": self.seed,
            "remove_unused_columns": self.remove_unused_columns,
        }
    
    def get_kto_config(self):
        """Get KTO-specific configuration dictionary."""
        return {
            "beta": self.beta,
            "desirable_weight": self.desirable_weight,
            "undesirable_weight": self.undesirable_weight,
            "max_prompt_length": self.max_prompt_length,
            "max_length": self.max_length,
            "force_use_ref_model": self.force_use_ref_model,
        }