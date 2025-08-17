"""Direct Preference Optimization (DPO) Configuration Module

This module contains configuration settings specific to DPO training
for Arabic Qwen models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from .base_config import BaseConfig


@dataclass
class DPOConfig(BaseConfig):
    """Configuration class for Direct Preference Optimization (DPO).
    
    This class extends BaseConfig with DPO-specific parameters
    for preference-based fine-tuning.
    """
    
    # DPO-specific Data Configuration
    dataset_name: str = field(default_factory=lambda: os.getenv("DPO_DATASET_NAME", "HuggingFaceH4/ultrafeedback_binarized"))
    dataset_config: Optional[str] = field(default_factory=lambda: os.getenv("DPO_DATASET_CONFIG"))
    dataset_split: str = field(default_factory=lambda: os.getenv("DPO_DATASET_SPLIT", "train_prefs"))
    validation_split: str = field(default_factory=lambda: os.getenv("DPO_VALIDATION_SPLIT", "test_prefs"))
    
    # DPO Algorithm Parameters
    beta: float = field(default_factory=lambda: float(os.getenv("DPO_BETA", "0.1")))
    loss_type: str = field(default_factory=lambda: os.getenv("DPO_LOSS_TYPE", "sigmoid"))
    label_smoothing: float = field(default_factory=lambda: float(os.getenv("DPO_LABEL_SMOOTHING", "0.0")))
    
    # Reference Model Configuration
    ref_model_name: Optional[str] = field(default_factory=lambda: os.getenv("DPO_REF_MODEL_NAME"))
    ref_model_path: Optional[str] = field(default_factory=lambda: os.getenv("DPO_REF_MODEL_PATH"))
    
    # Data processing
    max_samples: Optional[int] = field(default_factory=lambda: 
        int(os.getenv("DPO_MAX_SAMPLES")) if os.getenv("DPO_MAX_SAMPLES") else None
    )
    max_prompt_length: int = field(default_factory=lambda: int(os.getenv("DPO_MAX_PROMPT_LENGTH", "1024")))
    max_length: int = field(default_factory=lambda: int(os.getenv("DPO_MAX_LENGTH", "2048")))
    
    # Training-specific parameters
    remove_unused_columns: bool = field(default_factory=lambda: os.getenv("DPO_REMOVE_UNUSED_COLUMNS", "false").lower() == "true")
    force_use_ref_model: bool = field(default_factory=lambda: os.getenv("DPO_FORCE_USE_REF_MODEL", "false").lower() == "true")
    
    # LoRA Configuration
    use_lora: bool = field(default_factory=lambda: os.getenv("DPO_USE_LORA", "true").lower() == "true")
    lora_r: int = field(default_factory=lambda: int(os.getenv("DPO_LORA_R", "16")))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv("DPO_LORA_ALPHA", "32")))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv("DPO_LORA_DROPOUT", "0.1")))
    lora_target_modules: List[str] = field(default_factory=lambda: 
        os.getenv("DPO_LORA_TARGET_MODULES", "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
    )
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        super().__post_init__()
        self._validate_dpo_config()
    
    def _validate_dpo_config(self):
        """Validate DPO-specific configuration parameters."""
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        
        if self.loss_type not in ["sigmoid", "hinge", "ipo", "kto_pair"]:
            raise ValueError(f"loss_type must be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair'], got {self.loss_type}")
        
        if not (0 <= self.label_smoothing <= 1):
            raise ValueError("label_smoothing must be between 0 and 1")
        
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
        """Get training arguments specific to DPO."""
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
    
    def get_dpo_config(self):
        """Get DPO-specific configuration dictionary."""
        return {
            "beta": self.beta,
            "loss_type": self.loss_type,
            "label_smoothing": self.label_smoothing,
            "max_prompt_length": self.max_prompt_length,
            "max_length": self.max_length,
            "force_use_ref_model": self.force_use_ref_model,
        }