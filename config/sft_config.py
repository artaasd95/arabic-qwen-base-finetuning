"""Supervised Fine-Tuning (SFT) Configuration Module

This module contains configuration settings specific to supervised fine-tuning
of Arabic Qwen models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from .base_config import BaseConfig


@dataclass
class SFTConfig(BaseConfig):
    """Configuration class for Supervised Fine-Tuning (SFT).
    
    This class extends BaseConfig with SFT-specific parameters
    for instruction following and general language modeling tasks.
    """
    
    # SFT-specific Data Configuration
    dataset_name: str = field(default_factory=lambda: os.getenv("SFT_DATASET_NAME", "riotu-lab/InstructAr-500k"))
    dataset_config: Optional[str] = field(default_factory=lambda: os.getenv("SFT_DATASET_CONFIG"))
    dataset_split: str = field(default_factory=lambda: os.getenv("SFT_DATASET_SPLIT", "train"))
    validation_split: str = field(default_factory=lambda: os.getenv("SFT_VALIDATION_SPLIT", "validation"))
    
    # Text formatting
    instruction_template: str = field(default_factory=lambda: os.getenv(
        "SFT_INSTRUCTION_TEMPLATE", 
        "### التعليمات:\n{instruction}\n\n### الإجابة:\n{response}"
    ))
    
    # Data processing
    max_samples: Optional[int] = field(default_factory=lambda: 
        int(os.getenv("SFT_MAX_SAMPLES")) if os.getenv("SFT_MAX_SAMPLES") else None
    )
    shuffle_data: bool = field(default_factory=lambda: os.getenv("SFT_SHUFFLE_DATA", "true").lower() == "true")
    
    # Training-specific parameters
    packing: bool = field(default_factory=lambda: os.getenv("SFT_PACKING", "false").lower() == "true")
    response_template: str = field(default_factory=lambda: os.getenv("SFT_RESPONSE_TEMPLATE", "### الإجابة:"))
    
    # LoRA Configuration (if using parameter-efficient fine-tuning)
    use_lora: bool = field(default_factory=lambda: os.getenv("SFT_USE_LORA", "true").lower() == "true")
    lora_r: int = field(default_factory=lambda: int(os.getenv("SFT_LORA_R", "16")))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv("SFT_LORA_ALPHA", "32")))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv("SFT_LORA_DROPOUT", "0.1")))
    lora_target_modules: List[str] = field(default_factory=lambda: 
        os.getenv("SFT_LORA_TARGET_MODULES", "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
    )
    
    # Data collator settings
    mlm: bool = field(default_factory=lambda: os.getenv("SFT_MLM", "false").lower() == "true")
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        super().__post_init__()
        self._validate_sft_config()
    
    def _validate_sft_config(self):
        """Validate SFT-specific configuration parameters."""
        if self.use_lora:
            if self.lora_r <= 0:
                raise ValueError("lora_r must be positive when using LoRA")
            
            if self.lora_alpha <= 0:
                raise ValueError("lora_alpha must be positive when using LoRA")
            
            if not (0 <= self.lora_dropout <= 1):
                raise ValueError("lora_dropout must be between 0 and 1")
            
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules cannot be empty when using LoRA")
        
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
        """Get training arguments specific to SFT."""
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
            "remove_unused_columns": False,
        }