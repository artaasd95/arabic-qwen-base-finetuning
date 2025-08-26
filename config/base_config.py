"""Base Configuration Module

This module contains the base configuration class and common settings
for all training methods in the Arabic Qwen fine-tuning framework.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class for all training methods.
    
    This class contains common settings that are shared across
    different training methods (SFT, DPO, KTO, IPO, CPO).
    """
    
    # Model Configuration
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B"))
    model_path: Optional[str] = field(default_factory=lambda: os.getenv("MODEL_PATH"))
    tokenizer_name: Optional[str] = field(default_factory=lambda: os.getenv("TOKENIZER_NAME"))
    
    # Training Configuration
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./checkpoints"))
    logging_dir: str = field(default_factory=lambda: os.getenv("LOGGING_DIR", "./reports/logs"))
    
    # Hardware Configuration
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    fp16: bool = field(default_factory=lambda: os.getenv("FP16", "true").lower() == "true")
    bf16: bool = field(default_factory=lambda: os.getenv("BF16", "false").lower() == "true")
    gradient_checkpointing: bool = field(default_factory=lambda: os.getenv("GRADIENT_CHECKPOINTING", "true").lower() == "true")
    
    # Data Configuration
    max_seq_length: int = field(default_factory=lambda: int(os.getenv("MAX_SEQ_LENGTH", "2048")))
    train_batch_size: int = field(default_factory=lambda: int(os.getenv("TRAIN_BATCH_SIZE", "4")))
    eval_batch_size: int = field(default_factory=lambda: int(os.getenv("EVAL_BATCH_SIZE", "8")))
    
    # Optimization Configuration
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "5e-5")))
    weight_decay: float = field(default_factory=lambda: float(os.getenv("WEIGHT_DECAY", "0.01")))
    warmup_ratio: float = field(default_factory=lambda: float(os.getenv("WARMUP_RATIO", "0.1")))
    num_train_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_TRAIN_EPOCHS", "3")))
    
    # Evaluation Configuration
    eval_strategy: str = field(default_factory=lambda: os.getenv("EVAL_STRATEGY", "steps"))
    eval_steps: int = field(default_factory=lambda: int(os.getenv("EVAL_STEPS", "500")))
    save_strategy: str = field(default_factory=lambda: os.getenv("SAVE_STRATEGY", "steps"))
    save_steps: int = field(default_factory=lambda: int(os.getenv("SAVE_STEPS", "500")))
    save_total_limit: int = field(default_factory=lambda: int(os.getenv("SAVE_TOTAL_LIMIT", "3")))
    
    # Logging Configuration
    logging_steps: int = field(default_factory=lambda: int(os.getenv("LOGGING_STEPS", "100")))
    report_to: List[str] = field(default_factory=lambda: os.getenv("REPORT_TO", "tensorboard").split(","))
    
    # Reproducibility
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create directories if they don't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BaseConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from file."""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)