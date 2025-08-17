"""Environment Configuration Module

This module provides utilities for loading environment variables
and creating configuration objects for different training methods.
"""

import os
from typing import Union, Dict, Any
from pathlib import Path

from .base_config import BaseConfig
from .sft_config import SFTConfig
from .dpo_config import DPOConfig
from .kto_config import KTOConfig
from .ipo_config import IPOConfig
from .cpo_config import CPOConfig


class ConfigFactory:
    """Factory class for creating configuration objects."""
    
    _config_classes = {
        "sft": SFTConfig,
        "dpo": DPOConfig,
        "kto": KTOConfig,
        "ipo": IPOConfig,
        "cpo": CPOConfig,
    }
    
    @classmethod
    def create_config(cls, method: str, **kwargs) -> BaseConfig:
        """Create a configuration object for the specified training method.
        
        Args:
            method: Training method name (sft, dpo, kto, ipo, cpo)
            **kwargs: Additional configuration parameters
            
        Returns:
            Configuration object for the specified method
            
        Raises:
            ValueError: If method is not supported
        """
        method = method.lower()
        if method not in cls._config_classes:
            raise ValueError(
                f"Unsupported training method: {method}. "
                f"Supported methods: {list(cls._config_classes.keys())}"
            )
        
        config_class = cls._config_classes[method]
        return config_class(**kwargs)
    
    @classmethod
    def load_config_from_env(cls, method: str) -> BaseConfig:
        """Load configuration from environment variables.
        
        Args:
            method: Training method name (sft, dpo, kto, ipo, cpo)
            
        Returns:
            Configuration object loaded from environment variables
        """
        return cls.create_config(method)
    
    @classmethod
    def load_config_from_file(cls, method: str, config_path: Union[str, Path]) -> BaseConfig:
        """Load configuration from JSON file.
        
        Args:
            method: Training method name (sft, dpo, kto, ipo, cpo)
            config_path: Path to configuration file
            
        Returns:
            Configuration object loaded from file
        """
        config_class = cls._config_classes[method.lower()]
        return config_class.load(config_path)
    
    @classmethod
    def get_supported_methods(cls) -> list:
        """Get list of supported training methods."""
        return list(cls._config_classes.keys())


def load_env_file(env_path: Union[str, Path] = ".env"):
    """Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file
    """
    env_path = Path(env_path)
    if not env_path.exists():
        print(f"Warning: Environment file {env_path} not found")
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value


def get_env_var(key: str, default: Any = None, var_type: type = str) -> Any:
    """Get environment variable with type conversion.
    
    Args:
        key: Environment variable key
        default: Default value if key not found
        var_type: Type to convert the value to
        
    Returns:
        Environment variable value converted to specified type
    """
    value = os.getenv(key, default)
    
    if value is None:
        return None
    
    if var_type == bool:
        return str(value).lower() in ('true', '1', 'yes', 'on')
    elif var_type == list:
        return str(value).split(',')
    else:
        return var_type(value)


def validate_environment() -> Dict[str, bool]:
    """Validate that required environment variables are set.
    
    Returns:
        Dictionary with validation results for each method
    """
    validation_results = {}
    
    # Common required variables
    common_vars = [
        "MODEL_NAME",
        "OUTPUT_DIR",
        "LOGGING_DIR"
    ]
    
    # Method-specific required variables
    method_vars = {
        "sft": ["SFT_DATASET_NAME"],
        "dpo": ["DPO_DATASET_NAME", "DPO_BETA"],
        "kto": ["KTO_DATASET_NAME", "KTO_BETA"],
        "ipo": ["IPO_DATASET_NAME", "IPO_BETA", "IPO_TAU"],
        "cpo": ["CPO_DATASET_NAME", "CPO_BETA", "CPO_ALPHA"]
    }
    
    for method, required_vars in method_vars.items():
        all_vars = common_vars + required_vars
        missing_vars = [var for var in all_vars if not os.getenv(var)]
        validation_results[method] = len(missing_vars) == 0
        
        if missing_vars:
            print(f"Warning: Missing environment variables for {method}: {missing_vars}")
    
    return validation_results


def create_sample_env_file(output_path: Union[str, Path] = ".env.sample"):
    """Create a sample environment file with all configuration options.
    
    Args:
        output_path: Path to output sample file
    """
    sample_content = '''# Arabic Qwen Fine-tuning Environment Configuration
# Copy this file to .env and modify the values as needed

# =============================================================================
# COMMON CONFIGURATION
# =============================================================================

# Model Configuration
MODEL_NAME=Qwen/Qwen2.5-0.5B
# MODEL_PATH=./models/qwen-0.5b  # Optional: local model path
# TOKENIZER_NAME=Qwen/Qwen2.5-0.5B  # Optional: custom tokenizer

# Output and Logging
OUTPUT_DIR=./checkpoints
LOGGING_DIR=./reports/logs

# Hardware Configuration
DEVICE=auto
FP16=true
BF16=false
GRADIENT_CHECKPOINTING=true

# Training Configuration
MAX_SEQ_LENGTH=2048
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
NUM_TRAIN_EPOCHS=3

# Evaluation and Saving
EVAL_STRATEGY=steps
EVAL_STEPS=500
SAVE_STRATEGY=steps
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=100
REPORT_TO=tensorboard

# Reproducibility
SEED=42

# =============================================================================
# SFT (SUPERVISED FINE-TUNING) CONFIGURATION
# =============================================================================

# Dataset Configuration
SFT_DATASET_NAME=riotu-lab/InstructAr-500k
# SFT_DATASET_CONFIG=default
SFT_DATASET_SPLIT=train
SFT_VALIDATION_SPLIT=validation

# Text Formatting
SFT_INSTRUCTION_TEMPLATE="### التعليمات:\n{instruction}\n\n### الإجابة:\n{response}"
SFT_RESPONSE_TEMPLATE="### الإجابة:"

# Data Processing
# SFT_MAX_SAMPLES=10000  # Optional: limit number of samples
SFT_SHUFFLE_DATA=true
SFT_PACKING=false
SFT_MLM=false

# LoRA Configuration
SFT_USE_LORA=true
SFT_LORA_R=16
SFT_LORA_ALPHA=32
SFT_LORA_DROPOUT=0.1
SFT_LORA_TARGET_MODULES=q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

# =============================================================================
# DPO (DIRECT PREFERENCE OPTIMIZATION) CONFIGURATION
# =============================================================================

# Dataset Configuration
DPO_DATASET_NAME=HuggingFaceH4/ultrafeedback_binarized
DPO_DATASET_SPLIT=train_prefs
DPO_VALIDATION_SPLIT=test_prefs

# Algorithm Parameters
DPO_BETA=0.1
DPO_LOSS_TYPE=sigmoid
DPO_LABEL_SMOOTHING=0.0

# Reference Model
# DPO_REF_MODEL_NAME=Qwen/Qwen2.5-0.5B
# DPO_REF_MODEL_PATH=./models/qwen-0.5b-sft

# Data Processing
DPO_MAX_PROMPT_LENGTH=1024
DPO_MAX_LENGTH=2048
DPO_REMOVE_UNUSED_COLUMNS=false
DPO_FORCE_USE_REF_MODEL=false

# LoRA Configuration
DPO_USE_LORA=true
DPO_LORA_R=16
DPO_LORA_ALPHA=32
DPO_LORA_DROPOUT=0.1
DPO_LORA_TARGET_MODULES=q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

# =============================================================================
# KTO (KAHNEMAN-TVERSKY OPTIMIZATION) CONFIGURATION
# =============================================================================

# Dataset Configuration
KTO_DATASET_NAME=argilla/ultrafeedback-binarized-preferences-cleaned
KTO_DATASET_SPLIT=train
KTO_VALIDATION_SPLIT=test

# Algorithm Parameters
KTO_BETA=0.1
KTO_DESIRABLE_WEIGHT=1.0
KTO_UNDESIRABLE_WEIGHT=1.0

# Reference Model
# KTO_REF_MODEL_NAME=Qwen/Qwen2.5-0.5B
# KTO_REF_MODEL_PATH=./models/qwen-0.5b-sft

# Data Processing
KTO_MAX_PROMPT_LENGTH=1024
KTO_MAX_LENGTH=2048
KTO_REMOVE_UNUSED_COLUMNS=false
KTO_FORCE_USE_REF_MODEL=false

# LoRA Configuration
KTO_USE_LORA=true
KTO_LORA_R=16
KTO_LORA_ALPHA=32
KTO_LORA_DROPOUT=0.1
KTO_LORA_TARGET_MODULES=q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

# =============================================================================
# IPO (IDENTITY PREFERENCE OPTIMIZATION) CONFIGURATION
# =============================================================================

# Dataset Configuration
IPO_DATASET_NAME=HuggingFaceH4/ultrafeedback_binarized
IPO_DATASET_SPLIT=train_prefs
IPO_VALIDATION_SPLIT=test_prefs

# Algorithm Parameters
IPO_BETA=0.1
IPO_TAU=0.1
IPO_LABEL_SMOOTHING=0.0

# Reference Model
# IPO_REF_MODEL_NAME=Qwen/Qwen2.5-0.5B
# IPO_REF_MODEL_PATH=./models/qwen-0.5b-sft

# Data Processing
IPO_MAX_PROMPT_LENGTH=1024
IPO_MAX_LENGTH=2048
IPO_REMOVE_UNUSED_COLUMNS=false
IPO_FORCE_USE_REF_MODEL=false

# LoRA Configuration
IPO_USE_LORA=true
IPO_LORA_R=16
IPO_LORA_ALPHA=32
IPO_LORA_DROPOUT=0.1
IPO_LORA_TARGET_MODULES=q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

# =============================================================================
# CPO (CONSERVATIVE POLICY OPTIMIZATION) CONFIGURATION
# =============================================================================

# Dataset Configuration
CPO_DATASET_NAME=HuggingFaceH4/ultrafeedback_binarized
CPO_DATASET_SPLIT=train_prefs
CPO_VALIDATION_SPLIT=test_prefs

# Algorithm Parameters
CPO_BETA=0.1
CPO_LOSS_TYPE=sigmoid
CPO_LABEL_SMOOTHING=0.0
CPO_ALPHA=1.0
CPO_SIMPO_GAMMA=0.5

# Reference Model
# CPO_REF_MODEL_NAME=Qwen/Qwen2.5-0.5B
# CPO_REF_MODEL_PATH=./models/qwen-0.5b-sft

# Data Processing
CPO_MAX_PROMPT_LENGTH=1024
CPO_MAX_LENGTH=2048
CPO_REMOVE_UNUSED_COLUMNS=false
CPO_FORCE_USE_REF_MODEL=false

# LoRA Configuration
CPO_USE_LORA=true
CPO_LORA_R=16
CPO_LORA_ALPHA=32
CPO_LORA_DROPOUT=0.1
CPO_LORA_TARGET_MODULES=q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"Sample environment file created at: {output_path}")