"""Training Module for Arabic Qwen Fine-tuning

This module provides training implementations for various fine-tuning methods
including SFT, DPO, KTO, IPO, and CPO.
"""

from .base_trainer import BaseTrainer
from .sft_trainer import SFTTrainer, create_sft_trainer
from .dpo_trainer import DPOTrainer, create_dpo_trainer
from .preference_trainer import (
    PreferenceTrainer,
    create_preference_trainer,
    create_kto_trainer,
    create_ipo_trainer,
    create_cpo_trainer
)

__all__ = [
    # Base trainer
    "BaseTrainer",
    
    # Specific trainers
    "SFTTrainer",
    "DPOTrainer", 
    "PreferenceTrainer",
    
    # Factory functions
    "create_sft_trainer",
    "create_dpo_trainer",
    "create_preference_trainer",
    "create_kto_trainer",
    "create_ipo_trainer",
    "create_cpo_trainer",
    
    # Utility functions
    "get_trainer",
    "list_supported_training_methods"
]


def get_trainer(method: str, config):
    """Factory function to get the appropriate trainer based on method.
    
    Args:
        method: Training method ('sft', 'dpo', 'kto', 'ipo', 'cpo')
        config: Configuration object for the specified method
        
    Returns:
        Trainer instance for the specified method
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == "sft":
        return create_sft_trainer(config)
    elif method == "dpo":
        return create_dpo_trainer(config)
    elif method == "kto":
        return create_kto_trainer(config)
    elif method == "ipo":
        return create_ipo_trainer(config)
    elif method == "cpo":
        return create_cpo_trainer(config)
    else:
        raise ValueError(
            f"Unsupported training method: {method}. "
            f"Supported methods: {list_supported_training_methods()}"
        )


def list_supported_training_methods():
    """List all supported training methods.
    
    Returns:
        List of supported training method names
    """
    return ["sft", "dpo", "kto", "ipo", "cpo"]