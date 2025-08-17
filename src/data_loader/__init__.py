"""Data Loader Package for Arabic Qwen Fine-tuning

This package provides modular data loading functionality for different
training methods including SFT and preference optimization (DPO, KTO, IPO, CPO).
"""

from .base_loader import BaseDataLoader
from .sft_loader import SFTDataLoader
from .dpo_loader import DPODataLoader
from .preference_loader import PreferenceDataLoader

__version__ = "1.0.0"

__all__ = [
    "BaseDataLoader",
    "SFTDataLoader",
    "DPODataLoader",
    "PreferenceDataLoader",
]


def get_data_loader(method: str, tokenizer, **kwargs):
    """Factory function to get the appropriate data loader for a training method.
    
    Args:
        method: Training method ("sft", "dpo", "kto", "ipo", "cpo")
        tokenizer: Pre-trained tokenizer
        **kwargs: Additional arguments passed to the data loader
        
    Returns:
        Appropriate data loader instance
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == "sft":
        return SFTDataLoader(tokenizer, **kwargs)
    elif method == "dpo":
        return DPODataLoader(tokenizer, **kwargs)
    elif method in ["kto", "ipo", "cpo"]:
        return PreferenceDataLoader(tokenizer, method=method, **kwargs)
    else:
        raise ValueError(
            f"Unsupported training method: {method}. "
            f"Supported methods: sft, dpo, kto, ipo, cpo"
        )


def list_supported_methods():
    """Get a list of supported training methods.
    
    Returns:
        List of supported method names
    """
    return ["sft", "dpo", "kto", "ipo", "cpo"]