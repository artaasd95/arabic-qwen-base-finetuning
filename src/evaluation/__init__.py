"""Evaluation Module for Arabic Qwen Fine-tuning

This module provides evaluation implementations for various fine-tuning methods
including SFT, DPO, KTO, IPO, and CPO.
"""

from .base_evaluator import BaseEvaluator
from .sft_evaluator import SFTEvaluator, create_sft_evaluator
from .preference_evaluator import PreferenceEvaluator, create_preference_evaluator

__all__ = [
    # Base evaluator
    "BaseEvaluator",
    
    # Specific evaluators
    "SFTEvaluator",
    "PreferenceEvaluator",
    
    # Factory functions
    "create_sft_evaluator",
    "create_preference_evaluator",
    
    # Utility functions
    "get_evaluator",
    "list_supported_evaluation_methods"
]


def get_evaluator(method: str, **kwargs):
    """Factory function to get the appropriate evaluator based on method.
    
    Args:
        method: Evaluation method ('sft', 'dpo', 'kto', 'ipo', 'cpo')
        **kwargs: Additional parameters for evaluator initialization
        
    Returns:
        Evaluator instance for the specified method
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == "sft":
        return create_sft_evaluator(**kwargs)
    elif method in ["dpo", "kto", "ipo", "cpo"]:
        return create_preference_evaluator(method=method, **kwargs)
    else:
        raise ValueError(
            f"Unsupported evaluation method: {method}. "
            f"Supported methods: {list_supported_evaluation_methods()}"
        )


def list_supported_evaluation_methods():
    """List all supported evaluation methods.
    
    Returns:
        List of supported evaluation method names
    """
    return ["sft", "dpo", "kto", "ipo", "cpo"]