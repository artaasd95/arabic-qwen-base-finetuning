"""Arabic Qwen Fine-tuning Package

A comprehensive PyTorch-based framework for fine-tuning Qwen models on Arabic datasets
using various training methods including SFT and preference optimization (DPO, KTO, IPO, CPO).
"""

__version__ = "1.0.0"
__author__ = "Arabic Qwen Fine-tuning Team"
__email__ = "contact@example.com"

# Package imports
from .data_loader import *
from .training import *
from .evaluation import *
from .utils import *

__all__ = [
    "data_loader",
    "training", 
    "evaluation",
    "utils"
]