#!/usr/bin/env python3
"""
Model Merging Script for Arabic Qwen Fine-tuning Project

This script provides automated model merging capabilities to combine
different fine-tuned models (e.g., SFT + DPO, SFT + ORPO, etc.)
using various merging strategies.

Usage:
    python merge_models.py --strategy weighted --models model1 model2 --weights 0.7 0.3 --output merged_model
    python merge_models.py --strategy sequential --models sft_model dpo_model --output sft_dpo_merged
    python merge_models.py --strategy task_arithmetic --models model1 model2 model3 --output arithmetic_merged
"""

import argparse
import logging
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, LoraConfig
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.model_utils import load_model_and_tokenizer, save_model_config
from utils.common import setup_logging, ensure_dir, save_json, get_device

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    strategy: str  # 'weighted', 'sequential', 'task_arithmetic', 'slerp'
    models: List[str]  # List of model paths
    weights: Optional[List[float]] = None  # Weights for weighted merging
    output_dir: str = "merged_model"
    base_model: Optional[str] = None  # Base model for task arithmetic
    device: str = "auto"
    torch_dtype: str = "float16"
    safe_serialization: bool = True
    merge_lora: bool = True  # Whether to merge LoRA adapters first


class ModelMerger:
    """Advanced model merging utility."""
    
    def __init__(self, config: MergeConfig):
        self.config = config
        self.device = get_device() if config.device == "auto" else config.device
        self.torch_dtype = getattr(torch, config.torch_dtype)
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate merge configuration."""
        if len(self.config.models) < 2:
            raise ValueError("At least 2 models are required for merging")
            
        if self.config.strategy == "weighted":
            if self.config.weights is None:
                # Default to equal weights
                self.config.weights = [1.0 / len(self.config.models)] * len(self.config.models)
            elif len(self.config.weights) != len(self.config.models):
                raise ValueError("Number of weights must match number of models")
            elif abs(sum(self.config.weights) - 1.0) > 1e-6:
                logger.warning("Weights don't sum to 1.0, normalizing...")
                total = sum(self.config.weights)
                self.config.weights = [w / total for w in self.config.weights]
                
        if self.config.strategy == "task_arithmetic" and self.config.base_model is None:
            raise ValueError("Base model is required for task arithmetic merging")
            
        # Check if model paths exist
        for model_path in self.config.models:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
    
    def merge_models(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Merge models according to the specified strategy."""
        logger.info(f"Starting model merging with strategy: {self.config.strategy}")
        logger.info(f"Models to merge: {self.config.models}")
        
        if self.config.strategy == "weighted":
            return self._weighted_merge()
        elif self.config.strategy == "sequential":
            return self._sequential_merge()
        elif self.config.strategy == "task_arithmetic":
            return self._task_arithmetic_merge()
        elif self.config.strategy == "slerp":
            return self._slerp_merge()
        else:
            raise ValueError(f"Unknown merging strategy: {self.config.strategy}")
    
    def _load_model_safe(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Safely load a model and tokenizer."""
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Check if it's a PEFT model
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists() and self.config.merge_lora:
                # Load as PEFT model and merge
                logger.info(f"Detected PEFT model, merging LoRA adapters...")
                
                # Load adapter config to get base model
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')
                
                if base_model_name:
                    # Load base model
                    base_model, tokenizer = load_model_and_tokenizer(
                        base_model_name,
                        device=self.device,
                        torch_dtype=self.torch_dtype
                    )
                    
                    # Load PEFT model
                    peft_model = PeftModel.from_pretrained(base_model, model_path)
                    
                    # Merge and unload
                    merged_model = peft_model.merge_and_unload()
                    return merged_model, tokenizer
            
            # Load as regular model
            model, tokenizer = load_model_and_tokenizer(
                model_path,
                device=self.device,
                torch_dtype=self.torch_dtype
            )
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _weighted_merge(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Merge models using weighted averaging."""
        logger.info(f"Performing weighted merge with weights: {self.config.weights}")
        
        # Load first model as base
        merged_model, tokenizer = self._load_model_safe(self.config.models[0])
        merged_state_dict = merged_model.state_dict()
        
        # Initialize with first model weighted
        for key in merged_state_dict:
            merged_state_dict[key] = merged_state_dict[key] * self.config.weights[0]
        
        # Add other models with their weights
        for i, model_path in enumerate(self.config.models[1:], 1):
            model, _ = self._load_model_safe(model_path)
            model_state_dict = model.state_dict()
            
            for key in merged_state_dict:
                if key in model_state_dict:
                    merged_state_dict[key] += model_state_dict[key] * self.config.weights[i]
                else:
                    logger.warning(f"Key {key} not found in model {model_path}")
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load merged weights
        merged_model.load_state_dict(merged_state_dict)
        
        logger.info("Weighted merge completed successfully")
        return merged_model, tokenizer
    
    def _sequential_merge(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Merge models sequentially (fine-tune on top of each other)."""
        logger.info("Performing sequential merge")
        
        # Start with first model
        current_model, tokenizer = self._load_model_safe(self.config.models[0])
        
        # For sequential merging, we'll use a simple averaging approach
        # In practice, this would involve actual fine-tuning
        for model_path in self.config.models[1:]:
            next_model, _ = self._load_model_safe(model_path)
            
            # Average the parameters
            current_state = current_model.state_dict()
            next_state = next_model.state_dict()
            
            for key in current_state:
                if key in next_state:
                    current_state[key] = (current_state[key] + next_state[key]) / 2.0
            
            current_model.load_state_dict(current_state)
            
            # Free memory
            del next_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Sequential merge completed successfully")
        return current_model, tokenizer
    
    def _task_arithmetic_merge(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Merge models using task arithmetic (task vectors)."""
        logger.info("Performing task arithmetic merge")
        
        # Load base model
        base_model, tokenizer = self._load_model_safe(self.config.base_model)
        base_state_dict = base_model.state_dict()
        
        # Calculate task vectors
        task_vectors = []
        for model_path in self.config.models:
            model, _ = self._load_model_safe(model_path)
            model_state_dict = model.state_dict()
            
            # Calculate task vector (fine-tuned - base)
            task_vector = {}
            for key in base_state_dict:
                if key in model_state_dict:
                    task_vector[key] = model_state_dict[key] - base_state_dict[key]
            
            task_vectors.append(task_vector)
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average task vectors
        merged_task_vector = {}
        for key in base_state_dict:
            vectors = [tv[key] for tv in task_vectors if key in tv]
            if vectors:
                merged_task_vector[key] = torch.stack(vectors).mean(dim=0)
        
        # Apply merged task vector to base model
        merged_state_dict = {}
        for key in base_state_dict:
            if key in merged_task_vector:
                merged_state_dict[key] = base_state_dict[key] + merged_task_vector[key]
            else:
                merged_state_dict[key] = base_state_dict[key]
        
        base_model.load_state_dict(merged_state_dict)
        
        logger.info("Task arithmetic merge completed successfully")
        return base_model, tokenizer
    
    def _slerp_merge(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Merge models using Spherical Linear Interpolation (SLERP)."""
        if len(self.config.models) != 2:
            raise ValueError("SLERP merge requires exactly 2 models")
        
        logger.info("Performing SLERP merge")
        
        # Load models
        model1, tokenizer = self._load_model_safe(self.config.models[0])
        model2, _ = self._load_model_safe(self.config.models[1])
        
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        # SLERP interpolation factor
        t = self.config.weights[0] if self.config.weights else 0.5
        
        merged_state_dict = {}
        for key in state_dict1:
            if key in state_dict2:
                # Flatten tensors for SLERP
                tensor1 = state_dict1[key].flatten()
                tensor2 = state_dict2[key].flatten()
                
                # Compute SLERP
                dot_product = torch.dot(tensor1, tensor2)
                norm1 = torch.norm(tensor1)
                norm2 = torch.norm(tensor2)
                
                # Avoid division by zero
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cos_theta = dot_product / (norm1 * norm2)
                    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                    theta = torch.acos(cos_theta)
                    
                    if theta > 1e-6:
                        sin_theta = torch.sin(theta)
                        w1 = torch.sin((1 - t) * theta) / sin_theta
                        w2 = torch.sin(t * theta) / sin_theta
                        interpolated = w1 * tensor1 + w2 * tensor2
                    else:
                        # Linear interpolation for very small angles
                        interpolated = (1 - t) * tensor1 + t * tensor2
                else:
                    # Fallback to linear interpolation
                    interpolated = (1 - t) * tensor1 + t * tensor2
                
                # Reshape back to original shape
                merged_state_dict[key] = interpolated.reshape(state_dict1[key].shape)
            else:
                merged_state_dict[key] = state_dict1[key]
        
        model1.load_state_dict(merged_state_dict)
        
        # Free memory
        del model2
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("SLERP merge completed successfully")
        return model1, tokenizer
    
    def save_merged_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Save the merged model and tokenizer."""
        output_path = Path(self.config.output_dir)
        ensure_dir(output_path)
        
        logger.info(f"Saving merged model to: {output_path}")
        
        # Save model
        model.save_pretrained(
            output_path,
            safe_serialization=self.config.safe_serialization
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        
        # Save merge configuration
        merge_info = {
            "strategy": self.config.strategy,
            "source_models": self.config.models,
            "weights": self.config.weights,
            "base_model": self.config.base_model,
            "merge_timestamp": str(torch.datetime.now()),
            "device": self.device,
            "torch_dtype": self.config.torch_dtype
        }
        
        save_json(merge_info, output_path / "merge_info.json")
        
        logger.info(f"Merged model saved successfully to: {output_path}")


def create_merge_config_from_args(args) -> MergeConfig:
    """Create merge configuration from command line arguments."""
    return MergeConfig(
        strategy=args.strategy,
        models=args.models,
        weights=args.weights,
        output_dir=args.output,
        base_model=args.base_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        safe_serialization=args.safe_serialization,
        merge_lora=args.merge_lora
    )


def main():
    """Main function for model merging."""
    parser = argparse.ArgumentParser(
        description="Merge fine-tuned models using various strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Weighted merge of SFT and DPO models
  python merge_models.py --strategy weighted \\
    --models models/qwen-3-base-arabic-arabic-chat-SFT models/qwen-3-base-arabic-arabic-chat-DPO \\
    --weights 0.7 0.3 --output models/merged_sft_dpo

  # Sequential merge
  python merge_models.py --strategy sequential \\
    --models models/qwen-3-base-arabic-arabic-chat-SFT models/qwen-3-base-arabic-arabic-chat-DPO \\
    --output models/sequential_sft_dpo

  # Task arithmetic merge
  python merge_models.py --strategy task_arithmetic \\
    --models models/qwen-3-base-arabic-arabic-chat-SFT models/qwen-3-base-arabic-arabic-chat-DPO \\
    --base-model Qwen/Qwen2.5-3B-Instruct --output models/arithmetic_merged
        """
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["weighted", "sequential", "task_arithmetic", "slerp"],
        required=True,
        help="Merging strategy to use"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to models to merge"
    )
    
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        help="Weights for weighted merging (must sum to 1.0)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="merged_model",
        help="Output directory for merged model"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model for task arithmetic merging"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for merging"
    )
    
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for model loading"
    )
    
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        default=True,
        help="Use safe serialization for saving"
    )
    
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        default=True,
        help="Merge LoRA adapters before model merging"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    try:
        # Create merge configuration
        config = create_merge_config_from_args(args)
        
        # Initialize merger
        merger = ModelMerger(config)
        
        # Perform merge
        merged_model, tokenizer = merger.merge_models()
        
        # Save merged model
        merger.save_merged_model(merged_model, tokenizer)
        
        logger.info("Model merging completed successfully!")
        
    except Exception as e:
        logger.error(f"Model merging failed: {e}")
        raise


if __name__ == "__main__":
    main()