"""Model Utility Functions

This module contains utility functions for model operations in the Arabic Qwen
fine-tuning project.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import json

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    quantization_config: Optional[Dict[str, Any]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    use_auth_token: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer.
    
    Args:
        model_name: Model name or path
        device: Device to load model on
        quantization_config: Quantization configuration
        torch_dtype: Torch data type
        trust_remote_code: Whether to trust remote code
        use_auth_token: Hugging Face auth token
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Prepare model loading arguments
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "use_auth_token": use_auth_token
    }
    
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    
    if device != "auto":
        model_kwargs["device_map"] = device
    else:
        model_kwargs["device_map"] = "auto"
    
    # Add quantization config if provided
    if quantization_config is not None:
        bnb_config = create_bnb_config(**quantization_config)
        model_kwargs["quantization_config"] = bnb_config
        logger.info(f"Using quantization: {quantization_config}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    logger.info(f"Model loaded successfully: {model_name}")
    return model, tokenizer


def create_bnb_config(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True
) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization configuration.
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization
        load_in_8bit: Whether to use 8-bit quantization
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_quant_type: Quantization type for 4-bit
        bnb_4bit_use_double_quant: Whether to use double quantization
        
    Returns:
        BitsAndBytesConfig object
    """
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot use both 4-bit and 8-bit quantization")
    
    if not load_in_4bit and not load_in_8bit:
        raise ValueError("Must specify either 4-bit or 8-bit quantization")
    
    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    compute_dtype = dtype_mapping.get(bnb_4bit_compute_dtype, torch.float16)
    
    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
    )
    
    logger.info(f"Created BnB config: 4bit={load_in_4bit}, 8bit={load_in_8bit}")
    return config


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.1,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    modules_to_save: Optional[List[str]] = None
) -> LoraConfig:
    """Create LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: Target modules for LoRA
        lora_dropout: LoRA dropout rate
        bias: Bias configuration
        task_type: Task type
        modules_to_save: Additional modules to save
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default target modules for Qwen models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Convert string task type to TaskType enum
    task_type_mapping = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS
    }
    
    task_type_enum = task_type_mapping.get(task_type, TaskType.CAUSAL_LM)
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type_enum,
        modules_to_save=modules_to_save
    )
    
    logger.info(f"Created LoRA config: r={r}, alpha={lora_alpha}, targets={target_modules}")
    return config


def apply_lora_to_model(
    model: PreTrainedModel,
    lora_config: LoraConfig
) -> PeftModel:
    """Apply LoRA to a model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        PEFT model with LoRA applied
    """
    logger.info("Applying LoRA to model")
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return peft_model


def load_peft_model(
    base_model_name: str,
    peft_model_path: str,
    device: str = "auto",
    quantization_config: Optional[Dict[str, Any]] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """Load a PEFT model.
    
    Args:
        base_model_name: Base model name
        peft_model_path: Path to PEFT model
        device: Device to load model on
        quantization_config: Quantization configuration
        torch_dtype: Torch data type
        
    Returns:
        Tuple of (peft_model, tokenizer)
    """
    logger.info(f"Loading PEFT model from {peft_model_path}")
    
    # Load base model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(
        base_model_name,
        device=device,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype
    )
    
    # Load PEFT model
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    logger.info(f"PEFT model loaded successfully from {peft_model_path}")
    return peft_model, tokenizer


def merge_and_save_model(
    peft_model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    safe_serialization: bool = True
):
    """Merge PEFT model with base model and save.
    
    Args:
        peft_model: PEFT model to merge
        tokenizer: Tokenizer
        output_dir: Output directory
        safe_serialization: Whether to use safe serialization
    """
    logger.info(f"Merging and saving model to {output_dir}")
    
    # Merge model
    merged_model = peft_model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Merged model saved to {output_dir}")


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """Get model memory usage.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with memory usage info
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "param_size_mb": param_size / (1024 * 1024),
        "buffer_size_mb": buffer_size / (1024 * 1024),
        "total_size_mb": total_size / (1024 * 1024),
        "param_size_gb": param_size / (1024 * 1024 * 1024),
        "buffer_size_gb": buffer_size / (1024 * 1024 * 1024),
        "total_size_gb": total_size / (1024 * 1024 * 1024)
    }


def freeze_model_layers(
    model: nn.Module,
    freeze_embeddings: bool = True,
    freeze_layers: Optional[List[int]] = None,
    unfreeze_layers: Optional[List[int]] = None
):
    """Freeze specific layers of a model.
    
    Args:
        model: Model to freeze layers
        freeze_embeddings: Whether to freeze embedding layers
        freeze_layers: List of layer indices to freeze
        unfreeze_layers: List of layer indices to unfreeze
    """
    logger.info("Freezing model layers")
    
    # Freeze embeddings
    if freeze_embeddings:
        for name, param in model.named_parameters():
            if "embed" in name.lower():
                param.requires_grad = False
                logger.debug(f"Frozen embedding: {name}")
    
    # Freeze specific layers
    if freeze_layers is not None:
        for layer_idx in freeze_layers:
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}" in name or f"layer.{layer_idx}" in name:
                    param.requires_grad = False
                    logger.debug(f"Frozen layer {layer_idx}: {name}")
    
    # Unfreeze specific layers
    if unfreeze_layers is not None:
        for layer_idx in unfreeze_layers:
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}" in name or f"layer.{layer_idx}" in name:
                    param.requires_grad = True
                    logger.debug(f"Unfrozen layer {layer_idx}: {name}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Trainable parameters after freezing: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")


def save_model_config(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config_path: str,
    additional_info: Optional[Dict[str, Any]] = None
):
    """Save model configuration and metadata.
    
    Args:
        model: Model
        tokenizer: Tokenizer
        config_path: Path to save configuration
        additional_info: Additional information to save
    """
    config_data = {
        "model_config": model.config.to_dict(),
        "tokenizer_config": {
            "vocab_size": tokenizer.vocab_size,
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token,
            "unk_token": tokenizer.unk_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token_id": tokenizer.unk_token_id
        },
        "model_info": {
            "model_type": model.config.model_type,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    if additional_info is not None:
        config_data.update(additional_info)
    
    # Save configuration
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Model configuration saved to {config_path}")


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    logger.info(f"Model configuration loaded from {config_path}")
    return config_data


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """Compare two models.
    
    Args:
        model1: First model
        model2: Second model
        tolerance: Tolerance for parameter comparison
        
    Returns:
        Comparison results
    """
    logger.info("Comparing models")
    
    # Get parameter dictionaries
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # Compare parameter names
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())
    
    common_keys = keys1.intersection(keys2)
    only_in_model1 = keys1 - keys2
    only_in_model2 = keys2 - keys1
    
    # Compare parameter values
    different_params = []
    for key in common_keys:
        param1 = params1[key]
        param2 = params2[key]
        
        if param1.shape != param2.shape:
            different_params.append({
                "name": key,
                "reason": "shape_mismatch",
                "shape1": param1.shape,
                "shape2": param2.shape
            })
        elif not torch.allclose(param1, param2, atol=tolerance):
            max_diff = torch.max(torch.abs(param1 - param2)).item()
            different_params.append({
                "name": key,
                "reason": "value_difference",
                "max_difference": max_diff
            })
    
    results = {
        "total_params_model1": len(params1),
        "total_params_model2": len(params2),
        "common_params": len(common_keys),
        "only_in_model1": list(only_in_model1),
        "only_in_model2": list(only_in_model2),
        "different_params": different_params,
        "models_identical": len(different_params) == 0 and len(only_in_model1) == 0 and len(only_in_model2) == 0
    }
    
    logger.info(f"Model comparison complete: identical={results['models_identical']}")
    return results


def get_layer_names(model: nn.Module, include_weights: bool = True) -> List[str]:
    """Get all layer names in a model.
    
    Args:
        model: PyTorch model
        include_weights: Whether to include weight parameter names
        
    Returns:
        List of layer names
    """
    if include_weights:
        return [name for name, _ in model.named_parameters()]
    else:
        return [name for name, _ in model.named_modules() if name != ""]


def print_model_structure(model: nn.Module, max_depth: int = 3):
    """Print model structure.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to print
    """
    print("\nModel Structure:")
    print("=" * 50)
    
    for name, module in model.named_modules():
        if name == "":
            continue
        
        depth = name.count(".")
        if depth > max_depth:
            continue
        
        indent = "  " * depth
        module_type = type(module).__name__
        
        # Get parameter count for this module
        param_count = sum(p.numel() for p in module.parameters())
        
        print(f"{indent}{name}: {module_type} ({param_count:,} params)")
    
    print("=" * 50)