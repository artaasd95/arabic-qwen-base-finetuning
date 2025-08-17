"""Utility Functions Module

This module provides utility functions for the Arabic Qwen fine-tuning project.
It includes common utilities, model utilities, and data utilities.
"""

from .common import (
    set_seed,
    setup_logging,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    ensure_dir,
    get_device,
    get_gpu_memory_info,
    format_time,
    format_bytes,
    get_file_hash,
    backup_file,
    clean_directory,
    validate_config,
    merge_configs,
    get_model_size,
    print_model_info,
    create_experiment_name,
    safe_divide,
    truncate_text
)

from .model_utils import (
    load_model_and_tokenizer,
    create_bnb_config,
    create_lora_config,
    apply_lora_to_model,
    load_peft_model,
    merge_and_save_model,
    get_model_memory_usage,
    freeze_model_layers,
    save_model_config,
    load_model_config,
    compare_models,
    get_layer_names,
    print_model_structure
)

from .data_utils import (
    load_dataset_from_path,
    save_dataset,
    split_dataset,
    filter_dataset_by_length,
    clean_text,
    clean_dataset,
    analyze_dataset,
    has_arabic,
    sample_dataset,
    create_instruction_dataset,
    create_preference_dataset,
    visualize_dataset_stats,
    export_dataset_report
)

__all__ = [
    # Common utilities
    "set_seed",
    "setup_logging",
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "ensure_dir",
    "get_device",
    "get_gpu_memory_info",
    "format_time",
    "format_bytes",
    "get_file_hash",
    "backup_file",
    "clean_directory",
    "validate_config",
    "merge_configs",
    "get_model_size",
    "print_model_info",
    "create_experiment_name",
    "safe_divide",
    "truncate_text",
    
    # Model utilities
    "load_model_and_tokenizer",
    "create_bnb_config",
    "create_lora_config",
    "apply_lora_to_model",
    "load_peft_model",
    "merge_and_save_model",
    "get_model_memory_usage",
    "freeze_model_layers",
    "save_model_config",
    "load_model_config",
    "compare_models",
    "get_layer_names",
    "print_model_structure",
    
    # Data utilities
    "load_dataset_from_path",
    "save_dataset",
    "split_dataset",
    "filter_dataset_by_length",
    "clean_text",
    "clean_dataset",
    "analyze_dataset",
    "has_arabic",
    "sample_dataset",
    "create_instruction_dataset",
    "create_preference_dataset",
    "visualize_dataset_stats",
    "export_dataset_report"
]