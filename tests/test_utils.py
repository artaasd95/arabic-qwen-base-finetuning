"""Tests for Utility Functions

This module contains unit tests for the utility functions.
"""

import pytest
import tempfile
import json
import yaml
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from datasets import Dataset

from src.utils import (
    # Common utilities
    set_seed,
    setup_logging,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    ensure_dir,
    get_device,
    format_time,
    format_bytes,
    validate_config,
    merge_configs,
    create_experiment_name,
    safe_divide,
    truncate_text,
    
    # Data utilities
    clean_text,
    has_arabic,
    sample_dataset,
    
    # Model utilities (basic tests)
    create_bnb_config,
    create_lora_config
)


class TestCommonUtilities:
    """Test common utility functions."""
    
    @patch('random.seed')
    @patch('numpy.random.seed')
    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_set_seed(self, mock_cuda_seed, mock_torch_seed, mock_np_seed, mock_random_seed):
        """Test set_seed function."""
        set_seed(42)
        
        mock_random_seed.assert_called_once_with(42)
        mock_np_seed.assert_called_once_with(42)
        mock_torch_seed.assert_called_once_with(42)
        mock_cuda_seed.assert_called_once_with(42)
        
        # Check environment variable
        assert os.environ.get('PYTHONHASHSEED') == '42'
    
    def test_save_and_load_json(self):
        """Test JSON save and load functions."""
        test_data = {
            "name": "test",
            "value": 123,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "test.json"
            
            # Save JSON
            save_json(test_data, json_path)
            assert json_path.exists()
            
            # Load JSON
            loaded_data = load_json(json_path)
            assert loaded_data == test_data
    
    def test_load_json_file_not_found(self):
        """Test load_json with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_json("non_existent_file.json")
    
    def test_save_and_load_yaml(self):
        """Test YAML save and load functions."""
        test_data = {
            "name": "test",
            "value": 123,
            "list": [1, 2, 3]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "test.yaml"
            
            # Save YAML
            save_yaml(test_data, yaml_path)
            assert yaml_path.exists()
            
            # Load YAML
            loaded_data = load_yaml(yaml_path)
            assert loaded_data == test_data
    
    def test_load_yaml_file_not_found(self):
        """Test load_yaml with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml("non_existent_file.yaml")
    
    def test_ensure_dir(self):
        """Test ensure_dir function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "new_dir" / "nested_dir"
            
            # Directory doesn't exist
            assert not test_dir.exists()
            
            # Create directory
            result = ensure_dir(test_dir)
            
            assert test_dir.exists()
            assert test_dir.is_dir()
            assert result == test_dir
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda(self, mock_cuda_available):
        """Test get_device with CUDA available."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.get_device_name', return_value="Test GPU"):
            device = get_device()
            assert device == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_get_device_cpu(self, mock_cuda_available):
        """Test get_device with only CPU available."""
        mock_cuda_available.return_value = False
        
        device = get_device()
        assert device == "cpu"
    
    def test_format_time(self):
        """Test format_time function."""
        # Test seconds
        assert format_time(30.5) == "30.50s"
        
        # Test minutes
        assert format_time(90.5) == "1m 30.5s"
        
        # Test hours
        assert format_time(3661.5) == "1h 1m 1.5s"
    
    def test_format_bytes(self):
        """Test format_bytes function."""
        assert format_bytes(512) == "512.00 B"
        assert format_bytes(1536) == "1.50 KB"
        assert format_bytes(1048576) == "1.00 MB"
        assert format_bytes(1073741824) == "1.00 GB"
    
    def test_validate_config(self):
        """Test validate_config function."""
        # Valid config
        config = {"key1": "value1", "key2": "value2", "key3": "value3"}
        required_keys = ["key1", "key2"]
        
        assert validate_config(config, required_keys) == True
        
        # Invalid config (missing key)
        required_keys = ["key1", "missing_key"]
        assert validate_config(config, required_keys) == False
    
    def test_merge_configs(self):
        """Test merge_configs function."""
        base_config = {
            "key1": "value1",
            "nested": {"a": 1, "b": 2},
            "list": [1, 2, 3]
        }
        
        override_config = {
            "key2": "value2",
            "nested": {"b": 20, "c": 3},
            "list": [4, 5, 6]
        }
        
        merged = merge_configs(base_config, override_config)
        
        assert merged["key1"] == "value1"
        assert merged["key2"] == "value2"
        assert merged["nested"]["a"] == 1
        assert merged["nested"]["b"] == 20
        assert merged["nested"]["c"] == 3
        assert merged["list"] == [4, 5, 6]
    
    def test_create_experiment_name(self):
        """Test create_experiment_name function."""
        # Without timestamp
        name = create_experiment_name(
            method="SFT",
            model_name="Qwen/Qwen2.5-7B",
            dataset_name="my-dataset",
            timestamp=False
        )
        
        assert name == "sft_qwen2.5_7b_my_dataset"
        
        # With timestamp
        name_with_timestamp = create_experiment_name(
            method="DPO",
            model_name="test/model",
            dataset_name="test-data",
            timestamp=True
        )
        
        parts = name_with_timestamp.split("_")
        assert parts[0] == "dpo"
        assert parts[1] == "model"
        assert parts[2] == "test"
        assert len(parts) == 4  # method, model, dataset, timestamp
    
    def test_safe_divide(self):
        """Test safe_divide function."""
        # Normal division
        assert safe_divide(10, 2) == 5.0
        
        # Division by zero with default
        assert safe_divide(10, 0) == 0.0
        
        # Division by zero with custom default
        assert safe_divide(10, 0, default=-1.0) == -1.0
    
    def test_truncate_text(self):
        """Test truncate_text function."""
        # Text shorter than max length
        short_text = "Short text"
        assert truncate_text(short_text, 20) == "Short text"
        
        # Text longer than max length
        long_text = "This is a very long text that should be truncated"
        truncated = truncate_text(long_text, 20)
        assert len(truncated) == 20
        assert truncated.endswith("...")
        
        # Custom suffix
        truncated_custom = truncate_text(long_text, 20, suffix=" [more]")
        assert truncated_custom.endswith(" [more]")


class TestDataUtilities:
    """Test data utility functions."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "  This   has   extra   spaces  "
        cleaned = clean_text(text, remove_extra_whitespace=True)
        assert cleaned == "This has extra spaces"
    
    def test_clean_text_arabic_normalization(self):
        """Test Arabic text normalization."""
        # Test Alef normalization
        text = "أإآا"
        cleaned = clean_text(text, normalize_arabic=True)
        assert cleaned == "اااا"
        
        # Test Yeh normalization
        text = "ى"
        cleaned = clean_text(text, normalize_arabic=True)
        assert cleaned == "ي"
    
    def test_clean_text_remove_diacritics(self):
        """Test Arabic diacritics removal."""
        text = "مَرْحَبًا"
        cleaned = clean_text(text, remove_diacritics=True)
        # Should remove diacritics but keep base letters
        assert "َ" not in cleaned
        assert "ْ" not in cleaned
        assert "ً" not in cleaned
        assert "مرحبا" in cleaned
    
    def test_has_arabic(self):
        """Test has_arabic function."""
        # Text with Arabic
        assert has_arabic("مرحبا") == True
        assert has_arabic("Hello مرحبا") == True
        
        # Text without Arabic
        assert has_arabic("Hello World") == False
        assert has_arabic("123456") == False
        assert has_arabic("") == False
    
    def test_sample_dataset(self):
        """Test sample_dataset function."""
        # Create test dataset
        data = [{"text": f"Example {i}"} for i in range(100)]
        dataset = Dataset.from_list(data)
        
        # Sample smaller dataset
        sampled = sample_dataset(dataset, n_samples=10, seed=42)
        assert len(sampled) == 10
        
        # Sample larger than dataset
        sampled_large = sample_dataset(dataset, n_samples=150, seed=42)
        assert len(sampled_large) == 100  # Should return full dataset


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_create_bnb_config_4bit(self):
        """Test create_bnb_config for 4-bit quantization."""
        config = create_bnb_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4"
        )
        
        assert config.load_in_4bit == True
        assert config.load_in_8bit == False
        assert config.bnb_4bit_compute_dtype == torch.float16
        assert config.bnb_4bit_quant_type == "nf4"
    
    def test_create_bnb_config_8bit(self):
        """Test create_bnb_config for 8-bit quantization."""
        config = create_bnb_config(
            load_in_8bit=True
        )
        
        assert config.load_in_8bit == True
        assert config.load_in_4bit == False
    
    def test_create_bnb_config_invalid(self):
        """Test create_bnb_config with invalid parameters."""
        # Both 4-bit and 8-bit
        with pytest.raises(ValueError, match="Cannot use both 4-bit and 8-bit"):
            create_bnb_config(load_in_4bit=True, load_in_8bit=True)
        
        # Neither 4-bit nor 8-bit
        with pytest.raises(ValueError, match="Must specify either 4-bit or 8-bit"):
            create_bnb_config(load_in_4bit=False, load_in_8bit=False)
    
    def test_create_lora_config(self):
        """Test create_lora_config function."""
        config = create_lora_config(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
    
    def test_create_lora_config_defaults(self):
        """Test create_lora_config with default parameters."""
        config = create_lora_config()
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestUtilityIntegration:
    """Integration tests for utility functions."""
    
    def test_experiment_workflow(self):
        """Test a complete experiment workflow using utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup experiment
            experiment_name = create_experiment_name(
                method="sft",
                model_name="test/model",
                dataset_name="test-data",
                timestamp=False
            )
            
            experiment_dir = ensure_dir(Path(temp_dir) / experiment_name)
            
            # Save experiment config
            config = {
                "model_name": "test/model",
                "method": "sft",
                "parameters": {"lr": 0.001, "epochs": 3}
            }
            
            config_path = experiment_dir / "config.json"
            save_json(config, config_path)
            
            # Load and validate config
            loaded_config = load_json(config_path)
            assert validate_config(loaded_config, ["model_name", "method"])
            
            # Test file operations
            assert experiment_dir.exists()
            assert config_path.exists()
            assert loaded_config == config
    
    def test_text_processing_pipeline(self):
        """Test text processing pipeline."""
        # Raw text with various issues
        raw_texts = [
            "  مَرْحَبًا   بِكُمْ  ",  # Arabic with diacritics and extra spaces
            "Hello   World!  ",  # English with extra spaces
            "أإآا ى ة",  # Arabic letters to normalize
            "Mixed text مرحبا Hello"
        ]
        
        processed_texts = []
        for text in raw_texts:
            # Clean text
            cleaned = clean_text(
                text,
                remove_extra_whitespace=True,
                normalize_arabic=True,
                remove_diacritics=True
            )
            processed_texts.append(cleaned)
            
            # Check if contains Arabic
            contains_arabic = has_arabic(cleaned)
            
            # Truncate if needed
            truncated = truncate_text(cleaned, max_length=20)
            
            assert len(truncated) <= 20
        
        # Verify processing
        assert len(processed_texts) == len(raw_texts)
        assert all("  " not in text for text in processed_texts)  # No double spaces
    
    def test_config_management_workflow(self):
        """Test configuration management workflow."""
        # Base configuration
        base_config = {
            "model": {
                "name": "base/model",
                "quantization": {"enabled": False}
            },
            "training": {
                "epochs": 3,
                "lr": 0.001
            }
        }
        
        # Override configuration
        override_config = {
            "model": {
                "quantization": {"enabled": True, "bits": 4}
            },
            "training": {
                "lr": 0.0001
            },
            "new_section": {
                "value": "test"
            }
        }
        
        # Merge configurations
        merged_config = merge_configs(base_config, override_config)
        
        # Validate merged configuration
        required_keys = ["model", "training"]
        assert validate_config(merged_config, required_keys)
        
        # Check merge results
        assert merged_config["model"]["name"] == "base/model"  # Preserved
        assert merged_config["model"]["quantization"]["enabled"] == True  # Overridden
        assert merged_config["model"]["quantization"]["bits"] == 4  # Added
        assert merged_config["training"]["epochs"] == 3  # Preserved
        assert merged_config["training"]["lr"] == 0.0001  # Overridden
        assert merged_config["new_section"]["value"] == "test"  # Added


if __name__ == "__main__":
    pytest.main([__file__])