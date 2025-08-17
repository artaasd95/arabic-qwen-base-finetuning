"""Tests for Configuration Modules

This module contains unit tests for the configuration classes.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import (
    BaseConfig,
    SFTConfig,
    DPOConfig,
    KTOConfig,
    IPOConfig,
    CPOConfig,
    get_config,
    list_supported_methods
)


class TestBaseConfig:
    """Test BaseConfig class."""
    
    def test_base_config_initialization(self):
        """Test BaseConfig initialization."""
        config = BaseConfig()
        
        # Check default values
        assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.output_dir == "./checkpoints"
        assert config.logging_dir == "./reports/logs"
        assert config.seed == 42
        assert config.device == "auto"
    
    def test_base_config_with_custom_values(self):
        """Test BaseConfig with custom values."""
        config = BaseConfig(
            model_name="custom/model",
            output_dir="./custom_output",
            seed=123
        )
        
        assert config.model_name == "custom/model"
        assert config.output_dir == "./custom_output"
        assert config.seed == 123
    
    def test_base_config_from_dict(self):
        """Test BaseConfig.from_dict method."""
        config_dict = {
            "model_name": "test/model",
            "output_dir": "./test_output",
            "seed": 456
        }
        
        config = BaseConfig.from_dict(config_dict)
        
        assert config.model_name == "test/model"
        assert config.output_dir == "./test_output"
        assert config.seed == 456
    
    def test_base_config_to_dict(self):
        """Test BaseConfig.to_dict method."""
        config = BaseConfig(model_name="test/model", seed=789)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test/model"
        assert config_dict["seed"] == 789
    
    @patch.dict(os.environ, {"MODEL_NAME": "env/model", "SEED": "999"})
    def test_base_config_load_from_env(self):
        """Test BaseConfig.load_from_env method."""
        config = BaseConfig.load_from_env()
        
        assert config.model_name == "env/model"
        assert config.seed == 999
    
    def test_base_config_save_and_load(self):
        """Test BaseConfig save and load methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Create and save config
            original_config = BaseConfig(model_name="save/test", seed=111)
            original_config.save(str(config_path))
            
            # Load config
            loaded_config = BaseConfig.load(str(config_path))
            
            assert loaded_config.model_name == "save/test"
            assert loaded_config.seed == 111
    
    def test_base_config_validate(self):
        """Test BaseConfig.validate method."""
        # Valid config
        valid_config = BaseConfig()
        assert valid_config.validate() == True
        
        # Invalid config (empty model name)
        invalid_config = BaseConfig(model_name="")
        assert invalid_config.validate() == False


class TestSFTConfig:
    """Test SFTConfig class."""
    
    def test_sft_config_initialization(self):
        """Test SFTConfig initialization."""
        config = SFTConfig()
        
        # Check SFT-specific defaults
        assert config.max_seq_length == 2048
        assert config.packing == True
        assert config.dataset_text_field == "text"
        assert config.instruction_template is not None
        assert config.response_template is not None
    
    def test_sft_config_inheritance(self):
        """Test SFTConfig inherits from BaseConfig."""
        config = SFTConfig()
        
        # Check inherited attributes
        assert hasattr(config, "model_name")
        assert hasattr(config, "output_dir")
        assert hasattr(config, "seed")
    
    def test_sft_config_custom_values(self):
        """Test SFTConfig with custom values."""
        config = SFTConfig(
            max_seq_length=1024,
            packing=False,
            dataset_text_field="content"
        )
        
        assert config.max_seq_length == 1024
        assert config.packing == False
        assert config.dataset_text_field == "content"


class TestDPOConfig:
    """Test DPOConfig class."""
    
    def test_dpo_config_initialization(self):
        """Test DPOConfig initialization."""
        config = DPOConfig()
        
        # Check DPO-specific defaults
        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"
        assert config.max_length == 1024
        assert config.max_prompt_length == 512
        assert config.max_target_length == 512
    
    def test_dpo_config_custom_values(self):
        """Test DPOConfig with custom values."""
        config = DPOConfig(
            beta=0.2,
            loss_type="hinge",
            max_length=2048
        )
        
        assert config.beta == 0.2
        assert config.loss_type == "hinge"
        assert config.max_length == 2048


class TestPreferenceConfigs:
    """Test KTO, IPO, and CPO config classes."""
    
    def test_kto_config_initialization(self):
        """Test KTOConfig initialization."""
        config = KTOConfig()
        
        assert config.beta == 0.1
        assert config.desirable_weight == 1.0
        assert config.undesirable_weight == 1.0
    
    def test_ipo_config_initialization(self):
        """Test IPOConfig initialization."""
        config = IPOConfig()
        
        assert config.beta == 0.1
        assert config.loss_type == "ipo"
    
    def test_cpo_config_initialization(self):
        """Test CPOConfig initialization."""
        config = CPOConfig()
        
        assert config.beta == 0.1
        assert config.loss_type == "simpo"
        assert config.cpo_alpha == 1.0


class TestConfigFactory:
    """Test configuration factory functions."""
    
    def test_get_config_sft(self):
        """Test get_config for SFT."""
        config = get_config("sft")
        assert isinstance(config, SFTConfig)
    
    def test_get_config_dpo(self):
        """Test get_config for DPO."""
        config = get_config("dpo")
        assert isinstance(config, DPOConfig)
    
    def test_get_config_kto(self):
        """Test get_config for KTO."""
        config = get_config("kto")
        assert isinstance(config, KTOConfig)
    
    def test_get_config_ipo(self):
        """Test get_config for IPO."""
        config = get_config("ipo")
        assert isinstance(config, IPOConfig)
    
    def test_get_config_cpo(self):
        """Test get_config for CPO."""
        config = get_config("cpo")
        assert isinstance(config, CPOConfig)
    
    def test_get_config_invalid_method(self):
        """Test get_config with invalid method."""
        with pytest.raises(ValueError, match="Unsupported training method"):
            get_config("invalid_method")
    
    def test_get_config_with_kwargs(self):
        """Test get_config with additional kwargs."""
        config = get_config("sft", model_name="custom/model", seed=123)
        
        assert isinstance(config, SFTConfig)
        assert config.model_name == "custom/model"
        assert config.seed == 123
    
    def test_list_supported_methods(self):
        """Test list_supported_methods function."""
        methods = list_supported_methods()
        
        assert isinstance(methods, list)
        assert "sft" in methods
        assert "dpo" in methods
        assert "kto" in methods
        assert "ipo" in methods
        assert "cpo" in methods
        assert len(methods) == 5


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_sft_config_validation_valid(self):
        """Test SFT config validation with valid config."""
        config = SFTConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=2048,
            dataset_text_field="text"
        )
        
        assert config.validate() == True
    
    def test_sft_config_validation_invalid(self):
        """Test SFT config validation with invalid config."""
        # Invalid max_seq_length
        config = SFTConfig(max_seq_length=0)
        assert config.validate() == False
        
        # Empty dataset_text_field
        config = SFTConfig(dataset_text_field="")
        assert config.validate() == False
    
    def test_dpo_config_validation_valid(self):
        """Test DPO config validation with valid config."""
        config = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            max_length=1024
        )
        
        assert config.validate() == True
    
    def test_dpo_config_validation_invalid(self):
        """Test DPO config validation with invalid config."""
        # Invalid beta
        config = DPOConfig(beta=-0.1)
        assert config.validate() == False
        
        # Invalid loss_type
        config = DPOConfig(loss_type="invalid")
        assert config.validate() == False


class TestConfigEnvironmentVariables:
    """Test configuration with environment variables."""
    
    @patch.dict(os.environ, {
        "MODEL_NAME": "env/test-model",
        "OUTPUT_DIR": "./env_output",
        "SEED": "42",
        "MAX_SEQ_LENGTH": "1024",
        "PACKING": "false",
        "BETA": "0.2"
    })
    def test_sft_config_from_env(self):
        """Test SFT config loading from environment."""
        config = SFTConfig.load_from_env()
        
        assert config.model_name == "env/test-model"
        assert config.output_dir == "./env_output"
        assert config.seed == 42
        assert config.max_seq_length == 1024
        assert config.packing == False
    
    @patch.dict(os.environ, {
        "MODEL_NAME": "env/test-model",
        "BETA": "0.3",
        "LOSS_TYPE": "hinge",
        "MAX_LENGTH": "2048"
    })
    def test_dpo_config_from_env(self):
        """Test DPO config loading from environment."""
        config = DPOConfig.load_from_env()
        
        assert config.model_name == "env/test-model"
        assert config.beta == 0.3
        assert config.loss_type == "hinge"
        assert config.max_length == 2048


if __name__ == "__main__":
    pytest.main([__file__])