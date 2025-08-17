"""Tests for Data Loader Modules

This module contains unit tests for the data loader classes.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

from src.data_loader import (
    BaseDataLoader,
    SFTDataLoader,
    DPODataLoader,
    PreferenceDataLoader,
    get_data_loader,
    list_supported_methods
)
from src.config import SFTConfig, DPOConfig, KTOConfig


class TestBaseDataLoader:
    """Test BaseDataLoader class."""
    
    def test_base_data_loader_initialization(self):
        """Test BaseDataLoader initialization."""
        config = SFTConfig()
        loader = BaseDataLoader(config)
        
        assert loader.config == config
        assert loader.tokenizer is None
        assert loader.dataset is None
    
    def test_base_data_loader_abstract_methods(self):
        """Test that BaseDataLoader abstract methods raise NotImplementedError."""
        config = SFTConfig()
        loader = BaseDataLoader(config)
        
        with pytest.raises(NotImplementedError):
            loader.load_dataset("dummy_path")
        
        with pytest.raises(NotImplementedError):
            loader.prepare_dataset()
        
        with pytest.raises(NotImplementedError):
            loader.get_data_collator()
    
    @patch('src.data_loader.base_loader.AutoTokenizer')
    def test_setup_tokenizer(self, mock_tokenizer_class):
        """Test setup_tokenizer method."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        config = SFTConfig(model_name="test/model")
        loader = BaseDataLoader(config)
        
        loader.setup_tokenizer()
        
        # Check tokenizer setup
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "test/model",
            trust_remote_code=True
        )
        assert loader.tokenizer == mock_tokenizer
        assert mock_tokenizer.pad_token == "</s>"


class TestSFTDataLoader:
    """Test SFTDataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = SFTConfig(
            dataset_text_field="text",
            max_seq_length=512,
            packing=True
        )
        self.loader = SFTDataLoader(self.config)
    
    def test_sft_data_loader_initialization(self):
        """Test SFTDataLoader initialization."""
        assert isinstance(self.loader, BaseDataLoader)
        assert self.loader.config == self.config
    
    def create_test_dataset_file(self, temp_dir, data):
        """Helper to create test dataset file."""
        import json
        
        dataset_path = Path(temp_dir) / "test_dataset.jsonl"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return str(dataset_path)
    
    @patch('src.data_loader.sft_loader.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        """Test load_dataset method."""
        # Mock dataset
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        result = self.loader.load_dataset("test_path")
        
        mock_load_dataset.assert_called_once_with(
            "json",
            data_files="test_path",
            split="train"
        )
        assert result == mock_dataset
        assert self.loader.dataset == mock_dataset
    
    @patch('src.data_loader.sft_loader.AutoTokenizer')
    def test_prepare_dataset(self, mock_tokenizer_class):
        """Test prepare_dataset method."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.filter.return_value = mock_dataset
        
        self.loader.tokenizer = mock_tokenizer
        self.loader.dataset = mock_dataset
        
        result = self.loader.prepare_dataset()
        
        # Check that dataset methods were called
        assert mock_dataset.map.called
        assert mock_dataset.filter.called
        assert result == mock_dataset
    
    def test_tokenize_function(self):
        """Test _tokenize_function method."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1]
        }
        
        self.loader.tokenizer = mock_tokenizer
        
        example = {"text": "Test text"}
        result = self.loader._tokenize_function(example)
        
        mock_tokenizer.assert_called_once_with(
            "Test text",
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None
        )
        assert "input_ids" in result
        assert "attention_mask" in result
    
    @patch('src.data_loader.sft_loader.DataCollatorForLanguageModeling')
    def test_get_data_collator(self, mock_collator_class):
        """Test get_data_collator method."""
        mock_collator = Mock()
        mock_collator_class.return_value = mock_collator
        
        mock_tokenizer = Mock()
        self.loader.tokenizer = mock_tokenizer
        
        result = self.loader.get_data_collator()
        
        mock_collator_class.assert_called_once_with(
            tokenizer=mock_tokenizer,
            mlm=False
        )
        assert result == mock_collator


class TestDPODataLoader:
    """Test DPODataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = DPOConfig(
            max_length=1024,
            max_prompt_length=512,
            max_target_length=512
        )
        self.loader = DPODataLoader(self.config)
    
    def test_dpo_data_loader_initialization(self):
        """Test DPODataLoader initialization."""
        assert isinstance(self.loader, BaseDataLoader)
        assert self.loader.config == self.config
    
    @patch('src.data_loader.dpo_loader.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        """Test load_dataset method."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        result = self.loader.load_dataset("test_path")
        
        mock_load_dataset.assert_called_once_with(
            "json",
            data_files="test_path",
            split="train"
        )
        assert result == mock_dataset
    
    def test_format_dataset_function(self):
        """Test _format_dataset_function method."""
        example = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "AI is a robot."
        }
        
        result = self.loader._format_dataset_function(example)
        
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert result["prompt"] == "What is AI?"
        assert result["chosen"] == "AI is artificial intelligence."
        assert result["rejected"] == "AI is a robot."
    
    @patch('src.data_loader.dpo_loader.AutoTokenizer')
    def test_prepare_dataset(self, mock_tokenizer_class):
        """Test prepare_dataset method."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        
        self.loader.tokenizer = mock_tokenizer
        self.loader.dataset = mock_dataset
        
        result = self.loader.prepare_dataset()
        
        assert mock_dataset.map.called
        assert result == mock_dataset
    
    def test_get_data_collator(self):
        """Test get_data_collator method."""
        mock_tokenizer = Mock()
        self.loader.tokenizer = mock_tokenizer
        
        result = self.loader.get_data_collator()
        
        # DPO doesn't use a specific data collator
        assert result is None


class TestPreferenceDataLoader:
    """Test PreferenceDataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = KTOConfig(
            max_length=1024,
            max_prompt_length=512
        )
        self.loader = PreferenceDataLoader(self.config)
    
    def test_preference_data_loader_initialization(self):
        """Test PreferenceDataLoader initialization."""
        assert isinstance(self.loader, BaseDataLoader)
        assert self.loader.config == self.config
    
    def test_determine_method(self):
        """Test _determine_method method."""
        # Test with KTO config
        kto_config = KTOConfig()
        kto_loader = PreferenceDataLoader(kto_config)
        assert kto_loader._determine_method() == "kto"
        
        # Test with DPO config
        dpo_config = DPOConfig()
        dpo_loader = PreferenceDataLoader(dpo_config)
        assert dpo_loader._determine_method() == "dpo"
    
    def test_format_kto_function(self):
        """Test _format_kto_function method."""
        example = {
            "prompt": "What is ML?",
            "completion": "Machine Learning",
            "label": True
        }
        
        result = self.loader._format_kto_function(example)
        
        assert "prompt" in result
        assert "completion" in result
        assert "label" in result
        assert result["label"] == True
    
    def test_format_pairwise_function(self):
        """Test _format_pairwise_function method."""
        example = {
            "prompt": "What is DL?",
            "chosen": "Deep Learning",
            "rejected": "Data Lake"
        }
        
        result = self.loader._format_pairwise_function(example)
        
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result


class TestDataLoaderFactory:
    """Test data loader factory functions."""
    
    def test_get_data_loader_sft(self):
        """Test get_data_loader for SFT."""
        config = SFTConfig()
        loader = get_data_loader("sft", config)
        
        assert isinstance(loader, SFTDataLoader)
        assert loader.config == config
    
    def test_get_data_loader_dpo(self):
        """Test get_data_loader for DPO."""
        config = DPOConfig()
        loader = get_data_loader("dpo", config)
        
        assert isinstance(loader, DPODataLoader)
        assert loader.config == config
    
    def test_get_data_loader_preference_methods(self):
        """Test get_data_loader for preference methods."""
        methods = ["kto", "ipo", "cpo"]
        
        for method in methods:
            config = KTOConfig()  # Use KTO config as base
            loader = get_data_loader(method, config)
            
            assert isinstance(loader, PreferenceDataLoader)
            assert loader.config == config
    
    def test_get_data_loader_invalid_method(self):
        """Test get_data_loader with invalid method."""
        config = SFTConfig()
        
        with pytest.raises(ValueError, match="Unsupported data loading method"):
            get_data_loader("invalid_method", config)
    
    def test_list_supported_methods(self):
        """Test list_supported_methods function."""
        methods = list_supported_methods()
        
        assert isinstance(methods, list)
        expected_methods = ["sft", "dpo", "kto", "ipo", "cpo"]
        
        for method in expected_methods:
            assert method in methods
        
        assert len(methods) == len(expected_methods)


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""
    
    def create_test_sft_data(self):
        """Create test SFT data."""
        return [
            {"text": "This is a test sentence for SFT training."},
            {"text": "Another example text for supervised fine-tuning."},
            {"text": "Arabic text example: هذا نص تجريبي باللغة العربية."}
        ]
    
    def create_test_dpo_data(self):
        """Create test DPO data."""
        return [
            {
                "prompt": "What is the capital of France?",
                "chosen": "The capital of France is Paris.",
                "rejected": "The capital of France is London."
            },
            {
                "prompt": "Explain machine learning.",
                "chosen": "Machine learning is a subset of AI that enables computers to learn.",
                "rejected": "Machine learning is just programming."
            }
        ]
    
    def create_test_kto_data(self):
        """Create test KTO data."""
        return [
            {
                "prompt": "What is Python?",
                "completion": "Python is a programming language.",
                "label": True
            },
            {
                "prompt": "What is Java?",
                "completion": "Java is a coffee.",
                "label": False
            }
        ]
    
    @patch('src.data_loader.sft_loader.load_dataset')
    @patch('src.data_loader.sft_loader.AutoTokenizer')
    def test_sft_data_loader_full_pipeline(self, mock_tokenizer_class, mock_load_dataset):
        """Test full SFT data loader pipeline."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.return_value = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1]
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.filter.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        # Test pipeline
        config = SFTConfig()
        loader = SFTDataLoader(config)
        
        # Load and prepare dataset
        loader.load_dataset("test_path")
        loader.setup_tokenizer()
        prepared_dataset = loader.prepare_dataset()
        data_collator = loader.get_data_collator()
        
        # Verify results
        assert prepared_dataset == mock_dataset
        assert data_collator is not None
        assert loader.tokenizer == mock_tokenizer
    
    @patch('src.data_loader.dpo_loader.load_dataset')
    @patch('src.data_loader.dpo_loader.AutoTokenizer')
    def test_dpo_data_loader_full_pipeline(self, mock_tokenizer_class, mock_load_dataset):
        """Test full DPO data loader pipeline."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        # Test pipeline
        config = DPOConfig()
        loader = DPODataLoader(config)
        
        # Load and prepare dataset
        loader.load_dataset("test_path")
        loader.setup_tokenizer()
        prepared_dataset = loader.prepare_dataset()
        data_collator = loader.get_data_collator()
        
        # Verify results
        assert prepared_dataset == mock_dataset
        assert data_collator is None  # DPO doesn't use data collator
        assert loader.tokenizer == mock_tokenizer


if __name__ == "__main__":
    pytest.main([__file__])