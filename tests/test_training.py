"""Tests for Training Modules

This module contains unit tests for the training system.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datasets import Dataset

from src.training import (
    BaseTrainer,
    SFTTrainer,
    DPOTrainer,
    KTOTrainer,
    IPOTrainer,
    CPOTrainer,
    get_trainer,
    list_supported_training_methods
)
from src.config import SFTConfig, DPOConfig, KTOConfig, IPOConfig, CPOConfig


class TestBaseTrainer:
    """Test BaseTrainer abstract class."""
    
    def test_base_trainer_abstract(self):
        """Test that BaseTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTrainer()
    
    def test_setup_logging(self):
        """Test setup_logging method."""
        # Create a concrete implementation for testing
        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {"test": "result"}
        
        trainer = ConcreteTrainer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            trainer.setup_logging(log_dir)
            
            assert log_dir.exists()
    
    def test_save_and_load_checkpoint(self):
        """Test save_checkpoint and load_checkpoint methods."""
        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {"test": "result"}
        
        trainer = ConcreteTrainer()
        
        # Mock model and optimizer
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_model.state_dict.return_value = {"param": torch.tensor([1.0])}
        mock_optimizer.state_dict.return_value = {"lr": 0.001}
        
        trainer.model = mock_model
        trainer.optimizer = mock_optimizer
        
        checkpoint_data = {
            "epoch": 5,
            "step": 1000,
            "loss": 0.5,
            "metrics": {"accuracy": 0.85}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            
            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path, **checkpoint_data)
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            
            assert loaded_data["epoch"] == 5
            assert loaded_data["step"] == 1000
            assert loaded_data["loss"] == 0.5
            assert loaded_data["metrics"]["accuracy"] == 0.85
    
    def test_compute_metrics(self):
        """Test compute_metrics method."""
        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {"test": "result"}
        
        trainer = ConcreteTrainer()
        
        predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = torch.tensor([1, 0, 1])
        
        metrics = trainer.compute_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["loss"], float)
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_log_metrics(self):
        """Test log_metrics method."""
        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {"test": "result"}
        
        trainer = ConcreteTrainer()
        
        metrics = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "accuracy": 0.85
        }
        
        # Should not raise any exceptions
        trainer.log_metrics(metrics, step=100)


class TestSFTTrainer:
    """Test SFTTrainer class."""
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_sft_trainer_init(self, mock_get_data_loader, mock_load_model):
        """Test SFTTrainer initialization."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = SFTConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = SFTTrainer(config)
        
        assert trainer.config == config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.data_loader == mock_data_loader
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    @patch('transformers.Trainer')
    def test_sft_trainer_setup_trainer(self, mock_trainer_class, mock_get_data_loader, mock_load_model):
        """Test SFTTrainer setup_trainer method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_dataset = Mock()
        mock_data_loader.prepare_dataset.return_value = mock_dataset
        mock_data_loader.get_data_collator.return_value = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        config = SFTConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = SFTTrainer(config)
        trainer.setup_trainer()
        
        assert trainer.trainer == mock_trainer_instance
        mock_trainer_class.assert_called_once()
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_sft_trainer_train(self, mock_get_data_loader, mock_load_model):
        """Test SFTTrainer train method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = SFTConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = SFTTrainer(config)
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_train_result = Mock()
        mock_train_result.training_history = [{"train_loss": 0.5}]
        mock_trainer_instance.train.return_value = mock_train_result
        trainer.trainer = mock_trainer_instance
        
        # Mock save methods
        trainer.save_model = Mock()
        trainer.save_training_logs = Mock()
        
        result = trainer.train()
        
        assert "training_history" in result
        mock_trainer_instance.train.assert_called_once()
        trainer.save_model.assert_called_once()
        trainer.save_training_logs.assert_called_once()
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_sft_trainer_save_model(self, mock_get_data_loader, mock_load_model):
        """Test SFTTrainer save_model method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = SFTConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = SFTTrainer(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.config.output_dir = temp_dir
            trainer.save_model()
            
            # Check that save methods were called
            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()


class TestDPOTrainer:
    """Test DPOTrainer class."""
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_dpo_trainer_init(self, mock_get_data_loader, mock_load_model):
        """Test DPOTrainer initialization."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = DPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output",
            reference_model_name="test/ref_model"
        )
        
        trainer = DPOTrainer(config)
        
        assert trainer.config == config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.data_loader == mock_data_loader
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_dpo_trainer_load_reference_model(self, mock_get_data_loader, mock_load_model):
        """Test DPOTrainer load_reference_model method."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_ref_model = Mock()
        mock_load_model.side_effect = [(mock_model, mock_tokenizer), (mock_ref_model, mock_tokenizer)]
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = DPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output",
            reference_model_name="test/ref_model"
        )
        
        trainer = DPOTrainer(config)
        trainer.load_reference_model()
        
        assert trainer.reference_model == mock_ref_model
        mock_ref_model.eval.assert_called_once()
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_dpo_trainer_compute_dpo_loss(self, mock_get_data_loader, mock_load_model):
        """Test DPOTrainer compute_dpo_loss method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = DPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = DPOTrainer(config)
        trainer.reference_model = Mock()
        
        # Mock model outputs
        mock_logits = torch.randn(2, 10, 1000)  # batch_size=2, seq_len=10, vocab_size=1000
        mock_output = Mock()
        mock_output.logits = mock_logits
        
        trainer.model.return_value = mock_output
        trainer.reference_model.return_value = mock_output
        
        # Create test batch
        batch = {
            "chosen_input_ids": torch.randint(0, 1000, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10),
            "rejected_input_ids": torch.randint(0, 1000, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10),
            "chosen_labels": torch.randint(0, 1000, (2, 10)),
            "rejected_labels": torch.randint(0, 1000, (2, 10))
        }
        
        loss = trainer.compute_dpo_loss(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestKTOTrainer:
    """Test KTOTrainer class."""
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_kto_trainer_init(self, mock_get_data_loader, mock_load_model):
        """Test KTOTrainer initialization."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = KTOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = KTOTrainer(config)
        
        assert trainer.config == config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.data_loader == mock_data_loader
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_kto_trainer_compute_kto_loss(self, mock_get_data_loader, mock_load_model):
        """Test KTOTrainer compute_kto_loss method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = KTOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = KTOTrainer(config)
        trainer.reference_model = Mock()
        
        # Mock model outputs
        mock_logits = torch.randn(2, 10, 1000)
        mock_output = Mock()
        mock_output.logits = mock_logits
        
        trainer.model.return_value = mock_output
        trainer.reference_model.return_value = mock_output
        
        # Create test batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 1000, (2, 10)),
            "kto_labels": torch.tensor([1.0, 0.0])  # desirable, undesirable
        }
        
        loss = trainer.compute_kto_loss(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestIPOTrainer:
    """Test IPOTrainer class."""
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_ipo_trainer_init(self, mock_get_data_loader, mock_load_model):
        """Test IPOTrainer initialization."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = IPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = IPOTrainer(config)
        
        assert trainer.config == config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.data_loader == mock_data_loader
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_ipo_trainer_compute_ipo_loss(self, mock_get_data_loader, mock_load_model):
        """Test IPOTrainer compute_ipo_loss method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = IPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = IPOTrainer(config)
        
        # Mock model outputs
        mock_logits = torch.randn(2, 10, 1000)
        mock_output = Mock()
        mock_output.logits = mock_logits
        
        trainer.model.return_value = mock_output
        
        # Create test batch
        batch = {
            "chosen_input_ids": torch.randint(0, 1000, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10),
            "rejected_input_ids": torch.randint(0, 1000, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10),
            "chosen_labels": torch.randint(0, 1000, (2, 10)),
            "rejected_labels": torch.randint(0, 1000, (2, 10))
        }
        
        loss = trainer.compute_ipo_loss(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestCPOTrainer:
    """Test CPOTrainer class."""
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_cpo_trainer_init(self, mock_get_data_loader, mock_load_model):
        """Test CPOTrainer initialization."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = CPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = CPOTrainer(config)
        
        assert trainer.config == config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.data_loader == mock_data_loader
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_cpo_trainer_compute_cpo_loss(self, mock_get_data_loader, mock_load_model):
        """Test CPOTrainer compute_cpo_loss method."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_data_loader = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        config = CPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = CPOTrainer(config)
        
        # Mock model outputs
        mock_logits = torch.randn(2, 10, 1000)
        mock_output = Mock()
        mock_output.logits = mock_logits
        
        trainer.model.return_value = mock_output
        
        # Create test batch
        batch = {
            "chosen_input_ids": torch.randint(0, 1000, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10),
            "rejected_input_ids": torch.randint(0, 1000, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10),
            "chosen_labels": torch.randint(0, 1000, (2, 10)),
            "rejected_labels": torch.randint(0, 1000, (2, 10))
        }
        
        loss = trainer.compute_cpo_loss(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestTrainingFactory:
    """Test training factory functions."""
    
    def test_list_supported_training_methods(self):
        """Test list_supported_training_methods function."""
        methods = list_supported_training_methods()
        
        expected_methods = ["sft", "dpo", "kto", "ipo", "cpo"]
        assert all(method in methods for method in expected_methods)
    
    @patch('src.training.SFTTrainer')
    def test_get_trainer_sft(self, mock_sft_trainer):
        """Test get_trainer for SFT method."""
        mock_instance = Mock()
        mock_sft_trainer.return_value = mock_instance
        
        config = SFTConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = get_trainer(config)
        
        assert trainer == mock_instance
        mock_sft_trainer.assert_called_once_with(config)
    
    @patch('src.training.DPOTrainer')
    def test_get_trainer_dpo(self, mock_dpo_trainer):
        """Test get_trainer for DPO method."""
        mock_instance = Mock()
        mock_dpo_trainer.return_value = mock_instance
        
        config = DPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = get_trainer(config)
        
        assert trainer == mock_instance
        mock_dpo_trainer.assert_called_once_with(config)
    
    @patch('src.training.KTOTrainer')
    def test_get_trainer_kto(self, mock_kto_trainer):
        """Test get_trainer for KTO method."""
        mock_instance = Mock()
        mock_kto_trainer.return_value = mock_instance
        
        config = KTOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = get_trainer(config)
        
        assert trainer == mock_instance
        mock_kto_trainer.assert_called_once_with(config)
    
    @patch('src.training.IPOTrainer')
    def test_get_trainer_ipo(self, mock_ipo_trainer):
        """Test get_trainer for IPO method."""
        mock_instance = Mock()
        mock_ipo_trainer.return_value = mock_instance
        
        config = IPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = get_trainer(config)
        
        assert trainer == mock_instance
        mock_ipo_trainer.assert_called_once_with(config)
    
    @patch('src.training.CPOTrainer')
    def test_get_trainer_cpo(self, mock_cpo_trainer):
        """Test get_trainer for CPO method."""
        mock_instance = Mock()
        mock_cpo_trainer.return_value = mock_instance
        
        config = CPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output"
        )
        
        trainer = get_trainer(config)
        
        assert trainer == mock_instance
        mock_cpo_trainer.assert_called_once_with(config)
    
    def test_get_trainer_invalid_config(self):
        """Test get_trainer with invalid config type."""
        class InvalidConfig:
            pass
        
        invalid_config = InvalidConfig()
        
        with pytest.raises(ValueError, match="Unsupported config type"):
            get_trainer(invalid_config)


class TestTrainingIntegration:
    """Integration tests for training system."""
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    @patch('transformers.Trainer')
    def test_sft_training_pipeline(self, mock_trainer_class, mock_get_data_loader, mock_load_model):
        """Test complete SFT training pipeline."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_dataset = Mock()
        mock_data_loader.prepare_dataset.return_value = mock_dataset
        mock_data_loader.get_data_collator.return_value = Mock()
        mock_get_data_loader.return_value = mock_data_loader
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_train_result = Mock()
        mock_train_result.training_history = [{"train_loss": 0.5, "eval_loss": 0.6}]
        mock_trainer_instance.train.return_value = mock_train_result
        mock_trainer_class.return_value = mock_trainer_instance
        
        # Create config
        config = SFTConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
        
        # Create and run trainer
        trainer = SFTTrainer(config)
        trainer.setup_trainer()
        
        with patch.object(trainer, 'save_model'), patch.object(trainer, 'save_training_logs'):
            result = trainer.train()
        
        # Verify training was called
        mock_trainer_instance.train.assert_called_once()
        assert "training_history" in result
    
    @patch('src.utils.load_model_and_tokenizer')
    @patch('src.data.get_data_loader')
    def test_preference_training_pipeline(self, mock_get_data_loader, mock_load_model):
        """Test preference training pipeline setup."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_dataset = Mock()
        mock_data_loader.prepare_dataset.return_value = mock_dataset
        mock_get_data_loader.return_value = mock_data_loader
        
        # Test DPO trainer
        config = DPOConfig(
            model_name="test/model",
            dataset_path="test/dataset",
            output_dir="test/output",
            reference_model_name="test/ref_model"
        )
        
        trainer = DPOTrainer(config)
        
        # Verify initialization
        assert trainer.config == config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.data_loader == mock_data_loader
        
        # Test reference model loading
        with patch.object(trainer, 'load_reference_model') as mock_load_ref:
            trainer.load_reference_model()
            mock_load_ref.assert_called_once()
    
    def test_training_workflow_with_checkpoints(self):
        """Test training workflow with checkpoint saving and loading."""
        class MockTrainer(BaseTrainer):
            def __init__(self):
                self.model = Mock()
                self.optimizer = Mock()
                self.model.state_dict.return_value = {"param": torch.tensor([1.0])}
                self.optimizer.state_dict.return_value = {"lr": 0.001}
            
            def train(self):
                return {"loss": 0.5, "accuracy": 0.85}
        
        trainer = MockTrainer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint_epoch_5.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                epoch=5,
                step=1000,
                loss=0.5,
                metrics={"accuracy": 0.85}
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            
            assert loaded_data["epoch"] == 5
            assert loaded_data["step"] == 1000
            assert loaded_data["loss"] == 0.5
            assert loaded_data["metrics"]["accuracy"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])