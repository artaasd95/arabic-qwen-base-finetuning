"""Tests for Evaluation Modules

This module contains unit tests for the evaluation system.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

from src.evaluation import (
    BaseEvaluator,
    SFTEvaluator,
    PreferenceEvaluator,
    get_evaluator,
    list_supported_evaluation_methods
)


class TestBaseEvaluator:
    """Test BaseEvaluator abstract class."""
    
    def test_base_evaluator_abstract(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model_and_tokenizer(self, mock_tokenizer, mock_model):
        """Test load_model_and_tokenizer method."""
        # Create a concrete implementation for testing
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, dataset, **kwargs):
                return {"test": "result"}
        
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        evaluator = ConcreteEvaluator()
        model, tokenizer = evaluator.load_model_and_tokenizer("test/model")
        
        assert model == mock_model_instance
        assert tokenizer == mock_tokenizer_instance
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    def test_compute_perplexity(self):
        """Test compute_perplexity method."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, dataset, **kwargs):
                return {"test": "result"}
        
        evaluator = ConcreteEvaluator()
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        # Mock model output
        mock_logits = torch.randn(1, 4, 1000)  # batch_size=1, seq_len=4, vocab_size=1000
        mock_model.return_value.logits = mock_logits
        
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        
        texts = ["test text"]
        perplexity = evaluator.compute_perplexity(texts)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_generate_text(self):
        """Test generate_text method."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, dataset, **kwargs):
                return {"test": "result"}
        
        evaluator = ConcreteEvaluator()
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock tokenizer encode
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated text"
        
        # Mock model generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        evaluator.device = "cpu"
        
        result = evaluator.generate_text("input prompt")
        
        assert result == "generated text"
        mock_model.generate.assert_called_once()
    
    def test_compute_basic_metrics(self):
        """Test compute_basic_metrics method."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, dataset, **kwargs):
                return {"test": "result"}
        
        evaluator = ConcreteEvaluator()
        
        texts = [
            "This is a test text.",
            "Another test text with different words.",
            "Short text."
        ]
        
        metrics = evaluator.compute_basic_metrics(texts)
        
        assert "avg_length" in metrics
        assert "total_length" in metrics
        assert "unique_words" in metrics
        assert "avg_unique_words" in metrics
        assert "repetition_ratio" in metrics
        
        assert metrics["total_length"] > 0
        assert metrics["avg_length"] > 0
        assert metrics["unique_words"] > 0
    
    def test_save_and_load_results(self):
        """Test save_results and load_results methods."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, dataset, **kwargs):
                return {"test": "result"}
        
        evaluator = ConcreteEvaluator()
        
        results = {
            "perplexity": 15.5,
            "accuracy": 0.85,
            "metrics": {"bleu": 0.4}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "results.json"
            
            # Save results
            evaluator.save_results(results, results_path)
            assert results_path.exists()
            
            # Load results
            loaded_results = evaluator.load_results(results_path)
            assert loaded_results == results


class TestSFTEvaluator:
    """Test SFTEvaluator class."""
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_sft_evaluator_init(self, mock_tokenizer, mock_model):
        """Test SFTEvaluator initialization."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        evaluator = SFTEvaluator(
            model_path="test/model",
            device="cpu",
            instruction_template="### Instruction: {instruction}\n### Response:",
            response_template="{response}"
        )
        
        assert evaluator.model == mock_model_instance
        assert evaluator.tokenizer == mock_tokenizer_instance
        assert evaluator.device == "cpu"
        assert "### Instruction:" in evaluator.instruction_template
    
    def test_extract_instruction_and_response(self):
        """Test _extract_instruction and _extract_response methods."""
        evaluator = SFTEvaluator.__new__(SFTEvaluator)
        
        # Test with dict format
        example_dict = {
            "instruction": "What is AI?",
            "response": "AI is artificial intelligence."
        }
        
        instruction = evaluator._extract_instruction(example_dict)
        response = evaluator._extract_response(example_dict)
        
        assert instruction == "What is AI?"
        assert response == "AI is artificial intelligence."
        
        # Test with text format
        example_text = {
            "text": "### Instruction: What is AI?\n### Response: AI is artificial intelligence."
        }
        
        evaluator.instruction_template = "### Instruction: {instruction}\n### Response:"
        instruction = evaluator._extract_instruction(example_text)
        
        assert "What is AI?" in instruction
    
    def test_classify_instruction_type(self):
        """Test _classify_instruction_type method."""
        evaluator = SFTEvaluator.__new__(SFTEvaluator)
        
        # Test different instruction types
        question = "What is the capital of France?"
        task = "Translate this text to Arabic."
        creative = "Write a story about a robot."
        
        assert evaluator._classify_instruction_type(question) == "question"
        assert evaluator._classify_instruction_type(task) == "task"
        assert evaluator._classify_instruction_type(creative) == "creative"
    
    def test_evaluate_format_compliance(self):
        """Test _evaluate_format_compliance method."""
        evaluator = SFTEvaluator.__new__(SFTEvaluator)
        
        # Test compliant response
        compliant_response = "This is a well-formatted response with proper punctuation."
        compliance = evaluator._evaluate_format_compliance(compliant_response)
        
        assert "has_punctuation" in compliance
        assert "proper_capitalization" in compliance
        assert "reasonable_length" in compliance
        
        # Test non-compliant response
        non_compliant = "bad response no punctuation"
        compliance = evaluator._evaluate_format_compliance(non_compliant)
        
        assert compliance["has_punctuation"] == False
    
    @patch.object(SFTEvaluator, 'generate_text')
    def test_generate_instruction_response(self, mock_generate):
        """Test generate_instruction_response method."""
        evaluator = SFTEvaluator.__new__(SFTEvaluator)
        evaluator.instruction_template = "### Instruction: {instruction}\n### Response:"
        
        mock_generate.return_value = "Generated response"
        
        instruction = "What is machine learning?"
        response = evaluator.generate_instruction_response(instruction)
        
        assert response == "Generated response"
        mock_generate.assert_called_once()
    
    def test_evaluate_single_instruction(self):
        """Test evaluate_single_instruction method."""
        evaluator = SFTEvaluator.__new__(SFTEvaluator)
        
        # Mock dependencies
        evaluator.generate_instruction_response = Mock(return_value="Generated response")
        evaluator._classify_instruction_type = Mock(return_value="question")
        evaluator._evaluate_format_compliance = Mock(return_value={
            "has_punctuation": True,
            "proper_capitalization": True,
            "reasonable_length": True
        })
        
        instruction = "What is AI?"
        reference = "AI is artificial intelligence."
        
        result = evaluator.evaluate_single_instruction(instruction, reference)
        
        assert "generated_response" in result
        assert "instruction_type" in result
        assert "format_compliance" in result
        assert "response_length" in result
    
    def test_create_sft_evaluator_factory(self):
        """Test create_sft_evaluator factory function."""
        with patch('src.evaluation.sft_evaluator.SFTEvaluator') as mock_evaluator:
            mock_instance = Mock()
            mock_evaluator.return_value = mock_instance
            
            from src.evaluation.sft_evaluator import create_sft_evaluator
            
            evaluator = create_sft_evaluator(
                model_path="test/model",
                device="cpu"
            )
            
            assert evaluator == mock_instance
            mock_evaluator.assert_called_once_with(
                model_path="test/model",
                device="cpu"
            )


class TestPreferenceEvaluator:
    """Test PreferenceEvaluator class."""
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_preference_evaluator_init(self, mock_tokenizer, mock_model):
        """Test PreferenceEvaluator initialization."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        evaluator = PreferenceEvaluator(
            model_path="test/model",
            device="cpu",
            method="dpo"
        )
        
        assert evaluator.model == mock_model_instance
        assert evaluator.tokenizer == mock_tokenizer_instance
        assert evaluator.device == "cpu"
        assert evaluator.method == "dpo"
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_reference_model(self, mock_model):
        """Test load_reference_model method."""
        evaluator = PreferenceEvaluator.__new__(PreferenceEvaluator)
        evaluator.device = "cpu"
        
        mock_ref_model = Mock()
        mock_model.return_value = mock_ref_model
        
        evaluator.load_reference_model("reference/model")
        
        assert evaluator.reference_model == mock_ref_model
        mock_ref_model.eval.assert_called_once()
    
    def test_compute_response_score(self):
        """Test _compute_response_score method."""
        evaluator = PreferenceEvaluator.__new__(PreferenceEvaluator)
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        # Mock model output with logits
        mock_logits = torch.randn(1, 4, 1000)
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        evaluator.device = "cpu"
        
        prompt = "What is AI?"
        response = "AI is artificial intelligence."
        
        score = evaluator._compute_response_score(prompt, response)
        
        assert isinstance(score, float)
    
    def test_compute_kto_scores(self):
        """Test _compute_kto_scores method."""
        evaluator = PreferenceEvaluator.__new__(PreferenceEvaluator)
        evaluator._compute_response_score = Mock(side_effect=[0.5, -0.3, 0.8])
        
        examples = [
            {"prompt": "Q1", "completion": "A1", "label": True},
            {"prompt": "Q2", "completion": "A2", "label": False},
            {"prompt": "Q3", "completion": "A3", "label": True}
        ]
        
        scores = evaluator._compute_kto_scores(examples)
        
        assert "desirable_scores" in scores
        assert "undesirable_scores" in scores
        assert "avg_desirable" in scores
        assert "avg_undesirable" in scores
        
        assert len(scores["desirable_scores"]) == 2  # Two True labels
        assert len(scores["undesirable_scores"]) == 1  # One False label
    
    def test_compute_preference_accuracy(self):
        """Test _compute_preference_accuracy method."""
        evaluator = PreferenceEvaluator.__new__(PreferenceEvaluator)
        evaluator._compute_response_score = Mock(side_effect=[0.8, 0.3, 0.9, 0.2])
        
        examples = [
            {"prompt": "Q1", "chosen": "Good answer", "rejected": "Bad answer"},
            {"prompt": "Q2", "chosen": "Better answer", "rejected": "Worse answer"}
        ]
        
        accuracy = evaluator._compute_preference_accuracy(examples)
        
        assert "accuracy" in accuracy
        assert "correct_preferences" in accuracy
        assert "total_preferences" in accuracy
        
        assert accuracy["accuracy"] == 1.0  # Both preferences correct
        assert accuracy["correct_preferences"] == 2
        assert accuracy["total_preferences"] == 2
    
    def test_evaluate_kto(self):
        """Test _evaluate_kto method."""
        evaluator = PreferenceEvaluator.__new__(PreferenceEvaluator)
        evaluator._compute_kto_scores = Mock(return_value={
            "avg_desirable": 0.7,
            "avg_undesirable": -0.2,
            "desirable_scores": [0.8, 0.6],
            "undesirable_scores": [-0.2]
        })
        
        dataset = Dataset.from_list([
            {"prompt": "Q1", "completion": "A1", "label": True},
            {"prompt": "Q2", "completion": "A2", "label": False}
        ])
        
        results = evaluator._evaluate_kto(dataset)
        
        assert "kto_scores" in results
        assert "label_distribution" in results
        assert "avg_desirable_score" in results
        assert "avg_undesirable_score" in results
    
    def test_evaluate_pairwise(self):
        """Test _evaluate_pairwise method."""
        evaluator = PreferenceEvaluator.__new__(PreferenceEvaluator)
        evaluator._compute_preference_accuracy = Mock(return_value={
            "accuracy": 0.85,
            "correct_preferences": 17,
            "total_preferences": 20
        })
        evaluator._compute_preference_margins = Mock(return_value={
            "avg_margin": 0.3,
            "margins": [0.5, 0.1, 0.4]
        })
        
        dataset = Dataset.from_list([
            {"prompt": "Q1", "chosen": "Good", "rejected": "Bad"},
            {"prompt": "Q2", "chosen": "Better", "rejected": "Worse"}
        ])
        
        results = evaluator._evaluate_pairwise(dataset)
        
        assert "preference_accuracy" in results
        assert "preference_margins" in results
        assert "accuracy" in results
        assert "avg_margin" in results
    
    def test_create_preference_evaluator_factory(self):
        """Test create_preference_evaluator factory function."""
        with patch('src.evaluation.preference_evaluator.PreferenceEvaluator') as mock_evaluator:
            mock_instance = Mock()
            mock_evaluator.return_value = mock_instance
            
            from src.evaluation.preference_evaluator import create_preference_evaluator
            
            evaluator = create_preference_evaluator(
                model_path="test/model",
                method="dpo",
                device="cpu"
            )
            
            assert evaluator == mock_instance
            mock_evaluator.assert_called_once_with(
                model_path="test/model",
                method="dpo",
                device="cpu"
            )


class TestEvaluationFactory:
    """Test evaluation factory functions."""
    
    def test_list_supported_evaluation_methods(self):
        """Test list_supported_evaluation_methods function."""
        methods = list_supported_evaluation_methods()
        
        expected_methods = ["sft", "dpo", "kto", "ipo", "cpo"]
        assert all(method in methods for method in expected_methods)
    
    @patch('src.evaluation.SFTEvaluator')
    def test_get_evaluator_sft(self, mock_sft_evaluator):
        """Test get_evaluator for SFT method."""
        mock_instance = Mock()
        mock_sft_evaluator.return_value = mock_instance
        
        evaluator = get_evaluator(
            method="sft",
            model_path="test/model",
            device="cpu"
        )
        
        assert evaluator == mock_instance
        mock_sft_evaluator.assert_called_once()
    
    @patch('src.evaluation.PreferenceEvaluator')
    def test_get_evaluator_preference(self, mock_preference_evaluator):
        """Test get_evaluator for preference methods."""
        mock_instance = Mock()
        mock_preference_evaluator.return_value = mock_instance
        
        for method in ["dpo", "kto", "ipo", "cpo"]:
            evaluator = get_evaluator(
                method=method,
                model_path="test/model",
                device="cpu"
            )
            
            assert evaluator == mock_instance
    
    def test_get_evaluator_invalid_method(self):
        """Test get_evaluator with invalid method."""
        with pytest.raises(ValueError, match="Unsupported evaluation method"):
            get_evaluator(
                method="invalid_method",
                model_path="test/model",
                device="cpu"
            )


class TestEvaluationIntegration:
    """Integration tests for evaluation system."""
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_sft_evaluation_pipeline(self, mock_tokenizer, mock_model):
        """Test complete SFT evaluation pipeline."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock tokenizer methods
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Generated response"
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model methods
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_logits = torch.randn(1, 3, 1000)
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model_instance.return_value = mock_output
        
        # Create evaluator
        evaluator = SFTEvaluator(
            model_path="test/model",
            device="cpu"
        )
        
        # Create test dataset
        test_data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."},
            {"instruction": "Explain machine learning.", "response": "ML is a subset of AI."}
        ]
        dataset = Dataset.from_list(test_data)
        
        # Run evaluation
        results = evaluator.evaluate(dataset, sample_size=2)
        
        # Check results structure
        assert "perplexity" in results
        assert "generation_metrics" in results
        assert "instruction_following" in results
        assert isinstance(results["perplexity"], float)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_preference_evaluation_pipeline(self, mock_tokenizer, mock_model):
        """Test complete preference evaluation pipeline."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock tokenizer methods
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model methods
        mock_logits = torch.randn(1, 3, 1000)
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model_instance.return_value = mock_output
        
        # Create evaluator
        evaluator = PreferenceEvaluator(
            model_path="test/model",
            device="cpu",
            method="dpo"
        )
        
        # Create test dataset
        test_data = [
            {"prompt": "What is AI?", "chosen": "Good answer", "rejected": "Bad answer"},
            {"prompt": "Explain ML.", "chosen": "Better answer", "rejected": "Worse answer"}
        ]
        dataset = Dataset.from_list(test_data)
        
        # Run evaluation
        results = evaluator.evaluate(dataset, sample_size=2)
        
        # Check results structure
        assert "preference_accuracy" in results or "kto_scores" in results
        assert "win_rate" in results
        assert "preference_strength" in results


if __name__ == "__main__":
    pytest.main([__file__])