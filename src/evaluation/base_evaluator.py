"""Base Evaluator Module

This module contains the base evaluator class that provides common
evaluation functionalities for all training methods.
"""

import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for model evaluation.
    
    This class provides common evaluation functionalities that can be
    extended by specific evaluation implementations.
    """
    
    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        device: Optional[str] = None
    ):
        """Initialize the base evaluator.
        
        Args:
            model: Model instance or path to model
            tokenizer: Tokenizer instance or path to tokenizer
            device: Device to run evaluation on
        """
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is not None:
            self.load_model(model)
        
        if tokenizer is not None:
            self.load_tokenizer(tokenizer)
        
        logger.info(f"BaseEvaluator initialized on device: {self.device}")
    
    def load_model(self, model: Union[str, PreTrainedModel]):
        """Load the model for evaluation.
        
        Args:
            model: Model instance or path to model
        """
        if isinstance(model, str):
            logger.info(f"Loading model from: {model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = model
        
        self.model.eval()
        logger.info("Model loaded and set to evaluation mode")
    
    def load_tokenizer(self, tokenizer: Union[str, PreTrainedTokenizer]):
        """Load the tokenizer for evaluation.
        
        Args:
            tokenizer: Tokenizer instance or path to tokenizer
        """
        if isinstance(tokenizer, str):
            logger.info(f"Loading tokenizer from: {tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
    
    @abstractmethod
    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """Evaluate the model on the given dataset.
        
        Args:
            dataset: Dataset to evaluate on
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def compute_perplexity(self, dataset: Dataset, batch_size: int = 8) -> float:
        """Compute perplexity on the given dataset.
        
        Args:
            dataset: Dataset to compute perplexity on
            batch_size: Batch size for evaluation
            
        Returns:
            Perplexity score
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")
        
        logger.info("Computing perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                
                # Handle both single examples and batches
                if isinstance(batch, dict):
                    texts = [batch.get('text', batch.get('input', ''))]
                else:
                    texts = [item.get('text', item.get('input', '')) for item in batch]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Compute loss
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Accumulate loss and token count
                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length // 2
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                generated = self.generate_text(prompt, **kwargs)
                batch_results.append(generated)
            
            results.extend(batch_results)
            
            if i + batch_size < len(prompts):
                logger.info(f"Generated {i + batch_size}/{len(prompts)} responses")
        
        logger.info(f"Batch generation completed: {len(results)} responses")
        return results
    
    def compute_basic_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute basic text generation metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary of basic metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        metrics = {}
        
        # Average length metrics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        metrics["avg_prediction_length"] = np.mean(pred_lengths)
        metrics["avg_reference_length"] = np.mean(ref_lengths)
        metrics["length_ratio"] = metrics["avg_prediction_length"] / max(metrics["avg_reference_length"], 1)
        
        # Character-level metrics
        pred_char_lengths = [len(pred) for pred in predictions]
        ref_char_lengths = [len(ref) for ref in references]
        
        metrics["avg_prediction_char_length"] = np.mean(pred_char_lengths)
        metrics["avg_reference_char_length"] = np.mean(ref_char_lengths)
        
        # Diversity metrics
        unique_predictions = len(set(predictions))
        metrics["prediction_diversity"] = unique_predictions / len(predictions)
        
        logger.info(f"Basic metrics computed: {metrics}")
        return metrics
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_metadata: bool = True
    ):
        """Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
            include_metadata: Whether to include metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if include_metadata:
            results["metadata"] = {
                "device": self.device,
                "model_name": getattr(self.model, 'name_or_path', 'unknown'),
                "tokenizer_name": getattr(self.tokenizer, 'name_or_path', 'unknown'),
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {output_path}")
    
    def load_evaluation_results(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """Load evaluation results from a JSON file.
        
        Args:
            input_path: Path to load results from
            
        Returns:
            Evaluation results dictionary
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Evaluation results file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Evaluation results loaded from: {input_path}")
        return results
    
    def compare_models(
        self,
        other_evaluator: 'BaseEvaluator',
        dataset: Dataset,
        metrics_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare this model with another model.
        
        Args:
            other_evaluator: Another evaluator instance
            dataset: Dataset to evaluate both models on
            metrics_to_compare: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        logger.info("Starting model comparison...")
        
        # Evaluate both models
        results_1 = self.evaluate(dataset)
        results_2 = other_evaluator.evaluate(dataset)
        
        comparison = {
            "model_1_results": results_1,
            "model_2_results": results_2,
            "comparison": {}
        }
        
        # Compare specified metrics or all common metrics
        if metrics_to_compare is None:
            metrics_to_compare = set(results_1.keys()) & set(results_2.keys())
        
        for metric in metrics_to_compare:
            if metric in results_1 and metric in results_2:
                val_1 = results_1[metric]
                val_2 = results_2[metric]
                
                if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                    comparison["comparison"][metric] = {
                        "model_1": val_1,
                        "model_2": val_2,
                        "difference": val_1 - val_2,
                        "relative_improvement": ((val_1 - val_2) / max(abs(val_2), 1e-8)) * 100
                    }
        
        logger.info("Model comparison completed")
        return comparison
    
    def __repr__(self) -> str:
        """String representation of the evaluator."""
        model_name = getattr(self.model, 'name_or_path', 'None') if self.model else 'None'
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', 'None') if self.tokenizer else 'None'
        
        return (
            f"{self.__class__.__name__}("
            f"model={model_name}, "
            f"tokenizer={tokenizer_name}, "
            f"device={self.device})"
        )