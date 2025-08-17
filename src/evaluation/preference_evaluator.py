"""Preference Evaluator Module

This module contains the evaluator implementation for preference optimization
methods including DPO, KTO, IPO, and CPO.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datasets import Dataset
import json
from pathlib import Path

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class PreferenceEvaluator(BaseEvaluator):
    """Evaluator for preference optimization methods.
    
    This class provides evaluation methods for models trained with
    preference optimization techniques like DPO, KTO, IPO, and CPO.
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        ref_model=None,
        method: str = "dpo"
    ):
        """Initialize the preference evaluator.
        
        Args:
            model: Trained model instance or path
            tokenizer: Tokenizer instance or path
            device: Device to run evaluation on
            ref_model: Reference model for comparison
            method: Preference optimization method ('dpo', 'kto', 'ipo', 'cpo')
        """
        super().__init__(model, tokenizer, device)
        self.ref_model = ref_model
        self.method = method.lower()
        
        if self.method not in ['dpo', 'kto', 'ipo', 'cpo']:
            raise ValueError(f"Unsupported method: {method}. Supported: dpo, kto, ipo, cpo")
        
        logger.info(f"PreferenceEvaluator initialized for {self.method.upper()}")
    
    def load_reference_model(self, ref_model: Union[str, torch.nn.Module]):
        """Load reference model for comparison.
        
        Args:
            ref_model: Reference model instance or path
        """
        if isinstance(ref_model, str):
            logger.info(f"Loading reference model from: {ref_model}")
            from transformers import AutoModelForCausalLM
            
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.ref_model = ref_model
        
        self.ref_model.eval()
        
        # Disable gradients for reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        logger.info("Reference model loaded successfully")
    
    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """Evaluate the preference optimization model.
        
        Args:
            dataset: Dataset containing preference data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing preference evaluation metrics
        """
        logger.info(f"Starting {self.method.upper()} evaluation on {len(dataset)} samples")
        
        # Extract evaluation parameters
        batch_size = kwargs.get('batch_size', 8)
        sample_size = kwargs.get('sample_size', None)
        compute_win_rate = kwargs.get('compute_win_rate', True)
        compute_preference_strength = kwargs.get('compute_preference_strength', True)
        
        results = {}
        
        # Sample dataset if specified
        eval_dataset = dataset
        if sample_size and len(dataset) > sample_size:
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            eval_dataset = dataset.select(indices)
            logger.info(f"Sampled {sample_size} examples for evaluation")
        
        # Method-specific evaluation
        if self.method == "kto":
            results.update(self._evaluate_kto(eval_dataset, batch_size))
        else:
            results.update(self._evaluate_pairwise(eval_dataset, batch_size))
        
        # Compute win rate against reference model
        if compute_win_rate and self.ref_model is not None:
            win_rate_results = self.compute_win_rate(eval_dataset, batch_size)
            results.update(win_rate_results)
        
        # Compute preference strength
        if compute_preference_strength:
            strength_results = self.compute_preference_strength(eval_dataset, batch_size)
            results.update(strength_results)
        
        logger.info(f"{self.method.upper()} evaluation completed")
        return results
    
    def _evaluate_kto(self, dataset: Dataset, batch_size: int) -> Dict[str, Any]:
        """Evaluate KTO-specific metrics.
        
        Args:
            dataset: KTO dataset with prompt, completion, label
            batch_size: Batch size for evaluation
            
        Returns:
            KTO evaluation metrics
        """
        logger.info("Evaluating KTO metrics...")
        
        results = {'kto_metrics': {}}
        
        # Extract data
        prompts = []
        completions = []
        labels = []
        
        for example in dataset:
            prompt = example.get('prompt', '')
            completion = example.get('completion', '')
            label = example.get('label', None)
            
            if prompt and completion and label is not None:
                prompts.append(prompt)
                completions.append(completion)
                labels.append(bool(label))  # Convert to boolean
        
        if not prompts:
            logger.warning("No valid KTO examples found")
            return results
        
        # Compute label distribution
        desirable_count = sum(labels)
        undesirable_count = len(labels) - desirable_count
        
        results['kto_metrics']['label_distribution'] = {
            'desirable': desirable_count,
            'undesirable': undesirable_count,
            'desirable_ratio': desirable_count / len(labels)
        }
        
        # Compute KTO-specific scores
        kto_scores = self._compute_kto_scores(prompts, completions, labels, batch_size)
        results['kto_metrics'].update(kto_scores)
        
        return results
    
    def _evaluate_pairwise(self, dataset: Dataset, batch_size: int) -> Dict[str, Any]:
        """Evaluate pairwise preference metrics.
        
        Args:
            dataset: Pairwise dataset with prompt, chosen, rejected
            batch_size: Batch size for evaluation
            
        Returns:
            Pairwise evaluation metrics
        """
        logger.info(f"Evaluating {self.method.upper()} pairwise metrics...")
        
        results = {f'{self.method}_metrics': {}}
        
        # Extract data
        prompts = []
        chosen_responses = []
        rejected_responses = []
        
        for example in dataset:
            prompt = example.get('prompt', '')
            chosen = example.get('chosen', '')
            rejected = example.get('rejected', '')
            
            if prompt and chosen and rejected:
                prompts.append(prompt)
                chosen_responses.append(chosen)
                rejected_responses.append(rejected)
        
        if not prompts:
            logger.warning("No valid pairwise examples found")
            return results
        
        # Compute preference accuracy
        accuracy = self._compute_preference_accuracy(
            prompts, chosen_responses, rejected_responses, batch_size
        )
        results[f'{self.method}_metrics']['preference_accuracy'] = accuracy
        
        # Compute preference margins
        margins = self._compute_preference_margins(
            prompts, chosen_responses, rejected_responses, batch_size
        )
        results[f'{self.method}_metrics']['preference_margins'] = margins
        
        return results
    
    def _compute_kto_scores(
        self,
        prompts: List[str],
        completions: List[str],
        labels: List[bool],
        batch_size: int
    ) -> Dict[str, float]:
        """Compute KTO-specific scores.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            labels: List of desirability labels
            batch_size: Batch size for processing
            
        Returns:
            KTO scores
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")
        
        scores = {'desirable_scores': [], 'undesirable_scores': []}
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_completions = completions[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                # Compute log probabilities for completions
                for prompt, completion, label in zip(batch_prompts, batch_completions, batch_labels):
                    full_text = prompt + completion
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model outputs
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Compute log probability of completion
                    prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
                    completion_logits = logits[0, prompt_length-1:-1, :]
                    completion_tokens = inputs['input_ids'][0, prompt_length:]
                    
                    log_probs = torch.log_softmax(completion_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, completion_tokens.unsqueeze(1)).squeeze(1)
                    avg_log_prob = token_log_probs.mean().item()
                    
                    if label:
                        scores['desirable_scores'].append(avg_log_prob)
                    else:
                        scores['undesirable_scores'].append(avg_log_prob)
        
        # Compute statistics
        result = {}
        
        if scores['desirable_scores']:
            result['avg_desirable_score'] = np.mean(scores['desirable_scores'])
            result['std_desirable_score'] = np.std(scores['desirable_scores'])
        
        if scores['undesirable_scores']:
            result['avg_undesirable_score'] = np.mean(scores['undesirable_scores'])
            result['std_undesirable_score'] = np.std(scores['undesirable_scores'])
        
        if scores['desirable_scores'] and scores['undesirable_scores']:
            result['score_separation'] = (
                result['avg_desirable_score'] - result['avg_undesirable_score']
            )
        
        return result
    
    def _compute_preference_accuracy(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        batch_size: int
    ) -> float:
        """Compute preference accuracy.
        
        Args:
            prompts: List of prompts
            chosen_responses: List of chosen responses
            rejected_responses: List of rejected responses
            batch_size: Batch size for processing
            
        Returns:
            Preference accuracy
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")
        
        correct_preferences = 0
        total_preferences = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_chosen = chosen_responses[i:i + batch_size]
                batch_rejected = rejected_responses[i:i + batch_size]
                
                for prompt, chosen, rejected in zip(batch_prompts, batch_chosen, batch_rejected):
                    # Compute scores for chosen and rejected responses
                    chosen_score = self._compute_response_score(prompt, chosen)
                    rejected_score = self._compute_response_score(prompt, rejected)
                    
                    # Check if model prefers chosen over rejected
                    if chosen_score > rejected_score:
                        correct_preferences += 1
                    
                    total_preferences += 1
        
        accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0.0
        logger.info(f"Preference accuracy: {accuracy:.4f} ({correct_preferences}/{total_preferences})")
        
        return accuracy
    
    def _compute_preference_margins(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        batch_size: int
    ) -> Dict[str, float]:
        """Compute preference margins.
        
        Args:
            prompts: List of prompts
            chosen_responses: List of chosen responses
            rejected_responses: List of rejected responses
            batch_size: Batch size for processing
            
        Returns:
            Preference margin statistics
        """
        margins = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_chosen = chosen_responses[i:i + batch_size]
                batch_rejected = rejected_responses[i:i + batch_size]
                
                for prompt, chosen, rejected in zip(batch_prompts, batch_chosen, batch_rejected):
                    chosen_score = self._compute_response_score(prompt, chosen)
                    rejected_score = self._compute_response_score(prompt, rejected)
                    
                    margin = chosen_score - rejected_score
                    margins.append(margin)
        
        return {
            'avg_margin': np.mean(margins),
            'std_margin': np.std(margins),
            'min_margin': np.min(margins),
            'max_margin': np.max(margins),
            'positive_margin_ratio': np.mean([m > 0 for m in margins])
        }
    
    def _compute_response_score(self, prompt: str, response: str) -> float:
        """Compute score for a response given a prompt.
        
        Args:
            prompt: Input prompt
            response: Response to score
            
        Returns:
            Response score (log probability)
        """
        full_text = prompt + response
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Compute log probability of response
        prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
        response_logits = logits[0, prompt_length-1:-1, :]
        response_tokens = inputs['input_ids'][0, prompt_length:]
        
        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        
        return token_log_probs.mean().item()
    
    def compute_win_rate(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Compute win rate against reference model.
        
        Args:
            dataset: Evaluation dataset
            batch_size: Batch size for generation
            num_samples: Number of samples to evaluate
            
        Returns:
            Win rate metrics
        """
        if self.ref_model is None:
            logger.warning("Reference model not available for win rate computation")
            return {'win_rate_metrics': {}}
        
        logger.info("Computing win rate against reference model...")
        
        # Sample dataset
        eval_size = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), eval_size, replace=False)
        eval_dataset = dataset.select(indices)
        
        wins = 0
        losses = 0
        ties = 0
        
        for example in eval_dataset:
            prompt = example.get('prompt', '')
            if not prompt:
                continue
            
            # Generate responses from both models
            trained_response = self.generate_text(prompt, max_length=256, temperature=0.7)
            
            # Generate from reference model
            ref_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            ref_inputs = {k: v.to(next(self.ref_model.parameters()).device) for k, v in ref_inputs.items()}
            
            with torch.no_grad():
                ref_outputs = self.ref_model.generate(
                    **ref_inputs,
                    max_length=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            ref_response = self.tokenizer.decode(
                ref_outputs[0][ref_inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Simple comparison based on response quality heuristics
            trained_score = self._evaluate_response_quality(trained_response)
            ref_score = self._evaluate_response_quality(ref_response)
            
            if trained_score > ref_score:
                wins += 1
            elif trained_score < ref_score:
                losses += 1
            else:
                ties += 1
        
        total = wins + losses + ties
        
        return {
            'win_rate_metrics': {
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'win_rate': wins / total if total > 0 else 0.0,
                'loss_rate': losses / total if total > 0 else 0.0,
                'tie_rate': ties / total if total > 0 else 0.0
            }
        }
    
    def _evaluate_response_quality(self, response: str) -> float:
        """Simple heuristic to evaluate response quality.
        
        Args:
            response: Response text
            
        Returns:
            Quality score
        """
        if not response.strip():
            return 0.0
        
        score = 0.0
        
        # Length score (prefer reasonable length)
        length = len(response.split())
        if 10 <= length <= 100:
            score += 1.0
        elif 5 <= length <= 200:
            score += 0.5
        
        # Diversity score (prefer diverse vocabulary)
        words = response.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio
        
        # Coherence score (simple heuristic)
        sentences = response.split('.')
        if len(sentences) > 1:
            score += 0.5
        
        return score
    
    def compute_preference_strength(
        self,
        dataset: Dataset,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """Compute preference strength metrics.
        
        Args:
            dataset: Evaluation dataset
            batch_size: Batch size for processing
            
        Returns:
            Preference strength metrics
        """
        logger.info("Computing preference strength...")
        
        if self.method == "kto":
            return self._compute_kto_preference_strength(dataset, batch_size)
        else:
            return self._compute_pairwise_preference_strength(dataset, batch_size)
    
    def _compute_kto_preference_strength(
        self,
        dataset: Dataset,
        batch_size: int
    ) -> Dict[str, Any]:
        """Compute KTO preference strength.
        
        Args:
            dataset: KTO dataset
            batch_size: Batch size
            
        Returns:
            KTO preference strength metrics
        """
        # This would involve computing the KTO loss and related metrics
        # For now, return placeholder
        return {
            'preference_strength_metrics': {
                'kto_strength': 0.0,
                'desirable_confidence': 0.0,
                'undesirable_confidence': 0.0
            }
        }
    
    def _compute_pairwise_preference_strength(
        self,
        dataset: Dataset,
        batch_size: int
    ) -> Dict[str, Any]:
        """Compute pairwise preference strength.
        
        Args:
            dataset: Pairwise dataset
            batch_size: Batch size
            
        Returns:
            Pairwise preference strength metrics
        """
        # This would involve computing preference margins and confidence
        # For now, return placeholder
        return {
            'preference_strength_metrics': {
                'avg_preference_strength': 0.0,
                'preference_confidence': 0.0,
                'margin_consistency': 0.0
            }
        }


def create_preference_evaluator(
    model=None,
    tokenizer=None,
    device=None,
    ref_model=None,
    method: str = "dpo",
    **kwargs
) -> PreferenceEvaluator:
    """Factory function to create a preference evaluator.
    
    Args:
        model: Trained model instance or path
        tokenizer: Tokenizer instance or path
        device: Device to use
        ref_model: Reference model
        method: Preference optimization method
        **kwargs: Additional parameters
        
    Returns:
        PreferenceEvaluator instance
    """
    return PreferenceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        ref_model=ref_model,
        method=method,
        **kwargs
    )