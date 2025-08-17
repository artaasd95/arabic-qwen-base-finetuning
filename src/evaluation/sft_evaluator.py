"""SFT Evaluator Module

This module contains the evaluator implementation for Supervised Fine-Tuning (SFT)
models with instruction-following capabilities.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datasets import Dataset
import re

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class SFTEvaluator(BaseEvaluator):
    """Evaluator for Supervised Fine-Tuning models.
    
    This class provides evaluation methods specifically designed for
    instruction-following models trained with SFT.
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n",
        response_template: str = "{response}"
    ):
        """Initialize the SFT evaluator.
        
        Args:
            model: Model instance or path to model
            tokenizer: Tokenizer instance or path to tokenizer
            device: Device to run evaluation on
            instruction_template: Template for formatting instructions
            response_template: Template for formatting responses
        """
        super().__init__(model, tokenizer, device)
        self.instruction_template = instruction_template
        self.response_template = response_template
        
        logger.info("SFTEvaluator initialized")
    
    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """Evaluate the SFT model on the given dataset.
        
        Args:
            dataset: Dataset containing instruction-response pairs
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing SFT evaluation metrics
        """
        logger.info(f"Starting SFT evaluation on {len(dataset)} samples")
        
        # Extract evaluation parameters
        batch_size = kwargs.get('batch_size', 8)
        max_length = kwargs.get('max_length', 512)
        compute_perplexity = kwargs.get('compute_perplexity', True)
        compute_generation_metrics = kwargs.get('compute_generation_metrics', True)
        sample_size = kwargs.get('sample_size', None)
        
        results = {}
        
        # Sample dataset if specified
        eval_dataset = dataset
        if sample_size and len(dataset) > sample_size:
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            eval_dataset = dataset.select(indices)
            logger.info(f"Sampled {sample_size} examples for evaluation")
        
        # Compute perplexity
        if compute_perplexity:
            try:
                perplexity = self.compute_perplexity(eval_dataset, batch_size)
                results['perplexity'] = perplexity
            except Exception as e:
                logger.warning(f"Failed to compute perplexity: {e}")
                results['perplexity'] = None
        
        # Compute generation-based metrics
        if compute_generation_metrics:
            generation_results = self.evaluate_generation(
                eval_dataset,
                max_length=max_length,
                batch_size=batch_size
            )
            results.update(generation_results)
        
        # Compute instruction-following metrics
        instruction_results = self.evaluate_instruction_following(eval_dataset)
        results.update(instruction_results)
        
        logger.info("SFT evaluation completed")
        return results
    
    def evaluate_generation(
        self,
        dataset: Dataset,
        max_length: int = 512,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """Evaluate text generation quality.
        
        Args:
            dataset: Dataset with instruction-response pairs
            max_length: Maximum generation length
            batch_size: Batch size for generation
            
        Returns:
            Generation evaluation metrics
        """
        logger.info("Evaluating text generation quality...")
        
        # Extract instructions and reference responses
        instructions = []
        references = []
        
        for example in dataset:
            instruction = self._extract_instruction(example)
            response = self._extract_response(example)
            
            if instruction and response:
                instructions.append(instruction)
                references.append(response)
        
        if not instructions:
            logger.warning("No valid instruction-response pairs found")
            return {}
        
        # Generate responses
        logger.info(f"Generating responses for {len(instructions)} instructions")
        predictions = []
        
        for i in range(0, len(instructions), batch_size):
            batch_instructions = instructions[i:i + batch_size]
            
            for instruction in batch_instructions:
                formatted_prompt = self.instruction_template.format(instruction=instruction)
                generated = self.generate_text(
                    formatted_prompt,
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                predictions.append(generated)
            
            if i + batch_size < len(instructions):
                logger.info(f"Generated {i + batch_size}/{len(instructions)} responses")
        
        # Compute metrics
        metrics = self.compute_basic_metrics(predictions, references)
        
        # Add generation-specific metrics
        metrics.update(self._compute_generation_quality_metrics(predictions, references))
        
        return {
            'generation_metrics': metrics,
            'sample_predictions': predictions[:5],  # Store first 5 predictions
            'sample_references': references[:5]     # Store first 5 references
        }
    
    def evaluate_instruction_following(
        self,
        dataset: Dataset,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """Evaluate instruction-following capabilities.
        
        Args:
            dataset: Dataset with instruction-response pairs
            sample_size: Number of samples to evaluate
            
        Returns:
            Instruction-following metrics
        """
        logger.info("Evaluating instruction-following capabilities...")
        
        # Sample dataset for instruction evaluation
        eval_size = min(sample_size, len(dataset))
        indices = np.random.choice(len(dataset), eval_size, replace=False)
        eval_dataset = dataset.select(indices)
        
        metrics = {
            'instruction_types': {},
            'response_quality': {},
            'format_compliance': {}
        }
        
        instruction_types = []
        response_lengths = []
        format_scores = []
        
        for example in eval_dataset:
            instruction = self._extract_instruction(example)
            response = self._extract_response(example)
            
            if not instruction or not response:
                continue
            
            # Classify instruction type
            inst_type = self._classify_instruction_type(instruction)
            instruction_types.append(inst_type)
            
            # Analyze response
            response_lengths.append(len(response.split()))
            
            # Check format compliance
            format_score = self._evaluate_format_compliance(instruction, response)
            format_scores.append(format_score)
        
        # Aggregate metrics
        if instruction_types:
            type_counts = {}
            for inst_type in instruction_types:
                type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
            
            metrics['instruction_types'] = {
                'distribution': type_counts,
                'total_types': len(set(instruction_types))
            }
        
        if response_lengths:
            metrics['response_quality'] = {
                'avg_length': np.mean(response_lengths),
                'median_length': np.median(response_lengths),
                'std_length': np.std(response_lengths),
                'min_length': np.min(response_lengths),
                'max_length': np.max(response_lengths)
            }
        
        if format_scores:
            metrics['format_compliance'] = {
                'avg_score': np.mean(format_scores),
                'compliance_rate': np.mean([score > 0.5 for score in format_scores])
            }
        
        return {'instruction_following_metrics': metrics}
    
    def _extract_instruction(self, example: Dict) -> Optional[str]:
        """Extract instruction from dataset example.
        
        Args:
            example: Dataset example
            
        Returns:
            Extracted instruction or None
        """
        # Try common instruction field names
        for field in ['instruction', 'prompt', 'input', 'question', 'query']:
            if field in example and example[field]:
                return example[field].strip()
        
        # Try to extract from text field
        if 'text' in example:
            text = example['text']
            # Look for instruction patterns
            patterns = [
                r'### Instruction:\s*(.+?)\s*### Response:',
                r'Instruction:\s*(.+?)\s*Response:',
                r'Question:\s*(.+?)\s*Answer:',
                r'Input:\s*(.+?)\s*Output:'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def _extract_response(self, example: Dict) -> Optional[str]:
        """Extract response from dataset example.
        
        Args:
            example: Dataset example
            
        Returns:
            Extracted response or None
        """
        # Try common response field names
        for field in ['response', 'output', 'answer', 'completion', 'target']:
            if field in example and example[field]:
                return example[field].strip()
        
        # Try to extract from text field
        if 'text' in example:
            text = example['text']
            # Look for response patterns
            patterns = [
                r'### Response:\s*(.+?)(?:### |$)',
                r'Response:\s*(.+?)(?:Instruction:|$)',
                r'Answer:\s*(.+?)(?:Question:|$)',
                r'Output:\s*(.+?)(?:Input:|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def _classify_instruction_type(self, instruction: str) -> str:
        """Classify the type of instruction.
        
        Args:
            instruction: Instruction text
            
        Returns:
            Instruction type category
        """
        instruction_lower = instruction.lower()
        
        # Define instruction type patterns
        patterns = {
            'question_answering': [r'what', r'how', r'why', r'when', r'where', r'who', r'\?'],
            'text_generation': [r'write', r'generate', r'create', r'compose', r'draft'],
            'summarization': [r'summarize', r'summary', r'brief', r'outline'],
            'translation': [r'translate', r'translation', r'convert'],
            'explanation': [r'explain', r'describe', r'define', r'clarify'],
            'analysis': [r'analyze', r'compare', r'evaluate', r'assess'],
            'classification': [r'classify', r'categorize', r'identify', r'label'],
            'reasoning': [r'solve', r'calculate', r'reason', r'logic']
        }
        
        for inst_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                if re.search(pattern, instruction_lower):
                    return inst_type
        
        return 'other'
    
    def _evaluate_format_compliance(self, instruction: str, response: str) -> float:
        """Evaluate how well the response follows expected format.
        
        Args:
            instruction: Original instruction
            response: Generated response
            
        Returns:
            Format compliance score (0-1)
        """
        score = 0.0
        checks = 0
        
        # Check if response is not empty
        checks += 1
        if response.strip():
            score += 1
        
        # Check reasonable length (not too short or too long)
        checks += 1
        response_length = len(response.split())
        if 5 <= response_length <= 500:
            score += 1
        
        # Check if response doesn't repeat the instruction
        checks += 1
        if instruction.lower() not in response.lower():
            score += 1
        
        # Check if response doesn't contain template artifacts
        checks += 1
        artifacts = ['### instruction', '### response', '<|endoftext|>', '[INST]', '[/INST]']
        if not any(artifact in response.lower() for artifact in artifacts):
            score += 1
        
        return score / checks if checks > 0 else 0.0
    
    def _compute_generation_quality_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute generation quality metrics.
        
        Args:
            predictions: Generated texts
            references: Reference texts
            
        Returns:
            Generation quality metrics
        """
        metrics = {}
        
        # Compute repetition metrics
        repetition_scores = []
        for pred in predictions:
            words = pred.split()
            if len(words) > 1:
                unique_words = len(set(words))
                repetition_score = unique_words / len(words)
                repetition_scores.append(repetition_score)
        
        if repetition_scores:
            metrics['avg_repetition_score'] = np.mean(repetition_scores)
        
        # Compute coherence metrics (simple heuristics)
        coherence_scores = []
        for pred in predictions:
            sentences = pred.split('.')
            # Simple coherence: check if sentences are reasonably connected
            coherence_score = min(len(sentences) / max(len(pred.split()), 1), 1.0)
            coherence_scores.append(coherence_score)
        
        if coherence_scores:
            metrics['avg_coherence_score'] = np.mean(coherence_scores)
        
        # Compute relevance metrics (keyword overlap)
        relevance_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if ref_words:
                overlap = len(pred_words & ref_words)
                relevance_score = overlap / len(ref_words)
                relevance_scores.append(relevance_score)
        
        if relevance_scores:
            metrics['avg_relevance_score'] = np.mean(relevance_scores)
        
        return metrics
    
    def generate_instruction_response(
        self,
        instruction: str,
        max_length: int = 512,
        **kwargs
    ) -> str:
        """Generate a response for a given instruction.
        
        Args:
            instruction: Input instruction
            max_length: Maximum response length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        formatted_prompt = self.instruction_template.format(instruction=instruction)
        
        response = self.generate_text(
            formatted_prompt,
            max_length=max_length,
            **kwargs
        )
        
        return response
    
    def evaluate_single_instruction(
        self,
        instruction: str,
        reference_response: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate model performance on a single instruction.
        
        Args:
            instruction: Input instruction
            reference_response: Optional reference response
            **kwargs: Additional generation parameters
            
        Returns:
            Single instruction evaluation results
        """
        # Generate response
        generated_response = self.generate_instruction_response(instruction, **kwargs)
        
        results = {
            'instruction': instruction,
            'generated_response': generated_response,
            'reference_response': reference_response
        }
        
        # Compute metrics if reference is available
        if reference_response:
            basic_metrics = self.compute_basic_metrics([generated_response], [reference_response])
            results['metrics'] = basic_metrics
            
            # Add format compliance
            format_score = self._evaluate_format_compliance(instruction, generated_response)
            results['format_compliance'] = format_score
        
        return results


def create_sft_evaluator(
    model=None,
    tokenizer=None,
    device=None,
    **kwargs
) -> SFTEvaluator:
    """Factory function to create an SFT evaluator.
    
    Args:
        model: Model instance or path
        tokenizer: Tokenizer instance or path
        device: Device to use
        **kwargs: Additional evaluator parameters
        
    Returns:
        SFTEvaluator instance
    """
    return SFTEvaluator(model=model, tokenizer=tokenizer, device=device, **kwargs)