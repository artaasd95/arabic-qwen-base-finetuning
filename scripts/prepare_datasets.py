#!/usr/bin/env python3
"""
Arabic Dataset Preparation Script
Prepares various Arabic datasets for different training methods (SFT, DPO, KTO, IPO, CPO)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicDatasetPreparer:
    """Prepares Arabic datasets for fine-tuning"""
    
    def __init__(self, data_dir: str = "./data", cache_dir: str = "./cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Dataset configurations for different training methods
        self.dataset_configs = {
            "sft": {
                "datasets": [
                    "2A2I/Arabic-OpenHermes-2.5",  # 982k samples - excellent for SFT
                    "FreedomIntelligence/Alpaca-Arabic-GPT4",  # 50k samples - high quality
                    "arbml/alpaca_arabic",  # 52k samples - good baseline
                ],
                "format": "instruction_response"
            },
            "dpo": {
                "datasets": [
                    "2A2I/argilla-dpo-mix-7k-arabic",  # 7.5k samples - DPO specific
                    "FreedomIntelligence/Arabic-preference-data-RLHF",  # 11.5k samples
                ],
                "format": "preference_pairs"
            },
            "kto": {
                "datasets": [
                    "maghwa/10k_prompts_ranked_arabic",  # 10.3k samples with rankings
                    "sepidmnorozy/Arabic_sentiment",  # 3.53k samples with sentiment labels
                ],
                "format": "binary_feedback"
            },
            "evaluation": {
                "datasets": [
                    "OALL/AlGhafa-Arabic-LLM-Benchmark-Native",  # 23k samples
                    "MBZUAI/ArabicMMLU",  # 29.2k samples
                    "OALL/Arabic_MMLU",  # 14.3k samples
                ],
                "format": "multiple_choice"
            }
        }
    
    def download_dataset(self, dataset_name: str, split: Optional[str] = None) -> Dataset:
        """Download a dataset from HuggingFace"""
        try:
            logger.info(f"Downloading dataset: {dataset_name}")
            if split:
                dataset = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)
            else:
                dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            logger.info(f"Successfully downloaded {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return None
    
    def format_sft_data(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """Format dataset for SFT training"""
        logger.info(f"Formatting SFT data for {dataset_name}")
        
        def format_sample(example):
            # Handle different dataset formats
            if "2A2I/Arabic-OpenHermes-2.5" in dataset_name:
                return {
                    "instruction": example.get("instruction", ""),
                    "input": example.get("input", ""),
                    "output": example.get("output", "")
                }
            elif "Alpaca" in dataset_name:
                return {
                    "instruction": example.get("instruction", ""),
                    "input": example.get("input", ""),
                    "output": example.get("output", "")
                }
            else:
                # Generic format
                return {
                    "instruction": example.get("instruction", example.get("prompt", "")),
                    "input": example.get("input", ""),
                    "output": example.get("output", example.get("response", ""))
                }
        
        return dataset.map(format_sample)
    
    def format_dpo_data(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """Format dataset for DPO training"""
        logger.info(f"Formatting DPO data for {dataset_name}")
        
        def format_sample(example):
            if "argilla-dpo" in dataset_name:
                return {
                    "prompt": example.get("instruction", ""),
                    "chosen": example.get("chosen", ""),
                    "rejected": example.get("rejected", "")
                }
            else:
                return {
                    "prompt": example.get("prompt", example.get("instruction", "")),
                    "chosen": example.get("chosen", example.get("preferred", "")),
                    "rejected": example.get("rejected", example.get("dispreferred", ""))
                }
        
        return dataset.map(format_sample)
    
    def format_kto_data(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """Format dataset for KTO training"""
        logger.info(f"Formatting KTO data for {dataset_name}")
        
        def format_sample(example):
            if "sentiment" in dataset_name.lower():
                # Convert sentiment to binary feedback
                sentiment = example.get("label", example.get("sentiment", 0))
                is_positive = sentiment > 0 if isinstance(sentiment, (int, float)) else sentiment == "positive"
                return {
                    "prompt": example.get("text", example.get("input", "")),
                    "completion": example.get("text", example.get("input", "")),
                    "label": is_positive
                }
            else:
                # Handle ranked prompts
                return {
                    "prompt": example.get("prompt", ""),
                    "completion": example.get("response", ""),
                    "label": example.get("score", 0) > 0.5
                }
        
        return dataset.map(format_sample)
    
    def prepare_datasets_for_method(self, method: str, max_samples: Optional[int] = None) -> Dict[str, Dataset]:
        """Prepare datasets for a specific training method"""
        if method not in self.dataset_configs:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[method]
        prepared_datasets = {}
        
        for dataset_name in config["datasets"]:
            try:
                # Download dataset
                dataset = self.download_dataset(dataset_name)
                if dataset is None:
                    continue
                
                # Handle DatasetDict vs Dataset
                if isinstance(dataset, DatasetDict):
                    # Use train split if available, otherwise first available split
                    if "train" in dataset:
                        dataset = dataset["train"]
                    else:
                        dataset = dataset[list(dataset.keys())[0]]
                
                # Limit samples if specified
                if max_samples and len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                
                # Format based on method
                if method == "sft":
                    formatted_dataset = self.format_sft_data(dataset, dataset_name)
                elif method == "dpo":
                    formatted_dataset = self.format_dpo_data(dataset, dataset_name)
                elif method == "kto":
                    formatted_dataset = self.format_kto_data(dataset, dataset_name)
                else:
                    formatted_dataset = dataset
                
                # Save formatted dataset
                dataset_key = dataset_name.split("/")[-1]
                prepared_datasets[dataset_key] = formatted_dataset
                
                # Save to disk
                output_path = self.data_dir / method / dataset_key
                output_path.mkdir(parents=True, exist_ok=True)
                formatted_dataset.save_to_disk(str(output_path))
                
                logger.info(f"Prepared {len(formatted_dataset)} samples for {dataset_key}")
                
            except Exception as e:
                logger.error(f"Failed to prepare {dataset_name}: {e}")
                continue
        
        return prepared_datasets
    
    def create_combined_dataset(self, method: str, datasets: Dict[str, Dataset]) -> Dataset:
        """Combine multiple datasets into one"""
        if not datasets:
            raise ValueError("No datasets to combine")
        
        combined_samples = []
        for dataset_name, dataset in datasets.items():
            logger.info(f"Adding {len(dataset)} samples from {dataset_name}")
            for sample in dataset:
                sample["source_dataset"] = dataset_name
                combined_samples.append(sample)
        
        combined_dataset = Dataset.from_list(combined_samples)
        
        # Save combined dataset
        output_path = self.data_dir / method / "combined"
        output_path.mkdir(parents=True, exist_ok=True)
        combined_dataset.save_to_disk(str(output_path))
        
        logger.info(f"Created combined dataset with {len(combined_dataset)} samples")
        return combined_dataset
    
    def prepare_all_datasets(self, max_samples_per_dataset: Optional[int] = None):
        """Prepare datasets for all training methods"""
        results = {}
        
        for method in ["sft", "dpo", "kto", "evaluation"]:
            logger.info(f"Preparing datasets for {method.upper()}")
            try:
                datasets = self.prepare_datasets_for_method(method, max_samples_per_dataset)
                if datasets:
                    if method != "evaluation":  # Don't combine evaluation datasets
                        combined = self.create_combined_dataset(method, datasets)
                        results[method] = {"individual": datasets, "combined": combined}
                    else:
                        results[method] = {"individual": datasets}
                else:
                    logger.warning(f"No datasets prepared for {method}")
            except Exception as e:
                logger.error(f"Failed to prepare datasets for {method}: {e}")
        
        return results
    
    def generate_dataset_info(self, results: Dict) -> Dict:
        """Generate information about prepared datasets"""
        info = {}
        
        for method, data in results.items():
            method_info = {}
            
            if "individual" in data:
                method_info["individual_datasets"] = {}
                for name, dataset in data["individual"].items():
                    method_info["individual_datasets"][name] = {
                        "num_samples": len(dataset),
                        "columns": list(dataset.column_names),
                        "sample": dict(dataset[0]) if len(dataset) > 0 else {}
                    }
            
            if "combined" in data:
                method_info["combined_dataset"] = {
                    "num_samples": len(data["combined"]),
                    "columns": list(data["combined"].column_names)
                }
            
            info[method] = method_info
        
        # Save info to file
        info_path = self.data_dir / "dataset_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        return info

def main():
    """Main function to prepare all datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Arabic datasets for fine-tuning")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--max-samples", type=int, help="Maximum samples per dataset")
    parser.add_argument("--method", choices=["sft", "dpo", "kto", "evaluation", "all"], 
                       default="all", help="Training method to prepare data for")
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = ArabicDatasetPreparer(args.data_dir, args.cache_dir)
    
    # Prepare datasets
    if args.method == "all":
        results = preparer.prepare_all_datasets(args.max_samples)
    else:
        datasets = preparer.prepare_datasets_for_method(args.method, args.max_samples)
        if datasets and args.method != "evaluation":
            combined = preparer.create_combined_dataset(args.method, datasets)
            results = {args.method: {"individual": datasets, "combined": combined}}
        else:
            results = {args.method: {"individual": datasets}}
    
    # Generate and save dataset info
    info = preparer.generate_dataset_info(results)
    
    print("\n=== Dataset Preparation Complete ===")
    for method, method_info in info.items():
        print(f"\n{method.upper()}:")
        if "individual_datasets" in method_info:
            for name, dataset_info in method_info["individual_datasets"].items():
                print(f"  - {name}: {dataset_info['num_samples']} samples")
        if "combined_dataset" in method_info:
            print(f"  - Combined: {method_info['combined_dataset']['num_samples']} samples")

if __name__ == "__main__":
    main()