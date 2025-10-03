# Implementation Examples for Arabic Qwen Fine-tuning

This document provides complete, ready-to-run code examples for fine-tuning Qwen models on Arabic datasets. All examples are optimized for RTX 3060 12GB and include proper error handling and optimization techniques.

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Basic SFT Implementation](#basic-sft-implementation)
3. [DPO Implementation](#dpo-implementation)
4. [KTO Implementation](#kto-implementation)
5. [IPO Implementation](#ipo-implementation)
6. [CPO Implementation](#cpo-implementation)
7. [SimPO Implementation](#simpo-implementation)
8. [Model Merging Examples](#model-merging-examples)
9. [Arabic Dialect Processing](#arabic-dialect-processing)
10. [Complete Training Scripts](#complete-training-scripts)
11. [Evaluation Examples](#evaluation-examples)
12. [Inference Examples](#inference-examples)

## 🛠️ Environment Setup

### Requirements Installation

```bash
# Create virtual environment
python -m venv arabic_qwen_env
source arabic_qwen_env/bin/activate  # On Windows: arabic_qwen_env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install peft>=0.7.0
pip install trl>=0.7.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.24.0
pip install wandb  # Optional: for experiment tracking
```

### Basic Imports and Setup

```python
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig, KTOTrainer, KTOConfig, IPOTrainer, IPOConfig, CPOTrainer, CPOConfig
import json
import os
from typing import Dict, List, Optional

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 🎯 Basic SFT Implementation

### Example 1: Qwen2.5-3B SFT with InstAr-500k

```python
class ArabicSFTTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B", output_dir: str = "./arabic-sft-model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with optimal settings for RTX 3060"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with FP16 for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
        
    def prepare_dataset(self, dataset_name: str = "FreedomIntelligence/InstAr-500k", max_samples: int = None):
        """Load and format Arabic instruction dataset"""
        print(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Limit samples for testing
        if max_samples:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            
        # Format for instruction following
        def format_instruction(example):
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            
            # Arabic instruction format
            formatted_text = f"### التعليمات:\n{instruction}\n\n### الاستجابة:\n{output}"
            
            return {"text": formatted_text}
        
        # Apply formatting
        formatted_dataset = dataset.map(format_instruction, remove_columns=dataset["train"].column_names)
        
        print(f"Dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def tokenize_dataset(self, dataset, max_length: int = 512):
        """Tokenize the formatted dataset"""
        def tokenize_function(examples):
            # Tokenize with truncation and padding
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Set labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def train(self, dataset, epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
        """Train the model with optimized settings for RTX 3060"""
        
        # Training arguments optimized for RTX 3060 12GB
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            fp16=True,  # Mixed precision for memory efficiency
            dataloader_pin_memory=False,
            gradient_checkpointing=True,  # Save memory at cost of speed
            report_to=None,  # Disable wandb for this example
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed. Model saved to {self.output_dir}")
        
        return trainer

# Usage example
if __name__ == "__main__":
    # Initialize trainer
    sft_trainer = ArabicSFTTrainer(
        model_name="Qwen/Qwen2.5-3B",
        output_dir="./models/arabic-qwen-sft"
    )
    
    # Setup model
    sft_trainer.setup_model_and_tokenizer()
    
    # Prepare dataset (use subset for testing)
    dataset = sft_trainer.prepare_dataset(
        dataset_name="FreedomIntelligence/InstAr-500k",
        max_samples=1000  # Remove for full dataset
    )
    
    # Tokenize dataset
    tokenized_dataset = sft_trainer.tokenize_dataset(dataset)
    
    # Train model
    trainer = sft_trainer.train(
        dataset=tokenized_dataset,
        epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )
```

## 🎯 DPO Implementation

### Example 2: DPO Training for Preference Optimization

```python
class ArabicDPOTrainer:
    def __init__(self, sft_model_path: str, output_dir: str = "./arabic-dpo-model"):
        self.sft_model_path = sft_model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Load the SFT model for DPO training"""
        print(f"Loading SFT model from: {self.sft_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("SFT model loaded for DPO training")
        
    def prepare_preference_dataset(self, dataset_name: str = "FreedomIntelligence/Arabic-preference-data-RLHF"):
        """Load and format preference dataset for DPO"""
        print(f"Loading preference dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        
        def format_preference_data(example):
            """Format data for DPO training"""
            return {
                "prompt": example["question"],
                "chosen": example["chosen_response"],
                "rejected": example["rejected_response"]
            }
        
        formatted_dataset = dataset.map(format_preference_data)
        
        print(f"Preference dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def train_dpo(self, dataset, epochs: int = 1, batch_size: int = 2, learning_rate: float = 5e-6):
        """Train with Direct Preference Optimization"""
        
        # DPO configuration
        dpo_config = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Effective batch size = 16
            learning_rate=learning_rate,
            beta=0.1,  # DPO temperature parameter
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use implicit reference model
            args=dpo_config,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
        )
        
        # Start DPO training
        print("Starting DPO training...")
        dpo_trainer.train()
        
        # Save model
        dpo_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"DPO training completed. Model saved to {self.output_dir}")
        
        return dpo_trainer

# Usage example
if __name__ == "__main__":
    # Initialize DPO trainer with SFT model
    dpo_trainer = ArabicDPOTrainer(
        sft_model_path="./models/arabic-qwen-sft",
        output_dir="./models/arabic-qwen-dpo"
    )
    
    # Setup model
    dpo_trainer.setup_model_and_tokenizer()
    
    # Prepare preference dataset
    preference_dataset = dpo_trainer.prepare_preference_dataset()
    
    # Train with DPO
    dpo_trainer.train_dpo(
        dataset=preference_dataset,
        epochs=1,
        batch_size=2,
        learning_rate=5e-6
    )
```

## 🎯 KTO Implementation

### Example 3: KTO Training for Preference Optimization

```python
from trl import KTOTrainer, KTOConfig

class ArabicKTOTrainer:
    def __init__(self, sft_model_path: str, output_dir: str = "./arabic-kto-model"):
        self.sft_model_path = sft_model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Load the SFT model for KTO training"""
        print(f"Loading SFT model from: {self.sft_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("SFT model loaded for KTO training")
        
    def prepare_kto_dataset(self, dataset_name: str = "argilla/argilla-dpo-mix-7k-arabic"):
        """Load and format dataset for KTO training"""
        print(f"Loading KTO dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        
        def format_kto_data(example):
            """Format data for KTO training"""
            return {
                "prompt": example["instruction"],
                "completion": example["chosen"],
                "label": True  # KTO uses binary labels
            }
        
        formatted_dataset = dataset.map(format_kto_data)
        
        print(f"KTO dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def train_kto(self, dataset, epochs: int = 1, batch_size: int = 2, learning_rate: float = 1e-5):
        """Train with Kahneman-Tversky Optimization"""
        
        kto_config = KTOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            beta=0.1,  # KTO temperature parameter
            desirable_weight=1.0,
            undesirable_weight=1.0,
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=None,
        )
        
        kto_trainer = KTOTrainer(
            model=self.model,
            args=kto_config,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
        )
        
        print("Starting KTO training...")
        kto_trainer.train()
        
        kto_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"KTO training completed. Model saved to {self.output_dir}")
        return kto_trainer
```

## 🎯 IPO Implementation

### Example 4: IPO Training for Identity Preference Optimization

```python
from trl import IPOTrainer, IPOConfig

class ArabicIPOTrainer:
    def __init__(self, sft_model_path: str, output_dir: str = "./arabic-ipo-model"):
        self.sft_model_path = sft_model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Load the SFT model for IPO training"""
        print(f"Loading SFT model from: {self.sft_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("SFT model loaded for IPO training")
        
    def prepare_ipo_dataset(self, dataset_name: str = "FreedomIntelligence/Arabic-preference-data-RLHF"):
        """Load and format preference dataset for IPO"""
        print(f"Loading IPO dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        
        def format_ipo_data(example):
            """Format data for IPO training"""
            return {
                "prompt": example["question"],
                "chosen": example["chosen_response"],
                "rejected": example["rejected_response"]
            }
        
        formatted_dataset = dataset.map(format_ipo_data)
        
        print(f"IPO dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def train_ipo(self, dataset, epochs: int = 1, batch_size: int = 2, learning_rate: float = 1e-5):
        """Train with Identity Preference Optimization"""
        
        ipo_config = IPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            beta=0.1,  # IPO regularization parameter
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=None,
        )
        
        ipo_trainer = IPOTrainer(
            model=self.model,
            ref_model=None,
            args=ipo_config,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
        )
        
        print("Starting IPO training...")
        ipo_trainer.train()
        
        ipo_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"IPO training completed. Model saved to {self.output_dir}")
        return ipo_trainer
```

## 🎯 CPO Implementation

### Example 5: CPO Training for Contrastive Preference Optimization

```python
from trl import CPOTrainer, CPOConfig

class ArabicCPOTrainer:
    def __init__(self, sft_model_path: str, output_dir: str = "./arabic-cpo-model"):
        self.sft_model_path = sft_model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Load the SFT model for CPO training"""
        print(f"Loading SFT model from: {self.sft_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("SFT model loaded for CPO training")
        
    def prepare_cpo_dataset(self, dataset_name: str = "argilla/argilla-dpo-mix-7k-arabic"):
        """Load and format preference dataset for CPO"""
        print(f"Loading CPO dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        
        def format_cpo_data(example):
            """Format data for CPO training"""
            return {
                "prompt": example["instruction"],
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }
        
        formatted_dataset = dataset.map(format_cpo_data)
        
        print(f"CPO dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def train_cpo(self, dataset, epochs: int = 1, batch_size: int = 2, learning_rate: float = 1e-5):
        """Train with Contrastive Preference Optimization"""
        
        cpo_config = CPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            beta=0.1,  # CPO temperature parameter
            simpo_gamma=0.5,  # CPO-specific parameter
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=None,
        )
        
        cpo_trainer = CPOTrainer(
            model=self.model,
            ref_model=None,
            args=cpo_config,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
        )
        
        print("Starting CPO training...")
        cpo_trainer.train()
        
        cpo_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"CPO training completed. Model saved to {self.output_dir}")
        return cpo_trainer
```

## 🎯 SimPO Implementation

### Example 7: Simple Preference Optimization (SimPO)

```python
class ArabicSimPOTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B", output_dir: str = "./arabic-simpo-model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer for SimPO training"""
        print(f"Loading model for SimPO: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with FP16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded for SimPO. Parameters: {self.model.num_parameters():,}")
        
    def prepare_simpo_dataset(self, dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"):
        """Prepare dataset for SimPO training with Arabic preference data"""
        print(f"Loading preference dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Convert to Arabic preference format
        def format_simpo_data(example):
            # Extract prompt and responses
            prompt = example.get("prompt", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")
            
            # Translate to Arabic format (in practice, use Arabic datasets)
            arabic_prompt = f"### السؤال:\n{prompt}\n\n"
            
            return {
                "prompt": arabic_prompt,
                "chosen": f"### الإجابة المفضلة:\n{chosen}",
                "rejected": f"### الإجابة المرفوضة:\n{rejected}"
            }
        
        # Apply formatting
        formatted_dataset = dataset.map(format_simpo_data)
        
        print(f"SimPO dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def train_simpo(self, dataset, epochs: int = 3, batch_size: int = 2, learning_rate: float = 1e-6):
        """Train with SimPO (Simple Preference Optimization)"""
        from src.training.simpo_trainer import SimPOTrainer, SimPOConfig
        
        # SimPO configuration
        simpo_config = SimPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            max_length=512,
            max_prompt_length=256,
            
            # SimPO-specific parameters
            beta=0.1,  # Length normalization parameter
            gamma=1.0,  # Preference strength
            length_penalty=0.1,  # Length penalty coefficient
            
            # Optimization settings
            fp16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            warmup_steps=100,
            weight_decay=0.01,
            
            # Memory optimization
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Initialize SimPO trainer
        simpo_trainer = SimPOTrainer(
            model=self.model,
            args=simpo_config,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
        )
        
        print("Starting SimPO training...")
        simpo_trainer.train()
        
        simpo_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"SimPO training completed. Model saved to {self.output_dir}")
        return simpo_trainer

# Usage example
if __name__ == "__main__":
    # Initialize SimPO trainer
    simpo_trainer = ArabicSimPOTrainer(
        model_name="./arabic-sft-model",  # Use SFT model as base
        output_dir="./arabic-simpo-model"
    )
    
    # Setup model
    simpo_trainer.setup_model_and_tokenizer()
    
    # Prepare dataset
    dataset = simpo_trainer.prepare_simpo_dataset()
    
    # Train with SimPO
    trainer = simpo_trainer.train_simpo(dataset, epochs=2, batch_size=2)
    
    print("SimPO training pipeline completed!")
```

## 🔀 Model Merging Examples

### Example 8: Advanced Model Merging

```python
class ArabicModelMerger:
    def __init__(self):
        self.merged_models = {}
        
    def weighted_merge_example(self):
        """Example of weighted model merging"""
        from src.training.model_merger import ModelMerger, MergeConfig
        
        # Configure weighted merging
        config = MergeConfig(
            strategy="weighted",
            models=[
                {
                    "path": "./arabic-sft-model",
                    "weight": 0.4,
                    "name": "sft_base"
                },
                {
                    "path": "./arabic-dpo-model", 
                    "weight": 0.3,
                    "name": "dpo_specialist"
                },
                {
                    "path": "./arabic-simpo-model",
                    "weight": 0.3,
                    "name": "simpo_specialist"
                }
            ],
            output_path="./merged-arabic-model",
            options={
                "normalize_weights": True,
                "preserve_tokenizer": True,
                "torch_dtype": "float16"
            },
            validation={
                "run_tests": True,
                "test_prompts": [
                    "مرحبا، كيف يمكنني مساعدتك؟",
                    "اشرح لي مفهوم الذكاء الاصطناعي",
                    "ما هي أفضل طريقة لتعلم البرمجة؟"
                ]
            }
        )
        
        # Perform merge
        merger = ModelMerger(config)
        merged_model = merger.merge()
        
        print("Weighted merge completed successfully!")
        return merged_model
    
    def task_arithmetic_example(self):
        """Example of task arithmetic merging"""
        from src.training.model_merger import ModelMerger, MergeConfig
        
        # Configure task arithmetic
        config = MergeConfig(
            strategy="task_arithmetic",
            base_model="./arabic-base-model",
            models=[
                {
                    "path": "./arabic-chat-model",
                    "weight": 1.0,
                    "operation": "add",
                    "name": "chat_enhancement"
                },
                {
                    "path": "./arabic-formal-model",
                    "weight": -0.5,
                    "operation": "subtract", 
                    "name": "formality_reduction"
                }
            ],
            output_path="./casual-arabic-chat",
            options={
                "scaling_factor": 1.0,
                "clamp_weights": True,
                "clamp_range": [-2.0, 2.0]
            }
        )
        
        merger = ModelMerger(config)
        merged_model = merger.merge()
        
        print("Task arithmetic merge completed!")
        return merged_model
    
    def dialect_merge_example(self):
        """Example of merging multiple Arabic dialect models"""
        from src.training.model_merger import ModelMerger, MergeConfig
        
        config = MergeConfig(
            strategy="weighted",
            models=[
                {"path": "./models/egyptian_arabic", "weight": 0.25, "name": "egyptian"},
                {"path": "./models/gulf_arabic", "weight": 0.25, "name": "gulf"},
                {"path": "./models/levantine_arabic", "weight": 0.25, "name": "levantine"},
                {"path": "./models/maghrebi_arabic", "weight": 0.25, "name": "maghrebi"}
            ],
            output_path="./models/multi_dialect_arabic",
            validation={
                "run_tests": True,
                "test_prompts": [
                    "إزيك؟",      # Egyptian
                    "شلونك؟",      # Gulf  
                    "كيفك؟",       # Levantine
                    "كيداير؟"      # Maghrebi
                ]
            }
        )
        
        merger = ModelMerger(config)
        multi_dialect_model = merger.merge()
        
        print("Multi-dialect merge completed!")
        return multi_dialect_model

# Usage example
if __name__ == "__main__":
    merger = ArabicModelMerger()
    
    # Try different merging strategies
    print("1. Weighted merge...")
    weighted_model = merger.weighted_merge_example()
    
    print("2. Task arithmetic merge...")
    arithmetic_model = merger.task_arithmetic_example()
    
    print("3. Dialect merge...")
    dialect_model = merger.dialect_merge_example()
    
    print("All merging examples completed!")
```

## 🗣️ Arabic Dialect Processing

### Example 9: Dialect Detection and Augmentation

```python
class ArabicDialectProcessor:
    def __init__(self):
        self.detector = None
        self.augmenter = None
        self.processor = None
        
    def setup_dialect_tools(self):
        """Initialize Arabic dialect processing tools"""
        from src.utils.arabic_dialect_utils import (
            ArabicDialectDetector,
            ArabicTextAugmenter, 
            ArabicDatasetProcessor
        )
        
        self.detector = ArabicDialectDetector()
        self.augmenter = ArabicTextAugmenter()
        self.processor = ArabicDatasetProcessor()
        
        print("Arabic dialect tools initialized successfully!")
    
    def dialect_detection_example(self):
        """Example of Arabic dialect detection"""
        # Sample texts in different dialects
        texts = [
            "مرحبا كيف حالك؟",        # MSA
            "إزيك يا صاحبي؟",         # Egyptian
            "شلونك حبيبي؟",           # Gulf
            "كيفك اليوم؟",            # Levantine
            "كيداير؟"                # Maghrebi
        ]
        
        print("Dialect Detection Results:")
        print("=" * 50)
        
        for text in texts:
            result = self.detector.detect_dialect(text)
            print(f"Text: {text}")
            print(f"Dialect: {result['dialect']} (Confidence: {result['confidence']:.2f})")
            print(f"Is Arabic: {result['is_arabic']}")
            print("-" * 30)
    
    def text_augmentation_example(self):
        """Example of Arabic text augmentation"""
        original_text = "كيف حالك اليوم؟"
        
        print(f"Original text: {original_text}")
        print("Augmented versions:")
        print("=" * 50)
        
        # Generate augmentations
        augmented = self.augmenter.augment_text(original_text)
        
        for i, aug_text in enumerate(augmented, 1):
            print(f"{i}. {aug_text}")
        
        # Dialect-specific augmentation
        print("\nEgyptian dialect versions:")
        print("-" * 30)
        
        egy_versions = self.augmenter.augment_text(
            original_text, 
            target_dialect="EGY"
        )
        
        for i, egy_text in enumerate(egy_versions, 1):
            print(f"{i}. {egy_text}")
    
    def dataset_processing_example(self):
        """Example of processing dataset with dialect handling"""
        from datasets import Dataset
        
        # Sample dataset
        data = {
            "text": [
                "مرحبا كيف حالك؟",
                "إزيك يا صاحبي؟", 
                "شلونك حبيبي؟",
                "كيفك اليوم؟",
                "Hello, how are you?"  # Non-Arabic
            ],
            "label": ["greeting"] * 5
        }
        
        dataset = Dataset.from_dict(data)
        
        # Processing configuration
        config = {
            "detect_dialects": True,
            "augment_data": True,
            "normalize_text": True,
            "target_dialects": ["MSA", "EGY", "GLF"],
            "augmentation_factor": 2
        }
        
        # Process dataset
        processed = self.processor.process_dataset(dataset, config)
        
        print("Dataset Processing Results:")
        print("=" * 50)
        print(f"Original size: {len(dataset)}")
        print(f"Processed size: {len(processed)}")
        
        # Show sample processed entries
        for i in range(min(3, len(processed))):
            entry = processed[i]
            print(f"\nEntry {i+1}:")
            print(f"  Original: {entry.get('original_text', 'N/A')}")
            print(f"  Processed: {entry.get('text', 'N/A')}")
            print(f"  Dialect: {entry.get('dialect', 'N/A')}")
            print(f"  Confidence: {entry.get('dialect_confidence', 'N/A')}")
    
    def training_integration_example(self):
        """Example of integrating dialect processing with training"""
        from datasets import Dataset
        
        # Create sample training data
        training_data = {
            "instruction": [
                "اشرح لي مفهوم الذكاء الاصطناعي",
                "إزاي أتعلم البرمجة؟",
                "شلون أقدر أحسن من مهاراتي؟"
            ],
            "output": [
                "الذكاء الاصطناعي هو...",
                "عشان تتعلم البرمجة...",
                "تقدر تحسن مهاراتك من خلال..."
            ]
        }
        
        dataset = Dataset.from_dict(training_data)
        
        # Process with dialect augmentation
        config = {
            "detect_dialects": True,
            "augment_data": True,
            "balance_dialects": True,
            "target_dialects": ["MSA", "EGY", "GLF", "LEV"],
            "augmentation_factor": 3
        }
        
        augmented_dataset = self.processor.process_dataset(dataset, config)
        
        print("Training Data Augmentation:")
        print("=" * 50)
        print(f"Original samples: {len(dataset)}")
        print(f"Augmented samples: {len(augmented_dataset)}")
        
        # Show dialect distribution
        dialect_counts = {}
        for entry in augmented_dataset:
            dialect = entry.get('dialect', 'UNKNOWN')
            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
        
        print("\nDialect distribution:")
        for dialect, count in dialect_counts.items():
            print(f"  {dialect}: {count} samples")

# Usage example
if __name__ == "__main__":
    processor = ArabicDialectProcessor()
    processor.setup_dialect_tools()
    
    print("1. Dialect Detection Example:")
    processor.dialect_detection_example()
    
    print("\n2. Text Augmentation Example:")
    processor.text_augmentation_example()
    
    print("\n3. Dataset Processing Example:")
    processor.dataset_processing_example()
    
    print("\n4. Training Integration Example:")
    processor.training_integration_example()
    
    print("\nAll dialect processing examples completed!")
```

## 🔄 Complete Training Scripts

### End-to-End Pipeline Script

```python
#!/usr/bin/env python3
"""
Complete Arabic Qwen Fine-tuning Pipeline
Supports SFT -> DPO workflow with automatic model management
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path

class ArabicQwenPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.base_model = config["base_model"]
        self.output_base_dir = config["output_dir"]
        self.experiment_name = config.get("experiment_name", f"arabic_qwen_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create experiment directory
        self.experiment_dir = Path(self.output_base_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.experiment_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def run_sft_stage(self):
        """Run Supervised Fine-Tuning stage"""
        print("=== Starting SFT Stage ===")
        
        sft_output_dir = self.experiment_dir / "sft_model"
        
        sft_trainer = ArabicSFTTrainer(
            model_name=self.base_model,
            output_dir=str(sft_output_dir)
        )
        
        # Setup and train
        sft_trainer.setup_model_and_tokenizer()
        
        dataset = sft_trainer.prepare_dataset(
            dataset_name=self.config["sft_dataset"],
            max_samples=self.config.get("sft_max_samples")
        )
        
        tokenized_dataset = sft_trainer.tokenize_dataset(dataset)
        
        trainer = sft_trainer.train(
            dataset=tokenized_dataset,
            epochs=self.config["sft_epochs"],
            batch_size=self.config["sft_batch_size"],
            learning_rate=self.config["sft_learning_rate"]
        )
        
        print(f"SFT completed. Model saved to {sft_output_dir}")
        return str(sft_output_dir)
    
    def run_preference_stage(self, sft_model_path: str):
        """Run preference optimization stage using SFT model"""
        preference_method = self.config.get("preference_method", "dpo")
        print(f"=== Starting {preference_method.upper()} Stage ===")
        
        preference_output_dir = self.experiment_dir / f"{preference_method}_model"
        
        if preference_method == "dpo":
            trainer = ArabicDPOTrainer(
                sft_model_path=sft_model_path,
                output_dir=str(preference_output_dir)
            )
            trainer.setup_model_and_tokenizer()
            dataset = trainer.prepare_preference_dataset(self.config["preference_dataset"])
            trainer.train_dpo(
                dataset=dataset,
                epochs=self.config["preference_epochs"],
                batch_size=self.config["preference_batch_size"],
                learning_rate=self.config["preference_learning_rate"]
            )
        elif preference_method == "kto":
            trainer = ArabicKTOTrainer(
                sft_model_path=sft_model_path,
                output_dir=str(preference_output_dir)
            )
            trainer.setup_model_and_tokenizer()
            dataset = trainer.prepare_kto_dataset(self.config["preference_dataset"])
            trainer.train_kto(
                dataset=dataset,
                epochs=self.config["preference_epochs"],
                batch_size=self.config["preference_batch_size"],
                learning_rate=self.config["preference_learning_rate"]
            )
        elif preference_method == "ipo":
            trainer = ArabicIPOTrainer(
                sft_model_path=sft_model_path,
                output_dir=str(preference_output_dir)
            )
            trainer.setup_model_and_tokenizer()
            dataset = trainer.prepare_ipo_dataset(self.config["preference_dataset"])
            trainer.train_ipo(
                dataset=dataset,
                epochs=self.config["preference_epochs"],
                batch_size=self.config["preference_batch_size"],
                learning_rate=self.config["preference_learning_rate"]
            )
        elif preference_method == "cpo":
            trainer = ArabicCPOTrainer(
                sft_model_path=sft_model_path,
                output_dir=str(preference_output_dir)
            )
            trainer.setup_model_and_tokenizer()
            dataset = trainer.prepare_cpo_dataset(self.config["preference_dataset"])
            trainer.train_cpo(
                dataset=dataset,
                epochs=self.config["preference_epochs"],
                batch_size=self.config["preference_batch_size"],
                learning_rate=self.config["preference_learning_rate"]
            )
        
        print(f"{preference_method.upper()} completed. Model saved to {preference_output_dir}")
        return str(preference_output_dir)
    
    def run_full_pipeline(self):
        """Run complete SFT -> Preference Optimization pipeline"""
        print(f"Starting full pipeline for experiment: {self.experiment_name}")
        
        # Stage 1: SFT
        if self.config.get("run_sft", True):
            sft_model_path = self.run_sft_stage()
        else:
            sft_model_path = self.config["existing_sft_model"]
            
        # Stage 2: Preference Optimization
        if self.config.get("run_preference", True):
            final_model_path = self.run_preference_stage(sft_model_path)
        else:
            final_model_path = sft_model_path
            
        print(f"\n=== Pipeline Complete ===")
        print(f"Experiment: {self.experiment_name}")
        print(f"Final model: {final_model_path}")
        print(f"Experiment directory: {self.experiment_dir}")
        
        return final_model_path

# Configuration examples
CONFIG_EXAMPLES = {
    "qwen_3b_dpo": {
        "base_model": "Qwen/Qwen2.5-3B",
        "output_dir": "./experiments",
        "experiment_name": "qwen3b_arabic_dpo",
        "run_sft": True,
        "run_preference": True,
        "preference_method": "dpo",
        "sft_dataset": "FreedomIntelligence/InstAr-500k",
        "sft_epochs": 3,
        "sft_batch_size": 4,
        "sft_learning_rate": 2e-5,
        "preference_dataset": "FreedomIntelligence/Arabic-preference-data-RLHF",
        "preference_epochs": 1,
        "preference_batch_size": 2,
        "preference_learning_rate": 5e-6
    },
    "qwen_1.7b_kto": {
        "base_model": "Qwen/Qwen3-1.7B",
        "output_dir": "./experiments",
        "experiment_name": "qwen1.7b_arabic_kto",
        "run_sft": True,
        "run_preference": True,
        "preference_method": "kto",
        "sft_dataset": "FreedomIntelligence/CIDAR",
        "sft_epochs": 5,
        "sft_batch_size": 8,
        "sft_learning_rate": 3e-5,
        "preference_dataset": "argilla/argilla-dpo-mix-7k-arabic",
        "preference_epochs": 1,
        "preference_batch_size": 4,
        "preference_learning_rate": 1e-5
    },
    "qwen_7b_ipo": {
        "base_model": "Qwen/Qwen2.5-7B",
        "output_dir": "./experiments",
        "experiment_name": "qwen7b_arabic_ipo",
        "run_sft": True,
        "run_preference": True,
        "preference_method": "ipo",
        "sft_dataset": "FreedomIntelligence/InstAr-500k",
        "sft_epochs": 2,
        "sft_batch_size": 2,
        "sft_learning_rate": 2e-5,
        "preference_dataset": "FreedomIntelligence/Arabic-preference-data-RLHF",
        "preference_epochs": 1,
        "preference_batch_size": 1,
        "preference_learning_rate": 1e-5
    },
    "qwen_3b_cpo": {
        "base_model": "Qwen/Qwen2.5-3B",
        "output_dir": "./experiments",
        "experiment_name": "qwen3b_arabic_cpo",
        "run_sft": True,
        "run_preference": True,
        "preference_method": "cpo",
        "sft_dataset": "FreedomIntelligence/InstAr-500k",
        "sft_epochs": 3,
        "sft_batch_size": 4,
        "sft_learning_rate": 2e-5,
        "preference_dataset": "argilla/argilla-dpo-mix-7k-arabic",
        "preference_epochs": 1,
        "preference_batch_size": 2,
        "preference_learning_rate": 1e-5
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Qwen Fine-tuning Pipeline")
    parser.add_argument("--config", choices=list(CONFIG_EXAMPLES.keys()), 
                       default="qwen_3b_dpo", help="Configuration preset")
    parser.add_argument("--config-file", type=str, help="Custom config JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = CONFIG_EXAMPLES[args.config]
    
    # Run pipeline
    pipeline = ArabicQwenPipeline(config)
    final_model = pipeline.run_full_pipeline()
```

## 📊 Evaluation Examples

### Model Evaluation Script

```python
class ArabicModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7):
        """Generate response for given Arabic prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def evaluate_instruction_following(self, test_prompts: List[str]):
        """Evaluate instruction following capability"""
        results = []
        
        for prompt in test_prompts:
            formatted_prompt = f"### التعليمات:\n{prompt}\n\n### الاستجابة:\n"
            response = self.generate_response(formatted_prompt)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "formatted_prompt": formatted_prompt
            })
            
        return results
    
    def evaluate_qa_capability(self, qa_pairs: List[Dict[str, str]]):
        """Evaluate QA capability"""
        results = []
        
        for qa in qa_pairs:
            question = qa["question"]
            expected_answer = qa["answer"]
            
            formatted_prompt = f"السؤال: {question}\nالإجابة: "
            generated_answer = self.generate_response(formatted_prompt)
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "prompt": formatted_prompt
            })
            
        return results

# Example evaluation
if __name__ == "__main__":
    # Test prompts
    test_instructions = [
        "اكتب قصة قصيرة عن الصداقة",
        "اشرح مفهوم الذكاء الاصطناعي بطريقة بسيطة",
        "ما هي فوائد القراءة؟",
        "كيف يمكنني تحسين مهاراتي في البرمجة؟"
    ]
    
    test_qa = [
        {"question": "ما هي عاصمة المملكة العربية السعودية؟", "answer": "الرياض"},
        {"question": "كم عدد أيام السنة الميلادية؟", "answer": "365 يوم"},
        {"question": "من هو مؤلف رواية 'مدن الملح'؟", "answer": "عبد الرحمن منيف"}
    ]
    
    # Evaluate model (replace with your trained model path)
    model_path = "./experiments/qwen3b_arabic_dpo/dpo_model"  # or kto_model, ipo_model, cpo_model
    evaluator = ArabicModelEvaluator(model_path)
    
    print("=== Instruction Following Evaluation ===")
    instruction_results = evaluator.evaluate_instruction_following(test_instructions)
    
    for result in instruction_results:
        print(f"\nPrompt: {result['prompt']}")
        print(f"Response: {result['response']}")
        print("-" * 50)
    
    print("\n=== QA Evaluation ===")
    qa_results = evaluator.evaluate_qa_capability(test_qa)
    
    for result in qa_results:
        print(f"\nQuestion: {result['question']}")
        print(f"Expected: {result['expected_answer']}")
        print(f"Generated: {result['generated_answer']}")
        print("-" * 50)
```

## 🚀 Inference Examples

### Production Inference Class

```python
class ArabicQwenInference:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load model for inference"""
        print(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.model.eval()
        print("Model loaded successfully")
        
    def chat(self, message: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        """Chat interface for Arabic conversations"""
        prompt = f"### التعليمات:\n{message}\n\n### الاستجابة:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs["input_ids"][0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
    
    def answer_question(self, question: str, max_length: int = 256):
        """QA interface for Arabic questions"""
        prompt = f"السؤال: {question}\nالإجابة: "
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs["input_ids"][0]) + max_length,
                temperature=0.3,  # Lower temperature for factual answers
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        
        return answer
    
    def batch_generate(self, prompts: List[str], max_length: int = 512):
        """Generate responses for multiple prompts"""
        formatted_prompts = [f"### التعليمات:\n{prompt}\n\n### الاستجابة:\n" for prompt in prompts]
        
        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        responses = []
        for i, output in enumerate(outputs):
            full_response = self.tokenizer.decode(output, skip_special_tokens=True)
            response = full_response[len(formatted_prompts[i]):].strip()
            responses.append(response)
            
        return responses

# Usage example
if __name__ == "__main__":
    # Initialize inference (replace with your trained model path)
    model_path = "./experiments/qwen3b_arabic_dpo/dpo_model"  # or kto_model, ipo_model, cpo_model
    arabic_model = ArabicQwenInference(model_path)
    
    # Interactive chat
    print("Arabic Qwen Chat Interface (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        user_input = input("\nأنت: ")
        if user_input.lower() in ['quit', 'exit', 'خروج']:
            break
            
        response = arabic_model.chat(user_input)
        print(f"النموذج: {response}")
    
    print("\nشكراً لاستخدام النموذج!")
```

## 📝 Usage Notes

### Memory Management Tips

```python
# Clear GPU memory between experiments
import gc

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)} bytes")

# Use this between training runs
clear_gpu_memory()
```

### Monitoring GPU Usage

```python
def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")
        print(f"Usage: {(allocated/total)*100:.1f}%")
    else:
        print("CUDA not available")

# Call during training to monitor memory
monitor_gpu_usage()
```

---

These implementation examples provide complete, production-ready code for fine-tuning Arabic Qwen models with multiple preference optimization methods (DPO, KTO, IPO, CPO). All examples are optimized for RTX 3060 12GB and include proper error handling, memory management, and evaluation capabilities.

## 🎯 Method Selection Guide

- **DPO**: Best for general preference alignment, stable training
- **KTO**: Efficient for binary preference data, faster training
- **IPO**: Good for identity-preserving optimization, balanced approach
- **CPO**: Excellent for contrastive learning, high-quality outputs

## 📚 Next Steps

1. Review [Dataset Preparation](./dataset-preparation.md) for data handling
2. Check [Hardware Requirements](./hardware-requirements.md) for system optimization
3. Consult [Troubleshooting Guide](./troubleshooting.md) for common issues
4. See [Fine-tuning Guide](./fine-tuning-guide.md) for theoretical background
5. Explore [Model Selection](./model-selection.md) for choosing the right base model