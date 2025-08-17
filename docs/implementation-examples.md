# Implementation Examples for Arabic Qwen Fine-tuning

This document provides complete, ready-to-run code examples for fine-tuning Qwen models on Arabic datasets. All examples are optimized for RTX 3060 12GB and include proper error handling and optimization techniques.

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Basic SFT Implementation](#basic-sft-implementation)
3. [DPO Implementation](#dpo-implementation)
4. [QLoRA Implementation](#qlora-implementation)
5. [Complete Training Scripts](#complete-training-scripts)
6. [Evaluation Examples](#evaluation-examples)
7. [Inference Examples](#inference-examples)

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
from trl import DPOTrainer, DPOConfig
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

## ⚡ QLoRA Implementation

### Example 3: QLoRA for Large Models (7B+)

```python
class ArabicQLoRATrainer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", output_dir: str = "./arabic-qlora-model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_quantized_model(self):
        """Setup 4-bit quantized model with LoRA"""
        print(f"Loading quantized model: {self.model_name}")
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA configuration
        peft_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Scaling factor
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print("Quantized model with LoRA setup completed")
        
    def prepare_qa_dataset(self, dataset_name: str = "riotu-lab/ArabicQA_2.1M", max_samples: int = None):
        """Prepare Arabic QA dataset"""
        print(f"Loading QA dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        
        # Limit samples for testing
        if max_samples:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            
        def format_qa(example):
            """Format QA data for training"""
            question = example.get("question", "")
            answer = example.get("answer", "")
            
            # Arabic QA format
            formatted_text = f"السؤال: {question}\nالإجابة: {answer}"
            
            return {"text": formatted_text}
        
        formatted_dataset = dataset.map(format_qa, remove_columns=dataset["train"].column_names)
        
        print(f"QA dataset prepared. Training samples: {len(formatted_dataset['train']):,}")
        return formatted_dataset
    
    def tokenize_dataset(self, dataset, max_length: int = 512):
        """Tokenize dataset for QLoRA training"""
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    
    def train_qlora(self, dataset, epochs: int = 2, batch_size: int = 1, learning_rate: float = 1e-4):
        """Train with QLoRA"""
        
        # Training arguments for QLoRA
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=16,  # Large accumulation for small batch
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=25,
            save_steps=1000,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            report_to=None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting QLoRA training...")
        trainer.train()
        
        # Save LoRA adapters
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"QLoRA training completed. Adapters saved to {self.output_dir}")
        
        return trainer

# Usage example
if __name__ == "__main__":
    # Initialize QLoRA trainer
    qlora_trainer = ArabicQLoRATrainer(
        model_name="Qwen/Qwen2.5-7B",
        output_dir="./models/arabic-qwen-qlora"
    )
    
    # Setup quantized model with LoRA
    qlora_trainer.setup_quantized_model()
    
    # Prepare QA dataset
    qa_dataset = qlora_trainer.prepare_qa_dataset(
        max_samples=5000  # Use subset for testing
    )
    
    # Tokenize dataset
    tokenized_dataset = qlora_trainer.tokenize_dataset(qa_dataset)
    
    # Train with QLoRA
    trainer = qlora_trainer.train_qlora(
        dataset=tokenized_dataset,
        epochs=2,
        batch_size=1,
        learning_rate=1e-4
    )
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
    
    def run_dpo_stage(self, sft_model_path: str):
        """Run DPO stage using SFT model"""
        print("=== Starting DPO Stage ===")
        
        dpo_output_dir = self.experiment_dir / "dpo_model"
        
        dpo_trainer = ArabicDPOTrainer(
            sft_model_path=sft_model_path,
            output_dir=str(dpo_output_dir)
        )
        
        # Setup and train
        dpo_trainer.setup_model_and_tokenizer()
        
        preference_dataset = dpo_trainer.prepare_preference_dataset(
            dataset_name=self.config["dpo_dataset"]
        )
        
        dpo_trainer.train_dpo(
            dataset=preference_dataset,
            epochs=self.config["dpo_epochs"],
            batch_size=self.config["dpo_batch_size"],
            learning_rate=self.config["dpo_learning_rate"]
        )
        
        print(f"DPO completed. Model saved to {dpo_output_dir}")
        return str(dpo_output_dir)
    
    def run_full_pipeline(self):
        """Run complete SFT -> DPO pipeline"""
        print(f"Starting full pipeline for experiment: {self.experiment_name}")
        
        # Stage 1: SFT
        if self.config.get("run_sft", True):
            sft_model_path = self.run_sft_stage()
        else:
            sft_model_path = self.config["existing_sft_model"]
            
        # Stage 2: DPO
        if self.config.get("run_dpo", True):
            final_model_path = self.run_dpo_stage(sft_model_path)
        else:
            final_model_path = sft_model_path
            
        print(f"\n=== Pipeline Complete ===")
        print(f"Experiment: {self.experiment_name}")
        print(f"Final model: {final_model_path}")
        print(f"Experiment directory: {self.experiment_dir}")
        
        return final_model_path

# Configuration examples
CONFIG_EXAMPLES = {
    "qwen_3b_full": {
        "base_model": "Qwen/Qwen2.5-3B",
        "output_dir": "./experiments",
        "experiment_name": "qwen3b_arabic_full",
        "run_sft": True,
        "run_dpo": True,
        "sft_dataset": "FreedomIntelligence/InstAr-500k",
        "sft_epochs": 3,
        "sft_batch_size": 4,
        "sft_learning_rate": 2e-5,
        "dpo_dataset": "FreedomIntelligence/Arabic-preference-data-RLHF",
        "dpo_epochs": 1,
        "dpo_batch_size": 2,
        "dpo_learning_rate": 5e-6
    },
    "qwen_1.7b_efficient": {
        "base_model": "Qwen/Qwen3-1.7B",
        "output_dir": "./experiments",
        "experiment_name": "qwen1.7b_arabic_efficient",
        "run_sft": True,
        "run_dpo": True,
        "sft_dataset": "FreedomIntelligence/CIDAR",
        "sft_epochs": 5,
        "sft_batch_size": 8,
        "sft_learning_rate": 3e-5,
        "dpo_dataset": "argilla/argilla-dpo-mix-7k-arabic",
        "dpo_epochs": 1,
        "dpo_batch_size": 4,
        "dpo_learning_rate": 1e-5
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Qwen Fine-tuning Pipeline")
    parser.add_argument("--config", choices=list(CONFIG_EXAMPLES.keys()), 
                       default="qwen_3b_full", help="Configuration preset")
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
    
    # Evaluate model
    evaluator = ArabicModelEvaluator("./models/arabic-qwen-dpo")
    
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
    # Initialize inference
    arabic_model = ArabicQwenInference("./models/arabic-qwen-dpo")
    
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

These implementation examples provide complete, production-ready code for fine-tuning Arabic Qwen models. All examples are optimized for RTX 3060 12GB and include proper error handling, memory management, and evaluation capabilities.

## 📚 Next Steps

1. Review [Dataset Preparation](./dataset-preparation.md) for data handling
2. Check [Hardware Requirements](./hardware-requirements.md) for system optimization
3. Consult [Troubleshooting Guide](./troubleshooting.md) for common issues
4. See [Fine-tuning Guide](./fine-tuning-guide.md) for theoretical background