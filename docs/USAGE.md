# Usage Guide

This guide provides comprehensive instructions on how to use the Arabic Qwen fine-tuning framework.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Model Evaluation](#model-evaluation)
- [Inference and Generation](#inference-and-generation)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- 50GB+ free disk space

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/artaasd95/arabic-qwen-base-finetuning.git
cd arabic-qwen-base-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,docs,monitoring]"

# Install pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build Docker image
docker build -t arabic-qwen .

# Run container
docker run --gpus all -p 8000:8000 arabic-qwen
```

## Quick Start

### 1. Prepare Your Environment

```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your Hugging Face token and other settings
```

### 2. Run Training

```bash
# Train all methods with default settings
python scripts/real_training.py

# Or use the command-line interface
arabic-qwen-train --config config/training_config.yaml
```

### 3. Test Inference

```bash
# Interactive inference
python scripts/inference.py --interactive

# Single inference
python scripts/inference.py --model 1 --prompt "اكتب فقرة عن أهمية التعليم"
```

### 4. Upload to Hugging Face

```bash
# Upload trained models
python scripts/upload_to_huggingface.py
```

## Training Models

### Supported Training Methods

1. **SFT (Supervised Fine-Tuning)**: Standard instruction following
2. **DPO (Direct Preference Optimization)**: Preference-based training
3. **KTO (Kahneman-Tversky Optimization)**: Behavioral economics approach
4. **IPO (Identity Preference Optimization)**: Fast preference learning
5. **CPO (Conservative Preference Optimization)**: Stable preference training

### Training Configuration

#### Basic Training

```python
# scripts/train_custom.py
from src.training.sft_trainer import SFTTrainer
from src.data_loader.arabic_datasets import load_arabic_dataset

# Load dataset
dataset = load_arabic_dataset("arabic-instruct")

# Initialize trainer
trainer = SFTTrainer(
    model_name="Qwen/Qwen2.5-3B",
    output_dir="outputs/sft_model",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=4
)

# Train model
trainer.train(dataset)
```

#### Advanced Training with Custom Config

```yaml
# config/custom_training.yaml
model:
  name: "Qwen/Qwen2.5-3B"
  torch_dtype: "float16"
  device_map: "auto"

training:
  methods: ["SFT", "DPO", "KTO"]
  num_epochs: 3
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  save_steps: 500
  eval_steps: 250
  logging_steps: 50

datasets:
  - name: "arabic-instruct"
    type: "instruction"
    samples: 10000
  - name: "arabic-chat"
    type: "conversation"
    samples: 8000

optimization:
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  use_gradient_checkpointing: true
  fp16: true
```

```bash
# Train with custom config
python scripts/train_all_methods.py --config config/custom_training.yaml
```

### Dataset Preparation

#### Using Built-in Datasets

```python
from src.data_loader.arabic_datasets import ArabicDatasetLoader

# Load predefined datasets
loader = ArabicDatasetLoader()

# Available datasets
datasets = {
    "arabic-instruct": "FreedomIntelligence/Alpaca-Arabic-GPT4",
    "arabic-chat": "2A2I/argilla-dpo-mix-7k-arabic",
    "arabic-qa": "sadeem-ai/arabic-qna"
}

# Load specific dataset
dataset = loader.load_dataset("arabic-instruct")
```

#### Using Custom Datasets

```python
# Custom dataset format for SFT
sft_data = [
    {
        "instruction": "اكتب قصة قصيرة",
        "input": "",
        "output": "كان يا ما كان في قديم الزمان..."
    },
    # More examples...
]

# Custom dataset format for DPO
dpo_data = [
    {
        "prompt": "ما هي عاصمة مصر؟",
        "chosen": "عاصمة مصر هي القاهرة",
        "rejected": "لا أعرف"
    },
    # More examples...
]
```

### Training Monitoring

#### Weights & Biases Integration

```python
import wandb

# Initialize wandb
wandb.init(
    project="arabic-qwen-finetuning",
    name="sft-experiment-1",
    config={
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 4
    }
)

# Training with wandb logging
trainer = SFTTrainer(
    model_name="Qwen/Qwen2.5-3B",
    output_dir="outputs/sft_model",
    report_to="wandb"
)
```

#### TensorBoard Logging

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs

# View at http://localhost:6006
```

## Model Evaluation

### Automatic Evaluation

```python
from src.evaluation.evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate model
results = evaluator.evaluate(
    model_path="outputs/models/sft_model",
    test_dataset="arabic_test_set",
    metrics=["bleu", "rouge", "perplexity"]
)

print(f"BLEU Score: {results['bleu']}")
print(f"ROUGE-L: {results['rouge']['rougeL']}")
print(f"Perplexity: {results['perplexity']}")
```

### Manual Evaluation

```python
# Compare different models
from scripts.inference import load_available_models

models = load_available_models()
test_prompts = [
    "اكتب فقرة عن أهمية التعليم",
    "ما هي فوائد القراءة؟",
    "اشرح مفهوم الذكاء الاصطناعي"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    for model_id, model_info in models.items():
        response = generate_response(model_info, prompt)
        print(f"{model_info['method']}: {response[:100]}...")
```

### Benchmark Evaluation

```bash
# Run comprehensive benchmarks
python scripts/evaluate_models.py --benchmark all

# Specific benchmark
python scripts/evaluate_models.py --benchmark arabic_glue

# Custom evaluation
python scripts/evaluate_models.py --test_file data/custom_test.json
```

## Inference and Generation

### Basic Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_name = "your-org/qwen-3-base-arabic-instruct-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
response = generate_text("اكتب قصة قصيرة عن الصداقة")
print(response)
```

### Chat Interface

```python
from scripts.inference import interactive_mode
from scripts.inference import load_available_models

# Load models
models = load_available_models()

# Select model (e.g., first SFT model)
sft_models = [m for m in models.values() if m['method'] == 'SFT']
selected_model = sft_models[0] if sft_models else list(models.values())[0]

# Start interactive chat
interactive_mode(selected_model)
```

### Batch Processing

```python
def process_batch(prompts, model, tokenizer, batch_size=4):
    """Process multiple prompts efficiently"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results

# Usage
prompts = [
    "اكتب عن الطبيعة",
    "ما هي فوائد الرياضة؟",
    "اشرح الفيزياء الكمية"
]

results = process_batch(prompts, model, tokenizer)
for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Response: {result}\n")
```

## Configuration

### Environment Variables

```bash
# .env file
HUGGINGFACE_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_key_here
OUTPUT_DIR=outputs
CACHE_DIR=cache
LOG_LEVEL=INFO
DEVICE=cuda
MODEL_MAX_LENGTH=2048
BATCH_SIZE=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3
```

### Training Configuration

```yaml
# config/training_config.yaml
model:
  base_model: "Qwen/Qwen2.5-3B"
  torch_dtype: "float16"
  device_map: "auto"
  trust_remote_code: true

training:
  output_dir: "outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
  logging_steps: 50
  save_steps: 500
  eval_steps: 250
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  evaluation_strategy: "steps"
  save_strategy: "steps"
  fp16: true
  dataloader_num_workers: 4
  remove_unused_columns: false
  report_to: ["wandb", "tensorboard"]

lora:
  use_lora: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

datasets:
  arabic_instruct:
    name: "FreedomIntelligence/Alpaca-Arabic-GPT4"
    split: "train"
    max_samples: 10000
    text_column: "text"
  
  arabic_chat:
    name: "2A2I/argilla-dpo-mix-7k-arabic"
    split: "train"
    max_samples: 8000
    text_column: "text"

generation:
  max_length: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true
  pad_token_id: null
  eos_token_id: null
```

### Model Configuration

```yaml
# config/model_config.yaml
qwen:
  model_name: "Qwen/Qwen2.5-3B"
  revision: "main"
  torch_dtype: "float16"
  device_map: "auto"
  low_cpu_mem_usage: true
  trust_remote_code: true
  
  # Quantization settings
  quantization:
    load_in_8bit: false
    load_in_4bit: false
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_use_double_quant: true
    bnb_4bit_quant_type: "nf4"
  
  # Generation settings
  generation:
    max_new_tokens: 512
    temperature: 0.7
    top_p: 0.9
    top_k: 50
    repetition_penalty: 1.1
    do_sample: true
    num_beams: 1
    early_stopping: false
```

## Advanced Usage

### Custom Training Loop

```python
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from src.models.qwen_model import QwenForCausalLM
from src.data_loader.base_loader import BaseDataLoader

class CustomTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup optimizer
        self.optimizer = AdamW(model.parameters(), lr=2e-5)
        
        # Setup scheduler
        num_training_steps = len(train_dataset) * 3  # 3 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        
        for batch in train_loader:
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        eval_loader = DataLoader(self.eval_dataset, batch_size=4)
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
        
        return total_loss / len(eval_loader)
    
    def train(self, num_epochs=3):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            eval_loss = self.evaluate()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Eval Loss: {eval_loss:.4f}")
            
            # Save checkpoint
            self.model.save_pretrained(f"outputs/checkpoint-epoch-{epoch+1}")
```

### Multi-GPU Training

```python
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Prepare model, optimizer, and data
model, optimizer, train_loader, eval_loader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
    # Save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(f"outputs/epoch-{epoch}")
```

### Custom Data Processing

```python
from datasets import Dataset
from src.data_loader.preprocessing import ArabicTextProcessor

class CustomDataProcessor:
    def __init__(self):
        self.processor = ArabicTextProcessor()
    
    def process_instruction_data(self, raw_data):
        """Process instruction-following data"""
        processed_data = []
        
        for item in raw_data:
            # Clean and normalize Arabic text
            instruction = self.processor.clean_text(item['instruction'])
            response = self.processor.clean_text(item['response'])
            
            # Format for training
            formatted_text = f"### التعليمات:\n{instruction}\n\n### الاستجابة:\n{response}"
            
            processed_data.append({
                'text': formatted_text,
                'instruction': instruction,
                'response': response
            })
        
        return Dataset.from_list(processed_data)
    
    def process_chat_data(self, conversations):
        """Process conversational data"""
        processed_data = []
        
        for conversation in conversations:
            formatted_conversation = ""
            
            for turn in conversation['messages']:
                role = turn['role']
                content = self.processor.clean_text(turn['content'])
                
                if role == 'user':
                    formatted_conversation += f"المستخدم: {content}\n"
                elif role == 'assistant':
                    formatted_conversation += f"المساعد: {content}\n"
            
            processed_data.append({'text': formatted_conversation})
        
        return Dataset.from_list(processed_data)
```

## Examples

### Example 1: Fine-tune for Question Answering

```python
# examples/qa_finetuning.py
from src.training.sft_trainer import SFTTrainer
from datasets import load_dataset

# Load Arabic QA dataset
dataset = load_dataset("sadeem-ai/arabic-qna")

# Prepare data for QA format
def format_qa_data(examples):
    formatted_texts = []
    for question, answer in zip(examples['question'], examples['answer']):
        text = f"السؤال: {question}\nالجواب: {answer}"
        formatted_texts.append(text)
    return {'text': formatted_texts}

train_dataset = dataset['train'].map(format_qa_data, batched=True)

# Initialize trainer
trainer = SFTTrainer(
    model_name="Qwen/Qwen2.5-3B",
    output_dir="outputs/qa_model",
    num_train_epochs=3,
    learning_rate=2e-5
)

# Train
trainer.train(train_dataset)
```

### Example 2: Preference Optimization with DPO

```python
# examples/dpo_training.py
from src.training.dpo_trainer import DPOTrainer
from datasets import Dataset

# Prepare preference data
preference_data = [
    {
        "prompt": "اكتب قصة قصيرة",
        "chosen": "كان يا ما كان، في قديم الزمان، عاش ملك عادل...",
        "rejected": "قصة. النهاية."
    },
    # More examples...
]

dataset = Dataset.from_list(preference_data)

# Initialize DPO trainer
trainer = DPOTrainer(
    model_name="outputs/sft_model",  # Use SFT model as base
    output_dir="outputs/dpo_model",
    beta=0.1,  # DPO temperature parameter
    learning_rate=5e-7
)

# Train with preference optimization
trainer.train(dataset)
```

### Example 3: Multi-turn Conversation

```python
# examples/conversation_example.py
from transformers import AutoTokenizer, AutoModelForCausalLM

class ConversationManager:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.conversation_history = []
    
    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
    
    def generate_response(self, user_message):
        # Add user message
        self.add_message("user", user_message)
        
        # Format conversation
        conversation_text = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                conversation_text += f"المستخدم: {msg['content']}\n"
            else:
                conversation_text += f"المساعد: {msg['content']}\n"
        
        conversation_text += "المساعد: "
        
        # Generate response
        inputs = self.tokenizer(conversation_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response[len(conversation_text):].strip()
        
        # Add assistant response to history
        self.add_message("assistant", assistant_response)
        
        return assistant_response

# Usage
conversation = ConversationManager("outputs/chat_model")

response1 = conversation.generate_response("مرحباً، كيف حالك؟")
print(f"المساعد: {response1}")

response2 = conversation.generate_response("ما هي هواياتك؟")
print(f"المساعد: {response2}")
```

### Example 4: Batch Evaluation

```python
# examples/batch_evaluation.py
import json
from src.evaluation.metrics import calculate_bleu, calculate_rouge

def evaluate_model_on_dataset(model_path, test_file):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    predictions = []
    references = []
    
    for item in test_data:
        prompt = item['prompt']
        reference = item['reference']
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction[len(prompt):].strip()
        
        predictions.append(prediction)
        references.append(reference)
    
    # Calculate metrics
    bleu_score = calculate_bleu(predictions, references)
    rouge_scores = calculate_rouge(predictions, references)
    
    return {
        'bleu': bleu_score,
        'rouge': rouge_scores,
        'num_samples': len(test_data)
    }

# Usage
results = evaluate_model_on_dataset(
    "outputs/sft_model", 
    "data/arabic_test.json"
)

print(f"BLEU Score: {results['bleu']:.4f}")
print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
```

## Best Practices

1. **Data Quality**: Ensure high-quality, diverse Arabic training data
2. **Preprocessing**: Properly clean and normalize Arabic text
3. **Evaluation**: Use multiple metrics and human evaluation
4. **Monitoring**: Track training progress with logging tools
5. **Checkpointing**: Save regular checkpoints during training
6. **Validation**: Use held-out validation sets for model selection
7. **Documentation**: Document your experiments and configurations
8. **Version Control**: Track model versions and training configurations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size, use gradient accumulation
2. **Slow Training**: Enable mixed precision, use multiple GPUs
3. **Poor Quality**: Adjust hyperparameters, increase training data
4. **Convergence Issues**: Check learning rate, add warmup steps

### Getting Help

- Check the [FAQ](FAQ.md)
- Review [GitHub Issues](https://github.com/artaasd95/arabic-qwen-base-finetuning/issues)
- Join our [Discord Community](https://discord.gg/arabic-nlp)
- Read the [API Reference](API_REFERENCE.md)

## Next Steps

- [Deployment Guide](DEPLOYMENT.md)
- [API Reference](API_REFERENCE.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Advanced Examples](../examples/)