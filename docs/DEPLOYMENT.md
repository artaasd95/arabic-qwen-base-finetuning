# Model Deployment Guide

This guide covers how to deploy and use the fine-tuned Arabic Qwen models in various environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Model Loading](#model-loading)
- [Inference Methods](#inference-methods)
- [Deployment Options](#deployment-options)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch transformers accelerate

# For quantization support
pip install bitsandbytes optimum

# For serving
pip install fastapi uvicorn gradio
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "your-org/qwen-3-base-arabic-instruct-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text
input_text = "اكتب فقرة عن أهمية التعليم"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Model Loading

### Standard Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load with default settings
model = AutoModelForCausalLM.from_pretrained("model_path")
tokenizer = AutoTokenizer.from_pretrained("model_path")
```

### Optimized Loading

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load with optimizations
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",          # Automatic device placement
    low_cpu_mem_usage=True,     # Reduce CPU memory usage
    trust_remote_code=True      # Allow custom code
)
```

### Quantized Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    quantization_config=quantization_config,
    device_map="auto"
)

# 4-bit quantization (more aggressive)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

## Inference Methods

### Text Generation

```python
def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """Generate text using the model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Usage
prompt = "ما هي فوائد القراءة؟"
response = generate_text(model, tokenizer, prompt)
print(response)
```

### Chat Interface

```python
def chat_with_model(model, tokenizer, messages, max_length=500):
    """Chat interface for conversation"""
    # Format messages for chat
    chat_template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(chat_template, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(chat_template):].strip()

# Usage
messages = [
    {"role": "user", "content": "مرحباً، كيف حالك؟"}
]
response = chat_with_model(model, tokenizer, messages)
print(response)
```

### Batch Inference

```python
def batch_inference(model, tokenizer, prompts, batch_size=4):
    """Process multiple prompts in batches"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode results
        batch_results = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        results.extend(batch_results)
    
    return results
```

## Deployment Options

### 1. Local Deployment

#### Simple Script

```python
# deploy_local.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ArabicQwenModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, prompt, max_length=200, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model = ArabicQwenModel("path/to/model")
    
    while True:
        prompt = input("أدخل النص: ")
        if prompt.lower() in ['quit', 'exit', 'خروج']:
            break
        
        response = model.generate(prompt)
        print(f"الاستجابة: {response}")
```

### 2. FastAPI Web Service

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI(title="Arabic Qwen API", version="1.0.0")

# Load model at startup
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = "path/to/model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Gradio Interface

```python
# gradio_app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_path = "path/to/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_response(prompt, max_length, temperature, top_p):
    """Generate response using the model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="النص المدخل", placeholder="اكتب نصك هنا..."),
        gr.Slider(50, 500, value=200, label="الطول الأقصى"),
        gr.Slider(0.1, 2.0, value=0.7, label="درجة الحرارة"),
        gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
    ],
    outputs=gr.Textbox(label="النص المولد"),
    title="نموذج Qwen العربي",
    description="نموذج لغوي عربي مدرب على بيانات عربية"
)

if __name__ == "__main__":
    iface.launch(share=True)
```

### 4. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  arabic-qwen-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/qwen-arabic
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Performance Optimization

### 1. Memory Optimization

```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    outputs = model.generate(**inputs)
```

### 2. Speed Optimization

```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use optimized attention
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path)
config.use_flash_attention_2 = True
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    config=config
)
```

### 3. Batch Processing

```python
# Process multiple requests efficiently
def batch_generate(prompts, batch_size=4):
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```python
# Solution 1: Use quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Solution 2: Reduce batch size
batch_size = 1

# Solution 3: Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    offload_folder="offload"
)
```

#### 2. Slow Inference

```python
# Solution 1: Use half precision
model = model.half()

# Solution 2: Optimize generation parameters
outputs = model.generate(
    **inputs,
    max_length=100,  # Reduce max length
    num_beams=1,     # Use greedy search
    do_sample=False  # Disable sampling
)
```

#### 3. Poor Quality Output

```python
# Solution: Adjust generation parameters
outputs = model.generate(
    **inputs,
    temperature=0.8,        # Adjust creativity
    top_p=0.9,             # Nucleus sampling
    repetition_penalty=1.2, # Reduce repetition
    length_penalty=1.0      # Control length preference
)
```

### Monitoring and Logging

```python
import logging
import time
from functools import wraps

def monitor_inference(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logging.info(f"Inference completed in {end_time - start_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Inference failed: {str(e)}")
            raise
    
    return wrapper

@monitor_inference
def generate_text(prompt):
    # Your generation code here
    pass
```

## Best Practices

1. **Model Loading**: Load models once at startup, not per request
2. **Memory Management**: Use appropriate quantization for your hardware
3. **Batch Processing**: Process multiple requests together when possible
4. **Caching**: Cache frequently used responses
5. **Monitoring**: Monitor memory usage, inference time, and quality
6. **Error Handling**: Implement robust error handling and fallbacks
7. **Security**: Validate inputs and implement rate limiting
8. **Documentation**: Keep deployment documentation up to date

## Next Steps

- [Model Usage Examples](../examples/model_usage_examples.py)
- [API Reference](API_REFERENCE.md)
- [Performance Benchmarks](BENCHMARKS.md)
- [Contributing Guidelines](../CONTRIBUTING.md)