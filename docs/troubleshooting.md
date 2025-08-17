# Troubleshooting Guide for Arabic Qwen Fine-tuning

This comprehensive troubleshooting guide addresses common issues encountered during Arabic Qwen model fine-tuning, providing practical solutions and preventive measures.

## 📋 Table of Contents

1. [Memory Issues](#memory-issues)
2. [Training Problems](#training-problems)
3. [Model Loading Errors](#model-loading-errors)
4. [Dataset Issues](#dataset-issues)
5. [Performance Problems](#performance-problems)
6. [Arabic Text Issues](#arabic-text-issues)
7. [Hardware-Specific Issues](#hardware-specific-issues)
8. [Environment Setup Problems](#environment-setup-problems)
9. [Best Practices](#best-practices)

## 🧠 Memory Issues

### CUDA Out of Memory (OOM) Errors

#### Problem: `RuntimeError: CUDA out of memory`

**Common Causes:**
- Batch size too large
- Sequence length too long
- Model too large for available VRAM
- Memory leaks from previous runs

**Solutions:**

```python
# Solution 1: Reduce batch size and increase gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Reduce from 4 to 1
    gradient_accumulation_steps=16,  # Increase to maintain effective batch size
    dataloader_pin_memory=False,     # Reduce system RAM usage
    remove_unused_columns=False,     # Keep for debugging
)

# Solution 2: Enable gradient checkpointing
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Trade compute for memory
    fp16=True,                   # Use half precision
    tf32=True,                   # Use TensorFloat-32 on Ampere GPUs
)

# Solution 3: Use QLoRA for large models
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Solution 4: Clear cache and restart
import torch
torch.cuda.empty_cache()
import gc
gc.collect()

# Solution 5: Reduce sequence length
max_seq_length = 512  # Reduce from 2048
tokenizer.model_max_length = max_seq_length
```

#### Memory Monitoring Script

```python
import torch
import psutil
import time

class MemoryMonitor:
    def __init__(self, alert_threshold_gpu=0.9, alert_threshold_ram=0.85):
        self.alert_threshold_gpu = alert_threshold_gpu
        self.alert_threshold_ram = alert_threshold_ram
        self.monitoring = False
        
    def get_memory_info(self):
        """Get current memory usage"""
        info = {}
        
        # GPU Memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            total_gpu = torch.cuda.get_device_properties(0).total_memory
            allocated = gpu_memory['allocated_bytes.all.current']
            reserved = gpu_memory['reserved_bytes.all.current']
            
            info['gpu'] = {
                'allocated_gb': allocated / 1024**3,
                'reserved_gb': reserved / 1024**3,
                'total_gb': total_gpu / 1024**3,
                'utilization': reserved / total_gpu
            }
        
        # System RAM
        ram = psutil.virtual_memory()
        info['ram'] = {
            'used_gb': ram.used / 1024**3,
            'total_gb': ram.total / 1024**3,
            'utilization': ram.percent / 100
        }
        
        return info
    
    def check_memory_alerts(self):
        """Check for memory alerts"""
        info = self.get_memory_info()
        alerts = []
        
        if 'gpu' in info and info['gpu']['utilization'] > self.alert_threshold_gpu:
            alerts.append(f"⚠️ GPU memory high: {info['gpu']['utilization']*100:.1f}%")
            
        if info['ram']['utilization'] > self.alert_threshold_ram:
            alerts.append(f"⚠️ RAM usage high: {info['ram']['utilization']*100:.1f}%")
            
        return alerts
    
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        print("🚨 Performing emergency memory cleanup...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear Python cache
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
            
        print("✅ Emergency cleanup completed")
        
        # Show new memory status
        info = self.get_memory_info()
        if 'gpu' in info:
            print(f"GPU: {info['gpu']['allocated_gb']:.1f}GB / {info['gpu']['total_gb']:.1f}GB")
        print(f"RAM: {info['ram']['used_gb']:.1f}GB / {info['ram']['total_gb']:.1f}GB")

# Usage
monitor = MemoryMonitor()

# Check memory before training
alerts = monitor.check_memory_alerts()
if alerts:
    for alert in alerts:
        print(alert)
    monitor.emergency_cleanup()

# Use in training loop
def training_step_with_monitoring(model, batch, step):
    try:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Check memory every 50 steps
        if step % 50 == 0:
            alerts = monitor.check_memory_alerts()
            if alerts:
                print(f"Step {step}: {alerts}")
                
        return loss
        
    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM at step {step}")
        monitor.emergency_cleanup()
        raise
```

### System RAM Issues

#### Problem: System runs out of RAM during data loading

**Solutions:**

```python
# Solution 1: Optimize DataLoader
from torch.utils.data import DataLoader

# Reduce number of workers
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=2,  # Reduce from 8 to 2
    pin_memory=False,  # Disable pin memory
    persistent_workers=False  # Don't keep workers alive
)

# Solution 2: Use streaming datasets for large data
from datasets import load_dataset

# Load dataset in streaming mode
dataset = load_dataset(
    "FreedomIntelligence/InstAr-500k",
    streaming=True  # Don't load entire dataset into memory
)

# Solution 3: Implement custom data loading with memory management
class MemoryEfficientDataset:
    def __init__(self, dataset_path, max_cache_size=1000):
        self.dataset_path = dataset_path
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.cache_order = []
        
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        # Load item
        item = self._load_item(idx)
        
        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
            
        self.cache[idx] = item
        self.cache_order.append(idx)
        
        return item
    
    def _load_item(self, idx):
        # Implement actual data loading
        pass
```

## 🏋️ Training Problems

### Loss Not Decreasing

#### Problem: Training loss remains constant or increases

**Diagnostic Steps:**

```python
# Diagnostic script for training issues
class TrainingDiagnostic:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        
    def check_data_quality(self, num_samples=10):
        """Check if data is properly formatted"""
        print("=== Data Quality Check ===")
        
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            
            print(f"\nSample {i}:")
            print(f"Keys: {list(sample.keys())}")
            
            if 'text' in sample:
                text = sample['text']
                print(f"Text length: {len(text)} chars")
                print(f"First 100 chars: {text[:100]}...")
                
                # Check tokenization
                tokens = self.tokenizer(text, return_tensors="pt")
                print(f"Token count: {tokens.input_ids.shape[1]}")
                
                # Check for special tokens
                decoded = self.tokenizer.decode(tokens.input_ids[0])
                print(f"Decoded matches original: {text.strip() == decoded.strip()}")
    
    def check_model_gradients(self, sample_batch):
        """Check if gradients are flowing properly"""
        print("\n=== Gradient Check ===")
        
        self.model.train()
        
        # Forward pass
        outputs = self.model(**sample_batch, labels=sample_batch['input_ids'])
        loss = outputs.loss
        
        print(f"Loss value: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                if param_count <= 5:  # Show first 5 parameters
                    print(f"{name}: grad_norm = {param_norm:.6f}")
                    
        total_norm = total_norm ** (1. / 2)
        print(f"Total gradient norm: {total_norm:.6f}")
        print(f"Parameters with gradients: {param_count}")
        
        if total_norm < 1e-7:
            print("⚠️ Warning: Very small gradients detected")
        elif total_norm > 100:
            print("⚠️ Warning: Very large gradients detected (possible exploding gradients)")
    
    def check_learning_rate(self, optimizer):
        """Check learning rate settings"""
        print("\n=== Learning Rate Check ===")
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            print(f"Parameter group {i}: lr = {lr}")
            
            if lr > 1e-3:
                print(f"⚠️ Warning: Learning rate {lr} might be too high")
            elif lr < 1e-6:
                print(f"⚠️ Warning: Learning rate {lr} might be too low")
    
    def run_full_diagnostic(self, optimizer, num_samples=5):
        """Run complete diagnostic"""
        print("🔍 Running Training Diagnostic...\n")
        
        # Check data
        self.check_data_quality(num_samples)
        
        # Prepare sample batch
        sample_texts = [self.dataset[i]['text'] for i in range(min(2, len(self.dataset)))]
        sample_batch = self.tokenizer(
            sample_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            sample_batch = {k: v.cuda() for k, v in sample_batch.items()}
        
        # Check gradients
        self.check_model_gradients(sample_batch)
        
        # Check learning rate
        self.check_learning_rate(optimizer)
        
        print("\n✅ Diagnostic complete")

# Usage
diagnostic = TrainingDiagnostic(model, tokenizer, train_dataset)
diagnostic.run_full_diagnostic(optimizer)
```

**Common Solutions:**

```python
# Solution 1: Adjust learning rate
training_args = TrainingArguments(
    learning_rate=2e-5,  # Try different values: 1e-5, 5e-5, 1e-4
    lr_scheduler_type="cosine",  # Use cosine scheduler
    warmup_steps=100,  # Add warmup
)

# Solution 2: Check data preprocessing
def verify_data_format(dataset, tokenizer):
    """Verify data is properly formatted"""
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    if 'text' in sample:
        text = sample['text']
        print(f"Sample text: {text[:200]}...")
        
        # Check if text contains instruction format
        if "### التعليمات:" not in text and "### Instruction:" not in text:
            print("⚠️ Warning: Text doesn't seem to be in instruction format")
            
        # Check tokenization
        tokens = tokenizer(text, return_tensors="pt")
        if tokens.input_ids.shape[1] < 10:
            print("⚠️ Warning: Very short tokenized sequence")

# Solution 3: Implement loss tracking
class LossTracker:
    def __init__(self, window_size=100):
        self.losses = []
        self.window_size = window_size
        
    def add_loss(self, loss):
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
            
    def get_trend(self):
        if len(self.losses) < 10:
            return "insufficient_data"
            
        recent = self.losses[-10:]
        older = self.losses[-20:-10] if len(self.losses) >= 20 else self.losses[:-10]
        
        if not older:
            return "insufficient_data"
            
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg < older_avg * 0.95:
            return "decreasing"
        elif recent_avg > older_avg * 1.05:
            return "increasing"
        else:
            return "stable"
    
    def should_stop_training(self):
        trend = self.get_trend()
        return trend == "increasing" and len(self.losses) > 50

# Usage in training loop
loss_tracker = LossTracker()

for step, batch in enumerate(dataloader):
    outputs = model(**batch, labels=batch['input_ids'])
    loss = outputs.loss
    
    loss_tracker.add_loss(loss.item())
    
    if step % 50 == 0:
        trend = loss_tracker.get_trend()
        print(f"Step {step}, Loss: {loss.item():.4f}, Trend: {trend}")
        
        if loss_tracker.should_stop_training():
            print("⚠️ Loss increasing consistently, consider stopping")
```

### Gradient Explosion/Vanishing

#### Problem: Gradients become too large or too small

**Solutions:**

```python
# Solution 1: Gradient clipping
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Clip gradients to prevent explosion
    gradient_checkpointing=True,
)

# Solution 2: Custom gradient monitoring
class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.grad_norms = []
        
    def monitor_gradients(self):
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
        total_norm = total_norm ** (1. / 2)
        self.grad_norms.append(total_norm)
        
        return total_norm
    
    def check_gradient_health(self):
        if not self.grad_norms:
            return "no_data"
            
        recent_norms = self.grad_norms[-10:]
        avg_norm = sum(recent_norms) / len(recent_norms)
        
        if avg_norm > 100:
            return "exploding"
        elif avg_norm < 1e-6:
            return "vanishing"
        else:
            return "healthy"

# Usage
grad_monitor = GradientMonitor(model)

# In training loop
loss.backward()
grad_norm = grad_monitor.monitor_gradients()

if step % 100 == 0:
    health = grad_monitor.check_gradient_health()
    print(f"Gradient health: {health}, norm: {grad_norm:.6f}")
    
    if health == "exploding":
        print("⚠️ Reducing learning rate due to exploding gradients")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
    elif health == "vanishing":
        print("⚠️ Increasing learning rate due to vanishing gradients")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 1.5

optimizer.step()
optimizer.zero_grad()
```

## 📁 Model Loading Errors

### HuggingFace Model Loading Issues

#### Problem: `OSError: Can't load tokenizer/model`

**Solutions:**

```python
# Solution 1: Clear cache and retry
import os
from transformers.utils import TRANSFORMERS_CACHE

def clear_transformers_cache():
    """Clear HuggingFace transformers cache"""
    import shutil
    if os.path.exists(TRANSFORMERS_CACHE):
        shutil.rmtree(TRANSFORMERS_CACHE)
        print(f"Cleared cache: {TRANSFORMERS_CACHE}")

# Solution 2: Manual download with error handling
def safe_model_loading(model_name, max_retries=3):
    """Safely load model with retries"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time
    
    for attempt in range(max_retries):
        try:
            print(f"Loading attempt {attempt + 1}/{max_retries}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="./cache"
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./cache"
            )
            
            print(f"✅ Successfully loaded {model_name}")
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print("Waiting 10 seconds before retry...")
                time.sleep(10)
                
                # Clear cache on failure
                if "cache" in str(e).lower():
                    clear_transformers_cache()
            else:
                print("All attempts failed")
                raise e

# Usage
try:
    tokenizer, model = safe_model_loading("Qwen/Qwen2.5-3B")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback to different model or manual download

# Solution 3: Offline model loading
def setup_offline_model(model_name, local_path):
    """Setup model for offline use"""
    import os
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(local_path):
        print(f"Downloading {model_name} to {local_path}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
    
    # Load from local path
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model
```

### Tokenizer Issues

#### Problem: Tokenizer doesn't handle Arabic text properly

**Solutions:**

```python
# Solution 1: Verify tokenizer configuration
def check_tokenizer_arabic_support(tokenizer):
    """Check if tokenizer properly handles Arabic"""
    test_texts = [
        "مرحبا بك في عالم الذكاء الاصطناعي",
        "هذا نص تجريبي باللغة العربية",
        "السؤال: ما هو الذكاء الاصطناعي؟"
    ]
    
    print("=== Tokenizer Arabic Support Check ===")
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt")
        token_ids = tokens.input_ids[0]
        
        print(f"Token count: {len(token_ids)}")
        print(f"Token IDs: {token_ids[:10].tolist()}...")  # First 10 tokens
        
        # Decode
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"Decoded: {decoded}")
        
        # Check if decoding matches original
        matches = text.strip() == decoded.strip()
        print(f"Decoding matches: {matches}")
        
        if not matches:
            print("⚠️ Warning: Tokenization/decoding mismatch")
            
        # Check for unknown tokens
        unk_token_id = tokenizer.unk_token_id
        if unk_token_id and unk_token_id in token_ids:
            unk_count = (token_ids == unk_token_id).sum().item()
            print(f"⚠️ Warning: {unk_count} unknown tokens found")

# Solution 2: Custom tokenizer setup for Arabic
def setup_arabic_tokenizer(model_name):
    """Setup tokenizer optimized for Arabic"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add Arabic-specific tokens if needed
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
        "unk_token": "<|unknown|>"
    }
    
    # Add tokens that might be missing
    tokens_to_add = []
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            tokens_to_add.append(token)
            setattr(tokenizer, token_type, token)
    
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        print(f"Added special tokens: {tokens_to_add}")
    
    # Set reasonable defaults
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Usage
tokenizer = setup_arabic_tokenizer("Qwen/Qwen2.5-3B")
check_tokenizer_arabic_support(tokenizer)
```

## 📊 Dataset Issues

### Data Format Problems

#### Problem: Dataset not in expected format

**Solutions:**

```python
# Solution 1: Dataset format validator
class DatasetValidator:
    def __init__(self):
        self.required_formats = {
            "instruction": ["instruction", "output"],
            "chat": ["messages"],
            "text": ["text"]
        }
    
    def detect_format(self, dataset):
        """Detect dataset format"""
        if len(dataset) == 0:
            return "empty"
            
        sample = dataset[0]
        sample_keys = set(sample.keys())
        
        for format_name, required_keys in self.required_formats.items():
            if all(key in sample_keys for key in required_keys):
                return format_name
                
        return "unknown"
    
    def validate_dataset(self, dataset, expected_format=None):
        """Validate dataset format and content"""
        print("=== Dataset Validation ===")
        
        # Basic checks
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ Error: Empty dataset")
            return False
            
        # Format detection
        detected_format = self.detect_format(dataset)
        print(f"Detected format: {detected_format}")
        
        if expected_format and detected_format != expected_format:
            print(f"⚠️ Warning: Expected {expected_format}, got {detected_format}")
        
        # Sample validation
        issues = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            sample_issues = self.validate_sample(sample, detected_format)
            if sample_issues:
                issues.extend([f"Sample {i}: {issue}" for issue in sample_issues])
        
        if issues:
            print(f"❌ Found {len(issues)} issues:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
            return False
        else:
            print("✅ Dataset validation passed")
            return True
    
    def validate_sample(self, sample, format_type):
        """Validate individual sample"""
        issues = []
        
        if format_type == "instruction":
            if "instruction" not in sample or not sample["instruction"].strip():
                issues.append("Empty or missing instruction")
            if "output" not in sample or not sample["output"].strip():
                issues.append("Empty or missing output")
                
        elif format_type == "text":
            if "text" not in sample or not sample["text"].strip():
                issues.append("Empty or missing text")
            elif len(sample["text"]) < 10:
                issues.append("Text too short")
                
        return issues
    
    def fix_common_issues(self, dataset):
        """Fix common dataset issues"""
        print("=== Fixing Common Issues ===")
        
        fixed_dataset = []
        skipped_count = 0
        
        for i, sample in enumerate(dataset):
            # Skip empty samples
            if not any(v for v in sample.values() if isinstance(v, str) and v.strip()):
                skipped_count += 1
                continue
                
            # Fix text encoding issues
            fixed_sample = {}
            for key, value in sample.items():
                if isinstance(value, str):
                    # Remove excessive whitespace
                    value = ' '.join(value.split())
                    # Remove null characters
                    value = value.replace('\x00', '')
                    
                fixed_sample[key] = value
                
            fixed_dataset.append(fixed_sample)
        
        print(f"Skipped {skipped_count} empty samples")
        print(f"Fixed dataset size: {len(fixed_dataset)}")
        
        return fixed_dataset

# Usage
validator = DatasetValidator()

# Validate dataset
is_valid = validator.validate_dataset(train_dataset, expected_format="instruction")

if not is_valid:
    print("Attempting to fix issues...")
    fixed_dataset = validator.fix_common_issues(train_dataset)
    # Convert back to HuggingFace dataset
    from datasets import Dataset
    train_dataset = Dataset.from_list(fixed_dataset)
```

### Arabic Text Encoding Issues

#### Problem: Arabic text appears corrupted or as question marks

**Solutions:**

```python
# Solution 1: Text encoding fixer
import unicodedata
import re

class ArabicTextFixer:
    def __init__(self):
        # Arabic Unicode ranges
        self.arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]
        
    def is_arabic_char(self, char):
        """Check if character is Arabic"""
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.arabic_ranges)
    
    def detect_encoding_issues(self, text):
        """Detect common encoding issues"""
        issues = []
        
        # Check for question marks (encoding failure)
        if '?' in text and any(self.is_arabic_char(c) for c in text):
            issues.append("question_marks_with_arabic")
            
        # Check for mixed RTL/LTR issues
        if '\u202E' in text or '\u202D' in text:
            issues.append("rtl_ltr_override")
            
        # Check for broken Unicode
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("unicode_encode_error")
            
        return issues
    
    def fix_text_encoding(self, text):
        """Fix common Arabic text encoding issues"""
        if not isinstance(text, str):
            return str(text)
            
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove RTL/LTR override characters
        text = text.replace('\u202E', '').replace('\u202D', '')
        
        # Remove zero-width characters that can cause issues
        text = text.replace('\u200B', '')  # Zero-width space
        text = text.replace('\u200C', '')  # Zero-width non-joiner
        text = text.replace('\u200D', '')  # Zero-width joiner
        
        # Fix common character substitutions
        replacements = {
            'Ø§': 'ا',  # Fix common encoding issue
            'Ù…': 'م',  # Fix common encoding issue
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
            
        return text
    
    def fix_dataset_encoding(self, dataset):
        """Fix encoding issues in entire dataset"""
        print("=== Fixing Arabic Text Encoding ===")
        
        fixed_count = 0
        total_issues = 0
        
        def fix_sample(sample):
            nonlocal fixed_count, total_issues
            
            fixed_sample = {}
            sample_fixed = False
            
            for key, value in sample.items():
                if isinstance(value, str):
                    issues = self.detect_encoding_issues(value)
                    if issues:
                        total_issues += len(issues)
                        fixed_value = self.fix_text_encoding(value)
                        fixed_sample[key] = fixed_value
                        sample_fixed = True
                    else:
                        fixed_sample[key] = value
                else:
                    fixed_sample[key] = value
                    
            if sample_fixed:
                fixed_count += 1
                
            return fixed_sample
        
        fixed_dataset = dataset.map(fix_sample)
        
        print(f"Fixed {fixed_count} samples with {total_issues} total issues")
        return fixed_dataset

# Usage
text_fixer = ArabicTextFixer()

# Fix single text
fixed_text = text_fixer.fix_text_encoding("مرحبا بك في عالم الذكاء الاصطناعي")

# Fix entire dataset
fixed_dataset = text_fixer.fix_dataset_encoding(train_dataset)
```

## ⚡ Performance Problems

### Slow Training Speed

#### Problem: Training is much slower than expected

**Diagnostic and Solutions:**

```python
# Performance profiler
import time
import torch.profiler

class TrainingProfiler:
    def __init__(self):
        self.step_times = []
        self.bottlenecks = {}
        
    def profile_training_step(self, model, batch, optimizer):
        """Profile a single training step"""
        timings = {}
        
        # Data loading time (should be measured outside this function)
        start_time = time.time()
        
        # Forward pass
        forward_start = time.time()
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        timings['forward'] = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        timings['backward'] = time.time() - backward_start
        
        # Optimizer step
        optimizer_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        timings['optimizer'] = time.time() - optimizer_start
        
        # Total time
        timings['total'] = time.time() - start_time
        
        return timings, loss.item()
    
    def analyze_bottlenecks(self, timings_list):
        """Analyze performance bottlenecks"""
        if not timings_list:
            return
            
        avg_timings = {}
        for key in timings_list[0].keys():
            avg_timings[key] = sum(t[key] for t in timings_list) / len(timings_list)
            
        print("=== Performance Analysis ===")
        print(f"Average step time: {avg_timings['total']:.3f}s")
        print(f"Forward pass: {avg_timings['forward']:.3f}s ({avg_timings['forward']/avg_timings['total']*100:.1f}%)")
        print(f"Backward pass: {avg_timings['backward']:.3f}s ({avg_timings['backward']/avg_timings['total']*100:.1f}%)")
        print(f"Optimizer: {avg_timings['optimizer']:.3f}s ({avg_timings['optimizer']/avg_timings['total']*100:.1f}%)")
        
        # Identify bottlenecks
        if avg_timings['forward'] > avg_timings['total'] * 0.6:
            print("⚠️ Bottleneck: Forward pass (consider reducing model size or sequence length)")
        if avg_timings['backward'] > avg_timings['total'] * 0.4:
            print("⚠️ Bottleneck: Backward pass (consider gradient checkpointing)")
        if avg_timings['optimizer'] > avg_timings['total'] * 0.1:
            print("⚠️ Bottleneck: Optimizer (consider different optimizer or learning rate)")
    
    def suggest_optimizations(self, avg_step_time, target_time=1.0):
        """Suggest performance optimizations"""
        print("\n=== Optimization Suggestions ===")
        
        if avg_step_time > target_time:
            speedup_needed = avg_step_time / target_time
            print(f"Need {speedup_needed:.1f}x speedup")
            
            suggestions = [
                "Reduce batch size and increase gradient accumulation",
                "Reduce sequence length (max_length parameter)",
                "Enable gradient checkpointing",
                "Use mixed precision training (fp16)",
                "Consider QLoRA instead of full fine-tuning",
                "Optimize DataLoader (reduce num_workers, disable pin_memory)",
                "Use faster optimizer (AdamW with fused=True)"
            ]
            
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")

# Usage
profiler = TrainingProfiler()
timings_list = []

# Profile several steps
for step, batch in enumerate(dataloader):
    if step >= 10:  # Profile first 10 steps
        break
        
    timings, loss = profiler.profile_training_step(model, batch, optimizer)
    timings_list.append(timings)
    
    if step % 5 == 0:
        print(f"Step {step}: {timings['total']:.3f}s, Loss: {loss:.4f}")

# Analyze results
profiler.analyze_bottlenecks(timings_list)
avg_time = sum(t['total'] for t in timings_list) / len(timings_list)
profiler.suggest_optimizations(avg_time)
```

### GPU Utilization Issues

#### Problem: Low GPU utilization during training

**Solutions:**

```python
# GPU utilization monitor
import subprocess
import threading
import time

class GPUMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.monitoring = False
        self.utilization_history = []
        
    def get_gpu_utilization(self):
        """Get current GPU utilization"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = []
                
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_util = int(parts[0])
                        mem_used = int(parts[1])
                        mem_total = int(parts[2])
                        mem_util = (mem_used / mem_total) * 100
                        
                        gpu_data.append({
                            'gpu_utilization': gpu_util,
                            'memory_utilization': mem_util,
                            'memory_used_mb': mem_used,
                            'memory_total_mb': mem_total
                        })
                        
                return gpu_data
        except Exception as e:
            print(f"Error getting GPU utilization: {e}")
            
        return []
    
    def start_monitoring(self):
        """Start continuous GPU monitoring"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                gpu_data = self.get_gpu_utilization()
                if gpu_data:
                    self.utilization_history.append({
                        'timestamp': time.time(),
                        'gpus': gpu_data
                    })
                    
                    # Keep only last 100 measurements
                    if len(self.utilization_history) > 100:
                        self.utilization_history.pop(0)
                        
                time.sleep(self.interval)
                
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        
    def analyze_utilization(self):
        """Analyze GPU utilization patterns"""
        if not self.utilization_history:
            print("No utilization data available")
            return
            
        print("=== GPU Utilization Analysis ===")
        
        # Calculate averages
        gpu_utils = []
        mem_utils = []
        
        for record in self.utilization_history:
            for gpu in record['gpus']:
                gpu_utils.append(gpu['gpu_utilization'])
                mem_utils.append(gpu['memory_utilization'])
                
        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            avg_mem_util = sum(mem_utils) / len(mem_utils)
            
            print(f"Average GPU utilization: {avg_gpu_util:.1f}%")
            print(f"Average memory utilization: {avg_mem_util:.1f}%")
            
            # Provide recommendations
            if avg_gpu_util < 70:
                print("⚠️ Low GPU utilization detected")
                print("Suggestions:")
                print("- Increase batch size")
                print("- Reduce DataLoader num_workers")
                print("- Check for CPU bottlenecks")
                print("- Ensure data is on GPU")
                
            if avg_mem_util < 80:
                print("⚠️ Low memory utilization")
                print("Suggestions:")
                print("- Increase batch size")
                print("- Increase sequence length")
                print("- Use larger model if possible")

# Usage
gpu_monitor = GPUMonitor(interval=2)
gpu_monitor.start_monitoring()

# Run training for a while
time.sleep(60)  # Monitor for 1 minute

gpu_monitor.stop_monitoring()
gpu_monitor.analyze_utilization()

# Optimization based on utilization
def optimize_based_on_utilization():
    """Optimize training based on GPU utilization"""
    gpu_data = gpu_monitor.get_gpu_utilization()
    
    if gpu_data:
        gpu_util = gpu_data[0]['gpu_utilization']
        mem_util = gpu_data[0]['memory_utilization']
        
        print(f"Current GPU utilization: {gpu_util}%")
        print(f"Current memory utilization: {mem_util}%")
        
        # Dynamic batch size adjustment
        if gpu_util < 60 and mem_util < 70:
            print("💡 Consider increasing batch size")
            return "increase_batch_size"
        elif mem_util > 95:
            print("⚠️ Memory usage very high, consider reducing batch size")
            return "decrease_batch_size"
        else:
            print("✅ GPU utilization looks good")
            return "optimal"
    
    return "unknown"
```

## 🔤 Arabic Text Issues

### Text Direction and Display Problems

#### Problem: Arabic text displays incorrectly or in wrong direction

**Solutions:**

```python
# Arabic text handler
import unicodedata
from bidi.algorithm import get_display

class ArabicTextHandler:
    def __init__(self):
        self.arabic_reshaper = None
        try:
            import arabic_reshaper
            self.arabic_reshaper = arabic_reshaper.ArabicReshaper()
        except ImportError:
            print("arabic_reshaper not installed. Install with: pip install arabic-reshaper")
    
    def fix_arabic_display(self, text):
        """Fix Arabic text for proper display"""
        if not text:
            return text
            
        # Reshape Arabic text if reshaper is available
        if self.arabic_reshaper:
            reshaped_text = self.arabic_reshaper.reshape(text)
            # Apply bidirectional algorithm
            display_text = get_display(reshaped_text)
            return display_text
        else:
            # Basic bidirectional handling
            return get_display(text)
    
    def normalize_arabic_text(self, text):
        """Normalize Arabic text for processing"""
        if not text:
            return text
            
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove or normalize diacritics if needed
        # text = self.remove_diacritics(text)
        
        # Fix common character issues
        replacements = {
            'ي': 'ي',  # Normalize different forms of yeh
            'ك': 'ك',  # Normalize different forms of kaf
            'ة': 'ة',  # Normalize teh marbuta
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def remove_diacritics(self, text):
        """Remove Arabic diacritics (tashkeel)"""
        import re
        # Arabic diacritics Unicode range
        diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        return diacritics_pattern.sub('', text)
    
    def validate_arabic_content(self, text):
        """Validate Arabic text content"""
        if not text:
            return False, "Empty text"
            
        # Count Arabic characters
        arabic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if '\u0600' <= char <= '\u06FF':
                    arabic_chars += 1
                    
        if total_chars == 0:
            return False, "No alphabetic characters"
            
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio < 0.5:
            return False, f"Low Arabic content: {arabic_ratio:.2f}"
            
        return True, f"Valid Arabic content: {arabic_ratio:.2f}"

# Usage
arabic_handler = ArabicTextHandler()

# Fix display issues
original_text = "مرحبا بك في عالم الذكاء الاصطناعي"
fixed_text = arabic_handler.fix_arabic_display(original_text)
print(f"Original: {original_text}")
print(f"Fixed: {fixed_text}")

# Validate content
is_valid, message = arabic_handler.validate_arabic_content(original_text)
print(f"Validation: {is_valid}, {message}")

# Process dataset
def fix_arabic_in_dataset(dataset):
    """Fix Arabic text issues in dataset"""
    handler = ArabicTextHandler()
    
    def fix_sample(sample):
        fixed_sample = {}
        for key, value in sample.items():
            if isinstance(value, str) and value.strip():
                # Normalize and fix display
                normalized = handler.normalize_arabic_text(value)
                fixed_sample[key] = normalized
            else:
                fixed_sample[key] = value
        return fixed_sample
    
    return dataset.map(fix_sample)
```

## 🖥️ Hardware-Specific Issues

### RTX 3060 Specific Problems

#### Problem: Model doesn't fit in 12GB VRAM

**Solutions:**

```python
# RTX 3060 optimization strategies
class RTX3060Optimizer:
    def __init__(self):
        self.vram_limit = 12 * 1024 * 1024 * 1024  # 12GB in bytes
        self.safe_vram_limit = int(self.vram_limit * 0.9)  # Use 90% to be safe
        
    def estimate_model_memory(self, model_name):
        """Estimate memory requirements for different configurations"""
        
        # Model parameter counts (approximate)
        model_params = {
            "Qwen/Qwen2.5-3B": 3.0,
            "Qwen/Qwen2.5-7B": 7.0,
            "Qwen/Qwen2.5-14B": 14.0,
        }
        
        if model_name not in model_params:
            print(f"Unknown model: {model_name}")
            return None
            
        params_b = model_params[model_name]
        
        # Memory estimates in GB
        estimates = {
            "fp32_full": params_b * 4 * 4,  # 4 bytes per param, 4x for gradients/optimizer
            "fp16_full": params_b * 2 * 4,  # 2 bytes per param, 4x for gradients/optimizer
            "fp16_lora": params_b * 2 + 0.5,  # Model in fp16 + small LoRA params
            "4bit_qlora": params_b * 0.5 + 0.5,  # 4-bit model + LoRA params
        }
        
        return estimates
    
    def recommend_configuration(self, model_name):
        """Recommend optimal configuration for RTX 3060"""
        estimates = self.estimate_model_memory(model_name)
        
        if not estimates:
            return None
            
        vram_gb = self.safe_vram_limit / (1024**3)
        
        print(f"=== RTX 3060 Configuration for {model_name} ===")
        print(f"Available VRAM: {vram_gb:.1f}GB")
        print("\nMemory estimates:")
        
        recommendations = []
        
        for config, memory_gb in estimates.items():
            fits = memory_gb <= vram_gb
            status = "✅" if fits else "❌"
            print(f"{status} {config}: {memory_gb:.1f}GB")
            
            if fits:
                recommendations.append({
                    "config": config,
                    "memory_gb": memory_gb,
                    "utilization": memory_gb / vram_gb
                })
        
        if recommendations:
            # Sort by memory utilization (prefer higher utilization)
            recommendations.sort(key=lambda x: x["utilization"], reverse=True)
            best = recommendations[0]
            
            print(f"\n🎯 Recommended: {best['config']}")
            print(f"Memory usage: {best['memory_gb']:.1f}GB ({best['utilization']*100:.1f}%)")
            
            return self.get_training_config(best['config'], model_name)
        else:
            print("\n❌ No configuration fits in available VRAM")
            return None
    
    def get_training_config(self, config_type, model_name):
        """Get specific training configuration"""
        
        base_config = {
            "model_name": model_name,
            "output_dir": "./results",
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "warmup_steps": 100,
            "max_steps": 1000,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
        }
        
        if config_type == "fp16_full":
            config = {
                **base_config,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-5,
                "fp16": True,
                "gradient_checkpointing": True,
                "max_seq_length": 1024,
            }
            
        elif config_type == "fp16_lora":
            config = {
                **base_config,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1e-4,
                "fp16": True,
                "max_seq_length": 2048,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            }
            
        elif config_type == "4bit_qlora":
            config = {
                **base_config,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1e-4,
                "max_seq_length": 2048,
                "use_qlora": True,
                "qlora_r": 16,
                "qlora_alpha": 32,
                "qlora_dropout": 0.1,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
            }
        else:
            config = base_config
            
        return config

# Usage
optimizer = RTX3060Optimizer()

# Get recommendations for different models
models = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]

for model in models:
    config = optimizer.recommend_configuration(model)
    if config:
        print(f"\nTraining configuration for {model}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    print("\n" + "="*50)
```

## 🛠️ Environment Setup Problems

### Dependency Conflicts

#### Problem: Package version conflicts

**Solutions:**

```python
# Environment validator and fixer
import subprocess
import sys
import pkg_resources

class EnvironmentValidator:
    def __init__(self):
        self.required_packages = {
            "torch": ">=2.0.0",
            "transformers": ">=4.35.0",
            "datasets": ">=2.14.0",
            "accelerate": ">=0.20.0",
            "peft": ">=0.5.0",
            "bitsandbytes": ">=0.41.0",
        }
        
    def check_package_versions(self):
        """Check if required packages are installed with correct versions"""
        print("=== Package Version Check ===")
        
        issues = []
        
        for package, version_req in self.required_packages.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                print(f"✅ {package}: {installed_version}")
                
                # Check if version meets requirement
                if not self.version_satisfies(installed_version, version_req):
                    issues.append(f"{package} version {installed_version} doesn't meet {version_req}")
                    
            except pkg_resources.DistributionNotFound:
                print(f"❌ {package}: Not installed")
                issues.append(f"{package} is not installed")
                
        return issues
    
    def version_satisfies(self, installed, required):
        """Check if installed version satisfies requirement"""
        # Simple version comparison (for demonstration)
        # In practice, use packaging.version for proper comparison
        try:
            from packaging import version
            if required.startswith(">="):
                min_version = required[2:]
                return version.parse(installed) >= version.parse(min_version)
        except ImportError:
            # Fallback to simple string comparison
            return True
        except Exception:
            return False
    
    def fix_environment(self):
        """Attempt to fix environment issues"""
        issues = self.check_package_versions()
        
        if not issues:
            print("✅ Environment looks good!")
            return True
            
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
            
        print("\n🔧 Attempting to fix...")
        
        # Generate pip install commands
        install_commands = []
        for package, version_req in self.required_packages.items():
            if version_req.startswith(">="):
                min_version = version_req[2:]
                install_commands.append(f"pip install {package}>={min_version}")
            else:
                install_commands.append(f"pip install {package}")
                
        print("Run these commands to fix the environment:")
        for cmd in install_commands:
            print(f"  {cmd}")
            
        return False
    
    def check_cuda_compatibility(self):
        """Check CUDA compatibility"""
        print("\n=== CUDA Compatibility Check ===")
        
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU count: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"GPU {i}: {props.name}")
                    print(f"  Memory: {props.total_memory / 1024**3:.1f}GB")
                    print(f"  Compute capability: {props.major}.{props.minor}")
            else:
                print("❌ CUDA not available")
                print("Possible solutions:")
                print("  - Install CUDA-enabled PyTorch")
                print("  - Check NVIDIA drivers")
                print("  - Verify CUDA installation")
                
        except ImportError:
            print("❌ PyTorch not installed")

# Usage
validator = EnvironmentValidator()
validator.fix_environment()
validator.check_cuda_compatibility()
```

## 🎯 Best Practices

### Training Best Practices

1. **Start Small**: Begin with smaller models and datasets to validate your pipeline
2. **Monitor Everything**: Track loss, GPU utilization, memory usage, and training speed
3. **Save Frequently**: Use checkpointing to avoid losing progress
4. **Validate Early**: Run evaluation on small validation sets during training
5. **Document Everything**: Keep detailed logs of configurations and results

### Memory Management

```python
# Memory management best practices
class MemoryManager:
    @staticmethod
    def cleanup_memory():
        """Comprehensive memory cleanup"""
        import gc
        import torch
        
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Memory cleanup completed")
    
    @staticmethod
    def get_memory_summary():
        """Get detailed memory summary"""
        import torch
        import psutil
        
        summary = {}
        
        # System memory
        ram = psutil.virtual_memory()
        summary['system_ram'] = {
            'total_gb': ram.total / 1024**3,
            'used_gb': ram.used / 1024**3,
            'available_gb': ram.available / 1024**3,
            'percent': ram.percent
        }
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            summary['gpu_memory'] = {
                'total_gb': total_memory / 1024**3,
                'allocated_gb': gpu_memory['allocated_bytes.all.current'] / 1024**3,
                'reserved_gb': gpu_memory['reserved_bytes.all.current'] / 1024**3,
                'free_gb': (total_memory - gpu_memory['reserved_bytes.all.current']) / 1024**3
            }
        
        return summary

# Use before and after training
memory_manager = MemoryManager()
print("Before training:")
print(memory_manager.get_memory_summary())

# ... training code ...

print("After training:")
print(memory_manager.get_memory_summary())
memory_manager.cleanup_memory()
```

### Error Recovery

```python
# Robust training with error recovery
class RobustTrainer:
    def __init__(self, model, tokenizer, train_dataset, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.checkpoint_dir = f"{output_dir}/checkpoints"
        
    def train_with_recovery(self, training_args):
        """Train with automatic error recovery"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Training attempt {retry_count + 1}/{max_retries}")
                
                # Setup trainer
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    tokenizer=self.tokenizer,
                )
                
                # Start training
                trainer.train()
                
                # Save final model
                trainer.save_model(self.output_dir)
                print("✅ Training completed successfully")
                return True
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ CUDA OOM Error: {e}")
                
                # Cleanup and reduce batch size
                MemoryManager.cleanup_memory()
                training_args.per_device_train_batch_size = max(1, training_args.per_device_train_batch_size // 2)
                training_args.gradient_accumulation_steps *= 2
                
                print(f"Reduced batch size to {training_args.per_device_train_batch_size}")
                retry_count += 1
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                
                # Try to recover from checkpoint
                if self.try_resume_from_checkpoint():
                    print("Resumed from checkpoint")
                    retry_count += 1
                else:
                    print("Could not recover from checkpoint")
                    break
        
        print(f"❌ Training failed after {max_retries} attempts")
        return False
    
    def try_resume_from_checkpoint(self):
        """Try to resume from latest checkpoint"""
        import os
        
        if os.path.exists(self.checkpoint_dir):
            checkpoints = [d for d in os.listdir(self.checkpoint_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
                print(f"Found checkpoint: {checkpoint_path}")
                return True
        
        return False
```

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs**: Look for specific error messages and stack traces
2. **Search GitHub Issues**: Check the repositories of transformers, peft, and bitsandbytes
3. **Community Forums**: Ask on HuggingFace forums or Reddit r/MachineLearning
4. **Documentation**: Refer to official documentation for the libraries you're using

### Useful Commands for Debugging

```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi

# Check Python package versions
pip list | grep -E "torch|transformers|datasets|accelerate|peft|bitsandbytes"

# Check CUDA version
nvcc --version

# Check system memory
free -h

# Monitor system resources
htop
```

---

**Remember**: Fine-tuning Arabic models requires patience and experimentation. Start with smaller models and datasets, monitor your resources carefully, and don't hesitate to adjust configurations based on your hardware limitations.