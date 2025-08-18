# Benchmarking and Performance Optimization Guide

This document provides comprehensive guidance on benchmarking procedures and performance optimization techniques for the Arabic Qwen Base Fine-tuning project.

## Overview

Performance optimization is crucial for efficient training and inference of large language models. This guide covers:

- **Benchmarking Procedures**: Systematic performance measurement
- **Training Optimization**: Memory, speed, and resource optimization
- **Inference Optimization**: Fast and efficient model serving
- **Hardware Utilization**: GPU, CPU, and memory optimization
- **Monitoring and Profiling**: Performance tracking and analysis

## Benchmarking Procedures

### Performance Metrics

#### Training Metrics
- **Training Speed**: Tokens per second, samples per second
- **Memory Usage**: GPU memory, CPU memory, peak usage
- **Throughput**: Batch processing efficiency
- **Convergence Rate**: Loss reduction over time
- **Resource Utilization**: GPU utilization, CPU usage

#### Inference Metrics
- **Latency**: Time per inference request
- **Throughput**: Requests per second
- **Memory Footprint**: Model size in memory
- **Quality Metrics**: BLEU, ROUGE, perplexity
- **Scalability**: Performance under load

### Benchmarking Framework

#### Basic Benchmarking Setup
```python
"""Basic benchmarking utilities for the project."""

import time
import torch
import psutil
import GPUtil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    duration: float
    memory_peak: float
    gpu_memory_peak: float
    gpu_utilization: float
    throughput: float
    additional_metrics: Dict[str, Any]

@contextmanager
def benchmark_context(name: str = "benchmark"):
    """Context manager for benchmarking code blocks."""
    # Setup
    torch.cuda.empty_cache()
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    start_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        yield
    finally:
        # Cleanup and measurement
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        duration = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024**3  # GB
        gpu_memory_used = (end_gpu_memory - start_gpu_memory) / 1024**3  # GB
        
        print(f"{name} completed in {duration:.2f}s")
        print(f"Memory used: {memory_used:.2f}GB")
        print(f"GPU memory used: {gpu_memory_used:.2f}GB")

class PerformanceProfiler:
    """Comprehensive performance profiler."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.gpu_available = torch.cuda.is_available()
    
    def start_profiling(self, name: str):
        """Start profiling a specific operation."""
        self.start_time = time.time()
        self.metrics[name] = {
            'start_time': self.start_time,
            'start_memory': psutil.virtual_memory().used,
            'start_gpu_memory': torch.cuda.memory_allocated() if self.gpu_available else 0
        }
    
    def end_profiling(self, name: str, additional_metrics: Dict[str, Any] = None):
        """End profiling and record results."""
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_gpu_memory = torch.cuda.memory_allocated() if self.gpu_available else 0
        
        if name in self.metrics:
            start_data = self.metrics[name]
            
            self.metrics[name].update({
                'duration': end_time - start_data['start_time'],
                'memory_used': (end_memory - start_data['start_memory']) / 1024**3,
                'gpu_memory_used': (end_gpu_memory - start_data['start_gpu_memory']) / 1024**3,
                'additional_metrics': additional_metrics or {}
            })
    
    def get_results(self) -> Dict[str, BenchmarkResult]:
        """Get formatted benchmark results."""
        results = {}
        for name, data in self.metrics.items():
            if 'duration' in data:
                results[name] = BenchmarkResult(
                    duration=data['duration'],
                    memory_peak=data['memory_used'],
                    gpu_memory_peak=data['gpu_memory_used'],
                    gpu_utilization=0.0,  # Would need GPU monitoring
                    throughput=data['additional_metrics'].get('throughput', 0.0),
                    additional_metrics=data['additional_metrics']
                )
        return results
```

#### Training Benchmarks
```python
"""Training performance benchmarks."""

def benchmark_training_speed(trainer, dataset, num_batches: int = 10):
    """Benchmark training speed."""
    profiler = PerformanceProfiler()
    
    # Warmup
    for i, batch in enumerate(dataset):
        if i >= 3:  # 3 warmup batches
            break
        trainer.training_step(batch)
    
    # Actual benchmark
    profiler.start_profiling('training_speed')
    
    total_tokens = 0
    for i, batch in enumerate(dataset):
        if i >= num_batches:
            break
        
        batch_tokens = batch['input_ids'].numel()
        total_tokens += batch_tokens
        
        trainer.training_step(batch)
    
    profiler.end_profiling('training_speed', {
        'total_tokens': total_tokens,
        'throughput': total_tokens / profiler.metrics['training_speed']['duration']
    })
    
    return profiler.get_results()

def benchmark_memory_usage(trainer, dataset, batch_sizes: list = [1, 2, 4, 8]):
    """Benchmark memory usage across different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        
        # Create batch of specified size
        batch = create_batch(dataset, batch_size)
        
        with benchmark_context(f"batch_size_{batch_size}"):
            try:
                trainer.training_step(batch)
                results[batch_size] = {
                    'success': True,
                    'peak_memory': torch.cuda.max_memory_allocated() / 1024**3
                }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[batch_size] = {
                        'success': False,
                        'error': 'OOM'
                    }
                else:
                    raise
    
    return results
```

#### Inference Benchmarks
```python
"""Inference performance benchmarks."""

def benchmark_inference_latency(model, tokenizer, prompts: list, num_runs: int = 100):
    """Benchmark inference latency."""
    model.eval()
    latencies = []
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            inputs = tokenizer(prompts[0], return_tensors="pt")
            model.generate(**inputs, max_length=50)
    
    # Benchmark
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        
        start_time = time.time()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=50)
        end_time = time.time()
        
        latencies.append(end_time - start_time)
    
    return {
        'mean_latency': sum(latencies) / len(latencies),
        'p50_latency': sorted(latencies)[len(latencies) // 2],
        'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)],
        'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)]
    }

def benchmark_throughput(model, tokenizer, prompts: list, duration: int = 60):
    """Benchmark inference throughput."""
    model.eval()
    
    start_time = time.time()
    total_requests = 0
    total_tokens = 0
    
    while time.time() - start_time < duration:
        prompt = prompts[total_requests % len(prompts)]
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=50)
            
            total_tokens += outputs.shape[1]
            total_requests += 1
    
    actual_duration = time.time() - start_time
    
    return {
        'requests_per_second': total_requests / actual_duration,
        'tokens_per_second': total_tokens / actual_duration,
        'total_requests': total_requests,
        'total_tokens': total_tokens
    }
```

### Automated Benchmarking

#### Benchmark Suite
```python
"""Comprehensive benchmark suite."""

class BenchmarkSuite:
    """Automated benchmark suite for the project."""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.results = {}
    
    def run_all_benchmarks(self):
        """Run all configured benchmarks."""
        print("Starting comprehensive benchmark suite...")
        
        # Training benchmarks
        if self.config.get('training_benchmarks', True):
            self.run_training_benchmarks()
        
        # Inference benchmarks
        if self.config.get('inference_benchmarks', True):
            self.run_inference_benchmarks()
        
        # Memory benchmarks
        if self.config.get('memory_benchmarks', True):
            self.run_memory_benchmarks()
        
        # Generate report
        self.generate_report()
    
    def run_training_benchmarks(self):
        """Run training-specific benchmarks."""
        print("Running training benchmarks...")
        
        # Load training setup
        trainer = self.setup_trainer()
        dataset = self.load_benchmark_dataset()
        
        # Speed benchmark
        speed_results = benchmark_training_speed(trainer, dataset)
        self.results['training_speed'] = speed_results
        
        # Memory benchmark
        memory_results = benchmark_memory_usage(trainer, dataset)
        self.results['training_memory'] = memory_results
    
    def run_inference_benchmarks(self):
        """Run inference-specific benchmarks."""
        print("Running inference benchmarks...")
        
        # Load model for inference
        model, tokenizer = self.setup_inference_model()
        prompts = self.load_benchmark_prompts()
        
        # Latency benchmark
        latency_results = benchmark_inference_latency(model, tokenizer, prompts)
        self.results['inference_latency'] = latency_results
        
        # Throughput benchmark
        throughput_results = benchmark_throughput(model, tokenizer, prompts)
        self.results['inference_throughput'] = throughput_results
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        report_path = f"reports/benchmark_report_{int(time.time())}.json"
        
        report = {
            'timestamp': time.time(),
            'system_info': self.get_system_info(),
            'results': self.results,
            'config': self.config
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Benchmark report saved to {report_path}")
    
    def get_system_info(self):
        """Collect system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
```

## Performance Optimization

### Training Optimization

#### Memory Optimization
```python
"""Memory optimization techniques."""

# Gradient Checkpointing
class OptimizedSFTTrainer(SFTTrainer):
    """SFT Trainer with memory optimizations."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Enable gradient checkpointing
        if config.optimization.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Mixed precision training
        if config.optimization.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def training_step(self, batch):
        """Optimized training step with memory management."""
        # Clear cache before each step
        if self.config.optimization.clear_cache_frequency > 0:
            if self.global_step % self.config.optimization.clear_cache_frequency == 0:
                torch.cuda.empty_cache()
        
        # Mixed precision forward pass
        if self.config.optimization.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Standard training step
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return loss

# Memory-efficient data loading
class OptimizedDataLoader:
    """Memory-optimized data loader."""
    
    def __init__(self, dataset, batch_size, max_length):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
    
    def __iter__(self):
        """Iterate with dynamic batching for memory efficiency."""
        batch = []
        current_max_length = 0
        
        for item in self.dataset:
            item_length = len(item['input_ids'])
            
            # Check if adding this item would exceed memory limits
            new_max_length = max(current_max_length, item_length)
            estimated_memory = len(batch) * new_max_length * 4  # 4 bytes per token
            
            if estimated_memory > self.max_memory_per_batch and batch:
                yield self.collate_batch(batch)
                batch = [item]
                current_max_length = item_length
            else:
                batch.append(item)
                current_max_length = new_max_length
            
            if len(batch) >= self.batch_size:
                yield self.collate_batch(batch)
                batch = []
                current_max_length = 0
        
        if batch:
            yield self.collate_batch(batch)
```

#### Speed Optimization
```python
"""Speed optimization techniques."""

# Optimized model compilation
def optimize_model_for_training(model, config):
    """Apply various optimizations to the model."""
    
    # Torch compile (PyTorch 2.0+)
    if config.optimization.torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    # Flash Attention (if available)
    if config.optimization.flash_attention:
        # Enable flash attention in model config
        model.config.use_flash_attention_2 = True
    
    # Fused optimizers
    if config.optimization.fused_optimizer:
        from torch.optim import AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            fused=True  # Use fused implementation
        )
    
    return model, optimizer

# Efficient data preprocessing
class FastTokenizer:
    """Optimized tokenizer for batch processing."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def batch_encode(self, texts, max_length=512):
        """Efficient batch encoding."""
        # Use fast tokenizer with parallel processing
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            # Enable parallel processing
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        return encoded

# Asynchronous data loading
class AsyncDataLoader:
    """Asynchronous data loader for improved throughput."""
    
    def __init__(self, dataset, batch_size, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Use DataLoader with multiple workers
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2  # Prefetch batches
        )
    
    def __iter__(self):
        for batch in self.dataloader:
            # Move to GPU asynchronously
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            yield batch
```

### Inference Optimization

#### Model Optimization
```python
"""Inference optimization techniques."""

# Model quantization
def quantize_model(model, quantization_type="int8"):
    """Quantize model for faster inference."""
    
    if quantization_type == "int8":
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    elif quantization_type == "int4":
        # 4-bit quantization using bitsandbytes
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with quantization
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model.name_or_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    
    return quantized_model

# KV-cache optimization
class OptimizedInference:
    """Optimized inference with KV-cache management."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        # Enable KV-cache
        self.model.config.use_cache = True
    
    def generate_optimized(self, prompt, max_length=100, **kwargs):
        """Optimized generation with caching."""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate with optimizations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                # Optimization flags
                use_cache=True,
                early_stopping=True,
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate(self, prompts, max_length=100, **kwargs):
        """Optimized batch generation."""
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **kwargs
            )
        
        # Decode all outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            generated_text = self.tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
```

#### Serving Optimization
```python
"""Model serving optimization."""

# Optimized model server
class OptimizedModelServer:
    """High-performance model server."""
    
    def __init__(self, model_path, max_batch_size=8, max_sequence_length=512):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        # Load optimized model
        self.model = self.load_optimized_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Request queue for batching
        self.request_queue = asyncio.Queue()
        self.response_futures = {}
        
        # Start batch processing
        asyncio.create_task(self.batch_processor())
    
    def load_optimized_model(self, model_path):
        """Load model with all optimizations."""
        
        # Load with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Half precision
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply optimizations
        model = torch.compile(model, mode="reduce-overhead")
        model.eval()
        
        return model
    
    async def batch_processor(self):
        """Process requests in batches for efficiency."""
        
        while True:
            batch_requests = []
            
            # Collect requests for batching
            try:
                # Wait for first request
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=0.01  # 10ms timeout
                )
                batch_requests.append(request)
                
                # Collect additional requests (non-blocking)
                while len(batch_requests) < self.max_batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        batch_requests.append(request)
                    except asyncio.QueueEmpty:
                        break
                
                # Process batch
                await self.process_batch(batch_requests)
                
            except asyncio.TimeoutError:
                # No requests, continue
                continue
    
    async def process_batch(self, requests):
        """Process a batch of requests."""
        
        prompts = [req['prompt'] for req in requests]
        request_ids = [req['id'] for req in requests]
        
        # Generate responses
        try:
            responses = self.model.batch_generate(
                prompts,
                max_length=self.max_sequence_length
            )
            
            # Send responses
            for request_id, response in zip(request_ids, responses):
                if request_id in self.response_futures:
                    self.response_futures[request_id].set_result(response)
                    del self.response_futures[request_id]
        
        except Exception as e:
            # Handle errors
            for request_id in request_ids:
                if request_id in self.response_futures:
                    self.response_futures[request_id].set_exception(e)
                    del self.response_futures[request_id]
    
    async def generate(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        self.response_futures[request_id] = future
        
        # Add request to queue
        await self.request_queue.put({
            'id': request_id,
            'prompt': prompt
        })
        
        # Wait for response
        response = await future
        return response
```

## Hardware Optimization

### GPU Optimization

#### Multi-GPU Training
```python
"""Multi-GPU optimization strategies."""

# Data Parallel Training
class MultiGPUTrainer(SFTTrainer):
    """Multi-GPU trainer with optimizations."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Setup multi-GPU
        if torch.cuda.device_count() > 1:
            self.setup_multi_gpu()
    
    def setup_multi_gpu(self):
        """Setup multi-GPU training."""
        
        if self.config.training.strategy == "ddp":
            # Distributed Data Parallel
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Optimization
            )
        
        elif self.config.training.strategy == "dp":
            # Data Parallel (simpler but less efficient)
            self.model = torch.nn.DataParallel(self.model)
        
        elif self.config.training.strategy == "deepspeed":
            # DeepSpeed for very large models
            import deepspeed
            
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config=self.config.deepspeed
            )
    
    def training_step(self, batch):
        """Multi-GPU training step."""
        
        # Move batch to appropriate device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass with gradient synchronization
        if self.config.training.strategy == "ddp":
            # DDP handles gradient synchronization automatically
            loss.backward()
        else:
            # Manual gradient averaging for DP
            loss = loss.mean()  # Average across GPUs
            loss.backward()
        
        return loss

# GPU Memory Management
class GPUMemoryManager:
    """GPU memory optimization utilities."""
    
    @staticmethod
    def optimize_gpu_memory():
        """Apply GPU memory optimizations."""
        
        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory pool
            torch.cuda.empty_cache()
            
            # Set memory growth (if using TensorFlow backend)
            # tf.config.experimental.set_memory_growth(gpu, True)
    
    @staticmethod
    def monitor_gpu_memory():
        """Monitor GPU memory usage."""
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                
                print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager for GPU memory management."""
        
        # Clear cache before operation
        torch.cuda.empty_cache()
        
        try:
            yield
        finally:
            # Clear cache after operation
            torch.cuda.empty_cache()
```

### CPU Optimization

#### Threading and Parallelization
```python
"""CPU optimization techniques."""

# Optimized data loading
class OptimizedCPUDataLoader:
    """CPU-optimized data loader."""
    
    def __init__(self, dataset, batch_size, num_workers=None):
        # Auto-detect optimal number of workers
        if num_workers is None:
            num_workers = min(psutil.cpu_count(), 8)
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def __iter__(self):
        return iter(self.dataloader)

# CPU inference optimization
class CPUOptimizedInference:
    """CPU-optimized inference."""
    
    def __init__(self, model_path):
        # Set optimal thread count
        torch.set_num_threads(psutil.cpu_count())
        
        # Load model for CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"
        )
        
        # Optimize for CPU inference
        self.model = torch.jit.script(self.model)  # TorchScript compilation
    
    def generate(self, prompt, **kwargs):
        """CPU-optimized generation."""
        
        with torch.no_grad():
            # Use CPU-optimized generation
            outputs = self.model.generate(
                prompt,
                **kwargs,
                use_cache=True
            )
        
        return outputs
```

## Monitoring and Profiling

### Performance Monitoring

#### Real-time Monitoring
```python
"""Real-time performance monitoring."""

import wandb
import tensorboard
from torch.profiler import profile, record_function, ProfilerActivity

class PerformanceMonitor:
    """Comprehensive performance monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        
        # Setup monitoring backends
        if config.monitoring.wandb:
            wandb.init(project="arabic-qwen-finetuning")
        
        if config.monitoring.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter("logs/tensorboard")
    
    def log_training_metrics(self, step, metrics):
        """Log training metrics."""
        
        # Log to wandb
        if self.config.monitoring.wandb:
            wandb.log(metrics, step=step)
        
        # Log to tensorboard
        if self.config.monitoring.tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def log_system_metrics(self, step):
        """Log system performance metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # GPU metrics
        gpu_metrics = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_metrics[f'gpu_{i}_memory'] = torch.cuda.memory_allocated(i) / 1024**3
                gpu_metrics[f'gpu_{i}_utilization'] = torch.cuda.utilization(i)
        
        # Combine metrics
        system_metrics = {
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory_percent,
            **gpu_metrics
        }
        
        self.log_training_metrics(step, system_metrics)
    
    @contextmanager
    def profile_context(self, name):
        """Context manager for profiling code sections."""
        
        with record_function(name):
            start_time = time.time()
            yield
            duration = time.time() - start_time
            
            self.metrics[name] = duration

# Advanced profiling
class AdvancedProfiler:
    """Advanced profiling with PyTorch Profiler."""
    
    def __init__(self, output_dir="profiles"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def profile_training(self, trainer, dataset, num_steps=10):
        """Profile training with detailed analysis."""
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            
            for step, batch in enumerate(dataset):
                if step >= num_steps:
                    break
                
                with record_function(f"training_step_{step}"):
                    trainer.training_step(batch)
        
        # Export results
        prof.export_chrome_trace(f"{self.output_dir}/training_trace.json")
        prof.export_stacks(f"{self.output_dir}/training_stacks.txt", "self_cuda_time_total")
        
        # Print summary
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    def profile_inference(self, model, inputs, num_runs=100):
        """Profile inference performance."""
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            
            for i in range(num_runs):
                with record_function(f"inference_{i}"):
                    with torch.no_grad():
                        model(**inputs)
        
        # Export results
        prof.export_chrome_trace(f"{self.output_dir}/inference_trace.json")
        
        # Print summary
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Performance Analysis

#### Bottleneck Detection
```python
"""Performance bottleneck detection and analysis."""

class BottleneckAnalyzer:
    """Analyze performance bottlenecks."""
    
    def __init__(self):
        self.profiling_data = {}
    
    def analyze_training_bottlenecks(self, trainer, dataset):
        """Identify training bottlenecks."""
        
        bottlenecks = {}
        
        # Data loading bottleneck
        data_loading_time = self.measure_data_loading(dataset)
        bottlenecks['data_loading'] = data_loading_time
        
        # Forward pass bottleneck
        forward_time = self.measure_forward_pass(trainer, dataset)
        bottlenecks['forward_pass'] = forward_time
        
        # Backward pass bottleneck
        backward_time = self.measure_backward_pass(trainer, dataset)
        bottlenecks['backward_pass'] = backward_time
        
        # Optimizer step bottleneck
        optimizer_time = self.measure_optimizer_step(trainer)
        bottlenecks['optimizer_step'] = optimizer_time
        
        return self.analyze_bottlenecks(bottlenecks)
    
    def measure_data_loading(self, dataset, num_batches=10):
        """Measure data loading performance."""
        
        start_time = time.time()
        
        for i, batch in enumerate(dataset):
            if i >= num_batches:
                break
        
        total_time = time.time() - start_time
        return total_time / num_batches
    
    def analyze_bottlenecks(self, bottlenecks):
        """Analyze and rank bottlenecks."""
        
        total_time = sum(bottlenecks.values())
        
        analysis = {
            'total_time': total_time,
            'bottleneck_percentages': {
                name: (time / total_time) * 100
                for name, time in bottlenecks.items()
            },
            'primary_bottleneck': max(bottlenecks.items(), key=lambda x: x[1]),
            'recommendations': self.generate_recommendations(bottlenecks)
        }
        
        return analysis
    
    def generate_recommendations(self, bottlenecks):
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Data loading recommendations
        if bottlenecks.get('data_loading', 0) > 0.1:  # >100ms
            recommendations.append({
                'issue': 'Slow data loading',
                'solutions': [
                    'Increase num_workers in DataLoader',
                    'Use pin_memory=True',
                    'Preprocess data offline',
                    'Use faster storage (SSD)'
                ]
            })
        
        # Forward pass recommendations
        if bottlenecks.get('forward_pass', 0) > 0.5:  # >500ms
            recommendations.append({
                'issue': 'Slow forward pass',
                'solutions': [
                    'Enable mixed precision training',
                    'Use gradient checkpointing',
                    'Reduce batch size',
                    'Use model parallelism'
                ]
            })
        
        # Memory recommendations
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_usage > 0.9:
                recommendations.append({
                    'issue': 'High memory usage',
                    'solutions': [
                        'Reduce batch size',
                        'Enable gradient checkpointing',
                        'Use gradient accumulation',
                        'Clear cache more frequently'
                    ]
                })
        
        return recommendations
```

## Configuration for Optimization

### Optimization Configuration
```yaml
# config/optimization.yaml
optimization:
  # Memory optimizations
  gradient_checkpointing: true
  mixed_precision: true
  clear_cache_frequency: 100
  max_memory_per_batch: 2147483648  # 2GB
  
  # Speed optimizations
  torch_compile: true
  flash_attention: true
  fused_optimizer: true
  
  # Multi-GPU settings
  strategy: "ddp"  # ddp, dp, deepspeed
  find_unused_parameters: false
  
  # CPU optimizations
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  
  # Inference optimizations
  quantization:
    enabled: true
    type: "int8"  # int8, int4, fp16
  
  kv_cache: true
  batch_inference: true
  max_batch_size: 8

# Monitoring configuration
monitoring:
  wandb: true
  tensorboard: true
  profiling: true
  system_metrics: true
  
  # Profiling settings
  profile_frequency: 1000  # Every N steps
  profile_duration: 10     # Number of steps to profile
  
  # Alert thresholds
  alerts:
    memory_threshold: 0.9
    gpu_utilization_threshold: 0.8
    training_speed_threshold: 100  # tokens/sec
```

## Best Practices

### Performance Optimization Checklist

#### Training Optimization
- [ ] Enable mixed precision training (AMP)
- [ ] Use gradient checkpointing for large models
- [ ] Optimize batch size for your hardware
- [ ] Use gradient accumulation for effective large batches
- [ ] Enable torch.compile for PyTorch 2.0+
- [ ] Use efficient data loading (multiple workers, pin_memory)
- [ ] Clear GPU cache periodically
- [ ] Use appropriate learning rate scheduling

#### Inference Optimization
- [ ] Quantize models (int8/int4) for production
- [ ] Enable KV-cache for generation
- [ ] Use batch inference when possible
- [ ] Optimize sequence lengths
- [ ] Use TorchScript or ONNX for deployment
- [ ] Implement request batching for serving
- [ ] Use appropriate hardware (GPU/CPU) for workload

#### Memory Optimization
- [ ] Monitor memory usage continuously
- [ ] Use gradient checkpointing for large models
- [ ] Implement dynamic batching
- [ ] Clear unused variables and caches
- [ ] Use memory-efficient optimizers
- [ ] Consider model sharding for very large models

#### Hardware Optimization
- [ ] Use multiple GPUs with DDP when available
- [ ] Optimize CPU thread count
- [ ] Use fast storage (NVMe SSD) for data
- [ ] Ensure adequate RAM for dataset
- [ ] Monitor GPU utilization and memory
- [ ] Use appropriate CUDA versions

### Common Performance Issues

#### Slow Training
1. **Data Loading Bottleneck**: Increase num_workers, use pin_memory
2. **Small Batch Size**: Increase batch size or use gradient accumulation
3. **Inefficient Model**: Use mixed precision, gradient checkpointing
4. **Poor GPU Utilization**: Check batch size, sequence length

#### Memory Issues
1. **OOM Errors**: Reduce batch size, enable gradient checkpointing
2. **Memory Leaks**: Clear caches, delete unused variables
3. **Inefficient Memory Usage**: Use mixed precision, optimize data types

#### Slow Inference
1. **Large Model Size**: Use quantization, model pruning
2. **Inefficient Generation**: Enable KV-cache, optimize sequence length
3. **Single Request Processing**: Implement batch inference
4. **CPU Bottleneck**: Use GPU acceleration, optimize threading

## Related Documentation

- <mcfile name="docs/testing.md" path="docs/testing.md"></mcfile> - Testing procedures including performance tests
- <mcfile name="docs/api/training/index.md" path="docs/api/training/index.md"></mcfile> - Training system documentation
- <mcfile name="docs/api/config/index.md" path="docs/api/config/index.md"></mcfile> - Configuration system
- <mcfile name="CONTRIBUTING.md" path="CONTRIBUTING.md"></mcfile> - Development guidelines
- <mcfile name="requirements.txt" path="requirements.txt"></mcfile> - Project dependencies