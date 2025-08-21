#!/usr/bin/env python3
"""
Setup Validation Script
Tests CUDA availability, model loading, and basic functionality
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuda():
    """Test CUDA availability and setup"""
    print("\n" + "="*50)
    print("CUDA SETUP TEST")
    print("="*50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  Device {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Test basic operations
        try:
            device = torch.device("cuda:0")
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.mm(x, y)
            print(f"CUDA Operations: ✓ Working (result shape: {z.shape})")
            
            # Test memory
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
            print(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
            
        except Exception as e:
            print(f"CUDA Operations: ✗ Failed - {e}")
            return False
    else:
        print("CUDA not available - will use CPU")
        return False
    
    return True

def test_environment():
    """Test environment variables"""
    print("\n" + "="*50)
    print("ENVIRONMENT TEST")
    print("="*50)
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'TORCH_CUDA_ARCH_LIST',
        'PYTORCH_CUDA_ALLOC_CONF',
        'TOKENIZERS_PARALLELISM'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def test_model_loading():
    """Test model loading with the specified configuration"""
    print("\n" + "="*50)
    print("MODEL LOADING TEST")
    print("="*50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "Qwen/Qwen2.5-0.5B"
        print(f"Testing model: {model_name}")
        
        # Test tokenizer loading
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        # Test model loading
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print(f"✓ Model loaded (parameters: {model.num_parameters():,})")
        
        # Test tokenization
        test_text = "مرحبا، كيف حالك؟"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization test passed (tokens: {tokens['input_ids'].shape[1]})")
        
        # Test inference
        if torch.cuda.is_available():
            tokens = tokens.to(model.device)
        
        with torch.no_grad():
            outputs = model(**tokens)
            print(f"✓ Inference test passed (output shape: {outputs.logits.shape})")
        
        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_datasets():
    """Test dataset loading"""
    print("\n" + "="*50)
    print("DATASET TEST")
    print("="*50)
    
    try:
        import json
        
        # Check if datasets exist
        data_dir = Path("./data")
        if not data_dir.exists():
            print("✗ Data directory not found")
            return False
        
        # Test each dataset type
        dataset_files = {
            'SFT': 'sft/arabic_sft_samples.json',
            'DPO': 'dpo/arabic_dpo_samples.json',
            'KTO': 'kto/arabic_kto_samples.json',
            'Evaluation': 'evaluation/arabic_eval_samples.json'
        }
        
        for name, file_path in dataset_files.items():
            full_path = data_dir / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✓ {name}: {len(data)} samples")
            else:
                print(f"✗ {name}: File not found - {full_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n" + "="*50)
    print("DEPENDENCIES TEST")
    print("="*50)
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'peft',
        'trl',
        'wandb',
        'tensorboard',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + ' '.join(missing_packages))
        return False
    
    return True

def main():
    """Run all tests"""
    print("ARABIC QWEN FINE-TUNING SETUP VALIDATION")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Environment", test_environment),
        ("CUDA", test_cuda),
        ("Datasets", test_datasets),
        ("Model Loading", test_model_loading)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for training.")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)