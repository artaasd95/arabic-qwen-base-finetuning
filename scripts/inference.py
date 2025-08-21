#!/usr/bin/env python3
"""
Simple Inference Script for Arabic Qwen Models
Allows users to quickly test their trained models
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

def load_available_models() -> Dict[str, Dict]:
    """Load available trained models"""
    model_paths_file = Path("outputs/model_paths.json")
    
    if not model_paths_file.exists():
        print("❌ No trained models found. Please run training first.")
        return {}
    
    with open(model_paths_file, 'r', encoding='utf-8') as f:
        model_paths = json.load(f)
    
    available_models = {}
    for i, model_info in enumerate(model_paths):
        model_name = model_info['model_name']
        available_models[str(i+1)] = {
            'name': model_name,
            'path': model_info['path'],
            'method': model_info['method'],
            'dataset': model_info['dataset'],
            'samples': model_info['samples_trained']
        }
    
    return available_models

def display_models(models: Dict[str, Dict]):
    """Display available models"""
    print("\n📋 Available Models:")
    print("====================")
    
    for key, model_info in models.items():
        print(f"{key}. {model_info['name']}")
        print(f"   Method: {model_info['method']}")
        print(f"   Dataset: {model_info['dataset']}")
        print(f"   Samples: {model_info['samples']:,}")
        print(f"   Path: {model_info['path']}")
        print()

def mock_inference(model_info: Dict, prompt: str, task_type: str) -> str:
    """Mock inference function (replace with real implementation)"""
    model_name = model_info['name']
    method = model_info['method']
    
    # Mock responses based on method and task type
    if task_type == "chat":
        if "SFT" in method:
            return "مرحباً! أنا نموذج SFT المدرب على البيانات العربية. كيف يمكنني مساعدتك اليوم؟"
        elif "DPO" in method:
            return "أهلاً وسهلاً! أنا نموذج DPO محسّن للاستجابات المفضلة. أسعد بخدمتك."
        elif "KTO" in method:
            return "مرحباً بك! أنا نموذج KTO المحسّن بتقنية كانمان-تفرسكي. كيف أساعدك؟"
        elif "IPO" in method:
            return "أهلاً! أنا نموذج IPO السريع والفعال. ما الذي تحتاج إليه؟"
        elif "CPO" in method:
            return "مرحباً! أنا نموذج CPO المحسّن للفهم العميق. كيف يمكنني المساعدة؟"
    
    elif task_type == "qa":
        return "هذه إجابة تجريبية من النموذج. في التطبيق الحقيقي، سيقوم النموذج بتحليل السؤال وتقديم إجابة مناسبة."
    
    elif task_type == "instruction":
        return "تم تنفيذ التعليمات بنجاح. هذه استجابة تجريبية توضح كيفية عمل النموذج مع التعليمات المختلفة."
    
    elif task_type == "generation":
        return f"هذا نص مُولد من نموذج {method}. النص الحقيقي سيكون أكثر تماسكاً وصلة بالموضوع المطلوب."
    
    return "استجابة تجريبية من النموذج."

def interactive_mode(model_info: Dict):
    """Interactive chat mode"""
    print(f"\n🤖 Interactive Mode - {model_info['name']}")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'خروج']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n📚 Available commands:")
                print("  - Type any message to chat")
                print("  - 'quit' or 'exit' to leave")
                print("  - 'help' for this message")
                continue
            
            if not user_input:
                continue
            
            # Mock inference
            response = mock_inference(model_info, user_input, "chat")
            print(f"🤖 {model_info['method']}: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def single_inference(model_info: Dict, prompt: str, task_type: str):
    """Single inference mode"""
    print(f"\n🔍 Single Inference - {model_info['name']}")
    print(f"Task Type: {task_type}")
    print("=" * 50)
    
    print(f"📝 Input: {prompt}")
    
    # Mock inference
    response = mock_inference(model_info, prompt, task_type)
    print(f"🤖 Output: {response}")
    
    # Show model info
    print(f"\n📊 Model Info:")
    print(f"  Method: {model_info['method']}")
    print(f"  Dataset: {model_info['dataset']}")
    print(f"  Training Samples: {model_info['samples']:,}")

def benchmark_mode(models: Dict[str, Dict], prompt: str):
    """Benchmark multiple models with the same prompt"""
    print(f"\n⚡ Benchmark Mode")
    print(f"Testing prompt: {prompt}")
    print("=" * 50)
    
    for key, model_info in models.items():
        print(f"\n{key}. {model_info['method']} ({model_info['dataset']})")
        response = mock_inference(model_info, prompt, "generation")
        print(f"   Response: {response}")

def main():
    parser = argparse.ArgumentParser(description="Arabic Qwen Model Inference")
    parser.add_argument("--model", type=str, help="Model number to use (1, 2, 3, ...)")
    parser.add_argument("--prompt", type=str, help="Input prompt for inference")
    parser.add_argument("--task", type=str, choices=["chat", "qa", "instruction", "generation"], 
                       default="chat", help="Task type")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all models")
    
    args = parser.parse_args()
    
    # Load available models
    models = load_available_models()
    
    if not models:
        return
    
    print("🚀 Arabic Qwen Model Inference")
    print("==============================")
    
    # Display available models
    display_models(models)
    
    # Benchmark mode
    if args.benchmark:
        if not args.prompt:
            args.prompt = "اكتب فقرة قصيرة عن أهمية التعليم"
        benchmark_mode(models, args.prompt)
        return
    
    # Select model
    if args.model:
        if args.model not in models:
            print(f"❌ Model {args.model} not found. Available models: {list(models.keys())}")
            return
        selected_model = models[args.model]
    else:
        # Interactive model selection
        while True:
            try:
                choice = input("\n🔢 Select a model (number): ").strip()
                if choice in models:
                    selected_model = models[choice]
                    break
                else:
                    print(f"❌ Invalid choice. Please select from: {list(models.keys())}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
    
    print(f"\n✅ Selected: {selected_model['name']}")
    
    # Interactive mode
    if args.interactive or not args.prompt:
        interactive_mode(selected_model)
    else:
        # Single inference
        single_inference(selected_model, args.prompt, args.task)
    
    print("\n📚 Note: This is a demonstration script with mock responses.")
    print("For real inference, install transformers and torch, then update the inference functions.")

if __name__ == "__main__":
    main()