#!/usr/bin/env python3
"""
Arabic Qwen Model Usage Examples
Demonstrates how to use the fine-tuned Arabic Qwen models for various tasks
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional

# Mock imports for demonstration (replace with actual imports when packages are available)
class MockTokenizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
    
    def encode(self, text, return_tensors="pt", max_length=512, truncation=True, padding=True):
        # Mock encoding - returns dummy tensor
        return {"input_ids": torch.randint(0, 1000, (1, min(len(text.split()), max_length)))}
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Mock decoding
        return "هذا مثال على النص المُولد باللغة العربية."

class MockModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cpu"
    
    def generate(self, input_ids, max_length=256, temperature=0.7, do_sample=True, pad_token_id=None):
        # Mock generation - returns dummy tensor
        batch_size, seq_len = input_ids.shape
        new_length = min(max_length, seq_len + 50)
        return torch.randint(0, 1000, (batch_size, new_length))
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self

class ArabicQwenInference:
    """Arabic Qwen Model Inference Class"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the Arabic Qwen model for inference
        
        Args:
            model_path: Path to the trained model
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        
        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        print(f"✓ Loaded model from {model_path} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device for inference"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        # In real implementation, use:
        # from transformers import AutoTokenizer
        # return AutoTokenizer.from_pretrained(self.model_path)
        return MockTokenizer(self.model_path)
    
    def _load_model(self):
        """Load the model"""
        # In real implementation, use:
        # from transformers import AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained(self.model_path)
        # return model.to(self.device).eval()
        return MockModel(self.model_path).to(self.device).eval()
    
    def generate_text(self, 
                     prompt: str, 
                     max_length: int = 256, 
                     temperature: float = 0.7, 
                     do_sample: bool = True,
                     num_return_sequences: int = 1) -> List[str]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 = deterministic)
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Tokenize input
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token.encode() if hasattr(self.tokenizer.pad_token, 'encode') else 0,
                num_return_sequences=num_return_sequences
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the original prompt from the generated text
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            generated_texts.append(text)
        
        return generated_texts
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated response
        """
        # Format messages into a prompt
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        responses = self.generate_text(
            prompt, 
            max_length=512, 
            temperature=0.7, 
            do_sample=True
        )
        
        return responses[0] if responses else ""
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"النظام: {content}")
            elif role == 'user':
                prompt_parts.append(f"المستخدم: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"المساعد: {content}")
        
        prompt_parts.append("المساعد:")
        return "\n".join(prompt_parts)
    
    def question_answering(self, question: str, context: str = "") -> str:
        """
        Answer a question based on optional context
        
        Args:
            question: The question to answer
            context: Optional context for the question
            
        Returns:
            Generated answer
        """
        if context:
            prompt = f"السياق: {context}\n\nالسؤال: {question}\n\nالإجابة:"
        else:
            prompt = f"السؤال: {question}\n\nالإجابة:"
        
        responses = self.generate_text(
            prompt, 
            max_length=256, 
            temperature=0.3,  # Lower temperature for more focused answers
            do_sample=True
        )
        
        return responses[0] if responses else ""
    
    def instruction_following(self, instruction: str, input_text: str = "") -> str:
        """
        Follow an instruction with optional input
        
        Args:
            instruction: The instruction to follow
            input_text: Optional input text
            
        Returns:
            Generated response
        """
        if input_text:
            prompt = f"### التعليمات:\n{instruction}\n\n### المدخل:\n{input_text}\n\n### الاستجابة:\n"
        else:
            prompt = f"### التعليمات:\n{instruction}\n\n### الاستجابة:\n"
        
        responses = self.generate_text(
            prompt, 
            max_length=512, 
            temperature=0.7, 
            do_sample=True
        )
        
        return responses[0] if responses else ""

def load_available_models() -> Dict[str, str]:
    """Load available trained models"""
    model_paths_file = Path("outputs/model_paths.json")
    
    if not model_paths_file.exists():
        print("⚠️  No trained models found. Please run training first.")
        return {}
    
    with open(model_paths_file, 'r', encoding='utf-8') as f:
        model_paths = json.load(f)
    
    available_models = {}
    for model_info in model_paths:
        model_name = model_info['model_name']
        model_path = model_info['path']
        available_models[model_name] = model_path
    
    return available_models

def demonstrate_sft_model():
    """Demonstrate SFT model usage"""
    print("\n=== SFT Model Demonstration ===")
    print("SFT models are best for general instruction following and text generation.")
    
    # Load SFT model (using first available SFT model)
    available_models = load_available_models()
    sft_models = {k: v for k, v in available_models.items() if 'SFT' in k}
    
    if not sft_models:
        print("❌ No SFT models found.")
        return
    
    model_name = list(sft_models.keys())[0]
    model_path = sft_models[model_name]
    
    print(f"Loading model: {model_name}")
    model = ArabicQwenInference(model_path)
    
    # Example 1: Instruction following
    print("\n--- Example 1: Instruction Following ---")
    instruction = "اكتب قصة قصيرة عن طفل يحب القراءة"
    response = model.instruction_following(instruction)
    print(f"التعليمات: {instruction}")
    print(f"الاستجابة: {response}")
    
    # Example 2: Text completion
    print("\n--- Example 2: Text Completion ---")
    prompt = "الذكاء الاصطناعي هو"
    response = model.generate_text(prompt, max_length=200, temperature=0.7)
    print(f"البداية: {prompt}")
    print(f"النص المكتمل: {prompt} {response[0]}")

def demonstrate_dpo_model():
    """Demonstrate DPO model usage"""
    print("\n=== DPO Model Demonstration ===")
    print("DPO models are optimized for preference-aligned responses.")
    
    available_models = load_available_models()
    dpo_models = {k: v for k, v in available_models.items() if 'DPO' in k}
    
    if not dpo_models:
        print("❌ No DPO models found.")
        return
    
    model_name = list(dpo_models.keys())[0]
    model_path = dpo_models[model_name]
    
    print(f"Loading model: {model_name}")
    model = ArabicQwenInference(model_path)
    
    # Example: Chat completion
    print("\n--- Example: Chat Completion ---")
    messages = [
        {"role": "system", "content": "أنت مساعد ذكي ومفيد يتحدث العربية."},
        {"role": "user", "content": "ما هي أفضل طريقة لتعلم لغة جديدة؟"}
    ]
    
    response = model.chat_completion(messages)
    print(f"المستخدم: {messages[1]['content']}")
    print(f"المساعد: {response}")

def demonstrate_kto_model():
    """Demonstrate KTO model usage"""
    print("\n=== KTO Model Demonstration ===")
    print("KTO models use Kahneman-Tversky optimization for better preference learning.")
    
    available_models = load_available_models()
    kto_models = {k: v for k, v in available_models.items() if 'KTO' in k}
    
    if not kto_models:
        print("❌ No KTO models found.")
        return
    
    model_name = list(kto_models.keys())[0]
    model_path = kto_models[model_name]
    
    print(f"Loading model: {model_name}")
    model = ArabicQwenInference(model_path)
    
    # Example: Question answering
    print("\n--- Example: Question Answering ---")
    question = "ما هي عاصمة المملكة العربية السعودية؟"
    answer = model.question_answering(question)
    print(f"السؤال: {question}")
    print(f"الإجابة: {answer}")

def demonstrate_ipo_model():
    """Demonstrate IPO model usage"""
    print("\n=== IPO Model Demonstration ===")
    print("IPO models use Identity Preference Optimization for efficient training.")
    
    available_models = load_available_models()
    ipo_models = {k: v for k, v in available_models.items() if 'IPO' in k}
    
    if not ipo_models:
        print("❌ No IPO models found.")
        return
    
    model_name = list(ipo_models.keys())[0]
    model_path = ipo_models[model_name]
    
    print(f"Loading model: {model_name}")
    model = ArabicQwenInference(model_path)
    
    # Example: Context-based QA
    print("\n--- Example: Context-based Question Answering ---")
    context = "الرياض هي عاصمة المملكة العربية السعودية وأكبر مدنها. تقع في وسط شبه الجزيرة العربية."
    question = "أين تقع الرياض؟"
    answer = model.question_answering(question, context)
    print(f"السياق: {context}")
    print(f"السؤال: {question}")
    print(f"الإجابة: {answer}")

def demonstrate_cpo_model():
    """Demonstrate CPO model usage"""
    print("\n=== CPO Model Demonstration ===")
    print("CPO models use Contrastive Preference Optimization for better understanding.")
    
    available_models = load_available_models()
    cpo_models = {k: v for k, v in available_models.items() if 'CPO' in k}
    
    if not cpo_models:
        print("❌ No CPO models found.")
        return
    
    model_name = list(cpo_models.keys())[0]
    model_path = cpo_models[model_name]
    
    print(f"Loading model: {model_name}")
    model = ArabicQwenInference(model_path)
    
    # Example: Complex instruction
    print("\n--- Example: Complex Instruction Following ---")
    instruction = "قارن بين الطاقة الشمسية والطاقة النووية من ناحية التكلفة والأثر البيئي"
    response = model.instruction_following(instruction)
    print(f"التعليمات: {instruction}")
    print(f"الاستجابة: {response}")

def main():
    """Main demonstration function"""
    print("🚀 Arabic Qwen Model Usage Examples")
    print("====================================")
    
    # Check available models
    available_models = load_available_models()
    
    if not available_models:
        print("❌ No trained models found. Please run training first.")
        return
    
    print(f"\n📊 Found {len(available_models)} trained models:")
    for model_name in available_models.keys():
        print(f"  - {model_name}")
    
    # Demonstrate each model type
    try:
        demonstrate_sft_model()
        demonstrate_dpo_model()
        demonstrate_kto_model()
        demonstrate_ipo_model()
        demonstrate_cpo_model()
        
        print("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Note: This is a mock demonstration. Install transformers and torch for real usage.")
    
    # Usage instructions
    print("\n📚 Usage Instructions:")
    print("1. Install required packages: pip install transformers torch")
    print("2. Replace MockTokenizer and MockModel with real implementations")
    print("3. Use the ArabicQwenInference class for your applications")
    print("4. Adjust generation parameters based on your needs")

if __name__ == "__main__":
    main()