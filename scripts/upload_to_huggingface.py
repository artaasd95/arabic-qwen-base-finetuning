#!/usr/bin/env python3
"""
Upload Script for Arabic Qwen Base Fine-tuned Models
Uploads all trained models to Hugging Face Hub with proper model cards
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError
import shutil

MODEL_CONFIGS = {
    "sft": {
        "repo_name": "arabic-qwen-sft",
        "description": "Arabic Qwen model fine-tuned using Supervised Fine-Tuning (SFT)",
        "tags": ["arabic", "qwen", "sft", "supervised-fine-tuning", "text-generation", "conversational"]
    },
    "dpo": {
        "repo_name": "arabic-qwen-dpo",
        "description": "Arabic Qwen model fine-tuned using Direct Preference Optimization (DPO)",
        "tags": ["arabic", "qwen", "dpo", "direct-preference-optimization", "text-generation", "rlhf"]
    },
    "kto": {
        "repo_name": "arabic-qwen-kto",
        "description": "Arabic Qwen model fine-tuned using Kahneman-Tversky Optimization (KTO)",
        "tags": ["arabic", "qwen", "kto", "kahneman-tversky-optimization", "text-generation", "preference-learning"]
    },
    "ipo": {
        "repo_name": "arabic-qwen-ipo",
        "description": "Arabic Qwen model fine-tuned using Identity Preference Optimization (IPO)",
        "tags": ["arabic", "qwen", "ipo", "identity-preference-optimization", "text-generation", "preference-learning"]
    },
    "cpo": {
        "repo_name": "arabic-qwen-cpo",
        "description": "Arabic Qwen model fine-tuned using Contrastive Preference Optimization (CPO)",
        "tags": ["arabic", "qwen", "cpo", "contrastive-preference-optimization", "text-generation", "preference-learning"]
    }
}

class HuggingFaceUploader:
    def __init__(self, token: str, username: str = USERNAME):
        self.token = token
        self.username = username
        self.api = HfApi()
        
        # Login to Hugging Face
        login(token=token)
        print(f"✅ Logged in to Hugging Face as {username}")
    
def load_model_paths():
    """Load model paths from training results"""
    model_paths_file = Path("outputs/model_paths.json")
    if not model_paths_file.exists():
        print("❌ Model paths file not found. Please run training first.")
        return []
    
    with open(model_paths_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_repo_if_not_exists(api, repo_name, username):
    """Create repository if it doesn't exist"""
    try:
        api.repo_info(f"{username}/{repo_name}")
        print(f"  Repository {username}/{repo_name} already exists")
        return True
    except RepositoryNotFoundError:
        try:
            create_repo(
                repo_id=f"{username}/{repo_name}",
                token=api.token,
                private=False,
                repo_type="model"
            )
            print(f"  ✓ Created repository {username}/{repo_name}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to create repository {username}/{repo_name}: {e}")
            return False
    
def prepare_model_files(model_info, output_dir):
    """Prepare model files for upload"""
    method = model_info['method']
    dataset = model_info['dataset']
    model_path = Path(model_info['path'])
    
    # Create upload directory
    upload_dir = Path(output_dir) / model_info['model_name']
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files from training output
    if model_path.exists():
        for file in model_path.glob("*"):
            if file.is_file():
                shutil.copy2(file, upload_dir / file.name)
        print(f"  ✓ Copied model files from {model_path}")
    else:
        print(f"  ⚠️  Model path not found: {model_path}")
    
    # Copy model card from huggingface directory
    hf_card_path = Path(f"huggingface/{method.lower()}/README.md")
    if hf_card_path.exists():
        shutil.copy2(hf_card_path, upload_dir / "README.md")
        print(f"  ✓ Copied model card for {method}")
    else:
        print(f"  ⚠️  Model card not found for {method}")
    
    return upload_dir
    
def generate_model_config(model_dir, method, dataset, model_info):
    """Generate model configuration files"""
    
    # Model config.json (update if exists, create if not)
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
    else:
        model_config = {
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "intermediate_size": 18944,
            "max_position_embeddings": 32768
        }
    
    # Add training information
    model_config.update({
        "training_method": method,
        "dataset_name": dataset,
        "base_model": "Qwen/Qwen2.5-3B",
        "fine_tuned_by": "artaasd95",
        "samples_trained": model_info['samples_trained']
    })
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    
    # Tokenizer config (update if exists, create if not)
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        tokenizer_config = {
            "tokenizer_class": "Qwen2Tokenizer",
            "vocab_size": 151936,
            "model_max_length": 32768,
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "bos_token": None
        }
        
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Generated configuration files for {method}-{dataset}")
    
    def create_tokenizer_config(self, method: str) -> str:
        """Create tokenizer_config.json"""
        tokenizer_config = {
            "add_prefix_space": false,
            "added_tokens_decoder": {
                "151643": {
                    "content": "<|endoftext|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                },
                "151644": {
                    "content": "<|im_start|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                },
                "151645": {
                    "content": "<|im_end|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                }
            },
            "additional_special_tokens": [],
            "bos_token": "<|endoftext|>",
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            "clean_up_tokenization_spaces": false,
            "eos_token": "<|im_end|>",
            "errors": "replace",
            "model_max_length": 32768,
            "pad_token": "<|endoftext|>",
            "split_special_tokens": false,
            "tokenizer_class": "Qwen2Tokenizer",
            "unk_token": null
        }
        
        config_path = Path("huggingface") / method / "tokenizer_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        return str(config_path)
    
    def upload_model(self, method: str) -> bool:
        """Upload a single model to Hugging Face Hub"""
        print(f"\n🚀 Uploading {method.upper()} model...")
        
        # Create repository
        repo_id = self.create_model_repository(method)
        if not repo_id:
            return False
        
        try:
            # Create necessary config files
            config_path = self.create_config_json(method)
            tokenizer_config_path = self.create_tokenizer_config(method)
            
            # Prepare files for upload
            files_to_upload = self.prepare_model_files(method)
            files_to_upload["config.json"] = config_path
            files_to_upload["tokenizer_config.json"] = tokenizer_config_path
            
            # Upload files
            for filename, filepath in files_to_upload.items():
                if os.path.exists(filepath):
                    print(f"  📤 Uploading {filename}...")
                    self.api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                else:
                    print(f"  ⚠️  File not found: {filepath}")
            
            print(f"✅ Successfully uploaded {method.upper()} model to {repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading {method} model: {e}")
            return False
    
def upload_models_to_huggingface(username, token=None):
    """Upload all trained models to Hugging Face Hub"""
    if not token:
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            print("❌ Hugging Face token not found. Set HUGGINGFACE_TOKEN environment variable.")
            return
    
    # Load actual model paths from training
    model_paths = load_model_paths()
    if not model_paths:
        return
    
    api = HfApi(token=token)
    upload_dir = Path("upload_temp")
    upload_dir.mkdir(exist_ok=True)
    
    uploaded_models = []
    failed_uploads = []
    
    print(f"🚀 Starting upload of {len(model_paths)} models to Hugging Face Hub...\n")
    
    for model_info in model_paths:
        model_name = model_info['model_name']
        method = model_info['method']
        dataset = model_info['dataset']
        
        print(f"\nProcessing {model_name}...")
        print(f"  Method: {method}, Dataset: {dataset}, Samples: {model_info['samples_trained']:,}")
        
        # Create repository
        if not create_repo_if_not_exists(api, model_name, username):
            failed_uploads.append(model_name)
            continue
        
        # Prepare model files
        model_dir = prepare_model_files(model_info, upload_dir)
        
        # Generate model configuration
        generate_model_config(model_dir, method, dataset, model_info)
        
        # Upload to Hugging Face
        try:
            print(f"  📤 Uploading {model_name}...")
            upload_folder(
                folder_path=str(model_dir),
                repo_id=f"{username}/{model_name}",
                token=token,
                commit_message=f"Upload {method} model trained on {dataset} dataset ({model_info['samples_trained']:,} samples)"
            )
            print(f"  ✅ Successfully uploaded {model_name}")
            uploaded_models.append(f"{username}/{model_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to upload {model_name}: {e}")
            failed_uploads.append(model_name)
    
    # Cleanup
    shutil.rmtree(upload_dir, ignore_errors=True)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 UPLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Successfully uploaded: {len(uploaded_models)} models")
    print(f"❌ Failed uploads: {len(failed_uploads)} models")
    
    if uploaded_models:
        print("\n🎉 Uploaded Models:")
        for model in uploaded_models:
            print(f"  - https://huggingface.co/{model}")
    
    if failed_uploads:
        print("\n⚠️  Failed Models:")
        for model in failed_uploads:
            print(f"  - {model}")
    
    return uploaded_models, failed_uploads

def main():
    """Main function to upload models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload Arabic Qwen models to Hugging Face Hub")
    parser.add_argument(
        "--username", 
        type=str, 
        default="artaasd95",
        help="Hugging Face username"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        help="Hugging Face API token (or set HUGGINGFACE_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Upload models
    uploaded_models, failed_uploads = upload_models_to_huggingface(
        username=args.username,
        token=args.token
    )
    
    # Final message
    if uploaded_models and not failed_uploads:
        print("\n🎉 All uploads completed successfully!")
    elif uploaded_models:
        print("\n⚠️  Some uploads completed, but some failed. Check the summary above.")
    else:
        print("\n❌ All uploads failed. Please check the error messages and try again.")

if __name__ == "__main__":
    main()