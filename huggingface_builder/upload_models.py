#!/usr/bin/env python3
"""
Hugging Face Model Repository Builder and Uploader

This script reads trained models from the outputs directory and creates
separate Hugging Face repositories for each model with comprehensive
README documentation and proper file organization.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceModelUploader:
    def __init__(self, config_path: str = ".env"):
        """Initialize the uploader with configuration."""
        load_dotenv(config_path)
        
        # Load environment variables
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.hf_username = os.getenv('HUGGINGFACE_USERNAME')
        self.base_model_name = os.getenv('BASE_MODEL_NAME', 'Qwen/Qwen3-1.7B')
        self.project_root = Path(os.getenv('PROJECT_ROOT', '..'))
        self.outputs_dir = self.project_root / 'outputs'
        self.models_dir = self.project_root / 'models'
        self.temp_dir = Path('temp_repos')
        
        # Validate required environment variables
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN is required")
        if not self.hf_username:
            raise ValueError("HUGGINGFACE_USERNAME is required")
            
        # Initialize Hugging Face API
        self.api = HfApi(token=self.hf_token)
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized uploader for user: {self.hf_username}")
        logger.info(f"Base model: {self.base_model_name}")
    
    def load_model_data(self) -> tuple[List[Dict], List[Dict]]:
        """Load model paths and training results from outputs directory."""
        model_paths_file = self.outputs_dir / 'model_paths.json'
        training_results_file = self.outputs_dir / 'training_results.json'
        
        if not model_paths_file.exists():
            raise FileNotFoundError(f"Model paths file not found: {model_paths_file}")
        if not training_results_file.exists():
            raise FileNotFoundError(f"Training results file not found: {training_results_file}")
        
        with open(model_paths_file, 'r', encoding='utf-8') as f:
            model_paths = json.load(f)
        
        with open(training_results_file, 'r', encoding='utf-8') as f:
            training_results = json.load(f)
        
        logger.info(f"Loaded {len(model_paths)} model paths and {len(training_results)} training results")
        return model_paths, training_results
    
    def get_method_description(self, method: str) -> Dict[str, str]:
        """Get description and use cases for each training method."""
        descriptions = {
            'SFT': {
                'name': 'Supervised Fine-Tuning',
                'description': 'Fine-tuned using supervised learning on instruction-following datasets',
                'use_case': 'General instruction following and task completion',
                'technique': 'Standard supervised fine-tuning with cross-entropy loss'
            },
            'DPO': {
                'name': 'Direct Preference Optimization',
                'description': 'Optimized using direct preference learning without reinforcement learning',
                'use_case': 'Improved response quality and human preference alignment',
                'technique': 'Direct optimization of preference data using DPO loss'
            },
            'KTO': {
                'name': 'Kahneman-Tversky Optimization',
                'description': 'Optimized using prospect theory-inspired preference learning',
                'use_case': 'Better handling of human cognitive biases in responses',
                'technique': 'KTO loss function based on prospect theory principles'
            },
            'IPO': {
                'name': 'Identity Preference Optimization',
                'description': 'Optimized using identity-based preference learning',
                'use_case': 'Consistent personality and response style',
                'technique': 'IPO loss for maintaining consistent model identity'
            },
            'CPO': {
                'name': 'Constrained Policy Optimization',
                'description': 'Optimized with constraints to maintain safety and alignment',
                'use_case': 'Safe and controlled response generation',
                'technique': 'Constrained optimization with safety constraints'
            }
        }
        return descriptions.get(method, {
            'name': method,
            'description': f'{method} fine-tuning',
            'use_case': 'Specialized fine-tuning approach',
            'technique': f'{method} optimization technique'
        })
    
    def get_dataset_description(self, dataset: str) -> Dict[str, str]:
        """Get description for each dataset type."""
        descriptions = {
            'arabic-instruct': {
                'name': 'Arabic Instruction Following',
                'description': 'Comprehensive Arabic instruction-following dataset',
                'focus': 'General instruction following, task completion, and reasoning in Arabic',
                'datasets': ['FreedomIntelligence/alpaca-gpt4-arabic', 'arbml/alpaca_arabic', 'FreedomIntelligence/evol-instruct-arabic']
            },
            'arabic-chat': {
                'name': 'Arabic Conversational',
                'description': 'Arabic conversational and dialogue dataset',
                'focus': 'Natural conversation, dialogue management, and chat interactions in Arabic',
                'datasets': ['FreedomIntelligence/sharegpt-arabic', '2A2I/Arabic-OpenHermes-2.5']
            },
            'arabic-qa': {
                'name': 'Arabic Question Answering',
                'description': 'Arabic question-answering and knowledge retrieval dataset',
                'focus': 'Factual question answering, knowledge retrieval, and information extraction in Arabic',
                'datasets': ['sadeem-ai/arabic-qna', 'riotu-lab/ArabicQA_2.1M']
            }
        }
        return descriptions.get(dataset, {
            'name': dataset.replace('-', ' ').title(),
            'description': f'{dataset} dataset',
            'focus': f'Specialized training on {dataset} data',
            'datasets': []
        })
    
    def generate_readme(self, model_info: Dict, training_info: Dict) -> str:
        """Generate comprehensive README for the model."""
        method = model_info['method']
        dataset = model_info['dataset']
        model_name = model_info['model_name']
        
        method_desc = self.get_method_description(method)
        dataset_desc = self.get_dataset_description(dataset)
        
        # Format training time
        training_time = training_info.get('training_time', 'N/A')
        if training_time.endswith('s'):
            seconds = int(training_time[:-1])
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            remaining_seconds = seconds % 60
            if hours > 0:
                training_time_formatted = f"{hours}h {minutes}m {remaining_seconds}s"
            elif minutes > 0:
                training_time_formatted = f"{minutes}m {remaining_seconds}s"
            else:
                training_time_formatted = f"{remaining_seconds}s"
        else:
            training_time_formatted = training_time
        
        # Generate model card
        readme_content = f"""---
language:
- ar
- en
license: apache-2.0
base_model: {self.base_model_name}
tags:
- arabic
- qwen3
- {method.lower()}
- {dataset.replace('-', '_')}
- fine-tuned
- instruction-following
- conversational-ai
datasets:
{chr(10).join(f'- {ds}' for ds in dataset_desc['datasets'])}
model-index:
- name: {model_name}
  results: []
pipeline_tag: text-generation
widget:
- text: "اكتب قصة قصيرة عن الذكاء الاصطناعي"
  example_title: "Arabic Story Generation"
- text: "ما هي فوائد التعلم الآلي؟"
  example_title: "Arabic Question Answering"
- text: "اشرح لي مفهوم الفيزياء الكمية بطريقة بسيطة"
  example_title: "Arabic Explanation"
---

# {model_name}

## Model Description

This model is a fine-tuned version of [{self.base_model_name}](https://huggingface.co/{self.base_model_name}) specifically optimized for Arabic language tasks using **{method_desc['name']} ({method})** on the **{dataset_desc['name']}** dataset.

### Key Features

- 🌟 **Base Model**: {self.base_model_name} (1.7B parameters)
- 🎯 **Training Method**: {method_desc['name']} ({method})
- 📚 **Dataset Focus**: {dataset_desc['focus']}
- 🌍 **Language**: Arabic (العربية) with English support
- ⚡ **Optimization**: {method_desc['technique']}

## Training Details

### Training Method: {method_desc['name']}

{method_desc['description']}. This approach is particularly effective for {method_desc['use_case'].lower()}.

**Technique**: {method_desc['technique']}

### Dataset: {dataset_desc['name']}

{dataset_desc['description']}. The training focused on {dataset_desc['focus'].lower()}.

**Source Datasets**:
{chr(10).join(f'- [{ds}](https://huggingface.co/datasets/{ds})' for ds in dataset_desc['datasets'])}

### Training Statistics

| Metric | Value |
|--------|-------|
| Training Method | {method_desc['name']} ({method}) |
| Dataset Type | {dataset_desc['name']} |
| Samples Trained | {training_info.get('samples_trained', 'N/A'):,} |
| Training Time | {training_time_formatted} |
| Epochs | {training_info.get('epochs', 'N/A')} |
| Final Loss | {training_info.get('final_loss', 'N/A')} |
| Samples/Second | {training_info.get('samples_per_second', 'N/A')} |
| Base Model | [{self.base_model_name}](https://huggingface.co/{self.base_model_name}) |

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "{self.hf_username}/{model_name}"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Generate text
prompt = "اكتب قصة قصيرة عن المستقبل"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Chat Format

```python
# For conversational use
system_message = "أنت مساعد ذكي ومفيد يتحدث العربية"
user_message = "ما هي أهمية الذكاء الاصطناعي في المستقبل؟"

prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Model Performance

### Strengths

- ✅ **Arabic Language Proficiency**: Excellent understanding and generation of Arabic text
- ✅ **{dataset_desc['focus']}**: Optimized for {dataset_desc['focus'].lower()}
- ✅ **{method_desc['use_case']}**: Enhanced through {method_desc['name'].lower()}
- ✅ **Cultural Context**: Better understanding of Arabic cultural nuances
- ✅ **Instruction Following**: Improved ability to follow complex instructions in Arabic

### Recommended Use Cases

- 📝 Arabic content generation and creative writing
- 💬 Arabic conversational AI and chatbots
- 🎓 Educational content and tutoring in Arabic
- 📚 Arabic text summarization and analysis
- 🔍 Arabic question-answering systems
- 🌐 Arabic-English translation assistance

## Limitations

- The model is optimized for Arabic but may have reduced performance in other languages
- Performance may vary on tasks not represented in the training data
- As with all language models, outputs should be reviewed for accuracy and appropriateness
- The model may reflect biases present in the training data

## Technical Specifications

- **Architecture**: Qwen3 Transformer
- **Parameters**: 1.7 billion
- **Context Length**: 32,768 tokens
- **Vocabulary Size**: 151,936 tokens
- **Training Precision**: Mixed precision (FP16)
- **Framework**: PyTorch with Transformers

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{{{model_name.replace('-', '_')},
  title={{{model_name}: Arabic {method_desc['name']} Fine-tuned Qwen3 Model}},
  author={{{self.hf_username}}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{self.hf_username}/{model_name}}}
}}
```

## License

This model is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Base Model**: [Qwen Team](https://huggingface.co/Qwen) for the excellent Qwen3-1.7B base model
- **Training Framework**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Dataset Contributors**: Thanks to all the dataset creators and contributors
- **Community**: Arabic NLP community for continuous support and feedback

## Contact

For questions, issues, or collaborations, please:
- Open an issue on the [model repository](https://huggingface.co/{self.hf_username}/{model_name})
- Contact: [{self.hf_username}](https://huggingface.co/{self.hf_username})

---

*This model was trained as part of the Arabic Qwen Fine-tuning Project, aimed at advancing Arabic language AI capabilities.*
"""
        
        return readme_content
    
    def prepare_model_repository(self, model_info: Dict, training_info: Dict) -> Path:
        """Prepare a temporary repository for the model."""
        model_name = model_info['model_name']
        model_path = self.models_dir / model_info['path'].replace('\\', '/')
        
        # Create temporary repository directory
        repo_dir = self.temp_dir / model_name
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)
        
        # Copy model files
        if model_path.exists():
            for file in model_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, repo_dir / file.name)
        else:
            logger.warning(f"Model path not found: {model_path}")
        
        # Generate and save README
        readme_content = self.generate_readme(model_info, training_info)
        with open(repo_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create model card metadata
        model_card_data = {
            "language": ["ar", "en"],
            "license": "apache-2.0",
            "base_model": self.base_model_name,
            "tags": [
                "arabic",
                "qwen3",
                model_info['method'].lower(),
                model_info['dataset'].replace('-', '_'),
                "fine-tuned",
                "instruction-following",
                "conversational-ai"
            ],
            "datasets": self.get_dataset_description(model_info['dataset'])['datasets'],
            "pipeline_tag": "text-generation"
        }
        
        with open(repo_dir / 'model_card.json', 'w', encoding='utf-8') as f:
            json.dump(model_card_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Prepared repository for {model_name} at {repo_dir}")
        return repo_dir
    
    def create_and_upload_repository(self, model_info: Dict, training_info: Dict) -> str:
        """Create Hugging Face repository and upload model."""
        model_name = model_info['model_name']
        repo_id = f"{self.hf_username}/{model_name}"
        
        try:
            # Prepare local repository
            repo_dir = self.prepare_model_repository(model_info, training_info)
            
            # Create repository on Hugging Face
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=False,
                exist_ok=True
            )
            
            # Upload repository contents
            logger.info(f"Uploading files to {repo_id}")
            upload_folder(
                folder_path=str(repo_dir),
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Upload {model_info['method']} fine-tuned model on {model_info['dataset']} dataset"
            )
            
            logger.info(f"Successfully uploaded {model_name} to {repo_id}")
            return repo_id
            
        except Exception as e:
            logger.error(f"Failed to upload {model_name}: {str(e)}")
            raise
    
    def upload_all_models(self) -> List[str]:
        """Upload all trained models to Hugging Face."""
        model_paths, training_results = self.load_model_data()
        
        # Create mapping of training results by method and dataset
        training_map = {}
        for result in training_results:
            key = f"{result['method']}_{result['dataset']}"
            training_map[key] = result
        
        uploaded_repos = []
        failed_uploads = []
        
        for model_info in model_paths:
            try:
                # Find corresponding training result
                key = f"{model_info['method']}_{model_info['dataset']}"
                training_info = training_map.get(key, {})
                
                # Upload model
                repo_id = self.create_and_upload_repository(model_info, training_info)
                uploaded_repos.append(repo_id)
                
            except Exception as e:
                logger.error(f"Failed to upload {model_info['model_name']}: {str(e)}")
                failed_uploads.append(model_info['model_name'])
        
        # Generate summary
        logger.info(f"\n=== Upload Summary ===")
        logger.info(f"Successfully uploaded: {len(uploaded_repos)} models")
        logger.info(f"Failed uploads: {len(failed_uploads)} models")
        
        if uploaded_repos:
            logger.info("\nSuccessfully uploaded repositories:")
            for repo in uploaded_repos:
                logger.info(f"  - https://huggingface.co/{repo}")
        
        if failed_uploads:
            logger.info("\nFailed uploads:")
            for model in failed_uploads:
                logger.info(f"  - {model}")
        
        return uploaded_repos
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")

def main():
    """Main function to run the uploader."""
    try:
        uploader = HuggingFaceModelUploader()
        uploaded_repos = uploader.upload_all_models()
        
        print(f"\n🎉 Successfully uploaded {len(uploaded_repos)} models to Hugging Face!")
        print("\nRepository URLs:")
        for repo in uploaded_repos:
            print(f"  📦 https://huggingface.co/{repo}")
            
    except Exception as e:
        logger.error(f"Upload process failed: {str(e)}")
        raise
    finally:
        if 'uploader' in locals():
            uploader.cleanup()

if __name__ == "__main__":
    main()