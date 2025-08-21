# Third-Party Licenses and Attributions

This project incorporates or is built upon several third-party libraries and resources. The following is a comprehensive list of these dependencies and their respective licenses:

## Core Dependencies

### 1. Qwen Models
- **Source**: Alibaba Cloud
- **License**: Apache License 2.0
- **URL**: https://github.com/QwenLM/Qwen
- **Attribution**: This project uses Qwen models as base models for fine-tuning Arabic language tasks.

### 2. Transformers Library
- **Source**: Hugging Face
- **License**: Apache License 2.0
- **URL**: https://github.com/huggingface/transformers
- **Attribution**: Core library for model loading, training, and inference operations.

### 3. PyTorch
- **Source**: Meta (Facebook)
- **License**: BSD 3-Clause License
- **URL**: https://github.com/pytorch/pytorch
- **Attribution**: Deep learning framework used for model training and inference.

### 4. Datasets Library
- **Source**: Hugging Face
- **License**: Apache License 2.0
- **URL**: https://github.com/huggingface/datasets
- **Attribution**: Used for loading and processing training datasets from Hugging Face Hub.

## Training and Fine-tuning Libraries

### 5. TRL (Transformer Reinforcement Learning)
- **Source**: Hugging Face
- **License**: Apache License 2.0
- **URL**: https://github.com/huggingface/trl
- **Attribution**: Used for DPO, KTO, IPO, and CPO training methods implementation.

### 6. PEFT (Parameter-Efficient Fine-Tuning)
- **Source**: Hugging Face
- **License**: Apache License 2.0
- **URL**: https://github.com/huggingface/peft
- **Attribution**: Used for LoRA and other parameter-efficient fine-tuning methods.

### 7. Accelerate
- **Source**: Hugging Face
- **License**: Apache License 2.0
- **URL**: https://github.com/huggingface/accelerate
- **Attribution**: Used for distributed training, mixed precision, and hardware optimization.

## Optimization and Quantization

### 8. BitsAndBytes
- **Source**: Tim Dettmers
- **License**: MIT License
- **URL**: https://github.com/TimDettmers/bitsandbytes
- **Attribution**: Used for model quantization (4-bit, 8-bit) and memory optimization.

### 9. Flash Attention
- **Source**: Tri Dao, Stanford University
- **License**: BSD 3-Clause License
- **URL**: https://github.com/Dao-AILab/flash-attention
- **Attribution**: Used for memory-efficient attention computation.

## Arabic Text Processing

### 10. Arabic Text Processing Libraries
- **arabic-reshaper**: MIT License - Used for Arabic text reshaping
- **python-bidi**: LGPL License - Used for bidirectional text handling
- **pyarabic**: GPL License - Used for Arabic text preprocessing and normalization
- **camel-tools**: MIT License - Used for Arabic NLP preprocessing

## Training Datasets

### 11. Arabic Instruction Datasets
- **FreedomIntelligence/Alpaca-Arabic-GPT4**: Various licenses
  - Used for supervised fine-tuning (SFT) with Arabic instructions
- **2A2I/argilla-dpo-mix-7k-arabic**: Various licenses
  - Used for Direct Preference Optimization (DPO) training
- **sadeem-ai/arabic-qna**: Various licenses
  - Used for question-answering fine-tuning
- **OpenAssistant/oasst1**: Apache License 2.0
  - Used for conversational AI training (Arabic subset)

## Development and Testing Tools

### 12. Testing and Quality Assurance
- **pytest**: MIT License - Testing framework
- **black**: MIT License - Code formatting
- **isort**: MIT License - Import sorting
- **mypy**: MIT License - Type checking
- **ruff**: MIT License - Linting and code analysis

### 13. Documentation and Deployment
- **mkdocs**: BSD License - Documentation generation
- **gradio**: Apache License 2.0 - Web interface for model demos
- **fastapi**: MIT License - API framework for model serving
- **uvicorn**: BSD License - ASGI server

## Cloud and Infrastructure

### 14. Cloud Services
- **wandb**: MIT License - Experiment tracking and monitoring
- **tensorboard**: Apache License 2.0 - Training visualization
- **huggingface_hub**: Apache License 2.0 - Model repository management

## Important Notes

### License Compatibility
All dependencies have been selected to ensure license compatibility with the MIT License of this project. However, users should be aware of the following:

1. **GPL Licensed Components**: Some Arabic text processing libraries use GPL licenses, which may impose additional requirements for derivative works.
2. **Dataset Licenses**: Training datasets may have varying licenses and usage restrictions.
3. **Model Licenses**: Fine-tuned models inherit the licenses of their base models and training data.

### Commercial Use Considerations
Before using this project or its outputs for commercial purposes, please:

1. Review all third-party licenses for commercial use restrictions
2. Ensure compliance with dataset usage terms
3. Verify that your use case aligns with the Qwen model license
4. Consider the implications of GPL-licensed text processing tools

### Attribution Requirements
When using this project or its outputs, please:

1. Include this attribution file in your distribution
2. Acknowledge the use of Qwen models and Hugging Face libraries
3. Credit the Arabic dataset creators and contributors
4. Follow the attribution requirements of individual licenses

### Disclaimer
The fine-tuned models produced by this framework inherit the licenses and restrictions of their base models and training datasets. Users are solely responsible for ensuring compliance with all applicable licenses when using the models for any purpose.

For the most up-to-date license information for each dependency, please refer to their respective repositories and documentation.

---

*Last updated: January 2025*
*For questions about licensing, please open an issue in the project repository.*