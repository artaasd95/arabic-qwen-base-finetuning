# Hugging Face Model Repository Builder

This directory contains tools to automatically create and upload your fine-tuned Arabic Qwen models to Hugging Face Hub. The script reads trained models from the `outputs` directory and creates separate repositories for each model with comprehensive documentation.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to the builder directory
cd huggingface_builder

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment file
cp .env.example .env
```

### 2. Configure Credentials

Edit the `.env` file with your Hugging Face credentials:

```bash
# Required: Your Hugging Face token
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Required: Your Hugging Face username
HUGGINGFACE_USERNAME=your_username

# Optional: Base model name (default: Qwen/Qwen3-1.7B)
BASE_MODEL_NAME=Qwen/Qwen3-1.7B

# Optional: Project root (default: ..)
PROJECT_ROOT=..
```

### 3. Run the Uploader

```bash
# Upload all models
python upload_models.py
```

## 📋 Prerequisites

### Hugging Face Setup

1. **Create Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Generate Token**: Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Token Permissions**: Ensure your token has **write** permissions
4. **Username**: Note your exact username (case-sensitive)

### Project Requirements

The script expects the following project structure:

```
project_root/
├── outputs/
│   ├── model_paths.json      # Model information
│   └── training_results.json # Training metrics
├── models/
│   ├── qwen-3-base-arabic-arabic-instruct-SFT/
│   ├── qwen-3-base-arabic-arabic-chat-DPO/
│   └── ... (other model directories)
└── huggingface_builder/
    ├── upload_models.py
    ├── .env
    └── requirements.txt
```

## 🎯 What the Script Does

### 1. Repository Creation

For each trained model, the script:

- ✅ Creates a separate Hugging Face repository
- ✅ Uses naming convention: `{username}/{model-name}`
- ✅ Sets appropriate tags and metadata
- ✅ Configures model card information

### 2. File Upload

Uploads all model files:

- 🔧 `config.json` - Model configuration
- 🔧 `tokenizer_config.json` - Tokenizer settings
- 🔧 `training_args.json` - Training parameters
- 📝 `README.md` - Comprehensive documentation
- 📊 `model_card.json` - Metadata for Hugging Face

### 3. Documentation Generation

Generates comprehensive README files including:

- 📖 Model description and capabilities
- 🎯 Training method explanation (SFT/DPO/KTO/IPO/CPO)
- 📊 Training statistics and metrics
- 💻 Usage examples and code snippets
- 🏷️ Proper tags and categorization
- 📚 Dataset information and sources
- ⚖️ License and citation information

## 📊 Model Organization

The script organizes models using this matrix:

| Method | arabic-instruct | arabic-chat | arabic-qa |
|--------|----------------|-------------|----------|
| **SFT** | ✅ Instruction Following | ✅ Conversational | ✅ Question Answering |
| **DPO** | ✅ Preference Optimized | ✅ Chat Optimized | ✅ QA Optimized |
| **KTO** | ✅ Prospect Theory | ✅ Cognitive Bias Aware | ✅ Human-like Reasoning |
| **IPO** | ✅ Identity Consistent | ✅ Personality Stable | ✅ Style Consistent |
| **CPO** | ✅ Safety Constrained | ✅ Controlled Responses | ✅ Safe Answers |

**Total**: 15 separate repositories (3 datasets × 5 methods)

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | ✅ Yes | - | Your HF access token |
| `HUGGINGFACE_USERNAME` | ✅ Yes | - | Your HF username |
| `BASE_MODEL_NAME` | ❌ No | `Qwen/Qwen3-1.7B` | Base model reference |
| `PROJECT_ROOT` | ❌ No | `..` | Path to project root |
| `PRIVATE_REPOS` | ❌ No | `false` | Create private repos |

### Repository Naming

Repositories are named using the pattern:
```
{username}/{model-name}
```

Example repositories:
```
myusername/qwen-3-base-arabic-arabic-instruct-SFT
myusername/qwen-3-base-arabic-arabic-chat-DPO
myusername/qwen-3-base-arabic-arabic-qa-KTO
```

## 📝 Generated README Structure

Each model repository includes a comprehensive README with:

### Header Section
- Model card metadata (YAML frontmatter)
- Language tags, license, base model
- Dataset references and model index
- Interactive widgets for testing

### Description Section
- Model overview and key features
- Training method explanation
- Dataset focus and use cases
- Technical specifications

### Training Details
- Method-specific information
- Dataset descriptions and sources
- Training statistics table
- Performance metrics

### Usage Section
- Quick start code examples
- Chat format instructions
- Integration examples
- Best practices

### Additional Information
- Model strengths and limitations
- Recommended use cases
- Citation information
- License and acknowledgments

## 🚨 Troubleshooting

### Common Issues

#### Authentication Errors
```
Error: Invalid token
```
**Solution**: 
- Verify your token in `.env` file
- Check token has write permissions
- Ensure token is not expired

#### Repository Creation Fails
```
Error: Repository already exists
```
**Solution**:
- The script handles existing repos automatically
- Check if you have permissions to the existing repo
- Verify username is correct

#### File Upload Errors
```
Error: Model files not found
```
**Solution**:
- Ensure models are trained and saved
- Check `outputs/model_paths.json` exists
- Verify model directories contain required files

#### Network Issues
```
Error: Connection timeout
```
**Solution**:
- Check internet connection
- Verify Hugging Face Hub status
- Try again later if service is down

### Debug Mode

For detailed logging, modify the script:

```python
# Change logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)
```

### Manual Verification

After upload, verify your repositories:

1. Visit `https://huggingface.co/{username}`
2. Check each repository is created
3. Verify README renders correctly
4. Test model loading with provided code

## 🔒 Security Best Practices

### Token Security
- ✅ Never commit `.env` file to git
- ✅ Use tokens with minimal required permissions
- ✅ Regularly rotate access tokens
- ✅ Store tokens securely

### Repository Settings
- ✅ Review repository visibility settings
- ✅ Set appropriate licenses
- ✅ Configure access permissions
- ✅ Enable security features

## 📈 Advanced Usage

### Selective Upload

To upload specific models, modify the script:

```python
# Filter models by method
filtered_models = [m for m in model_paths if m['method'] == 'SFT']

# Filter models by dataset
filtered_models = [m for m in model_paths if m['dataset'] == 'arabic-chat']
```

### Custom README Templates

Modify the `generate_readme()` method to customize documentation:

```python
def generate_readme(self, model_info, training_info):
    # Add custom sections
    # Modify existing content
    # Include additional metadata
    pass
```

### Batch Operations

For large-scale operations:

```python
# Upload in batches
batch_size = 5
for i in range(0, len(models), batch_size):
    batch = models[i:i+batch_size]
    # Process batch
```

## 🤝 Contributing

To improve the uploader:

1. Fork the repository
2. Create feature branch
3. Add improvements
4. Test thoroughly
5. Submit pull request

## 📞 Support

For help and support:

- 📖 Check this README
- 🐛 Open GitHub issues
- 💬 Join community discussions
- 📧 Contact maintainers

---

**Happy uploading! 🚀**