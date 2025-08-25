# Arabic Qwen Fine-tuning Training and Upload Guide

This guide provides step-by-step instructions for training Arabic Qwen models and uploading them to Hugging Face Hub.

## Prerequisites

### 1. Environment Setup

```bash
# Install required dependencies
pip install -r requirements.txt

# Or install specific packages if needed
pip install torch transformers datasets accelerate peft trl huggingface_hub
```

### 2. Hugging Face Authentication

1. **Get your Hugging Face token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Write" permissions
   - Copy the token

2. **Set environment variable:**
   ```bash
   # Windows (PowerShell)
   $env:HUGGINGFACE_TOKEN="your_token_here"
   
   # Windows (Command Prompt)
   set HUGGINGFACE_TOKEN=your_token_here
   
   # Linux/Mac
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

3. **Alternative: Login via CLI**
   ```bash
   huggingface-cli login
   ```

## Training Process

### Step 1: Run the Training Script

The project includes a comprehensive training script that handles all fine-tuning methods:

```bash
# Navigate to project directory
cd G:\repositories\arabic-qwen-base-finetuning

# Run the training script with comprehensive reporting
python scripts/real_training.py
```

### What the Training Script Does:

1. **Loads Arabic Datasets:**
   - `arabic-instruct`: Instruction-following dataset
   - `arabic-chat`: Conversational dataset
   - `arabic-qa`: Question-answering dataset

2. **Trains Multiple Methods:**
   - **SFT (Supervised Fine-Tuning)**: Basic instruction following
   - **DPO (Direct Preference Optimization)**: Preference-based training
   - **KTO (Kahneman-Tversky Optimization)**: Human preference alignment
   - **IPO (Identity Preference Optimization)**: Identity-aware preferences
   - **CPO (Contrastive Preference Optimization)**: Contrastive learning

3. **Generates 15 Model Checkpoints:**
   - 5 methods × 3 datasets = 15 trained models
   - Each model is saved in `models/` directory
   - Training results saved to `outputs/training_results.json`
   - Model paths saved to `outputs/model_paths.json`

### Training Output Structure:

```
models/
├── qwen-3-base-arabic-arabic-instruct-SFT/
├── qwen-3-base-arabic-arabic-instruct-DPO/
├── qwen-3-base-arabic-arabic-instruct-KTO/
├── qwen-3-base-arabic-arabic-instruct-IPO/
├── qwen-3-base-arabic-arabic-instruct-CPO/
├── qwen-3-base-arabic-arabic-chat-SFT/
├── qwen-3-base-arabic-arabic-chat-DPO/
├── qwen-3-base-arabic-arabic-chat-KTO/
├── qwen-3-base-arabic-arabic-chat-IPO/
├── qwen-3-base-arabic-arabic-chat-CPO/
├── qwen-3-base-arabic-arabic-qa-SFT/
├── qwen-3-base-arabic-arabic-qa-DPO/
├── qwen-3-base-arabic-arabic-qa-KTO/
├── qwen-3-base-arabic-arabic-qa-IPO/
└── qwen-3-base-arabic-arabic-qa-CPO/
```

### Training Reports and Analytics

The training script now generates comprehensive reports and visualizations:

#### Generated Reports:

1. **Comprehensive Training Report** (`outputs/comprehensive_training_report.md`)
   - Executive summary with key metrics
   - Performance analysis by method and dataset
   - Detailed training results table
   - Recommendations for best performing configurations
   - Method-specific analysis and insights

2. **Training Summary JSON** (`outputs/training_summary.json`)
   - Structured data for programmatic access
   - Total models, samples, training time
   - Best performing model information
   - File paths for all generated outputs

3. **Raw Training Results** (`outputs/training_results.json`)
   - Detailed results for each model
   - Training metrics, timestamps, dataset information
   - Checkpoint paths and configuration details

#### Generated Charts:

All charts are saved in `outputs/plots/` directory:

1. **Final Loss Comparison** (`final_loss_comparison.png`)
   - Heatmap showing loss performance across methods and datasets
   - Helps identify best performing combinations

2. **Training Time Comparison** (`training_time_comparison.png`)
   - Bar chart comparing training duration
   - Grouped by method and dataset

3. **Training Throughput** (`inference_throughput_comparison.png`)
   - Box plot showing samples per second by method
   - Identifies most efficient training approaches

4. **Efficiency Score Comparison** (`efficiency_score_comparison.png`)
   - Heatmap of training efficiency (samples/time)
   - Combines speed and data processing metrics

#### Sample Output:

```
=== Training Summary ===
Total samples trained: 106,800
Total training time: 13347 seconds (222.4 minutes)
Average final loss: 0.7280
Total models created: 15

Generated Reports:
  - Comprehensive report: outputs/comprehensive_training_report.md
  - Training charts: outputs/plots/
  - Training summary: outputs/training_summary.json
  - Raw results: outputs/training_results.json
```

#### Dependencies for Reporting:

The reporting features require additional visualization libraries:

```bash
# Install visualization dependencies (included in requirements.txt)
pip install matplotlib seaborn pandas
```

If these packages are missing, the training will still complete successfully, but charts won't be generated.

## Uploading to Hugging Face

### Step 2: Upload Trained Models

After training completes successfully, upload all models to Hugging Face:

```bash
# Upload all trained models to Hugging Face Hub
python scripts/upload_to_huggingface.py
```

### What the Upload Script Does:

1. **Reads Training Results:**
   - Loads model paths from `outputs/model_paths.json`
   - Reads training metrics and configurations

2. **Creates Repositories:**
   - Creates a new repository for each trained model
   - Uses naming convention: `username/model-name`
   - Sets appropriate repository visibility

3. **Generates Model Cards:**
   - Creates comprehensive README.md for each model
   - Includes training details, usage examples, and metrics
   - Documents the fine-tuning method and dataset used

4. **Uploads Files:**
   - Model weights and configuration files
   - Tokenizer files
   - Training arguments and metadata
   - Generated model cards

### Expected Upload Results:

After successful upload, you'll have 15 repositories on Hugging Face:

- `your-username/qwen-3-base-arabic-arabic-instruct-SFT`
- `your-username/qwen-3-base-arabic-arabic-instruct-DPO`
- `your-username/qwen-3-base-arabic-arabic-instruct-KTO`
- `your-username/qwen-3-base-arabic-arabic-instruct-IPO`
- `your-username/qwen-3-base-arabic-arabic-instruct-CPO`
- ... (and 10 more for chat and qa datasets)

## Monitoring and Verification

### Check Training Progress:

```bash
# View training results
cat outputs/training_results.json

# Check model paths
cat outputs/model_paths.json

# View training summary
cat outputs/training_summary.json
```

### Test Trained Models:

```bash
# Run inference script to test models
python scripts/inference.py

# Run usage examples
python examples/model_usage_examples.py
```

## Troubleshooting

### Common Issues:

1. **Missing Dependencies:**
   ```bash
   pip install --upgrade transformers datasets torch accelerate
   ```

2. **Hugging Face Authentication:**
   ```bash
   # Re-login if token expires
   huggingface-cli login
   ```

3. **Memory Issues:**
   - Reduce batch size in training configurations
   - Use gradient checkpointing
   - Enable mixed precision training

4. **Network Issues:**
   - Check internet connection
   - Verify Hugging Face Hub access
   - Try uploading individual models if batch upload fails

### Manual Upload (if automated upload fails):

```bash
# Upload individual model
huggingface-cli upload your-username/model-name ./models/model-directory/
```

## Configuration Options

### Environment Variables:

```bash
# Required
HUGGINGFACE_TOKEN=your_token_here

# Optional
HF_USERNAME=your_username          # Default: uses token's username
MODEL_PREFIX=arabic-qwen          # Default: qwen-3-base-arabic
UPLOAD_PRIVATE=false              # Default: public repositories
```

### Training Configuration:

Modify training parameters in the script or create configuration files:

- `config/sft_config.py` - SFT training parameters
- `config/dpo_config.py` - DPO training parameters
- `config/kto_config.py` - KTO training parameters
- `config/ipo_config.py` - IPO training parameters
- `config/cpo_config.py` - CPO training parameters

## Best Practices

1. **Before Training:**
   - Ensure sufficient disk space (>50GB recommended)
   - Verify GPU memory availability
   - Test Hugging Face authentication

2. **During Training:**
   - Monitor training logs for errors
   - Check GPU utilization
   - Verify dataset loading

3. **After Training:**
   - Validate model outputs before uploading
   - Test inference on sample inputs
   - Review generated model cards

4. **Upload Strategy:**
   - Upload during off-peak hours for better performance
   - Verify each model upload completion
   - Keep local backups of trained models

## Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Hugging Face token
export HUGGINGFACE_TOKEN="your_token_here"

# 3. Run training
python scripts/real_training.py

# 4. Upload to Hugging Face
python scripts/upload_to_huggingface.py

# 5. Test models
python scripts/inference.py
```

That's it! You now have 15 fine-tuned Arabic Qwen models on Hugging Face Hub, ready for use in Arabic NLP applications.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation in `docs/`
3. Open an issue in the project repository
4. Consult Hugging Face documentation for platform-specific issues