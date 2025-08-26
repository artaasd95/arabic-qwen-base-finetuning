#!/usr/bin/env python3
"""
Hugging Face Setup Validator

This script validates the environment and configuration before running
the model uploader to prevent common issues and provide helpful feedback.
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv

try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

class SetupValidator:
    def __init__(self, config_path: str = ".env"):
        """Initialize the validator."""
        self.config_path = config_path
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Load environment variables if .env exists
        if os.path.exists(config_path):
            load_dotenv(config_path)
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(f"❌ ERROR: {message}")
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(f"⚠️  WARNING: {message}")
    
    def add_info(self, message: str):
        """Add an info message."""
        self.info.append(f"ℹ️  INFO: {message}")
    
    def validate_python_environment(self) -> bool:
        """Validate Python environment and dependencies."""
        print("🐍 Validating Python Environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.add_error(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
            return False
        else:
            self.add_info(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓")
        
        # Check required packages
        required_packages = [
            ('huggingface_hub', 'huggingface-hub'),
            ('transformers', 'transformers'),
            ('torch', 'torch'),
            ('dotenv', 'python-dotenv')
        ]
        
        missing_packages = []
        for package_name, pip_name in required_packages:
            try:
                __import__(package_name)
                self.add_info(f"{package_name} installed ✓")
            except ImportError:
                missing_packages.append(pip_name)
                self.add_error(f"{package_name} not installed")
        
        if missing_packages:
            self.add_error(f"Install missing packages: pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration file and environment variables."""
        print("⚙️  Validating Configuration...")
        
        # Check if .env file exists
        if not os.path.exists(self.config_path):
            self.add_error(f"Configuration file {self.config_path} not found")
            self.add_info("Copy .env.example to .env and configure your credentials")
            return False
        else:
            self.add_info(f"Configuration file {self.config_path} found ✓")
        
        # Validate required environment variables
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        hf_username = os.getenv('HUGGINGFACE_USERNAME')
        base_model = os.getenv('BASE_MODEL_NAME', 'Qwen/Qwen3-1.7B')
        project_root = os.getenv('PROJECT_ROOT', '..')
        
        # Check Hugging Face token
        if not hf_token or hf_token == 'your_huggingface_token_here':
            self.add_error("HUGGINGFACE_TOKEN not configured")
            self.add_info("Get your token from: https://huggingface.co/settings/tokens")
            return False
        elif not hf_token.startswith('hf_'):
            self.add_warning("HUGGINGFACE_TOKEN doesn't start with 'hf_' - this might be invalid")
        else:
            self.add_info(f"HUGGINGFACE_TOKEN configured (hf_****...****)✓")
        
        # Check Hugging Face username
        if not hf_username or hf_username == 'your_username_here':
            self.add_error("HUGGINGFACE_USERNAME not configured")
            return False
        else:
            self.add_info(f"HUGGINGFACE_USERNAME: {hf_username} ✓")
        
        # Check base model
        self.add_info(f"BASE_MODEL_NAME: {base_model} ✓")
        
        # Check project root
        project_path = Path(project_root)
        if not project_path.exists():
            self.add_error(f"PROJECT_ROOT path does not exist: {project_path.absolute()}")
            return False
        else:
            self.add_info(f"PROJECT_ROOT: {project_path.absolute()} ✓")
        
        return True
    
    def validate_huggingface_connection(self) -> bool:
        """Validate Hugging Face Hub connection and permissions."""
        print("🤗 Validating Hugging Face Connection...")
        
        if not HF_HUB_AVAILABLE:
            self.add_error("huggingface_hub package not available")
            return False
        
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            self.add_error("Cannot test connection without HUGGINGFACE_TOKEN")
            return False
        
        try:
            api = HfApi(token=hf_token)
            user_info = api.whoami()
            
            if user_info:
                username = user_info.get('name', 'Unknown')
                self.add_info(f"Connected to Hugging Face as: {username} ✓")
                
                # Check if configured username matches
                configured_username = os.getenv('HUGGINGFACE_USERNAME')
                if configured_username and configured_username != username:
                    self.add_warning(f"Configured username '{configured_username}' doesn't match token owner '{username}'")
                
                return True
            else:
                self.add_error("Failed to get user information from Hugging Face")
                return False
                
        except Exception as e:
            self.add_error(f"Failed to connect to Hugging Face: {str(e)}")
            self.add_info("Check your token permissions and internet connection")
            return False
    
    def validate_project_structure(self) -> bool:
        """Validate project structure and required files."""
        print("📁 Validating Project Structure...")
        
        project_root = Path(os.getenv('PROJECT_ROOT', '..'))
        
        # Check required directories
        required_dirs = [
            ('outputs', 'Training outputs directory'),
            ('models', 'Model checkpoints directory')
        ]
        
        for dir_name, description in required_dirs:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                self.add_error(f"{description} not found: {dir_path.absolute()}")
                return False
            else:
                self.add_info(f"{description} found ✓")
        
        # Check required files
        required_files = [
            ('outputs/model_paths.json', 'Model paths configuration'),
            ('outputs/training_results.json', 'Training results data')
        ]
        
        for file_path, description in required_files:
            full_path = project_root / file_path
            if not full_path.exists():
                self.add_error(f"{description} not found: {full_path.absolute()}")
                return False
            else:
                self.add_info(f"{description} found ✓")
        
        return True
    
    def validate_model_data(self) -> bool:
        """Validate model data and training results."""
        print("🤖 Validating Model Data...")
        
        project_root = Path(os.getenv('PROJECT_ROOT', '..'))
        
        try:
            # Load model paths
            model_paths_file = project_root / 'outputs' / 'model_paths.json'
            with open(model_paths_file, 'r', encoding='utf-8') as f:
                model_paths = json.load(f)
            
            if not model_paths:
                self.add_error("No models found in model_paths.json")
                return False
            
            self.add_info(f"Found {len(model_paths)} models in configuration ✓")
            
            # Load training results
            training_results_file = project_root / 'outputs' / 'training_results.json'
            with open(training_results_file, 'r', encoding='utf-8') as f:
                training_results = json.load(f)
            
            self.add_info(f"Found {len(training_results)} training results ✓")
            
            # Validate model directories exist
            models_dir = project_root / 'models'
            missing_models = []
            
            for model_info in model_paths:
                model_path = models_dir / model_info['path'].replace('\\', '/')
                if not model_path.exists():
                    missing_models.append(model_info['model_name'])
            
            if missing_models:
                self.add_error(f"Model directories not found: {', '.join(missing_models)}")
                return False
            else:
                self.add_info(f"All model directories exist ✓")
            
            # Check for required model files
            required_model_files = ['config.json', 'tokenizer_config.json']
            incomplete_models = []
            
            for model_info in model_paths:
                model_path = models_dir / model_info['path'].replace('\\', '/')
                for required_file in required_model_files:
                    if not (model_path / required_file).exists():
                        incomplete_models.append(f"{model_info['model_name']}: missing {required_file}")
            
            if incomplete_models:
                self.add_warning(f"Some models have missing files: {', '.join(incomplete_models)}")
            else:
                self.add_info("All models have required files ✓")
            
            return True
            
        except json.JSONDecodeError as e:
            self.add_error(f"Invalid JSON in model data files: {str(e)}")
            return False
        except Exception as e:
            self.add_error(f"Failed to validate model data: {str(e)}")
            return False
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🔍 Starting Setup Validation...\n")
        
        checks = [
            self.validate_python_environment,
            self.validate_configuration,
            self.validate_huggingface_connection,
            self.validate_project_structure,
            self.validate_model_data
        ]
        
        all_passed = True
        for check in checks:
            try:
                result = check()
                all_passed = all_passed and result
                print()  # Add spacing between checks
            except Exception as e:
                self.add_error(f"Validation check failed: {str(e)}")
                all_passed = False
                print()
        
        return all_passed
    
    def print_summary(self):
        """Print validation summary."""
        print("📋 Validation Summary")
        print("=" * 50)
        
        if self.info:
            print("\n✅ Success Messages:")
            for msg in self.info:
                print(f"  {msg}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for msg in self.warnings:
                print(f"  {msg}")
        
        if self.errors:
            print("\n❌ Errors:")
            for msg in self.errors:
                print(f"  {msg}")
        
        print("\n" + "=" * 50)
        
        if not self.errors:
            print("🎉 All validations passed! You're ready to upload models.")
            return True
        else:
            print(f"💥 {len(self.errors)} error(s) found. Please fix them before uploading.")
            return False

def main():
    """Main function to run validation."""
    validator = SetupValidator()
    
    try:
        validation_passed = validator.run_validation()
        summary_passed = validator.print_summary()
        
        if validation_passed and summary_passed:
            print("\n🚀 Ready to run: python upload_models.py")
            return 0
        else:
            print("\n🔧 Please fix the issues above and run validation again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())