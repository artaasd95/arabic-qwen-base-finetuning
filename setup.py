#!/usr/bin/env python3
"""Setup script for Arabic Qwen Base Fine-tuning project."""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("This package requires Python 3.8 or higher")

# Get the long description from README
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = here / filename
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []

# Core requirements
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "accelerate>=0.24.0",
    "peft>=0.6.0",
    "trl>=0.7.0",
    "bitsandbytes>=0.41.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.65.0",
    "click>=8.1.0",
    "rich>=13.5.0",
    "loguru>=0.7.0",
    "arabic-reshaper>=3.0.0",
    "python-bidi>=0.4.2",
    "pyarabic>=0.6.15",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.4.0",
    ],
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.2.0",
    ],
    "monitoring": [
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "mlflow>=2.7.0",
    ],
    "visualization": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
    ],
    "serving": [
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.26.0",
        "gradio>=3.44.0",
    ],
    "cloud": [
        "boto3>=1.28.0",
        "google-cloud-storage>=2.10.0",
        "azure-storage-blob>=12.17.0",
    ],
    "optimization": [
        "optuna>=3.3.0",
        "hyperopt>=0.2.7",
        "deepspeed>=0.10.0",
    ],
    "quantization": [
        "auto-gptq>=0.4.0",
        "optimum>=1.13.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
    ],
    "nlp": [
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "polyglot>=16.7.4",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.1.0",
    ],
    "testing": read_requirements("requirements-test.txt"),
}

# All extras
extras_require["all"] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

# Entry points for command-line scripts
entry_points = {
    "console_scripts": [
        "arabic-qwen-train=src.scripts.train:main",
        "arabic-qwen-evaluate=src.scripts.evaluate:main",
        "arabic-qwen-infer=src.scripts.inference:main",
        "arabic-qwen-data=src.scripts.data_processing:main",
        "arabic-qwen-config=src.scripts.config_manager:main",
    ],
}

# Package data
package_data = {
    "src": [
        "config/*.yaml",
        "config/*.json",
        "data/templates/*.txt",
        "data/examples/*.json",
    ],
}

# Classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: Linguistic",
    "Natural Language :: Arabic",
    "Natural Language :: English",
]

# Keywords
keywords = [
    "arabic", "nlp", "transformers", "fine-tuning", "qwen", "llm",
    "machine-learning", "deep-learning", "pytorch", "huggingface",
    "sft", "dpo", "kto", "ipo", "cpo", "preference-optimization",
    "language-model", "chatbot", "conversational-ai"
]

setup(
    name="arabic-qwen-base-finetuning",
    version="1.0.0",
    description="A comprehensive framework for fine-tuning Qwen models on Arabic datasets using various optimization methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arabic NLP Team",
    author_email="team@arabic-nlp.org",
    url="https://github.com/arabic-nlp/arabic-qwen-base-finetuning",
    project_urls={
        "Bug Reports": "https://github.com/arabic-nlp/arabic-qwen-base-finetuning/issues",
        "Source": "https://github.com/arabic-nlp/arabic-qwen-base-finetuning",
        "Documentation": "https://arabic-qwen-base-finetuning.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    package_data=package_data,
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=classifiers,
    keywords=keywords,
    license="MIT",
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
    tests_require=extras_require["testing"],
    
    # Options
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
    
    # Custom commands
    cmdclass={},
    
    # Data files
    data_files=[
        ("config", ["config/sft_config.yaml", "config/dpo_config.yaml"]),
        ("examples", ["examples/train_sft.py", "examples/train_dpo.py"]),
    ],
    
    # Namespace packages
    namespace_packages=[],
    
    # Dependency links (deprecated but kept for compatibility)
    dependency_links=[],
    
    # Eager resources
    eager_resources=[],
    
    # Exclude packages
    exclude_package_data={
        "": ["*.pyc", "*.pyo", "*~", "*.so", "*.dylib", "*.dll"],
    },
)

# Post-installation message
print("""
🎉 Arabic Qwen Base Fine-tuning has been installed successfully!

📚 Quick Start:
   1. Set up your environment variables (see .env.example)
   2. Prepare your dataset
   3. Configure your training parameters
   4. Run training: arabic-qwen-train --config config/sft_config.yaml

📖 Documentation: https://arabic-qwen-base-finetuning.readthedocs.io/
🐛 Issues: https://github.com/arabic-nlp/arabic-qwen-base-finetuning/issues
💬 Discussions: https://github.com/arabic-nlp/arabic-qwen-base-finetuning/discussions

Happy fine-tuning! 🚀
""")