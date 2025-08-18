# Testing Documentation

This document provides comprehensive guidance on testing the Arabic Qwen Base Fine-tuning project, including test structure, running tests, and writing new tests.

## Overview

The project uses a comprehensive testing framework built on pytest with extensive coverage for all components:

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Benchmark critical operations
- **Configuration Tests**: Validate configuration handling
- **End-to-End Tests**: Test complete training and evaluation pipelines

## Test Structure

### Directory Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_config.py           # Configuration system tests
├── test_data_loader.py      # Data loading system tests
├── test_training.py         # Training modules tests
├── test_evaluation.py       # Evaluation system tests
├── test_utils.py            # Utility functions tests
├── integration/             # Integration tests
│   ├── test_sft_pipeline.py
│   ├── test_dpo_pipeline.py
│   └── test_evaluation_pipeline.py
├── performance/             # Performance benchmarks
│   ├── test_training_speed.py
│   └── test_memory_usage.py
└── fixtures/                # Test data and fixtures
    ├── sample_configs/
    ├── sample_datasets/
    └── mock_models/
```

### Test Categories

Tests are organized using pytest markers:

```python
# Unit tests
@pytest.mark.unit
def test_config_validation():
    pass

# Integration tests
@pytest.mark.integration
def test_training_pipeline():
    pass

# Slow tests (require special handling)
@pytest.mark.slow
def test_full_training_cycle():
    pass

# GPU-dependent tests
@pytest.mark.gpu
def test_cuda_training():
    pass

# Model loading tests
@pytest.mark.model
def test_model_initialization():
    pass
```

## Running Tests

### Basic Test Execution

#### Run All Tests
```bash
# Using pytest directly
pytest

# Using make command
make test

# Using custom test runner
python run_tests.py
```

#### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit
make test-unit
python run_tests.py --unit

# Integration tests only
pytest -m integration
make test-integration
python run_tests.py --integration

# Fast tests (exclude slow tests)
pytest -m "not slow"
python run_tests.py --fast
```

#### Run Tests for Specific Modules
```bash
# Configuration tests
pytest tests/test_config.py
python run_tests.py --module config

# Training tests
pytest tests/test_training.py
python run_tests.py --module training

# Evaluation tests
pytest tests/test_evaluation.py
python run_tests.py --module evaluation
```

### Advanced Test Options

#### Coverage Reports
```bash
# Generate coverage report
pytest --cov=src --cov-report=html
make test-coverage
python run_tests.py --coverage

# View coverage report
open reports/coverage/index.html  # macOS/Linux
start reports/coverage/index.html  # Windows
```

#### Parallel Execution
```bash
# Run tests in parallel
pytest -n auto
pytest -n 4  # Use 4 processes

# Parallel execution with coverage
pytest -n auto --cov=src
```

#### Verbose Output
```bash
# Detailed test output
pytest -v
pytest -vv  # Extra verbose
python run_tests.py --verbose
```

#### Test Selection
```bash
# Run specific test function
pytest tests/test_config.py::test_sft_config_creation

# Run tests matching pattern
pytest -k "config and validation"

# Run tests in specific file
pytest tests/test_training.py
```

### Performance Testing

#### Benchmark Tests
```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Generate benchmark report
pytest tests/performance/ --benchmark-json=reports/benchmark.json
```

#### Memory Profiling
```bash
# Profile memory usage
pytest tests/test_training.py --profile

# Memory profiling with specific test
pytest tests/test_training.py::test_sft_training --profile-svg
```

## Test Configuration

### pytest.ini Configuration

The project uses <mcfile name="pytest.ini" path="pytest.ini"></mcfile> for test configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
    --durations=10
    --cov=src
    --cov-report=html:reports/coverage
    --cov-report=term-missing
    --junit-xml=reports/junit.xml

markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    gpu: marks tests that require GPU
    model: marks tests that require model loading
```

### Test Dependencies

Test dependencies are managed in <mcfile name="requirements-test.txt" path="requirements-test.txt"></mcfile>:

```txt
# Core testing framework
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
pytest-mock>=3.10.0

# Coverage and reporting
coverage>=7.0.0
pytest-html>=3.1.0
pytest-json-report>=1.5.0

# Performance testing
pytest-benchmark>=4.0.0
memory-profiler>=0.60.0

# Test utilities
factory-boy>=3.2.0
faker>=18.0.0
responses>=0.23.0
```

## Writing New Tests

### Test Structure Guidelines

#### Basic Test Structure
```python
"""Test module for [component name].

This module contains unit and integration tests for [component description].
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.component import ComponentClass


class TestComponentClass:
    """Test cases for ComponentClass."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.component = ComponentClass()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up resources
        pass
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality of the component."""
        # Arrange
        input_data = "test input"
        expected_output = "expected output"
        
        # Act
        result = self.component.process(input_data)
        
        # Assert
        assert result == expected_output
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test error handling in the component."""
        # Test that appropriate exceptions are raised
        with pytest.raises(ValueError, match="Invalid input"):
            self.component.process(None)
    
    @pytest.mark.integration
    def test_integration_with_other_components(self):
        """Test integration with other system components."""
        # Test component interactions
        pass
```

#### Fixture Usage
```python
@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "model": {"name": "test/model"},
        "training": {"epochs": 1, "learning_rate": 0.001}
    }

@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.forward.return_value = torch.tensor([1.0, 2.0, 3.0])
    return model

def test_with_fixtures(sample_config, mock_model):
    """Test using fixtures."""
    trainer = SFTTrainer(config=sample_config, model=mock_model)
    assert trainer.config == sample_config
```

### Testing Patterns

#### Configuration Testing
```python
class TestSFTConfig:
    """Test SFT configuration handling."""
    
    def test_valid_config_creation(self):
        """Test creation of valid SFT configuration."""
        config_dict = {
            "model": {"name": "Qwen/Qwen2-7B"},
            "training": {
                "epochs": 3,
                "learning_rate": 5e-5,
                "batch_size": 4
            }
        }
        
        config = SFTConfig.from_dict(config_dict)
        
        assert config.model.name == "Qwen/Qwen2-7B"
        assert config.training.epochs == 3
        assert config.training.learning_rate == 5e-5
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configurations."""
        invalid_config = {
            "model": {"name": ""},  # Empty model name
            "training": {"learning_rate": -0.001}  # Negative learning rate
        }
        
        with pytest.raises(ValueError):
            SFTConfig.from_dict(invalid_config)
```

#### Data Loading Testing
```python
class TestSFTDataLoader:
    """Test SFT data loading functionality."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return Dataset.from_dict({
            "instruction": ["What is AI?", "Explain ML"],
            "response": ["AI is...", "ML is..."]
        })
    
    def test_data_loading(self, sample_dataset):
        """Test basic data loading functionality."""
        loader = SFTDataLoader()
        processed_data = loader.process_dataset(sample_dataset)
        
        assert len(processed_data) == 2
        assert "input_ids" in processed_data[0]
        assert "attention_mask" in processed_data[0]
    
    def test_data_validation(self):
        """Test data validation during loading."""
        invalid_dataset = Dataset.from_dict({
            "instruction": ["What is AI?"],
            # Missing response field
        })
        
        loader = SFTDataLoader()
        
        with pytest.raises(KeyError):
            loader.process_dataset(invalid_dataset)
```

#### Training Testing
```python
class TestSFTTrainer:
    """Test SFT training functionality."""
    
    @pytest.fixture
    def mock_training_setup(self):
        """Set up mocked training environment."""
        with patch('torch.cuda.is_available', return_value=False):
            config = SFTConfig.from_dict({
                "model": {"name": "test/model"},
                "training": {"epochs": 1, "learning_rate": 0.001}
            })
            
            trainer = SFTTrainer(config=config)
            trainer.model = Mock()
            trainer.tokenizer = Mock()
            
            return trainer
    
    @pytest.mark.unit
    def test_trainer_initialization(self, mock_training_setup):
        """Test trainer initialization."""
        trainer = mock_training_setup
        
        assert trainer.config is not None
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_step(self, mock_training_setup):
        """Test a single training step."""
        trainer = mock_training_setup
        
        # Mock training data
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        
        # Mock model output
        trainer.model.return_value = Mock(
            loss=torch.tensor(0.5),
            logits=torch.randn(1, 3, 1000)
        )
        
        loss = trainer.training_step(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
```

#### Evaluation Testing
```python
class TestEvaluationSystem:
    """Test evaluation system functionality."""
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator for testing."""
        evaluator = SFTEvaluator(
            model=Mock(),
            tokenizer=Mock(),
            config=Mock()
        )
        return evaluator
    
    def test_metric_computation(self, mock_evaluator):
        """Test evaluation metric computation."""
        predictions = ["This is a test", "Another test"]
        references = ["This is a test", "Different text"]
        
        metrics = mock_evaluator.compute_metrics(predictions, references)
        
        assert "bleu_score" in metrics
        assert "rouge_scores" in metrics
        assert 0 <= metrics["bleu_score"] <= 1
    
    @pytest.mark.integration
    def test_evaluation_pipeline(self, mock_evaluator):
        """Test complete evaluation pipeline."""
        test_dataset = Dataset.from_dict({
            "instruction": ["Test instruction"],
            "response": ["Test response"]
        })
        
        results = mock_evaluator.evaluate(test_dataset)
        
        assert "perplexity" in results
        assert "generation_metrics" in results
        assert isinstance(results["perplexity"], float)
```

### Mocking and Fixtures

#### Common Fixtures
```python
# conftest.py
import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.vocab_size = 1000
    model.forward.return_value = Mock(
        loss=torch.tensor(0.5),
        logits=torch.randn(1, 10, 1000)
    )
    return model

@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer

@pytest.fixture
def sample_config():
    """Provide a sample configuration."""
    return {
        "model": {
            "name": "test/model",
            "quantization": {"enabled": False}
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 2
        },
        "data": {
            "max_length": 512,
            "train_file": "train.json"
        }
    }
```

#### Mocking External Dependencies
```python
# Mock transformers components
@patch('transformers.AutoModel.from_pretrained')
@patch('transformers.AutoTokenizer.from_pretrained')
def test_model_loading(mock_tokenizer, mock_model):
    """Test model loading with mocked transformers."""
    mock_model.return_value = Mock()
    mock_tokenizer.return_value = Mock()
    
    trainer = SFTTrainer(config=sample_config)
    trainer.load_model()
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()

# Mock file operations
@patch('pathlib.Path.exists')
@patch('json.load')
def test_config_loading(mock_json_load, mock_exists):
    """Test configuration loading with mocked file operations."""
    mock_exists.return_value = True
    mock_json_load.return_value = {"model": {"name": "test"}}
    
    config = load_config("test_config.json")
    
    assert config["model"]["name"] == "test"
```

### Performance Testing

#### Benchmark Tests
```python
import pytest
from time import time

class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_data_loading_speed(self, benchmark):
        """Benchmark data loading performance."""
        def load_data():
            loader = SFTDataLoader()
            dataset = create_large_dataset(1000)  # 1000 samples
            return loader.process_dataset(dataset)
        
        result = benchmark(load_data)
        assert len(result) == 1000
    
    @pytest.mark.benchmark
    def test_model_inference_speed(self, benchmark, mock_model):
        """Benchmark model inference performance."""
        def run_inference():
            inputs = torch.randint(0, 1000, (4, 512))  # Batch of 4
            return mock_model(inputs)
        
        result = benchmark(run_inference)
        assert result is not None
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage during training."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operation
        trainer = SFTTrainer(config=sample_config)
        trainer.load_model()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Assert memory increase is within reasonable bounds
        assert memory_increase < 2 * 1024 * 1024 * 1024  # Less than 2GB
```

## Continuous Integration

### GitHub Actions Integration

The project includes CI/CD integration for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: ["--tb=short", "-q"]
```

## Test Data Management

### Sample Data Creation
```python
# tests/fixtures/data_fixtures.py
def create_sample_sft_dataset(size=10):
    """Create a sample SFT dataset for testing."""
    instructions = [
        "What is machine learning?",
        "Explain neural networks",
        "How does fine-tuning work?"
    ] * (size // 3 + 1)
    
    responses = [
        "Machine learning is a subset of AI...",
        "Neural networks are computational models...",
        "Fine-tuning involves adapting a pre-trained model..."
    ] * (size // 3 + 1)
    
    return Dataset.from_dict({
        "instruction": instructions[:size],
        "response": responses[:size]
    })

def create_sample_preference_dataset(size=10):
    """Create a sample preference dataset for testing."""
    return Dataset.from_dict({
        "prompt": [f"Question {i}" for i in range(size)],
        "chosen": [f"Good answer {i}" for i in range(size)],
        "rejected": [f"Bad answer {i}" for i in range(size)]
    })
```

### Test Configuration Files
```python
# tests/fixtures/config_fixtures.py
SAMPLE_SFT_CONFIG = {
    "model": {
        "name": "microsoft/DialoGPT-small",
        "quantization": {"enabled": False}
    },
    "training": {
        "epochs": 1,
        "learning_rate": 5e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 1
    },
    "data": {
        "max_length": 128,
        "train_file": "tests/fixtures/sample_data/train.json"
    }
}

SAMPLE_DPO_CONFIG = {
    "model": {
        "name": "microsoft/DialoGPT-small",
        "quantization": {"enabled": False}
    },
    "training": {
        "epochs": 1,
        "learning_rate": 5e-6,
        "batch_size": 2,
        "beta": 0.1
    },
    "data": {
        "max_length": 128,
        "train_file": "tests/fixtures/sample_data/preference.json"
    }
}
```

## Debugging Tests

### Common Debugging Techniques

#### Using pytest with debugger
```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest --pdb -x

# Use ipdb for better debugging
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb
```

#### Verbose output for debugging
```bash
# Show all output (including print statements)
pytest -s

# Show local variables in tracebacks
pytest --tb=long

# Show only failed tests
pytest --tb=short --no-header -q
```

#### Debugging specific tests
```python
def test_debug_example():
    """Example test with debugging."""
    import pdb; pdb.set_trace()  # Debugger breakpoint
    
    # Test code here
    result = some_function()
    
    # Add logging for debugging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug(f"Result: {result}")
    
    assert result is not None
```

## Best Practices

### Test Organization
1. **Group related tests** in classes
2. **Use descriptive test names** that explain what is being tested
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Keep tests independent** - each test should be able to run in isolation
5. **Use fixtures** for common setup and teardown

### Test Quality
1. **Test both happy path and edge cases**
2. **Include error condition testing**
3. **Mock external dependencies** to ensure test isolation
4. **Use parametrized tests** for testing multiple scenarios
5. **Keep tests fast** - use mocks for expensive operations

### Coverage Guidelines
1. **Aim for high coverage** but focus on quality over quantity
2. **Test critical paths** thoroughly
3. **Include integration tests** for component interactions
4. **Test configuration validation** extensively
5. **Cover error handling** and edge cases

### Maintenance
1. **Update tests** when code changes
2. **Remove obsolete tests** when features are removed
3. **Refactor tests** to reduce duplication
4. **Document complex test scenarios**
5. **Review test failures** promptly

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest with proper path
pytest --import-mode=importlib
```

#### GPU Tests on CPU-only Systems
```python
# Skip GPU tests when CUDA is not available
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_training():
    pass
```

#### Memory Issues
```python
# Clean up after memory-intensive tests
def test_memory_intensive():
    try:
        # Test code
        pass
    finally:
        torch.cuda.empty_cache()
        gc.collect()
```

#### Slow Tests
```python
# Mark slow tests and run separately
@pytest.mark.slow
def test_full_training():
    pass

# Run without slow tests
# pytest -m "not slow"
```

### Test Environment Issues

#### Environment Variables
```bash
# Set test environment variables
export TESTING=true
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
```

#### Temporary Files
```python
# Proper cleanup of temporary files
import tempfile
import shutil

def test_with_temp_files():
    temp_dir = tempfile.mkdtemp()
    try:
        # Test code using temp_dir
        pass
    finally:
        shutil.rmtree(temp_dir)
```

## Related Documentation

- <mcfile name="CONTRIBUTING.md" path="CONTRIBUTING.md"></mcfile> - Contribution guidelines including testing requirements
- <mcfile name="Makefile" path="Makefile"></mcfile> - Build and test automation commands
- <mcfile name="pytest.ini" path="pytest.ini"></mcfile> - Pytest configuration
- <mcfile name="requirements-test.txt" path="requirements-test.txt"></mcfile> - Test dependencies
- <mcfile name="run_tests.py" path="run_tests.py"></mcfile> - Custom test runner script