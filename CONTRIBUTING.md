# Contributing to Arabic Qwen Base Fine-tuning 🤝

We welcome contributions from the community! This guide will help you get started with contributing to the Arabic Qwen Base Fine-tuning project.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Contributing Guidelines](#-contributing-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Issue Guidelines](#-issue-guidelines)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Community](#-community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [team@arabic-nlp.org](mailto:team@arabic-nlp.org).

## 🚀 Getting Started

### Ways to Contribute

- **🐛 Bug Reports**: Report bugs and issues
- **✨ Feature Requests**: Suggest new features or improvements
- **📝 Documentation**: Improve documentation and examples
- **🔧 Code Contributions**: Fix bugs, implement features, or optimize performance
- **🧪 Testing**: Add or improve tests
- **🌍 Translations**: Help with Arabic language support and localization
- **📊 Datasets**: Contribute Arabic datasets or data processing improvements
- **🎓 Examples**: Create tutorials, notebooks, or example scripts

### Before You Start

1. **Check existing issues**: Look through [existing issues](https://github.com/arabic-nlp/arabic-qwen-base-finetuning/issues) to see if your idea or bug has already been reported
2. **Read the documentation**: Familiarize yourself with the project structure and goals
3. **Join the discussion**: Participate in [GitHub Discussions](https://github.com/arabic-nlp/arabic-qwen-base-finetuning/discussions) to get feedback on your ideas

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA 11.8+ (for GPU development)
- Docker (optional, for containerized development)

### Local Development

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/arabic-qwen-base-finetuning.git
   cd arabic-qwen-base-finetuning
   ```

2. **Set up the development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev,testing]"
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests to verify setup**
   ```bash
   python run_tests.py
   ```

### Docker Development

```bash
# Build development image
docker-compose build dev

# Start development environment
docker-compose up dev

# Access the container
docker-compose exec dev bash
```

## 📋 Contributing Guidelines

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements
- `refactor/description` - Code refactoring
- `perf/description` - Performance improvements

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```bash
feat(training): add support for KTO optimization method
fix(data): resolve Arabic text encoding issues
docs(readme): update installation instructions
test(config): add tests for configuration validation
```

### Code Organization

```
src/
├── config/          # Configuration management
├── data/            # Data loading and processing
├── models/          # Model definitions and utilities
├── training/        # Training implementations
├── evaluation/      # Evaluation metrics and tools
├── utils/           # Utility functions
├── api/             # API endpoints
└── scripts/         # Command-line scripts
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create an issue** (if one doesn't exist) describing the problem or feature
2. **Fork the repository** and create a feature branch
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run the test suite** and ensure all tests pass
7. **Run code quality checks**

### Submitting Your PR

1. **Push your changes** to your fork
2. **Create a pull request** with a clear title and description
3. **Link the related issue** in the PR description
4. **Request review** from maintainers
5. **Address feedback** and update your PR as needed

### PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🐛 Issue Guidelines

### Bug Reports

When reporting bugs, please include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU, etc.)
- **Error messages** and stack traces
- **Minimal code example** if applicable

### Feature Requests

For feature requests, please provide:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation** approach
- **Alternatives considered**
- **Additional context** or examples

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements or additions to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `question` - Further information is requested
- `arabic` - Arabic language specific
- `training` - Related to model training
- `evaluation` - Related to model evaluation
- `performance` - Performance improvements

## 🎨 Code Style

### Python Code Style

We use the following tools for code formatting and quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check code quality
flake8 src/ tests/
mypy src/
```

### Style Guidelines

1. **Follow PEP 8** Python style guide
2. **Use type hints** for function parameters and return values
3. **Write docstrings** for all public functions and classes
4. **Keep functions small** and focused on a single responsibility
5. **Use meaningful variable names**
6. **Add comments** for complex logic
7. **Follow existing patterns** in the codebase

### Documentation Style

We use Google-style docstrings:

```python
def train_model(config: SFTConfig, dataset: Dataset) -> TrainingResults:
    """Train a model using the specified configuration.
    
    Args:
        config: Training configuration containing hyperparameters.
        dataset: Training dataset with input-output pairs.
        
    Returns:
        Training results including metrics and model path.
        
    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If training fails.
        
    Example:
        >>> config = SFTConfig.load("config.yaml")
        >>> dataset = load_dataset("arabic_data")
        >>> results = train_model(config, dataset)
    """
```

## 🧪 Testing

### Test Structure

```
tests/
├── test_config.py       # Configuration tests
├── test_data_loader.py  # Data loading tests
├── test_training.py     # Training tests
├── test_evaluation.py   # Evaluation tests
├── test_utils.py        # Utility function tests
└── conftest.py          # Test configuration
```

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --module config

# Run with coverage
python run_tests.py --coverage

# Run specific test file
pytest tests/test_config.py -v
```

### Writing Tests

1. **Write tests for new functionality**
2. **Use descriptive test names**
3. **Follow the AAA pattern** (Arrange, Act, Assert)
4. **Mock external dependencies**
5. **Test edge cases and error conditions**
6. **Keep tests independent** and isolated

```python
def test_sft_config_validation_with_invalid_learning_rate():
    """Test that SFTConfig raises ValueError for invalid learning rate."""
    # Arrange
    config_dict = {
        "model": {"name": "Qwen/Qwen2-7B"},
        "training": {"learning_rate": -0.001}  # Invalid negative LR
    }
    
    # Act & Assert
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        SFTConfig.from_dict(config_dict)
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **API Documentation**: Automatically generated from docstrings
3. **User Guides**: Step-by-step tutorials and examples
4. **Developer Documentation**: Architecture and contribution guides

### Documentation Guidelines

1. **Keep documentation up-to-date** with code changes
2. **Use clear, concise language**
3. **Include code examples** where appropriate
4. **Add screenshots** for UI-related features
5. **Link to related documentation**
6. **Test documentation examples** to ensure they work

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

## 🌍 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: [team@arabic-nlp.org](mailto:team@arabic-nlp.org)
- **Discord**: [Arabic NLP Community](https://discord.gg/arabic-nlp) (coming soon)

### Getting Help

1. **Check the documentation** first
2. **Search existing issues** and discussions
3. **Ask in GitHub Discussions** for general questions
4. **Create an issue** for bugs or specific problems
5. **Join community discussions** for broader topics

### Mentorship

We offer mentorship for new contributors:

- **Good First Issues**: Look for issues labeled `good first issue`
- **Mentorship Program**: Contact us for guided contribution opportunities
- **Code Reviews**: Learn from detailed feedback on your contributions
- **Pair Programming**: Schedule sessions with maintainers

## 🏆 Recognition

We recognize contributors in several ways:

- **Contributors List**: All contributors are listed in our README
- **Release Notes**: Significant contributions are highlighted in releases
- **Community Spotlight**: Featured contributors in our blog and social media
- **Swag**: Special contributors receive project merchandise

## 📝 License

By contributing to this project, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

## ❓ Questions?

If you have any questions about contributing, please:

1. Check this guide and the [FAQ](docs/FAQ.md)
2. Search [existing discussions](https://github.com/arabic-nlp/arabic-qwen-base-finetuning/discussions)
3. Create a new discussion or issue
4. Contact us at [team@arabic-nlp.org](mailto:team@arabic-nlp.org)

---

**Thank you for contributing to Arabic Qwen Base Fine-tuning! 🙏**

Your contributions help advance Arabic NLP research and make these tools accessible to the broader community.