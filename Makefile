# Arabic Qwen Base Fine-tuning Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev install-test clean test test-unit test-integration test-coverage lint format type-check security-check docs docs-serve build docker-build docker-run docker-dev setup-env validate-config train-sft train-dpo evaluate serve deploy-local deploy-k8s monitoring backup restore

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
POETRY := poetry
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
HELM := helm

# Project settings
PROJECT_NAME := arabic-qwen-base-finetuning
VERSION := $(shell python -c "import src; print(src.__version__)" 2>/dev/null || echo "1.0.0")
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_IMAGE_LATEST := $(PROJECT_NAME):latest

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config
DATA_DIR := data
CHECKPOINTS_DIR := checkpoints
LOGS_DIR := logs
REPORTS_DIR := reports

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(CYAN)Arabic Qwen Base Fine-tuning - Available Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup and Installation:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "install|setup|clean"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "test|lint|format|type-check|docs"
	@echo ""
	@echo "$(YELLOW)Training and Evaluation:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "train|evaluate|validate"
	@echo ""
	@echo "$(YELLOW)Docker and Deployment:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "docker|deploy|serve|monitoring"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "backup|restore|security"
	@echo ""

# =============================================================================
# Setup and Installation
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Production dependencies installed successfully!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"

install-test: ## Install test dependencies only
	@echo "$(BLUE)Installing test dependencies...$(NC)"
	$(PIP) install -r requirements-test.txt
	@echo "$(GREEN)Test dependencies installed successfully!$(NC)"

setup-env: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if [ ! -f .env ]; then cp .env.example .env; echo "$(YELLOW)Created .env file from template$(NC)"; fi
	@mkdir -p $(DATA_DIR) $(CHECKPOINTS_DIR) $(LOGS_DIR) $(REPORTS_DIR)
	@echo "$(GREEN)Development environment setup complete!$(NC)"

clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .tox/ htmlcov/
	@echo "$(GREEN)Cleanup complete!$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v
	@echo "$(GREEN)All tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "not integration"
	@echo "$(GREEN)Unit tests completed!$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "integration"
	@echo "$(GREEN)Integration tests completed!$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-fast: ## Run tests quickly (skip slow tests)
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "not slow"
	@echo "$(GREEN)Fast tests completed!$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 $(SRC_DIR) $(TEST_DIR)
	pylint $(SRC_DIR)
	@echo "$(GREEN)Linting checks completed!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Code formatting completed!$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Code formatting check completed!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy $(SRC_DIR)
	@echo "$(GREEN)Type checking completed!$(NC)"

security-check: ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r $(SRC_DIR)
	safety check
	@echo "$(GREEN)Security checks completed!$(NC)"

quality: format lint type-check security-check ## Run all code quality checks
	@echo "$(GREEN)All code quality checks completed!$(NC)"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd $(DOCS_DIR) && make html
	@echo "$(GREEN)Documentation built successfully!$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation build...$(NC)"
	cd $(DOCS_DIR) && make clean
	@echo "$(GREEN)Documentation cleaned!$(NC)"

# =============================================================================
# Configuration and Validation
# =============================================================================

validate-config: ## Validate configuration files
	@echo "$(BLUE)Validating configuration files...$(NC)"
	$(PYTHON) -m src.config.validator --config-dir $(CONFIG_DIR)
	@echo "$(GREEN)Configuration validation completed!$(NC)"

generate-config: ## Generate example configuration files
	@echo "$(BLUE)Generating example configuration files...$(NC)"
	$(PYTHON) -m src.config.generator --output-dir $(CONFIG_DIR)/examples
	@echo "$(GREEN)Example configurations generated!$(NC)"

# =============================================================================
# Training and Evaluation
# =============================================================================

train-sft: ## Run SFT training with default config
	@echo "$(BLUE)Starting SFT training...$(NC)"
	$(PYTHON) -m src.training.train --config $(CONFIG_DIR)/sft_config.yaml
	@echo "$(GREEN)SFT training completed!$(NC)"

train-dpo: ## Run DPO training with default config
	@echo "$(BLUE)Starting DPO training...$(NC)"
	$(PYTHON) -m src.training.train --config $(CONFIG_DIR)/dpo_config.yaml
	@echo "$(GREEN)DPO training completed!$(NC)"

train-kto: ## Run KTO training with default config
	@echo "$(BLUE)Starting KTO training...$(NC)"
	$(PYTHON) -m src.training.train --config $(CONFIG_DIR)/kto_config.yaml
	@echo "$(GREEN)KTO training completed!$(NC)"

evaluate: ## Run evaluation on trained model
	@echo "$(BLUE)Running model evaluation...$(NC)"
	$(PYTHON) -m src.evaluation.evaluate --config $(CONFIG_DIR)/eval_config.yaml
	@echo "$(GREEN)Evaluation completed!$(NC)"

benchmark: ## Run comprehensive benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) -m src.evaluation.benchmark --config $(CONFIG_DIR)/benchmark_config.yaml
	@echo "$(GREEN)Benchmarks completed!$(NC)"

# =============================================================================
# Model Serving
# =============================================================================

serve: ## Start model serving API
	@echo "$(BLUE)Starting model serving API...$(NC)"
	$(PYTHON) -m src.serving.api --config $(CONFIG_DIR)/serving_config.yaml

serve-gradio: ## Start Gradio interface
	@echo "$(BLUE)Starting Gradio interface...$(NC)"
	$(PYTHON) -m src.serving.gradio_app --config $(CONFIG_DIR)/serving_config.yaml

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t $(DOCKER_IMAGE) -t $(DOCKER_IMAGE_LATEST) .
	@echo "$(GREEN)Docker image built successfully!$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	$(DOCKER) build --target development -t $(PROJECT_NAME):dev .
	@echo "$(GREEN)Development Docker image built successfully!$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	$(DOCKER) run -it --rm -v $(PWD):/workspace $(DOCKER_IMAGE_LATEST)

docker-dev: ## Run development Docker container
	@echo "$(BLUE)Running development Docker container...$(NC)"
	$(DOCKER_COMPOSE) up dev

docker-train: ## Run training in Docker
	@echo "$(BLUE)Running training in Docker...$(NC)"
	$(DOCKER_COMPOSE) up train

docker-serve: ## Run serving in Docker
	@echo "$(BLUE)Running serving in Docker...$(NC)"
	$(DOCKER_COMPOSE) up api

docker-clean: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker images and containers...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f
	@echo "$(GREEN)Docker cleanup completed!$(NC)"

# =============================================================================
# Deployment
# =============================================================================

deploy-local: ## Deploy locally with docker-compose
	@echo "$(BLUE)Deploying locally...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Local deployment completed!$(NC)"

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	$(KUBECTL) apply -f k8s/
	@echo "$(GREEN)Kubernetes deployment completed!$(NC)"

deploy-helm: ## Deploy using Helm
	@echo "$(BLUE)Deploying with Helm...$(NC)"
	$(HELM) upgrade --install $(PROJECT_NAME) ./helm/$(PROJECT_NAME)
	@echo "$(GREEN)Helm deployment completed!$(NC)"

undeploy-k8s: ## Remove Kubernetes deployment
	@echo "$(BLUE)Removing Kubernetes deployment...$(NC)"
	$(KUBECTL) delete -f k8s/
	@echo "$(GREEN)Kubernetes deployment removed!$(NC)"

# =============================================================================
# Monitoring and Logging
# =============================================================================

monitoring: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) up -d prometheus grafana
	@echo "$(GREEN)Monitoring stack started!$(NC)"

logs: ## View application logs
	@echo "$(BLUE)Viewing application logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

metrics: ## View training metrics
	@echo "$(BLUE)Viewing training metrics...$(NC)"
	tensorboard --logdir=$(LOGS_DIR)

# =============================================================================
# Data Management
# =============================================================================

download-data: ## Download sample datasets
	@echo "$(BLUE)Downloading sample datasets...$(NC)"
	$(PYTHON) -m src.data.download --output-dir $(DATA_DIR)
	@echo "$(GREEN)Sample datasets downloaded!$(NC)"

process-data: ## Process raw data
	@echo "$(BLUE)Processing raw data...$(NC)"
	$(PYTHON) -m src.data.processor --input-dir $(DATA_DIR)/raw --output-dir $(DATA_DIR)/processed
	@echo "$(GREEN)Data processing completed!$(NC)"

validate-data: ## Validate dataset format
	@echo "$(BLUE)Validating dataset format...$(NC)"
	$(PYTHON) -m src.data.validator --data-dir $(DATA_DIR)
	@echo "$(GREEN)Data validation completed!$(NC)"

# =============================================================================
# Backup and Restore
# =============================================================================

backup: ## Backup models and checkpoints
	@echo "$(BLUE)Creating backup...$(NC)"
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz $(CHECKPOINTS_DIR) $(CONFIG_DIR) $(LOGS_DIR)
	@echo "$(GREEN)Backup created successfully!$(NC)"

restore: ## Restore from backup (specify BACKUP_FILE)
	@echo "$(BLUE)Restoring from backup...$(NC)"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "$(RED)Please specify BACKUP_FILE=<filename>$(NC)"; exit 1; fi
	tar -xzf $(BACKUP_FILE)
	@echo "$(GREEN)Restore completed!$(NC)"

# =============================================================================
# Utilities
# =============================================================================

version: ## Show project version
	@echo "$(CYAN)Project Version: $(VERSION)$(NC)"

status: ## Show project status
	@echo "$(CYAN)Project Status:$(NC)"
	@echo "  Version: $(VERSION)"
	@echo "  Python: $(shell python --version)"
	@echo "  Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "  Kubernetes: $(shell kubectl version --client --short 2>/dev/null || echo 'Not installed')"
	@echo "  GPU: $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not available')"

check-deps: ## Check if all dependencies are installed
	@echo "$(BLUE)Checking dependencies...$(NC)"
	$(PYTHON) -c "import pkg_resources; pkg_resources.require(open('requirements.txt').read().splitlines())"
	@echo "$(GREEN)All dependencies are satisfied!$(NC)"

update-deps: ## Update dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)Dependencies updated!$(NC)"

# =============================================================================
# Development Shortcuts
# =============================================================================

dev-setup: install-dev setup-env ## Complete development setup
	@echo "$(GREEN)Development setup completed!$(NC)"

dev-test: format lint test-fast ## Quick development test cycle
	@echo "$(GREEN)Development test cycle completed!$(NC)"

dev-full: format lint type-check test-coverage ## Full development check
	@echo "$(GREEN)Full development check completed!$(NC)"

ci: format-check lint type-check security-check test-coverage ## CI/CD pipeline simulation
	@echo "$(GREEN)CI/CD pipeline simulation completed!$(NC)"

# =============================================================================
# Release Management
# =============================================================================

release-check: ci docs ## Pre-release checks
	@echo "$(GREEN)Pre-release checks completed!$(NC)"

release-build: clean docker-build ## Build release artifacts
	@echo "$(GREEN)Release artifacts built!$(NC)"

release-tag: ## Tag current version for release
	@echo "$(BLUE)Tagging version $(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	@echo "$(GREEN)Version $(VERSION) tagged and pushed!$(NC)"

# =============================================================================
# Special Targets
# =============================================================================

# Target to run when setting up a new development environment
first-time-setup: clean install-dev setup-env validate-config generate-config
	@echo "$(GREEN)First-time setup completed! You're ready to start developing.$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Review and customize .env file"
	@echo "  2. Download or prepare your training data"
	@echo "  3. Run 'make train-sft' to start training"

# Target for continuous integration
ci-full: clean install-dev format-check lint type-check security-check test-coverage docs
	@echo "$(GREEN)Full CI pipeline completed!$(NC)"

# Target for production deployment preparation
prod-ready: ci-full docker-build validate-config
	@echo "$(GREEN)Production deployment ready!$(NC)"