# Arabic Qwen Base Fine-tuning Docker Image
# Multi-stage build for optimized production deployment

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# =============================================================================
# Development Stage - For development and testing
# =============================================================================
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    htop \
    tmux \
    screen \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-test.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-test.txt

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e ".[dev,testing,all]"

# Create necessary directories
RUN mkdir -p /app/cache /app/logs /app/checkpoints /app/reports /app/data

# Set permissions
RUN chmod +x /app/scripts/*.py 2>/dev/null || true

# Expose ports for development servers
EXPOSE 8000 8888 6006

# Default command for development
CMD ["bash"]

# =============================================================================
# Production Stage - Optimized for deployment
# =============================================================================
FROM base as production

# Copy only requirements first
COPY requirements.txt ./

# Install only production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (excluding development files)
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py ./
COPY README.md ./
COPY LICENSE ./

# Install the package
RUN pip install --no-cache-dir .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories and set permissions
RUN mkdir -p /app/cache /app/logs /app/checkpoints /app/reports /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Training Stage - Optimized for model training
# =============================================================================
FROM base as training

# Install additional training dependencies
RUN apt-get update && apt-get install -y \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install dependencies with optimizations for training
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        deepspeed \
        flash-attn \
        xformers \
        triton

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY setup.py ./
COPY README.md ./

# Install the package
RUN pip install --no-cache-dir .

# Create directories for training
RUN mkdir -p /app/cache /app/logs /app/checkpoints /app/reports /app/data

# Set environment variables for training
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA=1

# Default command for training
CMD ["python", "-m", "src.scripts.train"]

# =============================================================================
# Inference Stage - Optimized for model serving
# =============================================================================
FROM base as inference

# Install minimal dependencies for inference
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    && rm -rf /root/.cache/pip

# Copy only necessary files for inference
COPY src/models/ ./src/models/
COPY src/utils/ ./src/utils/
COPY src/api/ ./src/api/
COPY src/__init__.py ./src/
COPY setup.py ./

# Install the package
RUN pip install --no-cache-dir .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app/models /app/cache && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Jupyter Stage - For interactive development and research
# =============================================================================
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    jupyter-contrib-nbextensions \
    jupyterlab-git \
    && jupyter contrib nbextension install --user

# Configure Jupyter
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Copy notebooks
COPY notebooks/ ./notebooks/

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Build Arguments and Labels
# =============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="Arabic Qwen Base Fine-tuning" \
      org.label-schema.description="Framework for fine-tuning Qwen models on Arabic datasets" \
      org.label-schema.url="https://github.com/artaasd95/arabic-qwen-base-finetuning" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/artaasd95/arabic-qwen-base-finetuning" \
      org.label-schema.vendor="Arabic NLP Team" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      maintainer="team@arabic-nlp.org"

# =============================================================================
# Usage Examples:
# 
# Build development image:
# docker build --target development -t arabic-qwen-dev .
# 
# Build production image:
# docker build --target production -t arabic-qwen-prod .
# 
# Build training image:
# docker build --target training -t arabic-qwen-train .
# 
# Build inference image:
# docker build --target inference -t arabic-qwen-infer .
# 
# Build Jupyter image:
# docker build --target jupyter -t arabic-qwen-jupyter .
# 
# Run development container:
# docker run -it --gpus all -v $(pwd):/app -p 8000:8000 arabic-qwen-dev
# 
# Run training container:
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints arabic-qwen-train
# 
# Run inference container:
# docker run -d --gpus all -p 8000:8000 -v $(pwd)/models:/app/models arabic-qwen-infer
# 
# Run Jupyter container:
# docker run -d --gpus all -p 8888:8888 -v $(pwd):/app arabic-qwen-jupyter
# =============================================================================