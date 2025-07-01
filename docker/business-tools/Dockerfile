# ZamAI Business Tools - Production Docker Image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional business tools dependencies
RUN pip install --no-cache-dir \
    faiss-cpu \
    pandas \
    numpy \
    scikit-learn \
    nltk

# Copy application code
COPY demos/enhanced_business_tools.py ./demos/
COPY .env.example .env

# Create necessary directories
RUN mkdir -p business_data/documents business_data/embeddings business_data/faiss_index

# Build arguments for model configuration
ARG HF_TOKEN
ARG PHI3_MODEL=tasal9/ZamAI-Phi-3-Mini-Pashto
ARG EMBEDDINGS_MODEL=tasal9/Multilingual-ZamAI-Embeddings

# Set environment variables from build args
ENV HF_TOKEN=${HF_TOKEN}
ENV PHI3_BUSINESS_MODEL=${PHI3_MODEL}
ENV EMBEDDINGS_MODEL=${EMBEDDINGS_MODEL}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7866/health || exit 1

# Expose port
EXPOSE 7866

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash zamai && \
    chown -R zamai:zamai /app
USER zamai

# Start command
CMD ["python", "demos/enhanced_business_tools.py"]
