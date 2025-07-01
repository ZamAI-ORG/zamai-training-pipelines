#!/bin/bash
# Clean Repository Setup Script
# This script creates a clean repository without any token exposure

echo "🧹 Creating clean repository setup..."

# Create a new temporary directory
TEMP_DIR="ZamAI-Pro-Models-Strategy2-clean"
rm -rf "$TEMP_DIR"
mkdir "$TEMP_DIR"
cd "$TEMP_DIR"

echo "📁 Setting up clean directory structure..."

# Initialize new git repository
git init
git config user.name "ZamAI Developer"
git config user.email "dev@zamai.com"

# Create directory structure
mkdir -p {demos,scripts,api,tests,model_cards,business_data/{documents,embeddings,faiss_index},datasets/{cache,processed},docker/business-tools,.github/workflows,hf_space}

echo "📝 Creating clean configuration files..."

# Create clean .env.example (no actual tokens)
cat > .env.example << 'EOF'
# Environment Variables for ZamAI Pro Models Strategy

# Hugging Face Configuration
HF_TOKEN=your_hugging_face_token_here
HF_ORG=tasal9

# Your Existing Model Repositories
MISTRAL_EDU_MODEL=tasal9/ZamAI-Mistral-7B-Pashto
PHI3_BUSINESS_MODEL=tasal9/ZamAI-Phi-3-Mini-Pashto
WHISPER_MODEL=tasal9/ZamAI-Whisper-v3-Pashto
LLAMA3_MODEL=tasal9/ZamAI-LIama3-Pashto
BLOOM_MODEL=tasal9/pashto-base-bloom
EMBEDDINGS_MODEL=tasal9/Multilingual-ZamAI-Embeddings

# Dataset Configuration
PASHTO_DATASET_REPO=tasal9/Pashto-Dataset-Creating-Dataset
DATASET_CACHE_DIR=datasets/cache
PROCESSED_DATASET_DIR=datasets/processed

# Demo Configuration
ENHANCED_VOICE_PORT=7864
BASIC_VOICE_PORT=7861
ADVANCED_VOICE_PORT=7862
TUTOR_BOT_PORT=7865

# UI Configuration
GRADIO_THEME=default
GRADIO_SHARE=true
GRADIO_DEBUG=true

# HuggingFace Space Configuration
HF_SPACE_NAME=zamai-enhanced-tutor-bot
HF_SPACE_VISIBILITY=public

# Business Tools Configuration
BUSINESS_TOOLS_PORT=7866
BUSINESS_DATA_DIR=business_data
FAISS_INDEX_DIR=business_data/faiss_index
DOCUMENT_STORAGE_DIR=business_data/documents

# CI/CD Configuration
DOCKER_REGISTRY=ghcr.io
STAGING_SERVER=staging.zamai.internal
PRODUCTION_SERVER=production.zamai.internal
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
EOF

# Create comprehensive .gitignore
cat > .gitignore << 'EOF'
# Environment and Secrets
.env
.env.local
.env.production
.env.staging
*.key
*.pem
config.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# AI/ML
models/
checkpoints/
wandb/
.neptune/

# Data
business_data/
datasets/
*.csv
*.json
!requirements.txt
!package.json
!**/example*.json

# Temporary files
tmp/
temp/
.cache/
.pytest_cache/

# Gradio
gradio_cached_examples/
EOF

echo "✅ Clean repository structure created in $TEMP_DIR"
echo ""
echo "📋 Next steps:"
echo "1. Copy your source files to $TEMP_DIR (excluding .env)"
echo "2. Review all files to ensure no tokens are present"
echo "3. Initialize and push the clean repository"
echo ""
echo "🔒 Security checklist:"
echo "- ✅ .env file excluded from git"
echo "- ✅ Comprehensive .gitignore created"
echo "- ✅ Only .env.example with placeholder tokens"
echo "- ✅ Clean git history without any tokens"
