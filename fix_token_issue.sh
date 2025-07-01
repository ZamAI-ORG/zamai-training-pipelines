#!/bin/bash
# Quick Fix for Token Exposure Issue
# This script removes the token from the current repository

echo "🔒 Fixing HuggingFace token exposure issue..."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the ZamAI-Pro-Models-Strategy2 directory"
    exit 1
fi

echo "📋 Current git status:"
git status --short

echo ""
echo "🧹 Cleaning up token references..."

# 1. Remove .env file if it exists and contains real token
if [ -f ".env" ]; then
    if grep -q "hf_" .env; then
        echo "⚠️  Found HuggingFace token in .env file - removing from git"
        git rm --cached .env 2>/dev/null || true
        rm .env
    fi
fi

# 2. Create .env.example with safe placeholder
echo "📝 Creating safe .env.example..."
cat > .env.example << 'EOF'
# Environment Variables for ZamAI Pro Models Strategy

# Hugging Face Configuration (IMPORTANT: Add your real token here locally)
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

# 3. Ensure .gitignore exists and covers all secret files
echo "🛡️  Updating .gitignore for security..."
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

# 4. Check for any remaining token references in files
echo "🔍 Checking for remaining token references..."
TOKEN_FILES=$(grep -r "hf_" . --exclude-dir=.git --exclude="*.sh" 2>/dev/null || true)
if [ ! -z "$TOKEN_FILES" ]; then
    echo "⚠️  Found potential token references:"
    echo "$TOKEN_FILES"
    echo ""
    echo "Please manually review and replace any real tokens with 'your_hugging_face_token_here'"
fi

# 5. Add the safe files to git
echo "📦 Adding safe files to git..."
git add .gitignore .env.example

# 6. Create commit
echo "💾 Creating security fix commit..."
git commit -m "🔒 Security: Fix token exposure

- Remove .env file from repository
- Add comprehensive .gitignore
- Create safe .env.example template
- Ensure no real tokens in repository"

echo ""
echo "✅ Security fixes applied!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Add your real HuggingFace token to .env (locally only)"
echo "3. Try pushing again: git push origin main"
echo ""
echo "💡 Local setup command:"
echo "cp .env.example .env"
echo "# Then edit .env to add your real token"
