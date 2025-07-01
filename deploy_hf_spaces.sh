#!/bin/bash

# HuggingFace Spaces Deployment Script
set -e

echo "🤗 Starting HuggingFace Spaces Deployment..."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo "Please set your HuggingFace token: export HF_TOKEN=your_token_here"
    exit 1
fi

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
fi

# Your HuggingFace username
HF_USERNAME="tasal9"  # Replace with your actual username

# Deploy Voice Assistant Space
echo "🎤 Deploying Voice Assistant Space..."
cd hf_spaces/voice-assistant

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    git init
    git lfs install
    git remote add origin https://huggingface.co/spaces/$HF_USERNAME/zamai-voice-assistant
fi

# Commit and push
git add .
git commit -m "🎤 Deploy ZamAI Voice Assistant to HuggingFace Spaces" || echo "No changes to commit"
git push origin main

echo "✅ Voice Assistant Space deployed!"
cd ../..

# Deploy Business Tools Space
echo "📄 Deploying Business Tools Space..."
cd hf_spaces/business-tools

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    git init
    git lfs install
    git remote add origin https://huggingface.co/spaces/$HF_USERNAME/zamai-business-tools
fi

# Commit and push
git add .
git commit -m "📄 Deploy ZamAI Business Tools to HuggingFace Spaces" || echo "No changes to commit"
git push origin main

echo "✅ Business Tools Space deployed!"
cd ../..

# Deploy Enhanced Tutor Bot Space (using existing hf_space directory)
echo "🎓 Deploying Enhanced Tutor Bot Space..."
cd hf_space

# Update the space with latest changes
git add .
git commit -m "🎓 Update ZamAI Enhanced Tutor Bot" || echo "No changes to commit"
git push origin main

echo "✅ Enhanced Tutor Bot Space updated!"
cd ..

echo ""
echo "🌟 HuggingFace Spaces Deployment Complete!"
echo ""
echo "📱 Your Spaces are now live at:"
echo "├── 🎤 Voice Assistant: https://huggingface.co/spaces/$HF_USERNAME/zamai-voice-assistant"
echo "├── 📄 Business Tools: https://huggingface.co/spaces/$HF_USERNAME/zamai-business-tools"
echo "└── 🎓 Enhanced Tutor Bot: https://huggingface.co/spaces/$HF_USERNAME/zamai-enhanced-tutor-bot"
echo ""
echo "🔧 Space Management:"
echo "├── 📊 View analytics on HuggingFace Hub"
echo "├── ⚙️ Configure settings and visibility"
echo "├── 🔄 Auto-deploy on git push"
echo "└── 📝 Update README.md for better discoverability"
