#!/bin/bash
# Deploy fine-tuning Spaces to Hugging Face Hub
# Requires HF_TOKEN to be set

set -e

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set. Please set it in .env file"
    exit 1
fi

# Login to Hugging Face Hub
huggingface-cli login --token "$HF_TOKEN"

echo "🚀 Deploying Fine-tuning Spaces to Hugging Face Hub..."
echo ""

# Function to deploy a Space
deploy_space() {
    local SPACE_DIR=$1
    local SPACE_NAME=$2
    local SPACE_TITLE=$3
    local VISIBILITY=${4:-public}  # public or private
    
    echo "📦 Deploying $SPACE_TITLE..."
    
    # Create Space repository if it doesn't exist
    SPACE_REPO="${HF_ORG}/${SPACE_NAME}"
    
    echo "  Creating Space: $SPACE_REPO"
    huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk gradio --org "$HF_ORG" --exist_ok || true
    
    # Upload files to Space
    echo "  Uploading files from $SPACE_DIR..."
    huggingface-cli upload "$SPACE_REPO" "$SPACE_DIR/app.py" app.py --repo-type space
    huggingface-cli upload "$SPACE_REPO" "$SPACE_DIR/requirements.txt" requirements.txt --repo-type space
    huggingface-cli upload "$SPACE_REPO" "$SPACE_DIR/README.md" README.md --repo-type space
    
    echo "  ✅ Successfully deployed: https://huggingface.co/spaces/$SPACE_REPO"
    echo ""
}

# Deploy Phi-3 Fine-tuning Space
if [ -d "hf_spaces/phi3-finetuning" ]; then
    deploy_space "hf_spaces/phi3-finetuning" "zamai-phi3-finetuning" "Phi-3 Fine-tuning" "public"
fi

# Deploy MT5 Fine-tuning Space
if [ -d "hf_spaces/mt5-finetuning" ]; then
    deploy_space "hf_spaces/mt5-finetuning" "zamai-mt5-finetuning" "MT5 Multilingual Fine-tuning" "public"
fi

# Deploy existing Voice Assistant Space (if exists)
if [ -d "hf_spaces/voice-assistant" ]; then
    deploy_space "hf_spaces/voice-assistant" "zamai-voice-assistant" "Voice Assistant" "public"
fi

# Deploy existing Business Tools Space (if exists)
if [ -d "hf_spaces/business-tools" ]; then
    deploy_space "hf_spaces/business-tools" "zamai-business-tools" "Business Tools" "public"
fi

echo "🎉 All Spaces deployed successfully!"
echo ""
echo "📝 Your Spaces:"
echo "  - Phi-3 Fine-tuning: https://huggingface.co/spaces/${HF_ORG}/zamai-phi3-finetuning"
echo "  - MT5 Fine-tuning: https://huggingface.co/spaces/${HF_ORG}/zamai-mt5-finetuning"
echo "  - Voice Assistant: https://huggingface.co/spaces/${HF_ORG}/zamai-voice-assistant"
echo "  - Business Tools: https://huggingface.co/spaces/${HF_ORG}/zamai-business-tools"
echo ""
echo "💡 Pro Tips:"
echo "  - Upgrade Spaces to Pro for better GPU access"
echo "  - Enable analytics to track usage"
echo "  - Add custom domains for branding"
