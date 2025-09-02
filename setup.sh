#!/bin/bash

# ZamAI Pro Models Strategy Setup Script
# This script helps you set up the complete project environment

set -e  # Exit on any error

echo "🚀 Setting up ZamAI Pro Models Strategy..."
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

print_status "Python 3 found"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_MAJOR=${PYTHON_VERSION%%.*}
PYTHON_VERSION_MINOR=${PYTHON_VERSION##*.}
print_info "Python version: $PYTHON_VERSION_MAJOR.$PYTHON_VERSION_MINOR"

if [ "$PYTHON_VERSION_MAJOR" -ne 3 ] || [ "$PYTHON_VERSION_MINOR" -lt 9 ] || [ "$PYTHON_VERSION_MINOR" -ge 12 ]; then
    print_error "This project requires Python version 3.9, 3.10, or 3.11 for 'tts' package compatibility."
    print_error "Your version is $PYTHON_VERSION_MAJOR.$PYTHON_VERSION_MINOR. Please switch to a compatible Python version."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_info "Installing Python dependencies..."
pip install -r requirements.txt
print_status "Dependencies installed"

# Create .env file from template if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file and add your Hugging Face token!"
    print_info "You can get your token from: https://huggingface.co/settings/tokens"
else
    print_info ".env file already exists"
fi

# Create necessary directories
print_info "Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p outputs
print_status "Directories created"

# Check if HF token is set
if [ -f ".env" ]; then
    source .env
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_hugging_face_token_here" ]; then
        print_warning "Hugging Face token not set in .env file"
        print_info "Your existing models will be used:"
        echo "  - tasal9/ZamAI-Mistral-7B-Pashto"
        echo "  - tasal9/ZamAI-Phi-3-Mini-Pashto"
        echo "  - tasal9/ZamAI-Whisper-v3-Pashto"
        echo "  - tasal9/Multilingual-ZamAI-Embeddings"
        echo "  - tasal9/ZamAI-LIama3-Pashto"
        echo "  - tasal9/pashto-base-bloom"
    else
        print_status "Hugging Face token found in .env"
    fi
fi

# Test model access (if token is available)
if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_hugging_face_token_here" ]; then
    print_info "Testing model access..."
    python3 -c "
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(token=os.getenv('HF_TOKEN'))

models = [
    'tasal9/ZamAI-Mistral-7B-Pashto',
    'tasal9/ZamAI-Phi-3-Mini-Pashto',
    'tasal9/ZamAI-Whisper-v3-Pashto',
    'tasal9/Multilingual-ZamAI-Embeddings'
]

for model in models:
    try:
        # Just check if we can access the model info
        model_info = client.get_model_status(model)
        print(f'✅ {model} - Accessible')
    except Exception as e:
        print(f'⚠️  {model} - {str(e)[:50]}...')
"
    print_status "Model access test completed"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================="
echo ""
print_info "Next steps:"
echo "1. Edit .env file and add your Hugging Face token"
echo "2. Run demos:"
echo "   python demos/chatbot_demo.py"
echo "   python demos/voice_demo.py"
echo "   python demos/business_demo.py"
echo ""
echo "3. Start the API server:"
echo "   python api/main.py"
echo ""
echo "4. Run fine-tuning (if you have datasets):"
echo "   python scripts/fine_tune_mistral.py --dataset your-dataset"
echo "   python scripts/fine_tune_phi3.py --dataset your-dataset"
echo ""
print_info "Your existing models:"
echo "🤖 tasal9/ZamAI-Mistral-7B-Pashto (Text Generation)"
echo "🤖 tasal9/ZamAI-Phi-3-Mini-Pashto (Text Generation)"
echo "🎤 tasal9/ZamAI-Whisper-v3-Pashto (Speech-to-Text)"
echo "🔤 tasal9/Multilingual-ZamAI-Embeddings (Feature Extraction)"
echo "🦙 tasal9/ZamAI-LIama3-Pashto (Private)"
echo "🌸 tasal9/pashto-base-bloom (0.6B parameters)"
echo ""
print_status "Setup complete! Happy coding! 🚀"
