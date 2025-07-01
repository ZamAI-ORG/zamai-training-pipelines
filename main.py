"""
ZamAI Pro Models Strategy - Main Launcher
This script provides a unified interface to launch all components
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_banner():
    """Print the ZamAI banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                   🚀 ZamAI Pro Models Strategy            ║
    ║                                                           ║
    ║    Central hub for AI models with Pashto language        ║
    ║    support - Educational, Voice, and Business AI         ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    
    # Check if .env exists
    if not os.path.exists('.env'):
        issues.append("❌ .env file not found. Run setup.sh first.")
    
    # Check if HF token is set
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token or hf_token == 'your_hugging_face_token_here':
        issues.append("⚠️  Hugging Face token not set in .env file")
    
    # Check if virtual environment exists
    if not os.path.exists('venv'):
        issues.append("❌ Virtual environment not found. Run setup.sh first.")
    
    return issues

def list_available_models():
    """List your available models"""
    models = {
        "Text Generation Models": [
            "tasal9/ZamAI-Mistral-7B-Pashto",
            "tasal9/ZamAI-Phi-3-Mini-Pashto", 
            "tasal9/ZamAI-LIama3-Pashto (private)",
            "tasal9/pashto-base-bloom"
        ],
        "Speech Models": [
            "tasal9/ZamAI-Whisper-v3-Pashto"
        ],
        "Embedding Models": [
            "tasal9/Multilingual-ZamAI-Embeddings"
        ]
    }
    
    print("\n🤖 Your Available Models:")
    print("=" * 50)
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  • {model}")

def launch_demo(demo_type):
    """Launch a specific demo"""
    demo_commands = {
        "chatbot": "python demos/chatbot_demo.py",
        "voice": "python demos/voice_assistant_inference_api.py",
        "voice-basic": "python demos/voice_demo.py",
        "voice-advanced": "python demos/voice_assistant_advanced.py",
        "business": "python demos/business_demo.py",
        "all": "python demos/chatbot_demo.py & python demos/voice_assistant_inference_api.py & python demos/business_demo.py"
    }
    
    if demo_type not in demo_commands:
        print(f"❌ Unknown demo type: {demo_type}")
        print("Available demos:", ", ".join(demo_commands.keys()))
        return False
    
    command = demo_commands[demo_type]
    
    print(f"🚀 Launching {demo_type} demo...")
    print(f"Command: {command}")
    
    if demo_type == "voice":
        print("🎤 Voice Assistant Features:")
        print("   • Whisper → LLaMA-3 → TTS Pipeline")
        print("   • Inference API Integration")
        print("   • Context-aware responses")
        print("   • Performance metrics")
        print("   • Multi-language support (English/Pashto)")
        print("🌐 Interface: http://localhost:7862")
    elif demo_type == "voice-basic":
        print("🎤 Basic Voice Assistant")
        print("🌐 Interface: http://localhost:7861")
    elif demo_type == "voice-advanced":
        print("🎤 Advanced Voice Assistant with Analytics")
        print("🌐 Interface: http://localhost:7862")
    
    try:
        subprocess.run([sys.executable, demo_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching demo: {e}")
    except KeyboardInterrupt:
        print(f"\n✅ {demo_type} demo stopped by user")

def launch_api():
    """Launch the FastAPI server"""
    api_file = 'api/main.py'
    if not os.path.exists(api_file):
        print(f"❌ API file not found: {api_file}")
        return
    
    print("🚀 Launching FastAPI server...")
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'api.main:app', 
            '--host', '0.0.0.0', 
            '--port', '8000', 
            '--reload'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching API server: {e}")
    except KeyboardInterrupt:
        print("\n✅ API server stopped by user")

def run_fine_tuning(model_type, dataset=None):
    """Run fine-tuning for a specific model"""
    scripts = {
        'mistral': 'scripts/fine_tune_mistral.py',
        'phi3': 'scripts/fine_tune_phi3.py'
    }
    
    if model_type not in scripts:
        print(f"❌ Unknown model type: {model_type}")
        print(f"Available types: {', '.join(scripts.keys())}")
        return
    
    script_file = scripts[model_type]
    if not os.path.exists(script_file):
        print(f"❌ Fine-tuning script not found: {script_file}")
        return
    
    print(f"🔥 Starting fine-tuning for {model_type}...")
    
    cmd = [sys.executable, script_file]
    if dataset:
        cmd.extend(['--dataset', dataset])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during fine-tuning: {e}")
    except KeyboardInterrupt:
        print(f"\n✅ Fine-tuning stopped by user")

def show_help():
    """Show help information"""
    help_text = """
🎯 ZamAI Pro Models Strategy - Command Help

📋 Available Commands:
  
  🚀 Launch Demos:
    python main.py demo chatbot    # Educational chatbot with Pashto support
    python main.py demo voice      # Voice assistant (STT + LLM + TTS)
    python main.py demo business   # Business document processor
    
  🌐 Launch API Server:
    python main.py api             # Start FastAPI server on port 8000
    
  🔥 Fine-tuning:
    python main.py train mistral --dataset your-dataset
    python main.py train phi3 --dataset your-dataset
    
  📊 Information:
    python main.py models          # List all available models
    python main.py status          # Show system status
    python main.py help            # Show this help

🔧 Setup Commands:
    ./setup.sh                     # Initial project setup
    pip install -r requirements.txt # Install dependencies manually

📚 Your Models:
  • tasal9/ZamAI-Mistral-7B-Pashto (Educational Tutor)
  • tasal9/ZamAI-Phi-3-Mini-Pashto (Business Automation)
  • tasal9/ZamAI-Whisper-v3-Pashto (Speech Recognition)
  • tasal9/Multilingual-ZamAI-Embeddings (Text Embeddings)
  • tasal9/ZamAI-LIama3-Pashto (Private)
  • tasal9/pashto-base-bloom (Base Model)

🌐 Demo URLs (when running):
  • Chatbot: http://localhost:7860
  • Voice: http://localhost:7861  
  • Business: http://localhost:7862
  • API: http://localhost:8000
    """
    print(help_text)

def show_status():
    """Show current system status"""
    print("\n🔍 System Status Check:")
    print("=" * 50)
    
    issues = check_requirements()
    if not issues:
        print("✅ All requirements met!")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    # Check if demos exist
    demos = ['demos/chatbot_demo.py', 'demos/voice_demo.py', 'demos/business_demo.py']
    print(f"\n📱 Demo Files:")
    for demo in demos:
        status = "✅" if os.path.exists(demo) else "❌"
        print(f"  {status} {demo}")
    
    # Check API
    api_file = 'api/main.py'
    api_status = "✅" if os.path.exists(api_file) else "❌"
    print(f"\n🌐 API Server:")
    print(f"  {api_status} {api_file}")
    
    # Show environment info
    print(f"\n🔧 Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  HF Token: {'✅ Set' if os.getenv('HF_TOKEN') and os.getenv('HF_TOKEN') != 'your_hugging_face_token_here' else '❌ Not set'}")

def main():
    parser = argparse.ArgumentParser(description="ZamAI Pro Models Strategy Launcher")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Launch demo applications')
    demo_parser.add_argument('type', choices=['chatbot', 'voice', 'business'], help='Demo type to launch')
    
    # API command
    subparsers.add_parser('api', help='Launch FastAPI server')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run fine-tuning')
    train_parser.add_argument('model', choices=['mistral', 'phi3'], help='Model to fine-tune')
    train_parser.add_argument('--dataset', help='Dataset to use for training')
    
    # Info commands
    subparsers.add_parser('models', help='List available models')
    subparsers.add_parser('status', help='Show system status')
    subparsers.add_parser('help', help='Show detailed help')
    
    args = parser.parse_args()
    
    print_banner()
    
    if not args.command:
        show_help()
        return
    
    # Handle commands
    if args.command == 'demo':
        launch_demo(args.type)
    elif args.command == 'api':
        launch_api()
    elif args.command == 'train':
        run_fine_tuning(args.model, args.dataset)
    elif args.command == 'models':
        list_available_models()
    elif args.command == 'status':
        show_status()
    elif args.command == 'help':
        show_help()

if __name__ == "__main__":
    main()
