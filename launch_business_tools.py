#!/usr/bin/env python3
"""
Launch ZamAI Enhanced Business Tools Suite
Quick launcher with dependency checks and setup
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_dependencies():
    """Check if business tools dependencies are installed"""
    required_packages = [
        'gradio',
        'huggingface_hub',
        'faiss',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                import faiss
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package if package != 'faiss' else 'faiss-cpu')
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_directories():
    """Setup required directories for business tools"""
    directories = [
        "business_data",
        "business_data/documents", 
        "business_data/embeddings",
        "business_data/faiss_index"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def main():
    print("🏢 ZamAI Enhanced Business Tools Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return 1
    print("✅ All dependencies available")
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Check environment
    print("\n🔧 Checking environment...")
    hf_token = os.getenv("HF_TOKEN")
    phi3_model = os.getenv("PHI3_BUSINESS_MODEL", "tasal9/ZamAI-Phi-3-Mini-Pashto")
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "tasal9/Multilingual-ZamAI-Embeddings")
    
    print(f"HF Token: {'✅ Set' if hf_token else '⚠️ Not set (demo mode)'}")
    print(f"Phi-3 Model: {phi3_model}")
    print(f"Embeddings Model: {embeddings_model}")
    
    # Launch the business tools
    print("\n🚀 Launching Enhanced Business Tools Suite...")
    print("🌐 Interface will be available at: http://localhost:7866")
    print("\n📊 Features Available:")
    print("   • 📄 Document Processing & Analysis")
    print("   • 🔍 Intelligent Document Search")
    print("   • ✍️ Business Content Generation")
    print("   • 📚 Document Library Management")
    print("   • 📈 Performance Analytics")
    print("   • 🔐 Secure Document Storage")
    
    print(f"\n🤖 Models:")
    print(f"   • Document Processing: {phi3_model}")
    print(f"   • Embeddings: {embeddings_model}")
    
    print("\n⏹️ Press Ctrl+C to stop")
    
    try:
        # Launch the enhanced business tools
        subprocess.run([sys.executable, "demos/enhanced_business_tools.py"])
    except KeyboardInterrupt:
        print("\n👋 Business Tools stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching business tools: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
