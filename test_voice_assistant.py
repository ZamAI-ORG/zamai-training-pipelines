#!/usr/bin/env python3
"""
Test script for ZamAI Voice Assistant
Tests the Whisper → LLaMA-3 → TTS pipeline
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if environment is properly configured"""
    print("🔧 Testing Environment Configuration...")
    
    required_vars = [
        "HF_TOKEN",
        "WHISPER_MODEL", 
        "LLAMA3_MODEL"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
    
    return all(os.getenv(var) for var in required_vars)

def test_imports():
    """Test if required packages are available"""
    print("\n📦 Testing Package Imports...")
    
    try:
        import gradio as gr
        print("✅ Gradio:", gr.__version__)
    except ImportError as e:
        print("❌ Gradio not available:", e)
        return False
    
    try:
        from huggingface_hub import InferenceClient
        print("✅ HuggingFace Hub available")
    except ImportError as e:
        print("❌ HuggingFace Hub not available:", e)
        return False
    
    try:
        import librosa
        print("✅ Librosa available")
    except ImportError as e:
        print("❌ Librosa not available:", e)
        return False
    
    return True

def test_inference_client():
    """Test HuggingFace Inference Client"""
    print("\n🤖 Testing HuggingFace Inference Client...")
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        print("✅ Inference Client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Inference Client error: {e}")
        return False

def main():
    print("🎤 ZamAI Voice Assistant Test Suite")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test inference client
    client_ok = test_inference_client()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Inference Client: {'✅ PASS' if client_ok else '❌ FAIL'}")
    
    if all([env_ok, imports_ok, client_ok]):
        print("\n🎉 All tests passed! Voice Assistant is ready to run.")
        print("\n🚀 To start the voice assistant:")
        print("   python demos/voice_demo.py")
        print("   python demos/voice_assistant_advanced.py")
        return True
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        print("\n📝 Setup instructions:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your HF_TOKEN")
        print("   3. Install dependencies: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
