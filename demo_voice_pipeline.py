#!/usr/bin/env python3
"""
ZamAI Voice Assistant Demo Script
Quick demonstration of the Whisper → LLaMA-3 pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def demo_voice_pipeline():
    """Demonstrate the voice processing pipeline"""
    
    print("🎤 ZamAI Voice Assistant Pipeline Demo")
    print("=" * 50)
    
    # Check if we're in demo mode or production mode
    hf_token = os.getenv("HF_TOKEN")
    is_demo_mode = not hf_token or hf_token == "test_token_placeholder"
    
    if is_demo_mode:
        print("🟡 Running in DEMO MODE (no HF token)")
        print("   Add your HuggingFace token to .env for full functionality")
    else:
        print("🟢 Running in PRODUCTION MODE")
        print("   Connected to HuggingFace Inference API")
    
    print("\n🔧 Pipeline Components:")
    print(f"   🎤 STT Model: {os.getenv('WHISPER_MODEL', 'openai/whisper-base')}")
    print(f"   🧠 LLM Model: {os.getenv('LLAMA3_MODEL', 'meta-llama/Meta-Llama-3-8B-Instruct')}")
    print(f"   🔊 TTS: Coqui-TTS (Future Integration)")
    
    print("\n📋 Available Interfaces:")
    interfaces = [
        ("Basic Voice Demo", "demos/voice_demo.py", "http://localhost:7861"),
        ("Advanced Voice Assistant", "demos/voice_assistant_advanced.py", "http://localhost:7862"),
        ("Enhanced Inference API", "demos/voice_assistant_inference_api.py", "http://localhost:7863")
    ]
    
    for name, file, url in interfaces:
        exists = "✅" if Path(file).exists() else "❌"
        print(f"   {exists} {name}")
        print(f"      File: {file}")
        print(f"      URL: {url}")
        print()
    
    print("🚀 Quick Start Commands:")
    print("   python demos/voice_assistant_inference_api.py")
    print("   python launch_voice_assistant.py")
    print("   python main.py demo voice")
    
    return True

def demo_text_processing():
    """Demonstrate text processing without full UI"""
    
    print("\n💬 Text Processing Demo")
    print("-" * 30)
    
    # Sample inputs
    test_inputs = [
        ("Hello, how are you?", "English", "General"),
        ("What is machine learning?", "English", "Educational"),
        ("Explain quantum physics", "English", "Technical"),
        ("سلام وروره، څنګه یاست؟", "Pashto", "Casual")  # Hello brother, how are you?
    ]
    
    for text, language, context in test_inputs:
        print(f"\n📝 Input: '{text}'")
        print(f"   Language: {language}, Context: {context}")
        
        # Simulate processing
        if language == "Pashto":
            response = f"🤖 [Pashto Response] زه د {context} د برخې د غږیز ملګری یم. ستاسو ته مرسته کولی شم."
        else:
            response = f"🤖 [{context} Response] I'm an AI voice assistant specialized in {context.lower()} context. I'd be happy to help with your question about '{text[:30]}...'"
        
        print(f"   Response: {response}")

def demo_pipeline_features():
    """Demonstrate pipeline features"""
    
    print("\n⚡ Pipeline Features Demo")
    print("-" * 30)
    
    features = [
        ("🎤 Audio Input", "Microphone recording + file upload"),
        ("🎯 Context Modes", "Educational, Business, Technical, Casual, General"),
        ("🌐 Languages", "English and Pashto support"),
        ("📊 Metrics", "Real-time performance monitoring"),
        ("💾 History", "Conversation tracking and management"),
        ("🔄 Inference API", "HuggingFace model integration"),
        ("🎨 UI/UX", "Modern Gradio interface with tabs"),
        ("🔧 Extensible", "Ready for TTS and custom models")
    ]
    
    for feature, description in features:
        print(f"   {feature}: {description}")

def main():
    """Main demo function"""
    
    # Run demonstrations
    demo_voice_pipeline()
    demo_text_processing()
    demo_pipeline_features()
    
    print("\n" + "=" * 50)
    print("🎉 Demo Complete!")
    print("\n🚀 To launch the full voice assistant:")
    print("   python demos/voice_assistant_inference_api.py")
    print("\n📚 For more info, check:")
    print("   VOICE_ASSISTANT_SUMMARY.md")
    print("   README.md")

if __name__ == "__main__":
    main()
