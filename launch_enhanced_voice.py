#!/usr/bin/env python3
"""
ZamAI Enhanced Voice Assistant Launcher
Quick launcher for the improved UI version
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def launch_enhanced_voice_assistant():
    """Launch the enhanced voice assistant with improved UI"""
    
    print("🎤 ZamAI Enhanced Voice Assistant")
    print("=" * 50)
    
    # Check if the demo file exists
    demo_file = Path("demos/voice_assistant_enhanced_ui.py")
    if not demo_file.exists():
        print("❌ Enhanced voice assistant demo not found!")
        return False
    
    # Check environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "test_token_placeholder":
        print("⚠️  No HuggingFace token found. Demo will run in limited mode.")
    else:
        print("✅ HuggingFace token configured")
    
    # Display models
    print(f"🎤 STT Model: {os.getenv('WHISPER_MODEL', 'Default')}")
    print(f"🧠 LLM Model: {os.getenv('LLAMA3_MODEL', 'Default')}")
    
    # Display features
    print("\n🚀 Enhanced Features:")
    print("   • Professional UI with clear contrast")
    print("   • Whisper → LLaMA-3 → TTS Pipeline")
    print("   • Real-time performance analytics")
    print("   • Context-aware conversations")
    print("   • Multi-language support (English/Pashto)")
    print("   • Conversation history tracking")
    
    # Display access information
    port = os.getenv("ENHANCED_VOICE_PORT", "7864")
    print(f"\n🌐 Interface Access:")
    print(f"   • Local: http://localhost:{port}")
    print(f"   • Network: http://0.0.0.0:{port}")
    print(f"   • Share: {'Enabled' if os.getenv('GRADIO_SHARE', 'false').lower() == 'true' else 'Disabled'}")
    
    print("\n⏹️  Press Ctrl+C to stop the assistant")
    print("🚀 Launching Enhanced Voice Assistant...")
    
    try:
        # Launch the enhanced demo
        result = subprocess.run([
            sys.executable, 
            str(demo_file)
        ], cwd=".")
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n👋 Enhanced Voice Assistant stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching enhanced voice assistant: {e}")
        return False

def main():
    """Main launcher function"""
    success = launch_enhanced_voice_assistant()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
