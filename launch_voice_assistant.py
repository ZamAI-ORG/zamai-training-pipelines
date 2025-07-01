#!/usr/bin/env python3
"""
ZamAI Voice Assistant Launcher
Quick launcher for the Whisper → LLaMA-3 → TTS voice assistant
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'gradio',
        'huggingface_hub', 
        'python-dotenv',
        'librosa'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_environment():
    """Check environment configuration"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        example_file = Path('.env.example')
        if example_file.exists():
            example_file.read_text().replace('your_hugging_face_token_here', 'demo_mode')
            with open('.env', 'w') as f:
                f.write(example_file.read_text().replace('your_hugging_face_token_here', 'demo_mode'))
            print("✅ .env file created in demo mode")
        else:
            print("❌ .env.example not found")
            return False
    
    return True

def launch_voice_assistant(demo_type="enhanced"):
    """Launch the voice assistant"""
    
    demo_files = {
        "basic": "demos/voice_demo.py",
        "enhanced": "demos/voice_assistant_inference_api.py", 
        "advanced": "demos/voice_assistant_advanced.py"
    }
    
    demo_file = demo_files.get(demo_type, demo_files["enhanced"])
    
    if not Path(demo_file).exists():
        print(f"❌ Demo file not found: {demo_file}")
        return False
    
    print(f"🎤 Launching ZamAI Voice Assistant ({demo_type})...")
    print(f"📝 File: {demo_file}")
    print("🌐 Interface will be available at:")
    print("   - Local: http://localhost:7862")
    print("   - Network: http://0.0.0.0:7862")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        # Launch the demo
        result = subprocess.run([sys.executable, demo_file], cwd=".")
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n👋 Voice Assistant stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching voice assistant: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="ZamAI Voice Assistant Launcher")
    parser.add_argument(
        "--demo", 
        choices=["basic", "enhanced", "advanced"],
        default="enhanced",
        help="Choose demo type (default: enhanced)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and environment"
    )
    
    args = parser.parse_args()
    
    print("🎤 ZamAI Voice Assistant Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        return 1
    print("✅ Dependencies OK")
    
    # Check environment
    print("🔧 Checking environment...")
    if not check_environment():
        return 1
    print("✅ Environment OK")
    
    if args.check_only:
        print("\n✅ All checks passed! Ready to launch voice assistant.")
        print("🚀 Run without --check-only to start the interface")
        return 0
    
    # Launch voice assistant
    success = launch_voice_assistant(args.demo)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
