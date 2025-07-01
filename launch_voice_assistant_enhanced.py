#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['gradio', 'transformers', 'torch', 'librosa']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment variables"""
    required_vars = ["HF_TOKEN", "WHISPER_MODEL", "LLAMA3_MODEL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or set these variables")
        return False
    
    return True

def create_log_entry(demo_type, host, port, share):
    """Log the launch event"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "service": "voice_assistant",
        "demo_type": demo_type,
        "host": host,
        "port": port,
        "share": share,
        "status": "started"
    }
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Append to log file
    with open("logs/launch_history.json", "a") as f:
        f.write(json.dumps(log_data) + "\n")

def display_banner():
    """Display startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                🎤 ZamAI Voice Assistant 🎤                ║
║                                                          ║
║  Advanced multilingual AI voice interaction              ║
║  Whisper → LLaMA-3 → TTS Pipeline                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="Launch ZamAI Voice Assistant")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7861, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--demo", choices=["basic", "advanced", "inference"], 
                       default="advanced", help="Demo type to launch")
    parser.add_argument("--dev", action="store_true", help="Development mode with auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    display_banner()
    
    # Environment checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_environment():
        print("⚠️ Warning: Some environment variables are missing. Demo may have limited functionality.")
        time.sleep(2)
    
    # Log the launch
    create_log_entry(args.demo, args.host, args.port, args.share)
    
    print("🔧 Configuration:")
    print(f"├── 📡 Server: http://{args.host}:{args.port}")
    print(f"├── 🎯 Demo type: {args.demo}")
    print(f"├── 🌐 Public sharing: {'enabled' if args.share else 'disabled'}")
    print(f"├── 🔧 Development mode: {'enabled' if args.dev else 'disabled'}")
    print(f"└── 🐛 Debug mode: {'enabled' if args.debug else 'disabled'}")
    print()
    
    # Select demo file based on type
    demo_files = {
        "basic": "demos/voice_demo.py",
        "advanced": "demos/voice_assistant_advanced.py", 
        "inference": "demos/voice_assistant_inference_api.py"
    }
    
    demo_file = demo_files.get(args.demo)
    if not Path(demo_file).exists():
        print(f"❌ Demo file not found: {demo_file}")
        print("Available demo files:")
        for name, path in demo_files.items():
            status = "✅" if Path(path).exists() else "❌"
            print(f"  {status} {name}: {path}")
        sys.exit(1)
    
    print(f"🚀 Starting {args.demo} voice assistant...")
    print(f"📂 Loading: {demo_file}")
    print()
    
    # Launch the demo
    try:
        env = os.environ.copy()
        env["GRADIO_SERVER_NAME"] = args.host
        env["GRADIO_SERVER_PORT"] = str(args.port)
        if args.share:
            env["GRADIO_SHARE"] = "true"
        if args.debug:
            env["GRADIO_DEBUG"] = "true"
        
        print("🎤 Voice Assistant is starting...")
        print("   Press Ctrl+C to stop")
        print("=" * 60)
        
        subprocess.run([sys.executable, demo_file], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("👋 Voice Assistant stopped gracefully")
        
        # Log the stop event
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "service": "voice_assistant", 
            "status": "stopped",
            "reason": "user_interrupt"
        }
        with open("logs/launch_history.json", "a") as f:
            f.write(json.dumps(log_data) + "\n")
            
    except Exception as e:
        print(f"\n❌ Error launching voice assistant: {e}")
        
        # Log the error
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "service": "voice_assistant",
            "status": "error", 
            "error": str(e)
        }
        with open("logs/launch_history.json", "a") as f:
            f.write(json.dumps(log_data) + "\n")
            
        sys.exit(1)

if __name__ == "__main__":
    main()
