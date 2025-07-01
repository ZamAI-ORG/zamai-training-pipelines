#!/usr/bin/env python3
"""
Launch the ZamAI Enhanced Tutor Bot
Quick launcher with dataset integration
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    print("🎓 ZamAI Enhanced Tutor Bot Launcher")
    print("=" * 50)
    
    # Check if dataset is processed
    dataset_path = Path("datasets/processed/tutoring_qa.json")
    
    if not dataset_path.exists():
        print("📊 Dataset not found. Processing first...")
        print("🔄 Running dataset integration...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/dataset_integration.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dataset processed successfully!")
            else:
                print("⚠️ Dataset processing had issues, but continuing...")
                print("Note: Demo will work with sample data")
        except Exception as e:
            print(f"⚠️ Dataset processing failed: {e}")
            print("Note: Demo will work with sample data")
    else:
        print("✅ Dataset already processed")
    
    # Launch the Enhanced Tutor Bot
    print("\n🚀 Launching Enhanced Tutor Bot...")
    print("🌐 Interface will be available at: http://localhost:7865")
    print("📱 Features:")
    print("   • Dataset-aware responses")
    print("   • Performance evaluation")
    print("   • Multi-language support")
    print("   • Context-aware tutoring")
    
    try:
        subprocess.run([sys.executable, "demos/enhanced_tutor_bot.py"])
    except KeyboardInterrupt:
        print("\n👋 Enhanced Tutor Bot stopped by user")
    except Exception as e:
        print(f"❌ Error launching tutor bot: {e}")

if __name__ == "__main__":
    main()
