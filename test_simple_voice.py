#!/usr/bin/env python3
"""
Simple Voice Assistant Test
Test the core components without full Gradio UI
"""

import os
import sys
from pathlib import Path
import tempfile

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing Basic Imports...")
    
    try:
        import gradio as gr
        print(f"✅ Gradio: {gr.__version__}")
    except Exception as e:
        print(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv: Available")
    except Exception as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("🔧 Testing Environment...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    hf_token = os.getenv("HF_TOKEN")
    whisper_model = os.getenv("WHISPER_MODEL")
    llama3_model = os.getenv("LLAMA3_MODEL")
    
    print(f"HF_TOKEN: {'✅ Set' if hf_token else '❌ Not set'}")
    print(f"WHISPER_MODEL: {whisper_model or 'Default'}")
    print(f"LLAMA3_MODEL: {llama3_model or 'Default'}")
    
    return True

def test_simple_gradio():
    """Test simple Gradio interface"""
    print("🎨 Testing Simple Gradio Interface...")
    
    import gradio as gr
    
    def simple_echo(text):
        if not text:
            return "Please enter some text"
        return f"Echo: {text}"
    
    try:
        # Create simple interface
        with gr.Blocks(title="Simple Test") as demo:
            gr.Markdown("# Simple Voice Assistant Test")
            
            text_input = gr.Textbox(label="Input", placeholder="Enter text here...")
            text_output = gr.Textbox(label="Output", interactive=False)
            process_btn = gr.Button("Process", variant="primary")
            
            process_btn.click(
                fn=simple_echo,
                inputs=[text_input],
                outputs=[text_output]
            )
        
        print("✅ Gradio interface created successfully")
        return demo
        
    except Exception as e:
        print(f"❌ Gradio interface creation failed: {e}")
        return None

def main():
    print("🎤 ZamAI Voice Assistant - Simple Test")
    print("=" * 50)
    
    # Test imports
    if not test_basic_imports():
        return 1
    
    # Test environment
    test_environment()
    
    # Test simple Gradio
    demo = test_simple_gradio()
    
    if demo is None:
        print("❌ Failed to create Gradio interface")
        return 1
    
    print("\n✅ All basic tests passed!")
    print("🚀 Starting simple test interface...")
    print("🌐 Interface will be available at: http://localhost:7863")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7863,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
