"""
ZamAI Voice Assistant Demo
Gradio interface for speech-to-text and text-to-speech with Pashto support
"""

import gradio as gr
import os
from huggingface_hub import InferenceClient
import tempfile
import librosa
from dotenv import load_dotenv

load_dotenv()

# Initialize HF Inference Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def process_audio(audio_file, language="English"):
    """
    Process audio input: Whisper STT -> LLaMA-3 -> Response
    """
    try:
        if audio_file is None:
            return "Please record or upload an audio file.", ""
        
        # Step 1: Transcribe audio using your Whisper model
        transcription = client.automatic_speech_recognition(
            audio_file,
            model=os.getenv("WHISPER_MODEL", "tasal9/ZamAI-Whisper-v3-Pashto")
        )
        
        if not transcription.strip():
            return "Could not transcribe audio. Please try again.", ""
        
        # Step 2: Generate response using your LLaMA-3 model
        if language == "Pashto":
            model_name = os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto")
            system_prompt = "تاسو د غږیز ملګری یاست چې د زده کوونکو سره مرسته کوي. ډېر ګټور او دقیق ځوابونه ورکړئ."
        else:
            model_name = os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto")
            system_prompt = "You are an intelligent voice assistant. Provide helpful, accurate, and engaging responses."
        
        # Step 3: Use Inference API for LLaMA-3 response
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{transcription}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        response = client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        return transcription, response.strip()
        
    except Exception as e:
        return f"Transcription error: {str(e)}", "Please check your model access and try again."

def text_to_speech(text, language="English"):
    """
    Convert text to speech using TTS API
    """
    if not text.strip():
        return "No text to convert to speech."
    
    try:
        # For now, we'll use a placeholder TTS response
        # In production, you would integrate with:
        # - Coqui TTS for Pashto
        # - Azure Speech Services
        # - Google Text-to-Speech
        # - ElevenLabs API
        
        tts_message = f"🔊 TTS Ready: '{text[:100]}{'...' if len(text) > 100 else ''}'"
        
        if language == "Pashto":
            tts_message += "\n\n💡 For Pashto TTS, integrate with Coqui TTS or custom voice synthesis"
        else:
            tts_message += "\n\n💡 For English TTS, integrate with standard TTS services"
            
        return tts_message
        
    except Exception as e:
        return f"TTS Error: {str(e)}"

# Create Gradio interface
def create_voice_demo():
    with gr.Blocks(
        title="ZamAI Voice Assistant",
        theme=gr.themes.Default(),
        css="""
        /* Main container */
        .gradio-container { background: #f8fafc !important; }
        
        /* Header */
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white !important;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
        }
        
        /* Model info panel */
        .model-info { 
            background: #ffffff !important; 
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            color: #374151 !important;
        }
        
        /* Audio section */
        .audio-section { 
            background: #f0fdf4 !important; 
            padding: 25px; 
            border-radius: 12px; 
            margin: 15px 0; 
            border: 1px solid #22c55e;
            color: #166534 !important;
        }
        
        /* Buttons */
        .gr-button {
            background: linear-gradient(45deg, #059669, #10b981) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
        }
        
        .gr-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4) !important;
        }
        
        /* Input fields */
        .gr-textbox, .gr-dropdown, .gr-radio {
            background: white !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            color: #374151 !important;
        }
        
        /* Tabs */
        .gr-tab {
            background: white !important;
            color: #374151 !important;
            border-radius: 8px !important;
            border: 1px solid #e5e7eb !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 25px;
            background: #ffffff !important;
            border-radius: 12px;
            border-top: 4px solid #4f46e5;
            color: #374151 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎤 ZamAI Voice Assistant</h1>
            <p>Whisper → LLaMA-3 → TTS Pipeline with Pashto support</p>
            <p><strong>Architecture:</strong> Speech Recognition → AI Reasoning → Voice Synthesis</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="model-info">
                    <h3>🤖 Voice Pipeline Models</h3>
                    <p><strong>STT:</strong> ZamAI-Whisper-v3-Pashto</p>
                    <p><strong>LLM:</strong> ZamAI-LLaMA3-Pashto (Private)</p>
                    <p><strong>TTS:</strong> Coqui-TTS + Standard APIs</p>
                    <p><strong>Languages:</strong> English, Pashto</p>
                    <p><strong>API:</strong> HF Inference API</p>
                </div>
                """)
                
                language_selector = gr.Radio(
                    choices=["English", "Pashto"],
                    value="English",
                    label="🌐 Language Preference"
                )
                
                gr.HTML("""
                <div style="margin-top: 20px;">
                    <h4>🎯 How to use:</h4>
                    <ol>
                        <li>Select your language preference</li>
                        <li>Record audio or upload a file</li>
                        <li>Get transcription and AI response</li>
                    </ol>
                </div>
                """)
            
            with gr.Column(scale=2):
                with gr.Tab("🎙️ Voice Input"):
                    gr.HTML('<div class="audio-section">')
                    
                    audio_input = gr.Audio(
                        label="Record or Upload Audio",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                    
                    process_btn = gr.Button("Process Audio 🔄", variant="primary", size="lg")
                    
                    with gr.Row():
                        transcription_output = gr.Textbox(
                            label="📝 Transcription",
                            lines=3,
                            interactive=False
                        )
                        response_output = gr.Textbox(
                            label="🤖 AI Response", 
                            lines=3,
                            interactive=False
                        )
                    
                    gr.HTML('</div>')
                
                with gr.Tab("💬 Text Input"):
                    text_input = gr.Textbox(
                        label="Type your message",
                        lines=3,
                        placeholder="Type your question here..."
                    )
                    
                    text_process_btn = gr.Button("Get Response 💬", variant="primary")
                    
                    text_response = gr.Textbox(
                        label="🤖 Response",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Tab("🔊 Text-to-Speech"):
                    tts_input = gr.Textbox(
                        label="Text to Convert to Speech",
                        lines=3,
                        placeholder="Enter text to convert to speech..."
                    )
                    
                    tts_btn = gr.Button("Convert to Speech 🔊", variant="primary")
                    
                    tts_output = gr.Textbox(
                        label="TTS Status",
                        interactive=False
                    )
        
        # Event handlers
        def process_audio_interface(audio, language):
            transcription, response = process_audio(audio, language)
            return transcription, response
        
        def process_text_interface(text, language):
            if not text.strip():
                return "Please enter some text."
            
            # Process text directly with LLaMA-3 (skip STT)
            try:
                if language == "Pashto":
                    model_name = os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto")
                    system_prompt = "تاسو د غږیز ملګری یاست چې ډېر ګټور ځوابونه ورکوي."
                else:
                    model_name = os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto")
                    system_prompt = "You are an intelligent voice assistant providing helpful responses."
                
                # Use LLaMA-3 chat format
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                
                response = client.text_generation(
                    model=model_name,
                    prompt=prompt,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                
                return response.strip()
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Connect events
        process_btn.click(
            process_audio_interface,
            inputs=[audio_input, language_selector],
            outputs=[transcription_output, response_output]
        )
        
        text_process_btn.click(
            process_text_interface,
            inputs=[text_input, language_selector],
            outputs=[text_response]
        )
        
        tts_btn.click(
            text_to_speech,
            inputs=[tts_input, language_selector],
            outputs=[tts_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p>Powered by <strong>ZamAI Pro Models Strategy</strong></p>
            <p><strong>Voice Pipeline:</strong> 
                <a href="https://huggingface.co/tasal9/ZamAI-Whisper-v3-Pashto" target="_blank">Whisper-v3-Pashto</a> → 
                <a href="https://huggingface.co/tasal9/ZamAI-LIama3-Pashto" target="_blank">LLaMA3-Pashto</a> → TTS
            </p>
            <p>🔊 <strong>Features:</strong> Real-time processing | Inference API powered | Multilingual support</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_voice_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
