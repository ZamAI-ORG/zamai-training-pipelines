#!/usr/bin/env python3

import os
import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import json

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ZamAIVoiceAssistant:
    def __init__(self):
        self.setup_models()
        self.conversation_history = []
        self.performance_metrics = {
            "total_interactions": 0,
            "avg_response_time": 0,
            "languages_detected": set()
        }
    
    def setup_models(self):
        """Initialize AI models"""
        try:
            # Speech Recognition (Whisper)
            self.whisper_model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",  # Fallback model
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Language Model (LLaMA-3 compatible)
            model_name = "microsoft/DialoGPT-medium"  # Fallback model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def transcribe_audio(self, audio_file):
        """Convert speech to text"""
        if audio_file is None:
            return "No audio provided"
        
        try:
            # Load and process audio
            audio_array, sample_rate = librosa.load(audio_file, sr=16000)
            
            # Transcribe using Whisper
            result = self.whisper_model(audio_array)
            transcript = result['text'].strip()
            
            return transcript
        
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
    
    def generate_response(self, text):
        """Generate intelligent response"""
        try:
            # Encode input
            inputs = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                outputs = self.language_model.generate(
                    inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(text):].strip()
            
            return response if response else "I understand. Please tell me more."
        
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def process_voice_input(self, audio_file):
        """Main voice processing pipeline"""
        start_time = datetime.now()
        
        # Step 1: Speech to Text
        transcript = self.transcribe_audio(audio_file)
        if transcript.startswith("Error") or transcript == "No audio provided":
            return transcript, "", self.get_conversation_html(), self.get_metrics_html()
        
        # Step 2: Generate Response
        response = self.generate_response(transcript)
        
        # Step 3: Update conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "user": transcript,
            "assistant": response
        })
        
        # Step 4: Update metrics
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        self.update_metrics(response_time)
        
        # Step 5: Generate audio response (placeholder)
        audio_response = self.text_to_speech(response)
        
        return transcript, response, self.get_conversation_html(), self.get_metrics_html()
    
    def text_to_speech(self, text):
        """Convert text to speech (placeholder)"""
        # In a real implementation, you would use a TTS model
        # For now, return empty audio
        return None
    
    def update_metrics(self, response_time):
        """Update performance metrics"""
        self.performance_metrics["total_interactions"] += 1
        total = self.performance_metrics["total_interactions"]
        current_avg = self.performance_metrics["avg_response_time"]
        self.performance_metrics["avg_response_time"] = (current_avg * (total - 1) + response_time) / total
    
    def get_conversation_html(self):
        """Generate conversation history HTML"""
        if not self.conversation_history:
            return "<p>No conversation yet. Start by recording your voice!</p>"
        
        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for entry in self.conversation_history[-10:]:  # Show last 10 exchanges
            html += f"""
            <div style='margin-bottom: 15px; padding: 10px; border-left: 3px solid #007acc;'>
                <div style='color: #666; font-size: 0.8em;'>{entry['timestamp']}</div>
                <div style='color: #2196F3; font-weight: bold;'>🎤 You:</div>
                <div style='margin: 5px 0;'>{entry['user']}</div>
                <div style='color: #4CAF50; font-weight: bold;'>🤖 Assistant:</div>
                <div>{entry['assistant']}</div>
            </div>
            """
        html += "</div>"
        return html
    
    def get_metrics_html(self):
        """Generate metrics HTML"""
        metrics = self.performance_metrics
        return f"""
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.5em; font-weight: bold;'>{metrics['total_interactions']}</div>
                <div>Total Interactions</div>
            </div>
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.5em; font-weight: bold;'>{metrics['avg_response_time']:.2f}s</div>
                <div>Avg Response Time</div>
            </div>
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.5em; font-weight: bold;'>Active</div>
                <div>System Status</div>
            </div>
        </div>
        """

# Initialize the assistant
assistant = ZamAIVoiceAssistant()

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="ZamAI Voice Assistant",
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎤 ZamAI Voice Assistant</h1>
            <p>Advanced multilingual AI voice interaction with Whisper → LLaMA-3 → TTS pipeline</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🎙️ Voice Input</h3>")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record your voice"
                )
                
                process_btn = gr.Button(
                    "🚀 Process Voice",
                    variant="primary",
                    size="lg"
                )
                
                gr.HTML("<h3>📝 Results</h3>")
                transcript_output = gr.Textbox(
                    label="🎤 What you said",
                    placeholder="Your speech will appear here...",
                    lines=2
                )
                
                response_output = gr.Textbox(
                    label="🤖 Assistant Response",
                    placeholder="AI response will appear here...", 
                    lines=3
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3>💬 Conversation History</h3>")
                conversation_html = gr.HTML(
                    value="<p>No conversation yet. Start by recording your voice!</p>"
                )
                
                gr.HTML("<h3>📊 Performance Metrics</h3>")
                metrics_html = gr.HTML(
                    value=assistant.get_metrics_html()
                )
        
        # Event handlers
        process_btn.click(
            fn=assistant.process_voice_input,
            inputs=[audio_input],
            outputs=[transcript_output, response_output, conversation_html, metrics_html]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>🌟 Powered by ZamAI Pro Models Strategy | 🔧 Built with Gradio</p>
        </div>
        """)
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
