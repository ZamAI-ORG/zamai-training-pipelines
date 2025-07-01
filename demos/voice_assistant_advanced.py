"""
Enhanced ZamAI Voice Assistant Demo
Advanced Gradio test UI for Whisper → LLaMA-3 → TTS pipeline
"""

import gradio as gr
import os
import tempfile
import time
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize HF Inference Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

class VoiceAssistant:
    def __init__(self):
        self.conversation_history = []
        self.performance_metrics = {
            "total_requests": 0,
            "avg_response_time": 0,
            "success_rate": 100.0
        }
    
    def process_audio_advanced(self, audio_file, language="English", context="General"):
        """
        Advanced audio processing with context awareness
        """
        start_time = time.time()
        
        try:
            if audio_file is None:
                return "Please record or upload an audio file.", "", self._get_metrics_display()
            
            # Step 1: Speech-to-Text with Whisper
            transcription = client.automatic_speech_recognition(
                audio_file,
                model=os.getenv("WHISPER_MODEL", "tasal9/ZamAI-Whisper-v3-Pashto")
            )
            
            if not transcription.strip():
                return "Could not transcribe audio. Please try again.", "", self._get_metrics_display()
            
            # Step 2: Context-aware LLaMA-3 processing
            response = self._generate_contextual_response(transcription, language, context)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, True)
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "input": transcription,
                "output": response,
                "language": language,
                "context": context,
                "processing_time": f"{processing_time:.2f}s"
            })
            
            return transcription, response, self._get_metrics_display()
            
        except Exception as e:
            self._update_metrics(time.time() - start_time, False)
            return f"Error: {str(e)}", "Processing failed", self._get_metrics_display()
    
    def _generate_contextual_response(self, text, language, context):
        """Generate context-aware response using LLaMA-3"""
        
        # Context-specific system prompts
        context_prompts = {
            "Educational": "You are an educational voice assistant. Provide clear, informative explanations suitable for learning.",
            "Business": "You are a professional business assistant. Provide concise, professional responses.",
            "Casual": "You are a friendly conversational assistant. Be natural and engaging.",
            "Technical": "You are a technical expert. Provide detailed, accurate technical information.",
            "General": "You are an intelligent voice assistant. Provide helpful and accurate responses."
        }
        
        if language == "Pashto":
            system_prompt = f"تاسو د غږیز ملګری یاست. {context_prompts.get(context, context_prompts['General'])} په پښتو کې ځواب ورکړئ."
        else:
            system_prompt = context_prompts.get(context, context_prompts["General"])
        
        # Build conversation context
        conversation_context = ""
        if len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                conversation_context += f"Previous: {exchange['input']} -> {exchange['output'][:50]}...\n"
        
        # Use LLaMA-3 chat format with context
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n\nContext: {conversation_context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        response = client.text_generation(
            model=os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto"),
            prompt=full_prompt,
            max_new_tokens=350,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        return response.strip()
    
    def _update_metrics(self, processing_time, success):
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            # Update average response time
            current_avg = self.performance_metrics["avg_response_time"]
            total = self.performance_metrics["total_requests"]
            self.performance_metrics["avg_response_time"] = (current_avg * (total - 1) + processing_time) / total
        
        # Update success rate
        successful_requests = sum(1 for h in self.conversation_history if "Error" not in h.get("output", ""))
        self.performance_metrics["success_rate"] = (successful_requests / self.performance_metrics["total_requests"]) * 100
    
    def _get_metrics_display(self):
        """Get formatted metrics display"""
        metrics = self.performance_metrics
        return f"""📊 Performance Metrics:
• Total Requests: {metrics['total_requests']}
• Avg Response Time: {metrics['avg_response_time']:.2f}s
• Success Rate: {metrics['success_rate']:.1f}%
• Active Model: LLaMA-3-Pashto"""
    
    def get_conversation_history_json(self):
        """Get conversation history as JSON"""
        return json.dumps(self.conversation_history, indent=2)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."

# Initialize voice assistant
assistant = VoiceAssistant()

# Sample audio test data
SAMPLE_PROMPTS = {
    "English Educational": "Explain how photosynthesis works in plants",
    "English Technical": "What are the benefits of using microservices architecture?",
    "English Casual": "Tell me a joke about artificial intelligence",
    "Pashto General": "زموږ د ژوند معنا څه ده؟",  # What is the meaning of our life?
    "Pashto Educational": "د کمپیوټر د کار طریقه تشریح کړئ"  # Explain how computers work
}

def create_advanced_voice_demo():
    with gr.Blocks(
        title="ZamAI Advanced Voice Assistant",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 30px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; }
        .pipeline-info { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 5px solid #007bff; }
        .metrics-display { background: #e8f4fd; padding: 15px; border-radius: 10px; font-family: monospace; }
        .test-section { background: #fff3cd; padding: 15px; border-radius: 10px; margin: 10px 0; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎤 ZamAI Advanced Voice Assistant</h1>
            <p><strong>Whisper → LLaMA-3 → TTS Pipeline</strong></p>
            <p>Comprehensive Gradio Test UI with Performance Analytics</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="pipeline-info">
                    <h3>🔄 Processing Pipeline</h3>
                    <p><strong>1. STT:</strong> Whisper-v3-Pashto</p>
                    <p><strong>2. LLM:</strong> LLaMA-3-Pashto (Private)</p>
                    <p><strong>3. TTS:</strong> Context-aware synthesis</p>
                    <p><strong>API:</strong> HF Inference API</p>
                </div>
                """)
                
                # Configuration controls
                language_selector = gr.Radio(
                    choices=["English", "Pashto"],
                    value="English",
                    label="🌐 Language"
                )
                
                context_selector = gr.Radio(
                    choices=["General", "Educational", "Business", "Technical", "Casual"],
                    value="General",
                    label="🎯 Context Mode"
                )
                
                # Sample prompts
                gr.HTML('<div class="test-section"><h4>🧪 Test Prompts</h4></div>')
                sample_prompt = gr.Dropdown(
                    choices=list(SAMPLE_PROMPTS.keys()),
                    label="📝 Sample Prompts",
                    value="English Educational"
                )
                
                sample_text_display = gr.Textbox(
                    label="Selected Prompt Text",
                    value=SAMPLE_PROMPTS["English Educational"],
                    interactive=False
                )
                
                # Performance metrics display
                metrics_display = gr.Textbox(
                    label="📊 Real-time Metrics",
                    value=assistant._get_metrics_display(),
                    lines=5,
                    interactive=False
                )
            
            with gr.Column(scale=2):
                with gr.Tab("🎙️ Voice Processing"):
                    gr.HTML("""
                    <div style="text-align: center; padding: 15px; background: #f0f8f0; border-radius: 10px; margin-bottom: 15px;">
                        <h3>🔊 Advanced Voice Interface</h3>
                        <p>Real-time audio processing with context awareness</p>
                    </div>
                    """)
                    
                    audio_input = gr.Audio(
                        label="🎤 Record or Upload Audio",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("🔄 Process Audio", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Row():
                        transcription_output = gr.Textbox(
                            label="📝 Speech-to-Text (Whisper)",
                            lines=3,
                            interactive=False,
                            placeholder="Transcription will appear here..."
                        )
                        
                        response_output = gr.Textbox(
                            label="🤖 AI Response (LLaMA-3)",
                            lines=3,
                            interactive=False,
                            placeholder="AI response will appear here..."
                        )
                
                with gr.Tab("💬 Text Testing"):
                    text_input = gr.Textbox(
                        label="Direct Text Input",
                        lines=3,
                        placeholder="Type your message here for direct testing..."
                    )
                    
                    text_process_btn = gr.Button("💬 Process Text", variant="primary")
                    
                    text_response = gr.Textbox(
                        label="🤖 LLaMA-3 Response",
                        lines=5,
                        interactive=False
                    )
                
                with gr.Tab("📊 Analytics & History"):
                    gr.HTML("""
                    <h3>📈 Conversation Analytics</h3>
                    <p>Track performance and review conversation history</p>
                    """)
                    
                    with gr.Row():
                        export_btn = gr.Button("📥 Export History", variant="secondary")
                        clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
                    
                    history_output = gr.TextArea(
                        label="📋 Conversation History (JSON)",
                        lines=10,
                        placeholder="Conversation history will appear here...",
                        interactive=False
                    )
                    
                    history_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                
                with gr.Tab("🔧 API Testing"):
                    gr.HTML("""
                    <div style="padding: 15px; background: #f8f9fa; border-radius: 10px;">
                        <h3>🌐 API Integration Examples</h3>
                        <p>Test the voice assistant programmatically</p>
                    </div>
                    """)
                    
                    api_code_example = gr.Code(
                        label="Python API Usage Example",
                        language="python",
                        value='''import requests

# Voice processing API
api_url = "http://localhost:8000/voice/process"

# Upload audio file
with open("audio.wav", "rb") as f:
    files = {"audio_file": f}
    response = requests.post(api_url, files=files)
    
result = response.json()
print(f"Transcription: {result['transcription']}")
print(f"Response: {result['response']}")''',
                        interactive=False
                    )
                    
                    curl_example = gr.Code(
                        label="cURL Example",
                        language="bash",
                        value='''# Test voice processing endpoint
curl -X POST "http://localhost:8000/voice/process" \\
     -H "Content-Type: multipart/form-data" \\
     -F "audio_file=@audio.wav"''',
                        interactive=False
                    )
        
        # Event handlers
        def update_sample_text(prompt_key):
            return SAMPLE_PROMPTS.get(prompt_key, "")
        
        def process_audio_handler(audio, language, context):
            transcription, response, metrics = assistant.process_audio_advanced(audio, language, context)
            return transcription, response, metrics
        
        def process_text_handler(text, language, context):
            if not text.strip():
                return "Please enter some text."
            
            # Simulate text processing through the pipeline
            try:
                response = assistant._generate_contextual_response(text, language, context)
                assistant.conversation_history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "input": text,
                    "output": response,
                    "language": language,
                    "context": context,
                    "processing_time": "N/A (text input)"
                })
                return response
            except Exception as e:
                return f"Error: {str(e)}"
        
        def export_history():
            return assistant.get_conversation_history_json()
        
        def clear_history():
            return assistant.clear_history(), ""
        
        def clear_all():
            return "", "", "", assistant._get_metrics_display()
        
        # Connect events
        sample_prompt.change(update_sample_text, inputs=[sample_prompt], outputs=[sample_text_display])
        
        process_btn.click(
            process_audio_handler,
            inputs=[audio_input, language_selector, context_selector],
            outputs=[transcription_output, response_output, metrics_display]
        )
        
        text_process_btn.click(
            process_text_handler,
            inputs=[text_input, language_selector, context_selector],
            outputs=[text_response]
        )
        
        export_btn.click(export_history, outputs=[history_output])
        
        clear_history_btn.click(clear_history, outputs=[history_status, history_output])
        
        clear_btn.click(clear_all, outputs=[audio_input, transcription_output, response_output, metrics_display])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;">
            <h3>🚀 ZamAI Pro Models Strategy</h3>
            <p><strong>Voice Pipeline:</strong> Whisper-v3-Pashto → LLaMA-3-Pashto → TTS</p>
            <p><strong>Features:</strong> Context-aware | Performance tracking | Inference API powered | Multilingual</p>
            <p><strong>Models:</strong> 
                <a href="https://huggingface.co/tasal9/ZamAI-Whisper-v3-Pashto" style="color: #ffeb3b;">Whisper</a> | 
                <a href="https://huggingface.co/tasal9/ZamAI-LIama3-Pashto" style="color: #ffeb3b;">LLaMA-3</a>
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_advanced_voice_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
