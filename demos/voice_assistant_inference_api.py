"""
Enhanced ZamAI Voice Assistant with Inference API Integration
Whisper → LLaMA-3 → TTS Pipeline with Advanced Gradio UI
"""

import gradio as gr
import os
import tempfile
import time
import json
from typing import Optional, Tuple, Dict, Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EnhancedVoiceAssistant:
    """Enhanced Voice Assistant with comprehensive pipeline management"""
    
    def __init__(self):
        self.client = None
        self.conversation_history = []
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
        
        # Initialize HF client if token is available
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize HuggingFace Inference Client"""
        try:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token and hf_token != "test_token_placeholder":
                self.client = InferenceClient(token=hf_token)
                logger.info("HuggingFace Inference Client initialized successfully")
            else:
                logger.warning("HF_TOKEN not found or is placeholder - using demo mode")
        except Exception as e:
            logger.error(f"Failed to initialize HF client: {e}")
    
    def process_voice_pipeline(
        self, 
        audio_file: Optional[str], 
        language: str = "English",
        context: str = "General"
    ) -> Tuple[str, str, str]:
        """
        Complete voice processing pipeline: Whisper → LLaMA-3 → Response
        
        Args:
            audio_file: Path to audio file
            language: Language preference (English/Pashto)
            context: Context for response generation
            
        Returns:
            Tuple of (transcription, response, metrics)
        """
        start_time = time.time()
        
        try:
            # Validate input
            if audio_file is None:
                return "❌ No audio file provided", "Please record or upload an audio file", self._get_metrics_display()
            
            # Step 1: Speech-to-Text (Whisper)
            transcription = self._transcribe_audio(audio_file)
            
            if not transcription or transcription.startswith("❌"):
                return transcription, "Transcription failed", self._get_metrics_display()
            
            # Step 2: Generate AI Response (LLaMA-3)
            response = self._generate_response(transcription, language, context)
            
            # Update metrics and history
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, True)
            self._add_to_history(transcription, response, language, context, processing_time)
            
            return transcription, response, self._get_metrics_display()
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)
            error_msg = f"❌ Pipeline Error: {str(e)}"
            logger.error(error_msg)
            return error_msg, "Please try again", self._get_metrics_display()
    
    def _transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio using Whisper model"""
        if not self.client:
            return "🎤 Demo Mode: Simulated transcription of your audio input"
        
        try:
            whisper_model = os.getenv("WHISPER_MODEL", "openai/whisper-base")
            result = self.client.automatic_speech_recognition(
                audio_file,
                model=whisper_model
            )
            
            if isinstance(result, dict) and 'text' in result:
                return result['text'].strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                return "❌ Unexpected transcription format"
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"❌ Transcription failed: {str(e)}"
    
    def _generate_response(self, text: str, language: str, context: str) -> str:
        """Generate contextual response using LLaMA-3"""
        if not self.client:
            return f"🤖 Demo Mode Response: I heard '{text[:50]}...' and would respond based on {context} context in {language}."
        
        try:
            # Context-aware system prompts
            context_prompts = {
                "Educational": "You are an educational voice assistant specializing in clear, informative explanations.",
                "Business": "You are a professional business assistant providing concise, actionable responses.",
                "Casual": "You are a friendly conversational assistant with a natural, engaging personality.",
                "Technical": "You are a technical expert providing detailed, accurate information.",
                "General": "You are an intelligent voice assistant providing helpful responses."
            }
            
            if language == "Pashto":
                system_prompt = f"تاسو د پښتو ژبې غږیز ملګری یاست. {context_prompts.get(context, context_prompts['General'])} په پښتو کې ځواب ورکړئ."
            else:
                system_prompt = context_prompts.get(context, context_prompts["General"])
            
            # Build conversation context
            conversation_context = self._build_conversation_context()
            
            # LLaMA-3 chat format
            prompt = self._build_llama3_prompt(system_prompt, text, conversation_context)
            
            # Generate response
            llama3_model = os.getenv("LLAMA3_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
            response = self.client.text_generation(
                model=llama3_model,
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            return response.strip() if response else "❌ No response generated"
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"❌ Response generation failed: {str(e)}"
    
    def _build_llama3_prompt(self, system_prompt: str, user_text: str, context: str = "") -> str:
        """Build properly formatted LLaMA-3 prompt"""
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}"
        
        if context:
            prompt += f"\n\nConversation Context:\n{context}"
        
        prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        return prompt
    
    def _build_conversation_context(self) -> str:
        """Build context from recent conversation history"""
        if not self.conversation_history:
            return ""
        
        # Get last 3 exchanges
        recent_history = self.conversation_history[-3:]
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['input'][:100]}")
            context_parts.append(f"Assistant: {exchange['output'][:100]}")
        
        return "\n".join(context_parts)
    
    def _add_to_history(self, input_text: str, output_text: str, language: str, context: str, processing_time: float):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "input": input_text,
            "output": output_text,
            "language": language,
            "context": context,
            "processing_time": f"{processing_time:.2f}s"
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["total_response_time"] += processing_time
        
        if success:
            self.performance_metrics["successful_requests"] += 1
        
        # Calculate average response time
        self.performance_metrics["avg_response_time"] = (
            self.performance_metrics["total_response_time"] / 
            self.performance_metrics["total_requests"]
        )
    
    def _get_metrics_display(self) -> str:
        """Generate metrics display string"""
        metrics = self.performance_metrics
        success_rate = (metrics["successful_requests"] / max(metrics["total_requests"], 1)) * 100
        
        return f"""📊 **Performance Metrics**
• Total Requests: {metrics["total_requests"]}
• Success Rate: {success_rate:.1f}%
• Avg Response Time: {metrics["avg_response_time"]:.2f}s
• Client Status: {'🟢 Connected' if self.client else '🟡 Demo Mode'}"""
    
    def get_conversation_history_display(self) -> str:
        """Get formatted conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        history_text = "📝 **Recent Conversations**\n\n"
        for i, exchange in enumerate(self.conversation_history[-5:], 1):
            history_text += f"**{i}. [{exchange['timestamp']}] ({exchange['language']}, {exchange['context']})**\n"
            history_text += f"🎤 Input: {exchange['input'][:100]}{'...' if len(exchange['input']) > 100 else ''}\n"
            history_text += f"🤖 Response: {exchange['output'][:100]}{'...' if len(exchange['output']) > 100 else ''}\n"
            history_text += f"⏱️ Time: {exchange['processing_time']}\n\n"
        
        return history_text

def create_enhanced_voice_assistant_ui():
    """Create the enhanced Gradio interface"""
    
    # Initialize voice assistant
    assistant = EnhancedVoiceAssistant()
    
    # Custom CSS
    custom_css = """
    .header { text-align: center; margin-bottom: 30px; }
    .model-info { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0; }
    .audio-section { background: #f8f9ff; padding: 20px; border-radius: 15px; margin: 10px 0; border: 2px solid #e0e7ff; }
    .metrics-box { background: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; }
    .demo-warning { background: #fef3c7; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b; margin: 10px 0; }
    """
    
    with gr.Blocks(
        title="ZamAI Enhanced Voice Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎤 ZamAI Enhanced Voice Assistant</h1>
            <p><strong>Whisper → LLaMA-3 → TTS Pipeline</strong></p>
            <p>Advanced AI Voice Processing with Inference API Integration</p>
        </div>
        """)
        
        # Demo mode warning
        gr.HTML("""
        <div class="demo-warning">
            <h4>🚀 Getting Started</h4>
            <p><strong>Demo Mode:</strong> If you see "Demo Mode" responses, add your HuggingFace token to .env file:</p>
            <code>HF_TOKEN=your_huggingface_token_here</code>
            <p><strong>Production Mode:</strong> Requires access to private models: tasal9/ZamAI-Whisper-v3-Pashto, tasal9/ZamAI-LIama3-Pashto</p>
        </div>
        """)
        
        with gr.Row():
            # Left panel - Configuration and Info
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="model-info">
                    <h3>🤖 Pipeline Components</h3>
                    <p><strong>🎤 STT:</strong> Whisper-v3-Pashto</p>
                    <p><strong>🧠 LLM:</strong> LLaMA-3-Pashto</p>
                    <p><strong>🔊 TTS:</strong> Coqui-TTS (Future)</p>
                    <p><strong>🌐 Languages:</strong> English, Pashto</p>
                    <p><strong>⚡ API:</strong> HF Inference API</p>
                </div>
                """)
                
                # Configuration options
                language_selector = gr.Radio(
                    choices=["English", "Pashto"],
                    value="English",
                    label="🌐 Language Preference"
                )
                
                context_selector = gr.Dropdown(
                    choices=["General", "Educational", "Business", "Technical", "Casual"],
                    value="General",
                    label="🎯 Context Mode"
                )
                
                # Metrics display
                metrics_display = gr.Markdown(
                    assistant._get_metrics_display(),
                    elem_classes=["metrics-box"]
                )
            
            # Right panel - Main interface
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Voice Input Tab
                    with gr.TabItem("🎙️ Voice Input", id="voice"):
                        gr.HTML('<div class="audio-section">')
                        
                        audio_input = gr.Audio(
                            label="🎤 Record or Upload Audio",
                            type="filepath",
                            sources=["microphone", "upload"],
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#3b82f6",
                                waveform_progress_color="#1e40af"
                            )
                        )
                        
                        process_btn = gr.Button(
                            "🔄 Process Voice Input", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        with gr.Row():
                            transcription_output = gr.Textbox(
                                label="📝 Speech Transcription",
                                lines=3,
                                interactive=False,
                                placeholder="Audio transcription will appear here..."
                            )
                            
                            response_output = gr.Textbox(
                                label="🤖 AI Response",
                                lines=3,
                                interactive=False,
                                placeholder="AI response will appear here..."
                            )
                        
                        gr.HTML('</div>')
                    
                    # Text Input Tab
                    with gr.TabItem("💬 Text Chat", id="text"):
                        text_input = gr.Textbox(
                            label="💬 Type your message",
                            lines=3,
                            placeholder="Type your question here...",
                            max_lines=5
                        )
                        
                        text_process_btn = gr.Button("📤 Send Message", variant="primary")
                        
                        text_response = gr.Textbox(
                            label="🤖 Response",
                            lines=5,
                            interactive=False,
                            placeholder="Response will appear here..."
                        )
                    
                    # History Tab
                    with gr.TabItem("📜 Conversation History", id="history"):
                        history_display = gr.Markdown(
                            "No conversation history yet.",
                            label="Recent Conversations"
                        )
                        
                        refresh_history_btn = gr.Button("🔄 Refresh History")
                        clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
        
        # Event handlers
        def process_voice_input(audio, language, context):
            transcription, response, metrics = assistant.process_voice_pipeline(audio, language, context)
            return transcription, response, metrics
        
        def process_text_input(text, language, context):
            if not text.strip():
                return "Please enter some text."
            
            # Use text processing (skip STT)
            response = assistant._generate_response(text, language, context)
            assistant._add_to_history(text, response, language, context, 0.0)
            
            return response
        
        def refresh_history():
            return assistant.get_conversation_history_display()
        
        def clear_history():
            assistant.conversation_history.clear()
            return "Conversation history cleared."
        
        # Connect events
        process_btn.click(
            fn=process_voice_input,
            inputs=[audio_input, language_selector, context_selector],
            outputs=[transcription_output, response_output, metrics_display]
        )
        
        text_process_btn.click(
            fn=process_text_input,
            inputs=[text_input, language_selector, context_selector],
            outputs=[text_response]
        )
        
        refresh_history_btn.click(
            fn=refresh_history,
            outputs=[history_display]
        )
        
        clear_history_btn.click(
            fn=clear_history,
            outputs=[history_display]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 15px;">
            <h4>🚀 ZamAI Pro Models Strategy - Voice Assistant</h4>
            <p><strong>Pipeline:</strong> 
                <a href="https://huggingface.co/tasal9/ZamAI-Whisper-v3-Pashto" target="_blank">Whisper-v3-Pashto</a> → 
                <a href="https://huggingface.co/tasal9/ZamAI-LIama3-Pashto" target="_blank">LLaMA3-Pashto</a> → TTS
            </p>
            <p>🔧 <strong>Features:</strong> Real-time Processing | Context Awareness | Performance Metrics | Inference API</p>
            <p>🌐 <strong>Languages:</strong> English & Pashto | 🎯 <strong>Contexts:</strong> Educational, Business, Technical, Casual</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the enhanced voice assistant
    demo = create_enhanced_voice_assistant_ui()
    
    print("🎤 Starting ZamAI Enhanced Voice Assistant...")
    print("🌐 Access the interface at: http://localhost:7862")
    print("📝 Features: Whisper → LLaMA-3 pipeline, Context awareness, Performance metrics")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,  # Set to True for public sharing
        debug=True,
        show_error=True
    )
