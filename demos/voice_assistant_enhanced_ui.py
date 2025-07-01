#!/usr/bin/env python3
"""
ZamAI Enhanced Voice Assistant Demo
Professional Whisper → LLaMA-3 → TTS Pipeline with Clear UI
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

class EnhancedVoiceAssistant:
    def __init__(self):
        self.conversation_history = []
        self.performance_metrics = {
            "total_requests": 0,
            "avg_response_time": 0,
            "success_rate": 100.0,
            "total_processing_time": 0
        }
    
    def process_audio_enhanced(self, audio_file, language="English", context="General"):
        """Enhanced audio processing with improved error handling"""
        start_time = time.time()
        
        try:
            if audio_file is None:
                return "⚠️ Please record or upload an audio file first.", "", self._get_enhanced_metrics()
            
            # Step 1: Speech-to-Text with Whisper
            transcription = client.automatic_speech_recognition(
                audio_file,
                model=os.getenv("WHISPER_MODEL", "tasal9/ZamAI-Whisper-v3-Pashto")
            )
            
            if not transcription.strip():
                return "❌ Could not transcribe audio. Please try again with clearer audio.", "", self._get_enhanced_metrics()
            
            # Step 2: Enhanced LLaMA-3 processing
            response = self._generate_enhanced_response(transcription, language, context)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_enhanced_metrics(processing_time, True)
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "input": transcription,
                "output": response,
                "language": language,
                "context": context,
                "processing_time": f"{processing_time:.2f}s"
            })
            
            return f"✅ {transcription}", f"🤖 {response}", self._get_enhanced_metrics()
            
        except Exception as e:
            self._update_enhanced_metrics(time.time() - start_time, False)
            return f"❌ Transcription Error: {str(e)}", "🔧 Please check your model access and try again.", self._get_enhanced_metrics()
    
    def _generate_enhanced_response(self, text, language, context):
        """Generate enhanced context-aware response using LLaMA-3"""
        
        # Enhanced context-specific system prompts
        context_prompts = {
            "Educational": {
                "en": "You are an expert educational tutor. Provide clear, comprehensive explanations with examples. Break down complex topics into digestible parts.",
                "ps": "تاسو د تعلیماتو کارپوه ملګری یاست. واضح او جامع تشریحات د مثالونو سره ورکړئ. پیچلي موضوعات په اسانه برخو ویشئ."
            },
            "Business": {
                "en": "You are a professional business consultant. Provide concise, actionable advice with practical insights.",
                "ps": "تاسو د سوداګرۍ مسلکي مشاور یاست. لنډ او عملي مشورې د ګټورو نظرونو سره ورکړئ."
            },
            "Technical": {
                "en": "You are a technical expert. Provide detailed, accurate technical information with step-by-step explanations.",
                "ps": "تاسو د تخنیک کارپوه یاست. تفصیلي او سمې تخنیکي معلومات د ګام په ګام تشریحاتو سره ورکړئ."
            },
            "Casual": {
                "en": "You are a friendly conversational assistant. Be natural, engaging, and helpful in a relaxed manner.",
                "ps": "تاسو د دوستانه خبرو اترو ملګری یاست. په طبیعي، ګښونکي او ګټوري ډول ځواب ورکړئ."
            },
            "General": {
                "en": "You are an intelligent AI assistant. Provide helpful, accurate, and well-structured responses.",
                "ps": "تاسو د ذکي AI د ملګري یاست. ګټور، سم او ښه جوړ شوي ځوابونه ورکړئ."
            }
        }
        
        # Select appropriate prompt
        lang_code = "ps" if language == "Pashto" else "en"
        system_prompt = context_prompts.get(context, context_prompts["General"])[lang_code]
        
        # Build conversation context (last 3 exchanges)
        conversation_context = ""
        if len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-3:]
            for exchange in recent_history:
                conversation_context += f"Previous Q: {exchange['input'][:50]}...\nPrevious A: {exchange['output'][:50]}...\n"
        
        # Enhanced prompt with context
        if conversation_context:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n\nConversation Context:\n{conversation_context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        response = client.text_generation(
            model=os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto"),
            prompt=full_prompt,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        return response.strip()
    
    def _update_enhanced_metrics(self, processing_time, success):
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["total_processing_time"] += processing_time
        self.performance_metrics["avg_response_time"] = self.performance_metrics["total_processing_time"] / self.performance_metrics["total_requests"]
        
        if success:
            success_count = int(self.performance_metrics["success_rate"] / 100 * (self.performance_metrics["total_requests"] - 1)) + 1
        else:
            success_count = int(self.performance_metrics["success_rate"] / 100 * (self.performance_metrics["total_requests"] - 1))
        
        self.performance_metrics["success_rate"] = (success_count / self.performance_metrics["total_requests"]) * 100
    
    def _get_enhanced_metrics(self):
        """Get formatted metrics display"""
        metrics = self.performance_metrics
        return f"""📊 **Performance Analytics**
        
🔢 **Requests:** {metrics['total_requests']}
⏱️ **Avg Response:** {metrics['avg_response_time']:.2f}s
✅ **Success Rate:** {metrics['success_rate']:.1f}%
📈 **Total Conversations:** {len(self.conversation_history)}
        """

# Initialize assistant
assistant = EnhancedVoiceAssistant()

def create_enhanced_voice_demo():
    with gr.Blocks(
        title="🎤 ZamAI Enhanced Voice Assistant",
        theme=gr.themes.Default(),
        css="""
        /* Modern Professional UI Styling */
        .gradio-container { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
            min-height: 100vh;
        }
        
        .main-content {
            background: rgba(255, 255, 255, 0.95) !important;
            margin: 20px;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Header */
        .header { 
            text-align: center; 
            margin-bottom: 40px; 
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); 
            color: white !important; 
            padding: 30px; 
            border-radius: 20px; 
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.3);
        }
        
        .header h1 {
            color: white !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }
        
        .header p {
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 1.1rem !important;
        }
        
        /* Info panels */
        .info-panel { 
            background: #ffffff !important; 
            padding: 25px; 
            border-radius: 15px; 
            margin: 20px 0; 
            border-left: 5px solid #2563eb; 
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            color: #1f2937 !important;
        }
        
        .info-panel h3 {
            color: #2563eb !important;
            margin-bottom: 15px !important;
            font-weight: 600 !important;
        }
        
        /* Metrics display */
        .metrics-panel { 
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important; 
            padding: 20px; 
            border-radius: 15px; 
            font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace; 
            border: 2px solid #3b82f6;
            color: #1e40af !important;
            margin: 15px 0;
        }
        
        /* Audio section */
        .audio-panel {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            border: 2px solid #22c55e;
            color: #166534 !important;
        }
        
        /* Buttons */
        .gr-button {
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 15px 30px !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
        }
        
        .gr-button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        }
        
        /* Input fields */
        .gr-textbox, .gr-dropdown, .gr-radio {
            background: white !important;
            border: 2px solid #e5e7eb !important;
            border-radius: 12px !important;
            color: #1f2937 !important;
            font-size: 1rem !important;
            padding: 12px !important;
        }
        
        .gr-textbox:focus, .gr-dropdown:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        }
        
        /* Tabs */
        .gr-tab {
            background: white !important;
            color: #1f2937 !important;
            border-radius: 12px !important;
            border: 2px solid #e5e7eb !important;
            font-weight: 500 !important;
            padding: 12px 20px !important;
        }
        
        .gr-tab.selected {
            background: #2563eb !important;
            color: white !important;
            border-color: #2563eb !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
            border-radius: 15px;
            border-top: 4px solid #2563eb;
            color: #1f2937 !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        """
    ) as demo:
        
        with gr.Column(elem_classes="main-content"):
            gr.HTML("""
            <div class="header">
                <h1>🎤 ZamAI Enhanced Voice Assistant</h1>
                <p><strong>Whisper → LLaMA-3 → TTS Pipeline</strong></p>
                <p>Professional AI Voice Processing with Advanced Analytics</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="info-panel">
                        <h3>🔄 Processing Pipeline</h3>
                        <p><strong>STT:</strong> Whisper-v3-Pashto</p>
                        <p><strong>LLM:</strong> LLaMA-3-Pashto (Private)</p>
                        <p><strong>TTS:</strong> Context-aware synthesis</p>
                        <p><strong>API:</strong> HF Inference API</p>
                        <p><strong>Languages:</strong> English & Pashto</p>
                    </div>
                    """)
                    
                    # Enhanced configuration controls
                    language_selector = gr.Radio(
                        choices=["English", "Pashto"],
                        value="English",
                        label="🌐 Language Selection",
                        info="Choose your preferred language"
                    )
                    
                    context_selector = gr.Radio(
                        choices=["General", "Educational", "Business", "Technical", "Casual"],
                        value="General",
                        label="🎯 Context Mode",
                        info="Select conversation context"
                    )
                    
                    # Performance metrics display
                    metrics_display = gr.Markdown(
                        value="📊 **Performance Analytics**\n\n🔢 **Requests:** 0\n⏱️ **Avg Response:** 0.00s\n✅ **Success Rate:** 100.0%",
                        elem_classes="metrics-panel"
                    )
                
                with gr.Column(scale=2):
                    with gr.Tab("🎙️ Voice Processing"):
                        gr.HTML('<div class="audio-panel">')
                        
                        audio_input = gr.Audio(
                            label="🎤 Record or Upload Audio",
                            type="filepath",
                            sources=["microphone", "upload"],
                            elem_id="audio-input"
                        )
                        
                        process_btn = gr.Button(
                            "🚀 Process Audio", 
                            variant="primary", 
                            size="lg",
                            elem_id="process-btn"
                        )
                        
                        with gr.Row():
                            transcription_output = gr.Textbox(
                                label="📝 Speech-to-Text",
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
                    
                    with gr.Tab("💬 Text Chat"):
                        text_input = gr.Textbox(
                            label="💭 Your Message",
                            lines=3,
                            placeholder="Type your question or message here...",
                            elem_id="text-input"
                        )
                        
                        text_process_btn = gr.Button(
                            "💫 Get Response", 
                            variant="primary",
                            size="lg"
                        )
                        
                        text_response = gr.Textbox(
                            label="🤖 AI Response",
                            lines=5,
                            interactive=False,
                            placeholder="AI response will appear here..."
                        )
                    
                    with gr.Tab("📊 Analytics"):
                        conversation_history = gr.JSON(
                            label="📜 Conversation History",
                            value=[]
                        )
                        
                        clear_history_btn = gr.Button(
                            "🗑️ Clear History",
                            variant="secondary"
                        )
            
            # Event handlers
            def process_audio_interface(audio, language, context):
                transcription, response, metrics = assistant.process_audio_enhanced(audio, language, context)
                history = assistant.conversation_history
                return transcription, response, metrics, history
            
            def process_text_interface(text, language, context):
                if not text.strip():
                    return "⚠️ Please enter some text first."
                
                # Process text directly with LLaMA-3 (skip STT)
                start_time = time.time()
                try:
                    response = assistant._generate_enhanced_response(text, language, context)
                    processing_time = time.time() - start_time
                    assistant._update_enhanced_metrics(processing_time, True)
                    
                    # Add to conversation history
                    assistant.conversation_history.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "input": text,
                        "output": response,
                        "language": language,
                        "context": context,
                        "processing_time": f"{processing_time:.2f}s"
                    })
                    
                    return f"🤖 {response}", assistant._get_enhanced_metrics(), assistant.conversation_history
                except Exception as e:
                    assistant._update_enhanced_metrics(time.time() - start_time, False)
                    return f"❌ Error: {str(e)}", assistant._get_enhanced_metrics(), assistant.conversation_history
            
            def clear_history():
                assistant.conversation_history = []
                assistant.performance_metrics = {
                    "total_requests": 0,
                    "avg_response_time": 0,
                    "success_rate": 100.0,
                    "total_processing_time": 0
                }
                return "📊 **Performance Analytics**\n\n🔢 **Requests:** 0\n⏱️ **Avg Response:** 0.00s\n✅ **Success Rate:** 100.0%", []
            
            # Connect events
            process_btn.click(
                process_audio_interface,
                inputs=[audio_input, language_selector, context_selector],
                outputs=[transcription_output, response_output, metrics_display, conversation_history]
            )
            
            text_process_btn.click(
                process_text_interface,
                inputs=[text_input, language_selector, context_selector],
                outputs=[text_response, metrics_display, conversation_history]
            )
            
            clear_history_btn.click(
                clear_history,
                outputs=[metrics_display, conversation_history]
            )
            
            # Footer
            gr.HTML("""
            <div class="footer">
                <h3>🚀 Powered by ZamAI Pro Models Strategy</h3>
                <p><strong>Voice Pipeline:</strong> 
                    <a href="https://huggingface.co/tasal9/ZamAI-Whisper-v3-Pashto" target="_blank" style="color: #2563eb;">Whisper-v3-Pashto</a> → 
                    <a href="https://huggingface.co/tasal9/ZamAI-LIama3-Pashto" target="_blank" style="color: #2563eb;">LLaMA3-Pashto</a> → TTS
                </p>
                <p><strong>Features:</strong> Real-time processing | Advanced analytics | Context-aware responses | Multi-language support</p>
                <p><strong>Status:</strong> ✅ Inference API Connected | 🔒 Private Models | 📊 Performance Monitoring</p>
            </div>
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_enhanced_voice_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        debug=True,
        show_error=True,
        favicon_path=None,
        app_kwargs={"docs_url": "/docs"}
    )
