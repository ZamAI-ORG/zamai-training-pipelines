#!/usr/bin/env python3
"""
HuggingFace Space App for ZamAI Enhanced Tutor Bot
"""

import gradio as gr
import os
import json
import random
from huggingface_hub import InferenceClient
import time
from typing import List, Dict, Any

# Initialize the Inference Client
client = InferenceClient()

# Model configuration
MISTRAL_MODEL = "tasal9/ZamAI-Mistral-7B-Pashto"

class SpaceTutorBot:
    def __init__(self):
        self.client = client
        self.session_stats = {
            "questions_asked": 0,
            "session_start": time.time()
        }
        
        # Sample QA data (since we can't load files in HF Spaces easily)
        self.sample_qa_data = [
            {
                "question": "What is the capital of Afghanistan?",
                "answer": "The capital of Afghanistan is Kabul (کابل).",
                "language": "english",
                "category": "geography"
            },
            {
                "question": "د افغانستان پلازمینه کومه ده؟",
                "answer": "د افغانستان پلازمینه کابل دی.",
                "language": "pashto",
                "category": "geography"
            },
            {
                "question": "What does 'Salam' mean in Pashto?",
                "answer": "Salam (سلام) means 'peace' or 'hello' in Pashto. It's a common greeting.",
                "language": "mixed",
                "category": "language"
            }
        ]
    
    def generate_response(self, question: str, language: str = "Mixed", 
                         difficulty: str = "Medium", category: str = "General"):
        """Generate response using Mistral model"""
        start_time = time.time()
        
        try:
            # Language-specific system prompt
            if language == "Pashto":
                system_prompt = """تاسو د پښتو ژبې د تعلیماتو ملګری یاست. ډیر ځیرک او مرسته کوونکی یاست. 
د زده کوونکو سره په پښتو کې خبرې کوئ او ځیرک ځوابونه ورکړئ."""
            else:
                system_prompt = """You are an intelligent educational tutor specialized in Pashto language and culture. 
You provide helpful, accurate, and engaging responses to help students learn."""
            
            # Difficulty-specific instructions
            difficulty_instructions = {
                "Easy": "Keep explanations simple and use basic vocabulary.",
                "Medium": "Provide balanced explanations with examples.",
                "Hard": "Give detailed, comprehensive explanations with advanced concepts."
            }
            
            instruction = difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])
            
            # Build the prompt
            full_prompt = f"""<s>[INST] {system_prompt}
            
Instructions: {instruction}
Focus: {category} education

Student Question: {question}

Please provide a helpful and educational response. [/INST]"""
            
            # Generate response
            response = self.client.text_generation(
                model=MISTRAL_MODEL,
                prompt=full_prompt,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            # Update stats
            response_time = time.time() - start_time
            self.session_stats["questions_asked"] += 1
            
            return response.strip(), f"Response time: {response_time:.2f}s | Questions asked: {self.session_stats['questions_asked']}"
            
        except Exception as e:
            return f"Error: {str(e)}", "Error occurred"
    
    def get_random_example(self):
        """Get a random example from sample data"""
        example = random.choice(self.sample_qa_data)
        return example["question"], example["answer"]
    
    def get_session_stats(self):
        """Get current session statistics"""
        session_duration = time.time() - self.session_stats["session_start"]
        
        stats_text = f"""
📈 **Session Statistics:**
• Questions Asked: {self.session_stats['questions_asked']}
• Session Duration: {session_duration/60:.1f} minutes
• Model: {MISTRAL_MODEL}
• Status: Running on HuggingFace Spaces
"""
        return stats_text

def create_space_interface():
    """Create the HuggingFace Space interface"""
    
    tutor = SpaceTutorBot()
    
    # Custom CSS for the Space
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .main-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .header-section {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .textbox textarea {
        background: white !important;
        color: #2d3748 !important;
        border: 2px solid #e2e8f0 !important;
    }
    """
    
    with gr.Blocks(
        title="🎓 ZamAI Enhanced Tutor Bot",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header-section">
            <h1>🎓 ZamAI Enhanced Tutor Bot</h1>
            <h3>Mistral-7B-Instruct Fine-tuned on Pashto QA Dataset</h3>
            <p><strong>Model:</strong> tasal9/ZamAI-Mistral-7B-Pashto</p>
            <p><strong>Dataset:</strong> tasal9/Pashto-Dataset-Creating-Dataset</p>
            <p><strong>Features:</strong> Multilingual Support | Adaptive Difficulty | Educational Focus</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3>🌟 Features</h3>
                    <ul>
                        <li>🧠 Fine-tuned Mistral-7B</li>
                        <li>🗃️ Pashto QA Dataset</li>
                        <li>🌐 Multi-language Support</li>
                        <li>📚 Adaptive Difficulty</li>
                        <li>🎯 Educational Focus</li>
                    </ul>
                </div>
                """)
                
                language_selector = gr.Radio(
                    choices=["Mixed", "Pashto", "English"],
                    value="Mixed",
                    label="🌐 Language Preference"
                )
                
                difficulty_selector = gr.Radio(
                    choices=["Easy", "Medium", "Hard"],
                    value="Medium",
                    label="📚 Difficulty Level"
                )
                
                category_selector = gr.Radio(
                    choices=["General", "Language", "Culture", "History", "Geography"],
                    value="General",
                    label="📂 Category Focus"
                )
                
                example_btn = gr.Button("🎲 Random Example", variant="secondary")
                
            with gr.Column(scale=2):
                with gr.Tab("💬 Ask the Tutor"):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything in English or Pashto...",
                        lines=3
                    )
                    
                    ask_btn = gr.Button("Ask Tutor 🎓", variant="primary", size="lg")
                    
                    response_output = gr.Textbox(
                        label="🤖 Tutor Response",
                        lines=10,
                        interactive=False
                    )
                    
                    status_output = gr.Textbox(
                        label="📊 Status",
                        lines=2,
                        interactive=False
                    )
                
                with gr.Tab("📚 Examples"):
                    gr.HTML("""
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h4>📖 Sample Questions</h4>
                        <p>Explore example questions and answers from the dataset</p>
                    </div>
                    """)
                    
                    example_question = gr.Textbox(
                        label="Example Question",
                        interactive=False,
                        lines=2
                    )
                    
                    example_answer = gr.Textbox(
                        label="Example Answer",
                        interactive=False,
                        lines=4
                    )
                
                with gr.Tab("📈 Session Info"):
                    stats_btn = gr.Button("Refresh Stats 📊", variant="secondary")
                    
                    stats_display = gr.Textbox(
                        label="Session Statistics",
                        lines=8,
                        interactive=False
                    )
        
        # Event handlers
        ask_btn.click(
            fn=tutor.generate_response,
            inputs=[question_input, language_selector, difficulty_selector, category_selector],
            outputs=[response_output, status_output]
        )
        
        example_btn.click(
            fn=tutor.get_random_example,
            outputs=[example_question, example_answer]
        )
        
        stats_btn.click(
            fn=tutor.get_session_stats,
            outputs=[stats_display]
        )
        
        # Sample questions for easy testing
        gr.HTML("""
        <div style="background: #e8f4fd; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4>🎯 Try These Sample Questions:</h4>
            <ul>
                <li><strong>English:</strong> "What is the history of Pashto language?"</li>
                <li><strong>Pashto:</strong> "د پښتو ژبې تاریخ څه دی؟"</li>
                <li><strong>Culture:</strong> "Tell me about Afghan traditions"</li>
                <li><strong>Language:</strong> "How do you say 'thank you' in Pashto?"</li>
            </ul>
        </div>
        """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p><strong>🎓 ZamAI Enhanced Tutor Bot</strong> - Powered by Mistral-7B & Pashto Educational Dataset</p>
            <p>🔗 <a href="https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto" target="_blank">Model</a> | 
               <a href="https://huggingface.co/datasets/tasal9/Pashto-Dataset-Creating-Dataset" target="_blank">Dataset</a> | 
               <a href="https://github.com/ZamAI-Pro-Models-Strategy2" target="_blank">GitHub</a></p>
        </div>
        """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_space_interface()
    demo.launch()
