#!/usr/bin/env python3
"""
ZamAI Enhanced Tutor Bot
Mistral-7B-Instruct fine-tuned on Pashto QA Dataset
"""

import gradio as gr
import os
import json
import pandas as pd
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import time
import random
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class ZamAITutorBot:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.mistral_model = os.getenv("MISTRAL_EDU_MODEL", "tasal9/ZamAI-Mistral-7B-Pashto")
        self.client = InferenceClient(token=self.hf_token)
        
        # Load processed dataset
        self.dataset_path = Path("datasets/processed")
        self.qa_data = self.load_qa_dataset()
        self.evaluation_data = self.load_evaluation_data()
        
        # Performance tracking
        self.session_stats = {
            "questions_asked": 0,
            "correct_answers": 0,
            "avg_response_time": 0,
            "session_start": time.time()
        }
    
    def load_qa_dataset(self):
        """Load the processed QA dataset"""
        try:
            qa_file = self.dataset_path / "tutoring_qa.json"
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded {len(data)} QA examples from dataset")
                return data
            else:
                print("⚠️ QA dataset not found. Generate it first with dataset_integration.py")
                return []
        except Exception as e:
            print(f"❌ Error loading QA dataset: {e}")
            return []
    
    def load_evaluation_data(self):
        """Load evaluation data for testing"""
        try:
            eval_file = self.dataset_path / "eval_qa.json"
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded {len(data)} evaluation examples")
                return data
            else:
                print("⚠️ Evaluation dataset not found")
                return []
        except Exception as e:
            print(f"❌ Error loading evaluation data: {e}")
            return []
    
    def get_similar_examples(self, question: str, limit: int = 3) -> List[Dict]:
        """Find similar examples from the dataset"""
        if not self.qa_data:
            return []
        
        # Simple similarity based on keyword matching
        question_lower = question.lower()
        scored_examples = []
        
        for example in self.qa_data:
            example_q = example.get('question', '').lower()
            
            # Simple scoring based on common words
            common_words = set(question_lower.split()) & set(example_q.split())
            score = len(common_words)
            
            if score > 0:
                scored_examples.append((score, example))
        
        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:limit]]
    
    def generate_contextualized_response(self, question: str, language: str = "Mixed", 
                                       difficulty: str = "Medium", category: str = "General"):
        """Generate response using dataset context and Mistral model"""
        start_time = time.time()
        
        try:
            # Get similar examples from dataset
            similar_examples = self.get_similar_examples(question, limit=2)
            
            # Build context from similar examples
            context = ""
            if similar_examples:
                context = "Here are some related examples from my knowledge:\n\n"
                for i, example in enumerate(similar_examples, 1):
                    context += f"Example {i}:\n"
                    context += f"Q: {example['question']}\n"
                    context += f"A: {example['answer']}\n\n"
            
            # Create language-specific system prompt
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
            
            # Category-specific guidance
            category_guidance = {
                "Language": "Focus on Pashto grammar, vocabulary, and usage.",
                "Culture": "Explain Pashto/Afghan cultural concepts and traditions.",
                "History": "Provide historical context and background information.",
                "Literature": "Discuss Pashto literature, poetry, and writings.",
                "General": "Provide comprehensive educational support."
            }
            
            guidance = category_guidance.get(category, category_guidance["General"])
            
            # Build the complete prompt
            full_prompt = f"""<s>[INST] {system_prompt}
            
Instructions: {instruction}
Focus: {guidance}

{context}

Student Question: {question}

Please provide a helpful and educational response. [/INST]"""
            
            # Generate response using Mistral
            response = self.client.text_generation(
                model=self.mistral_model,
                prompt=full_prompt,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            # Update performance stats
            response_time = time.time() - start_time
            self.session_stats["questions_asked"] += 1
            self.session_stats["avg_response_time"] = (
                (self.session_stats["avg_response_time"] * (self.session_stats["questions_asked"] - 1) + response_time) 
                / self.session_stats["questions_asked"]
            )
            
            return response.strip(), f"Response time: {response_time:.2f}s"
            
        except Exception as e:
            return f"Error generating response: {str(e)}", "Error occurred"
    
    def get_random_example(self):
        """Get a random example from the dataset"""
        if not self.qa_data:
            return "No dataset loaded", "Please load the QA dataset first"
        
        example = random.choice(self.qa_data)
        return example.get('question', ''), example.get('answer', '')
    
    def run_evaluation_test(self, num_samples: int = 5):
        """Run evaluation on a sample of the evaluation dataset"""
        if not self.evaluation_data:
            return "No evaluation data available"
        
        if num_samples > len(self.evaluation_data):
            num_samples = len(self.evaluation_data)
        
        test_samples = random.sample(self.evaluation_data, num_samples)
        results = []
        
        for i, example in enumerate(test_samples, 1):
            question = example.get('question', '')
            expected_answer = example.get('answer', '')
            
            # Generate response
            generated_answer, _ = self.generate_contextualized_response(
                question, 
                language=example.get('language', 'Mixed').title(),
                category=example.get('category', 'General').title()
            )
            
            results.append({
                'question': question,
                'expected': expected_answer[:100] + "..." if len(expected_answer) > 100 else expected_answer,
                'generated': generated_answer[:100] + "..." if len(generated_answer) > 100 else generated_answer,
                'language': example.get('language', 'mixed'),
                'category': example.get('category', 'general')
            })
        
        # Format results for display
        result_text = f"📊 Evaluation Results ({num_samples} samples):\n\n"
        for i, result in enumerate(results, 1):
            result_text += f"**Test {i}:**\n"
            result_text += f"Question: {result['question']}\n"
            result_text += f"Expected: {result['expected']}\n"
            result_text += f"Generated: {result['generated']}\n"
            result_text += f"Language: {result['language']}, Category: {result['category']}\n\n"
        
        return result_text
    
    def get_session_stats(self):
        """Get current session statistics"""
        session_duration = time.time() - self.session_stats["session_start"]
        
        stats_text = f"""
📈 **Session Statistics:**
• Questions Asked: {self.session_stats['questions_asked']}
• Session Duration: {session_duration/60:.1f} minutes
• Avg Response Time: {self.session_stats['avg_response_time']:.2f}s
• Dataset Examples: {len(self.qa_data)}
• Evaluation Examples: {len(self.evaluation_data)}
"""
        return stats_text
    
def create_tutor_interface():
    """Create the enhanced Gradio interface for the Tutor Bot"""
    
    # Initialize the tutor bot
    tutor = ZamAITutorBot()
    
    # Custom CSS for better visibility
    custom_css = """
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .gradio-container {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 20px;
        padding: 20px;
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
    .info-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stats-card {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .example-card {
        background: #fff3cd;
        border: 2px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .textbox textarea {
        background: white !important;
        color: #2d3748 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px;
    }
    .button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
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
            <p><strong>Features:</strong> Dataset-Aware Responses | Contextual Learning | Performance Evaluation</p>
            <p><strong>Model:</strong> tasal9/ZamAI-Mistral-7B-Pashto | <strong>Dataset:</strong> tasal9/Pashto-Dataset-Creating-Dataset</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-card">
                    <h3>🗃️ Dataset Information</h3>
                    <p><strong>Source:</strong> tasal9/Pashto-Dataset-Creating-Dataset</p>
                    <p><strong>Processing:</strong> Automatic QA extraction</p>
                    <p><strong>Languages:</strong> Pashto, English, Mixed</p>
                    <p><strong>Categories:</strong> Language, Culture, History, Literature</p>
                </div>
                """)
                
                # Configuration controls
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
                    choices=["General", "Language", "Culture", "History", "Literature"],
                    value="General",
                    label="📂 Category Focus"
                )
                
                # Example button
                example_btn = gr.Button("🎲 Get Random Example", variant="secondary")
                
            with gr.Column(scale=2):
                with gr.Tab("💬 Tutoring Chat"):
                    question_input = gr.Textbox(
                        label="Ask Your Question",
                        placeholder="Enter your question in English or Pashto...",
                        lines=3
                    )
                    
                    ask_btn = gr.Button("Ask Tutor 🎓", variant="primary", size="lg")
                    
                    with gr.Row():
                        response_output = gr.Textbox(
                            label="🤖 Tutor Response",
                            lines=8,
                            interactive=False
                        )
                        timing_output = gr.Textbox(
                            label="⏱️ Performance",
                            lines=2,
                            interactive=False
                        )
                
                with gr.Tab("📊 Dataset Examples"):
                    gr.HTML("""
                    <div class="example-card">
                        <h4>📚 Explore Dataset Examples</h4>
                        <p>Click below to see random examples from the processed QA dataset</p>
                    </div>
                    """)
                    
                    example_question = gr.Textbox(
                        label="📝 Example Question",
                        interactive=False,
                        lines=2
                    )
                    
                    example_answer = gr.Textbox(
                        label="💡 Example Answer",
                        interactive=False,
                        lines=4
                    )
                
                with gr.Tab("🧪 Model Evaluation"):
                    gr.HTML("""
                    <div class="info-card">
                        <h4>🔬 Evaluation Testing</h4>
                        <p>Test the model performance on evaluation dataset samples</p>
                    </div>
                    """)
                    
                    eval_samples = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of Test Samples"
                    )
                    
                    eval_btn = gr.Button("Run Evaluation Test 🧪", variant="primary")
                    
                    eval_results = gr.Textbox(
                        label="📋 Evaluation Results",
                        lines=15,
                        interactive=False
                    )
                
                with gr.Tab("📈 Statistics"):
                    stats_btn = gr.Button("Refresh Stats 📊", variant="secondary")
                    
                    stats_display = gr.Textbox(
                        label="📈 Session Statistics",
                        lines=10,
                        interactive=False
                    )
        
        # Event handlers
        ask_btn.click(
            fn=tutor.generate_contextualized_response,
            inputs=[question_input, language_selector, difficulty_selector, category_selector],
            outputs=[response_output, timing_output]
        )
        
        example_btn.click(
            fn=tutor.get_random_example,
            outputs=[example_question, example_answer]
        )
        
        eval_btn.click(
            fn=tutor.run_evaluation_test,
            inputs=[eval_samples],
            outputs=[eval_results]
        )
        
        stats_btn.click(
            fn=tutor.get_session_stats,
            outputs=[stats_display]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p><strong>🎓 ZamAI Enhanced Tutor Bot</strong></p>
            <p>Powered by <strong>Mistral-7B-Instruct</strong> fine-tuned on Pashto educational datasets</p>
            <p><strong>🔧 Features:</strong> Dataset Integration | Context-Aware Responses | Performance Evaluation | Multi-language Support</p>
            <p><a href="https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto" target="_blank">Model Repository</a> | 
               <a href="https://huggingface.co/tasal9/Pashto-Dataset-Creating-Dataset" target="_blank">Dataset Repository</a></p>
        </div>
        """)
    
    return demo

def main():
    """Main function to launch the Enhanced Tutor Bot"""
    print("🎓 Starting ZamAI Enhanced Tutor Bot...")
    
    # Create and launch the interface
    demo = create_tutor_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=True,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
