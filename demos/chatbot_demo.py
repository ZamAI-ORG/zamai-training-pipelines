"""
ZamAI Educational Chatbot Demo
Gradio interface for educational tutoring with Pashto support
"""

import gradio as gr
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Initialize HF Inference Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def educational_chat(message, history, language="English"):
    """
    Educational chatbot function with Pashto support
    """
    try:
        # Select appropriate model based on language preference
        if language == "Pashto":
            model_name = os.getenv("MISTRAL_EDU_MODEL", "tasal9/ZamAI-Mistral-7B-Pashto")
            system_prompt = "تاسو د ښوونې او روزنې ملګری یاست. د زده کوونکو سره په پښتو کې مرسته وکړئ."
        else:
            model_name = os.getenv("MISTRAL_EDU_MODEL", "tasal9/ZamAI-Mistral-7B-Pashto")
            system_prompt = "You are an educational tutor. Help students learn in a clear and engaging way."
        
        # Construct the prompt
        prompt = f"System: {system_prompt}\n\nStudent: {message}\n\nTutor:"
        
        # Generate response
        response = client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        return response.strip()
        
    except Exception as e:
        return f"Error: {str(e)}. Please check your Hugging Face token and model access."

# Create Gradio interface
def create_demo():
    with gr.Blocks(
        title="ZamAI Educational Tutor",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 30px; }
        .model-info { background: #f0f0f0; padding: 15px; border-radius: 10px; margin: 10px 0; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎓 ZamAI Educational Tutor</h1>
            <p>AI-powered educational assistant with Pashto language support</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="model-info">
                    <h3>📚 Model Information</h3>
                    <p><strong>Model:</strong> ZamAI-Mistral-7B-Pashto</p>
                    <p><strong>Languages:</strong> English, Pashto</p>
                    <p><strong>Specialization:</strong> Educational content</p>
                </div>
                """)
                
                language_selector = gr.Radio(
                    choices=["English", "Pashto"],
                    value="English",
                    label="🌐 Language Preference"
                )
                
                gr.HTML("""
                <div style="margin-top: 20px;">
                    <h4>📝 Sample Questions:</h4>
                    <ul>
                        <li>Explain quantum physics in simple terms</li>
                        <li>What is machine learning?</li>
                        <li>د فزیک د کوانټم نظریه تشریح کړئ (Pashto)</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=400,
                    placeholder="Ask me anything about education, science, or learning! 📚"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        label="Your Question",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 📤", scale=1, variant="primary")
                
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
        
        # Chat functionality
        def respond(message, chat_history, language):
            if not message.strip():
                return "", chat_history
            
            bot_message = educational_chat(message, chat_history, language)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        def clear_chat():
            return []
        
        # Event handlers
        send_btn.click(
            respond,
            inputs=[msg, chatbot, language_selector],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, language_selector],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(clear_chat, outputs=[chatbot])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p>Powered by <strong>ZamAI Pro Models Strategy</strong> | Built with 🤗 Hugging Face</p>
            <p>Model: <a href="https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto" target="_blank">tasal9/ZamAI-Mistral-7B-Pashto</a></p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
