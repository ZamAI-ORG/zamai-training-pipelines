"""
ZamAI Business Document Processor Demo
Gradio interface for automated document processing and form extraction
"""

import gradio as gr
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize HF Inference Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def process_document(document_text, processing_type="Information Extraction"):
    """
    Process business documents using Phi-3 model
    """
    try:
        if not document_text.strip():
            return "Please provide document text to process.", ""
        
        model_name = os.getenv("PHI3_BUSINESS_MODEL", "tasal9/ZamAI-Phi-3-Mini-Pashto")
        
        # Different prompts based on processing type
        if processing_type == "Information Extraction":
            prompt = f"""Extract key information from this business document. Provide a structured summary including:
- Document type
- Key parties involved
- Important dates
- Financial information
- Key terms and conditions

Document: {document_text}

Extracted Information:"""
        
        elif processing_type == "Form Analysis":
            prompt = f"""Analyze this form and extract all field values in a structured format:

Form Content: {document_text}

Field Analysis:"""
        
        elif processing_type == "Contract Summary":
            prompt = f"""Summarize this contract/agreement highlighting:
- Parties involved
- Main obligations
- Key terms
- Important clauses
- Risks and considerations

Contract: {document_text}

Summary:"""
        
        else:  # Document Classification
            prompt = f"""Classify this document and provide analysis:

Document: {document_text}

Classification and Analysis:"""
        
        response = client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=400,
            temperature=0.1,  # Low temperature for more consistent extraction
            do_sample=True,
            top_p=0.9
        )
        
        return response.strip(), "✅ Document processed successfully"
        
    except Exception as e:
        return f"Error processing document: {str(e)}", "❌ Processing failed"

def generate_document_insights(extracted_info):
    """
    Generate additional insights from extracted information
    """
    try:
        if not extracted_info.strip():
            return "No information to analyze."
        
        model_name = os.getenv("PHI3_BUSINESS_MODEL", "tasal9/ZamAI-Phi-3-Mini-Pashto")
        
        prompt = f"""Based on this extracted information, provide business insights and recommendations:

Extracted Information: {extracted_info}

Business Insights and Recommendations:"""
        
        insights = client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.3
        )
        
        return insights.strip()
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Sample documents for testing
SAMPLE_DOCUMENTS = {
    "Sample Contract": """
    SERVICE AGREEMENT
    
    This Service Agreement is entered into on January 15, 2024, between:
    
    Client: ABC Corporation, located at 123 Business St, City, State 12345
    Service Provider: ZamAI Solutions, located at 456 Tech Ave, Innovation City, TC 67890
    
    Terms:
    - Service Period: 12 months starting February 1, 2024
    - Monthly Fee: $5,000
    - Payment Terms: Net 30 days
    - Services: AI model development and deployment
    - Termination: 30 days written notice required
    
    The Service Provider agrees to deliver custom AI solutions including model training, deployment, and maintenance.
    """,
    
    "Sample Invoice": """
    INVOICE #2024-001
    
    From: ZamAI Solutions
    456 Tech Ave, Innovation City, TC 67890
    
    To: ABC Corporation
    123 Business St, City, State 12345
    
    Date: March 1, 2024
    Due Date: March 31, 2024
    
    Services:
    - AI Model Development: $3,000
    - Deployment Services: $1,500
    - Support & Maintenance: $500
    
    Subtotal: $5,000
    Tax (8%): $400
    Total: $5,400
    
    Payment Terms: Net 30 days
    """,
    
    "Sample Form": """
    EMPLOYEE REGISTRATION FORM
    
    Employee ID: EMP-2024-0123
    Full Name: John Smith
    Department: Engineering
    Position: Senior AI Engineer
    Start Date: 2024-02-15
    Salary: $85,000
    Manager: Jane Doe
    Email: john.smith@company.com
    Phone: (555) 123-4567
    Emergency Contact: Mary Smith (555) 987-6543
    Benefits Enrolled: Health, Dental, 401k
    """
}

# Create Gradio interface
def create_business_demo():
    with gr.Blocks(
        title="ZamAI Business Document Processor",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 30px; }
        .model-info { background: #fff3cd; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .document-section { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>📄 ZamAI Business Document Processor</h1>
            <p>AI-powered document analysis and information extraction</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="model-info">
                    <h3>🤖 Model Information</h3>
                    <p><strong>Model:</strong> ZamAI-Phi-3-Mini-Pashto</p>
                    <p><strong>Specialization:</strong> Business documents</p>
                    <p><strong>Capabilities:</strong> Information extraction, Form analysis, Contract summarization</p>
                </div>
                """)
                
                processing_type = gr.Radio(
                    choices=[
                        "Information Extraction",
                        "Form Analysis", 
                        "Contract Summary",
                        "Document Classification"
                    ],
                    value="Information Extraction",
                    label="🔧 Processing Type"
                )
                
                sample_selector = gr.Dropdown(
                    choices=list(SAMPLE_DOCUMENTS.keys()) + ["Custom Document"],
                    value="Custom Document",
                    label="📋 Sample Documents"
                )
                
                gr.HTML("""
                <div style="margin-top: 20px;">
                    <h4>💡 Use Cases:</h4>
                    <ul>
                        <li>Contract analysis and summarization</li>
                        <li>Invoice and form data extraction</li>
                        <li>Document classification</li>
                        <li>Key information identification</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=2):
                with gr.Tab("📝 Document Input"):
                    gr.HTML('<div class="document-section">')
                    
                    document_input = gr.Textbox(
                        label="Document Text",
                        lines=10,
                        placeholder="Paste your document content here or select a sample document...",
                        max_lines=15
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("Process Document 🔄", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear 🗑️", variant="secondary")
                    
                    gr.HTML('</div>')
                
                with gr.Tab("📊 Results"):
                    status_output = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        lines=1
                    )
                    
                    results_output = gr.Textbox(
                        label="Extracted Information",
                        lines=12,
                        interactive=False
                    )
                    
                    insights_btn = gr.Button("Generate Business Insights 💡", variant="primary")
                    
                    insights_output = gr.Textbox(
                        label="Business Insights & Recommendations",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Tab("📈 Analytics"):
                    gr.HTML("""
                    <div style="padding: 20px;">
                        <h3>Document Processing Analytics</h3>
                        <p>This section would show:</p>
                        <ul>
                            <li>Processing time and efficiency</li>
                            <li>Confidence scores for extractions</li>
                            <li>Document type classification accuracy</li>
                            <li>Historical processing statistics</li>
                        </ul>
                        <p><em>Analytics features would be implemented with usage tracking.</em></p>
                    </div>
                    """)
        
        # Event handlers
        def load_sample_document(sample_name):
            if sample_name in SAMPLE_DOCUMENTS:
                return SAMPLE_DOCUMENTS[sample_name]
            return ""
        
        def process_document_interface(doc_text, proc_type):
            if not doc_text.strip():
                return "❌ No document provided", "Please provide document text to process."
            
            result, status = process_document(doc_text, proc_type)
            return status, result
        
        def clear_all():
            return "", "", "", ""
        
        # Connect events
        sample_selector.change(
            load_sample_document,
            inputs=[sample_selector],
            outputs=[document_input]
        )
        
        process_btn.click(
            process_document_interface,
            inputs=[document_input, processing_type],
            outputs=[status_output, results_output]
        )
        
        insights_btn.click(
            generate_document_insights,
            inputs=[results_output],
            outputs=[insights_output]
        )
        
        clear_btn.click(
            clear_all,
            outputs=[document_input, status_output, results_output, insights_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p>Powered by <strong>ZamAI Pro Models Strategy</strong></p>
            <p>Model: <a href="https://huggingface.co/tasal9/ZamAI-Phi-3-Mini-Pashto" target="_blank">tasal9/ZamAI-Phi-3-Mini-Pashto</a></p>
            <p>🔒 Secure • ⚡ Fast • 🎯 Accurate</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_business_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )
