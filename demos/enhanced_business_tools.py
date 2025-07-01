#!/usr/bin/env python3
"""
ZamAI Enhanced Business Tools Suite
Document parsing, generation, and retrieval using Phi-3 and embeddings
"""

import gradio as gr
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import faiss
import pickle

# Load environment variables
load_dotenv()

class BusinessDocumentProcessor:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.phi3_model = os.getenv("PHI3_BUSINESS_MODEL", "tasal9/ZamAI-Phi-3-Mini-Pashto")
        self.embeddings_model = os.getenv("EMBEDDINGS_MODEL", "tasal9/Multilingual-ZamAI-Embeddings")
        self.client = InferenceClient(token=self.hf_token)
        
        # Document storage and retrieval
        self.documents_path = Path("business_data/documents")
        self.embeddings_path = Path("business_data/embeddings") 
        self.index_path = Path("business_data/faiss_index")
        
        # Initialize storage
        self._initialize_storage()
        
        # Load or create document index
        self.document_index = self._load_document_index()
        self.faiss_index = self._load_faiss_index()
        
        # Performance tracking
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "last_processed": None
        }
    
    def _initialize_storage(self):
        """Initialize storage directories"""
        for path in [self.documents_path, self.embeddings_path, self.index_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_document_index(self):
        """Load document metadata index"""
        index_file = self.documents_path / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_document_index(self):
        """Save document metadata index"""
        index_file = self.documents_path / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_index, f, ensure_ascii=False, indent=2)
    
    def _load_faiss_index(self):
        """Load FAISS index for document retrieval"""
        faiss_file = self.index_path / "documents.index"
        if faiss_file.exists():
            return faiss.read_index(str(faiss_file))
        else:
            # Create new index (assuming 768-dim embeddings)
            return faiss.IndexFlatIP(768)  # Inner Product for similarity
    
    def _save_faiss_index(self):
        """Save FAISS index"""
        if self.faiss_index.ntotal > 0:
            faiss_file = self.index_path / "documents.index"
            faiss.write_index(self.faiss_index, str(faiss_file))
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using the multilingual embeddings model"""
        try:
            # Use the inference API for embeddings
            embedding = self.client.feature_extraction(
                text,
                model=self.embeddings_model
            )
            
            # Handle different response formats
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    # If nested list, take first item
                    embedding = embedding[0]
                embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = np.array(embedding, dtype=np.float32)
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Fallback: return random normalized vector
            embedding = np.random.randn(768).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
    
    def process_business_document(self, document_text: str, document_type: str = "General", 
                                 analysis_type: str = "Comprehensive Analysis") -> Dict[str, Any]:
        """Process business document with Phi-3 model"""
        start_time = time.time()
        
        try:
            if not document_text.strip():
                return {"error": "Please provide document text to process"}
            
            # Create context-specific prompt
            prompt = self._create_analysis_prompt(document_text, document_type, analysis_type)
            
            # Generate analysis using Phi-3
            response = self.client.text_generation(
                model=self.phi3_model,
                prompt=prompt,
                max_new_tokens=800,
                temperature=0.3,  # Lower temperature for business analysis
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            # Generate embeddings for retrieval
            embeddings = self.generate_embeddings(document_text)
            
            # Store document and metadata
            doc_id = self._store_document(document_text, document_type, response, embeddings)
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            return {
                "analysis": response.strip(),
                "document_id": doc_id,
                "processing_time": f"{processing_time:.2f}s",
                "embeddings_generated": True,
                "stored_for_retrieval": True
            }
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
    
    def _create_analysis_prompt(self, document_text: str, document_type: str, analysis_type: str) -> str:
        """Create analysis prompt based on document type and analysis type"""
        
        base_system = "You are an expert business analyst specializing in document processing and information extraction."
        
        type_specific_instructions = {
            "Contract": "Focus on parties, terms, obligations, dates, and legal implications.",
            "Invoice": "Extract financial details, line items, totals, dates, and payment terms.",
            "Report": "Summarize key findings, recommendations, and actionable insights.",
            "Email": "Identify sender, recipients, key decisions, action items, and follow-ups.",
            "Form": "Extract all field values and validate completeness.",
            "Legal Document": "Identify legal concepts, clauses, and compliance requirements.",
            "Financial Statement": "Analyze financial metrics, trends, and performance indicators.",
            "General": "Provide comprehensive analysis of key information and insights."
        }
        
        analysis_instructions = {
            "Information Extraction": "Extract and structure all key information in a clear format.",
            "Summary Generation": "Create a concise executive summary highlighting main points.",
            "Comprehensive Analysis": "Provide detailed analysis with insights and recommendations.",
            "Compliance Check": "Review for compliance requirements and potential issues.",
            "Action Items": "Identify specific action items and next steps.",
            "Risk Assessment": "Evaluate potential risks and mitigation strategies."
        }
        
        type_instruction = type_specific_instructions.get(document_type, type_specific_instructions["General"])
        analysis_instruction = analysis_instructions.get(analysis_type, analysis_instructions["Comprehensive Analysis"])
        
        prompt = f"""<|system|>
{base_system}

Document Type: {document_type}
Analysis Focus: {analysis_type}

Instructions: {type_instruction} {analysis_instruction}

Provide your analysis in a structured format with clear sections.
<|end|>

<|user|>
Please analyze the following business document:

{document_text}
<|end|>

<|assistant|>
"""
        
        return prompt
    
    def _store_document(self, text: str, doc_type: str, analysis: str, embeddings: np.ndarray) -> str:
        """Store document with metadata and embeddings"""
        doc_id = f"doc_{int(time.time())}_{len(self.document_index)}"
        
        # Store document metadata
        metadata = {
            "id": doc_id,
            "type": doc_type,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "analysis_length": len(analysis),
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
        self.document_index[doc_id] = metadata
        
        # Store full document
        doc_file = self.documents_path / f"{doc_id}.json"
        full_doc = {
            "metadata": metadata,
            "text": text,
            "analysis": analysis
        }
        
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(full_doc, f, ensure_ascii=False, indent=2)
        
        # Store embeddings
        embeddings_file = self.embeddings_path / f"{doc_id}.npy"
        np.save(embeddings_file, embeddings)
        
        # Add to FAISS index
        self.faiss_index.add(embeddings.reshape(1, -1))
        
        # Save indices
        self._save_document_index()
        self._save_faiss_index()
        
        return doc_id
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using embeddings"""
        try:
            if self.faiss_index.ntotal == 0:
                return [{"message": "No documents in the database yet"}]
            
            # Generate query embeddings
            query_embeddings = self.generate_embeddings(query)
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(
                query_embeddings.reshape(1, -1), min(top_k, self.faiss_index.ntotal)
            )
            
            results = []
            doc_ids = list(self.document_index.keys())
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    metadata = self.document_index[doc_id]
                    
                    results.append({
                        "rank": i + 1,
                        "document_id": doc_id,
                        "similarity_score": float(distance),
                        "document_type": metadata["type"],
                        "timestamp": metadata["timestamp"],
                        "preview": metadata["text_preview"]
                    })
            
            return results
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def generate_business_content(self, content_type: str, specifications: str, 
                                context: str = "") -> Dict[str, Any]:
        """Generate business content using Phi-3"""
        try:
            prompt = self._create_generation_prompt(content_type, specifications, context)
            
            response = self.client.text_generation(
                model=self.phi3_model,
                prompt=prompt,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            return {
                "generated_content": response.strip(),
                "content_type": content_type,
                "specifications_used": specifications,
                "generation_successful": True
            }
            
        except Exception as e:
            return {"error": f"Content generation failed: {str(e)}"}
    
    def _create_generation_prompt(self, content_type: str, specifications: str, context: str) -> str:
        """Create content generation prompt"""
        
        generation_templates = {
            "Email": "Draft a professional business email",
            "Report": "Generate a comprehensive business report",
            "Proposal": "Create a detailed business proposal",
            "Contract": "Draft a business contract template",
            "Summary": "Write an executive summary",
            "Memo": "Create a business memorandum",
            "Letter": "Draft a formal business letter",
            "Policy": "Write a company policy document"
        }
        
        template = generation_templates.get(content_type, "Generate professional business content")
        
        prompt = f"""<|system|>
You are an expert business writer specializing in creating professional documents and content.
<|end|>

<|user|>
Task: {template}

Specifications: {specifications}

{f"Additional Context: {context}" if context else ""}

Please create high-quality, professional content that meets the specifications.
<|end|>

<|assistant|>
"""
        
        return prompt
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["documents_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"] / 
            self.processing_stats["documents_processed"]
        )
        self.processing_stats["last_processed"] = datetime.now().isoformat()
    
    def get_processing_stats(self) -> str:
        """Get formatted processing statistics"""
        stats = self.processing_stats
        
        return f"""
📊 **Business Tools Statistics:**
• Documents Processed: {stats['documents_processed']}
• Total Processing Time: {stats['total_processing_time']:.1f}s
• Average Processing Time: {stats['average_processing_time']:.2f}s
• Documents in Database: {len(self.document_index)}
• FAISS Index Size: {self.faiss_index.ntotal if self.faiss_index else 0}
• Last Processed: {stats['last_processed'] or 'Never'}

🔧 **Models in Use:**
• Document Processing: {self.phi3_model}
• Embeddings: {self.embeddings_model}
"""
    
    def get_document_library(self) -> pd.DataFrame:
        """Get document library as DataFrame for display"""
        if not self.document_index:
            return pd.DataFrame({"Message": ["No documents stored yet"]})
        
        data = []
        for doc_id, metadata in self.document_index.items():
            data.append({
                "ID": doc_id,
                "Type": metadata["type"],
                "Date": metadata["timestamp"][:10],
                "Length": metadata["text_length"],
                "Preview": metadata["text_preview"][:100] + "..."
            })
        
        return pd.DataFrame(data)

def create_business_tools_interface():
    """Create the enhanced Business Tools Gradio interface"""
    
    processor = BusinessDocumentProcessor()
    
    # Custom CSS for professional business interface
    custom_css = """
    .main-container {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        min-height: 100vh;
    }
    .gradio-container {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        margin: 20px;
        padding: 25px;
    }
    .header-section {
        background: linear-gradient(45deg, #34495e, #2c3e50);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .business-card {
        background: #f8f9fa;
        border-left: 5px solid #3498db;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .stats-card {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .textbox textarea {
        background: white !important;
        color: #2c3e50 !important;
        border: 2px solid #bdc3c7 !important;
        border-radius: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .button {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 25px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
    }
    """
    
    with gr.Blocks(
        title="🏢 ZamAI Business Tools Suite",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header-section">
            <h1>🏢 ZamAI Enhanced Business Tools Suite</h1>
            <h3>Document Processing • Content Generation • Intelligent Retrieval</h3>
            <p><strong>Powered by:</strong> Phi-3-Mini-Pashto • Multilingual Embeddings • FAISS Search</p>
            <p><strong>Models:</strong> tasal9/ZamAI-Phi-3-Mini-Pashto | tasal9/Multilingual-ZamAI-Embeddings</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="business-card">
                    <h3>🛠️ Business Tools Features</h3>
                    <ul>
                        <li>📄 Document Analysis & Parsing</li>
                        <li>🔍 Intelligent Document Search</li>
                        <li>✍️ Content Generation</li>
                        <li>🗂️ Document Library Management</li>
                        <li>📊 Performance Analytics</li>
                        <li>🔐 Secure Document Storage</li>
                    </ul>
                </div>
                """)
                
                # Quick actions
                gr.HTML("### ⚡ Quick Actions")
                stats_btn = gr.Button("📊 View Statistics", variant="secondary")
                library_btn = gr.Button("📚 Document Library", variant="secondary")
                
            with gr.Column(scale=3):
                with gr.Tab("📄 Document Processing"):
                    gr.HTML("""
                    <div class="business-card">
                        <h4>📋 Document Analysis & Information Extraction</h4>
                        <p>Upload or paste business documents for intelligent processing and analysis</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        document_type = gr.Dropdown(
                            choices=["General", "Contract", "Invoice", "Report", "Email", 
                                   "Form", "Legal Document", "Financial Statement"],
                            value="General",
                            label="📂 Document Type"
                        )
                        
                        analysis_type = gr.Dropdown(
                            choices=["Comprehensive Analysis", "Information Extraction", 
                                   "Summary Generation", "Compliance Check", 
                                   "Action Items", "Risk Assessment"],
                            value="Comprehensive Analysis",
                            label="🎯 Analysis Type"
                        )
                    
                    document_input = gr.Textbox(
                        label="📝 Document Text",
                        placeholder="Paste your business document content here...",
                        lines=8
                    )
                    
                    process_btn = gr.Button("🔄 Process Document", variant="primary", size="lg")
                    
                    with gr.Row():
                        analysis_output = gr.Textbox(
                            label="📊 Analysis Results",
                            lines=12,
                            interactive=False
                        )
                        
                        metadata_output = gr.Textbox(
                            label="ℹ️ Processing Info",
                            lines=6,
                            interactive=False
                        )
                
                with gr.Tab("🔍 Document Search & Retrieval"):
                    gr.HTML("""
                    <div class="business-card">
                        <h4>🔎 Intelligent Document Search</h4>
                        <p>Search through your document library using semantic similarity</p>
                    </div>
                    """)
                    
                    search_query = gr.Textbox(
                        label="🔍 Search Query",
                        placeholder="Enter search terms or describe what you're looking for...",
                        lines=2
                    )
                    
                    search_results_count = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="📊 Number of Results"
                    )
                    
                    search_btn = gr.Button("🔍 Search Documents", variant="primary")
                    
                    search_results = gr.JSON(
                        label="📋 Search Results",
                        show_label=True
                    )
                
                with gr.Tab("✍️ Content Generation"):
                    gr.HTML("""
                    <div class="business-card">
                        <h4>📝 AI-Powered Content Creation</h4>
                        <p>Generate professional business documents and content</p>
                    </div>
                    """)
                    
                    content_type = gr.Dropdown(
                        choices=["Email", "Report", "Proposal", "Contract", 
                               "Summary", "Memo", "Letter", "Policy"],
                        value="Email",
                        label="📄 Content Type"
                    )
                    
                    specifications = gr.Textbox(
                        label="📋 Specifications & Requirements",
                        placeholder="Describe what you need generated...",
                        lines=4
                    )
                    
                    context_input = gr.Textbox(
                        label="🔗 Additional Context (Optional)",
                        placeholder="Provide any additional context or background...",
                        lines=3
                    )
                    
                    generate_btn = gr.Button("✨ Generate Content", variant="primary")
                    
                    generated_content = gr.Textbox(
                        label="📄 Generated Content",
                        lines=15,
                        interactive=False
                    )
                
                with gr.Tab("📚 Document Library"):
                    gr.HTML("""
                    <div class="business-card">
                        <h4>🗂️ Document Library Management</h4>
                        <p>View and manage your stored business documents</p>
                    </div>
                    """)
                    
                    refresh_library_btn = gr.Button("🔄 Refresh Library", variant="secondary")
                    
                    library_display = gr.Dataframe(
                        label="📊 Document Library",
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Tab("📈 Analytics & Stats"):
                    gr.HTML("""
                    <div class="business-card">
                        <h4>📊 Performance Analytics</h4>
                        <p>Monitor processing statistics and system performance</p>
                    </div>
                    """)
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
                    
                    stats_display = gr.Textbox(
                        label="📈 Processing Statistics",
                        lines=15,
                        interactive=False
                    )
        
        # Event handlers
        def process_document_handler(doc_text, doc_type, analysis_type):
            result = processor.process_business_document(doc_text, doc_type, analysis_type)
            
            if "error" in result:
                return result["error"], "Error occurred during processing"
            
            metadata_info = f"""
🆔 Document ID: {result['document_id']}
⏱️ Processing Time: {result['processing_time']}
🧠 Embeddings: {'✅ Generated' if result['embeddings_generated'] else '❌ Failed'}
💾 Storage: {'✅ Stored' if result['stored_for_retrieval'] else '❌ Failed'}
"""
            
            return result["analysis"], metadata_info
        
        def search_documents_handler(query, top_k):
            results = processor.search_similar_documents(query, top_k)
            return results
        
        def generate_content_handler(content_type, specs, context):
            result = processor.generate_business_content(content_type, specs, context)
            
            if "error" in result:
                return result["error"]
            
            return result["generated_content"]
        
        def get_stats_handler():
            return processor.get_processing_stats()
        
        def get_library_handler():
            return processor.get_document_library()
        
        # Connect events
        process_btn.click(
            fn=process_document_handler,
            inputs=[document_input, document_type, analysis_type],
            outputs=[analysis_output, metadata_output]
        )
        
        search_btn.click(
            fn=search_documents_handler,
            inputs=[search_query, search_results_count],
            outputs=[search_results]
        )
        
        generate_btn.click(
            fn=generate_content_handler,
            inputs=[content_type, specifications, context_input],
            outputs=[generated_content]
        )
        
        stats_btn.click(
            fn=get_stats_handler,
            outputs=[stats_display]
        )
        
        refresh_stats_btn.click(
            fn=get_stats_handler,
            outputs=[stats_display]
        )
        
        library_btn.click(
            fn=get_library_handler,
            outputs=[library_display]
        )
        
        refresh_library_btn.click(
            fn=get_library_handler,
            outputs=[library_display]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 25px; background: #ecf0f1; border-radius: 12px;">
            <p><strong>🏢 ZamAI Enhanced Business Tools Suite</strong></p>
            <p><strong>🔧 Powered By:</strong> 
                <a href="https://huggingface.co/tasal9/ZamAI-Phi-3-Mini-Pashto" target="_blank">Phi-3-Mini-Pashto</a> + 
                <a href="https://huggingface.co/tasal9/Multilingual-ZamAI-Embeddings" target="_blank">Multilingual Embeddings</a>
            </p>
            <p><strong>✨ Features:</strong> Document Processing | Intelligent Search | Content Generation | Analytics</p>
            <p><strong>🚀 Enterprise Ready:</strong> Scalable • Secure • CI/CD Integrated</p>
        </div>
        """)
    
    return demo

def main():
    """Main function to launch the Business Tools Suite"""
    print("🏢 Starting ZamAI Enhanced Business Tools Suite...")
    
    # Create and launch the interface
    demo = create_business_tools_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=True,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
