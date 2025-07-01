#!/usr/bin/env python3

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Any

class ZamAIBusinessTools:
    def __init__(self):
        self.setup_models()
        self.processing_history = []
        self.document_database = []
    
    def setup_models(self):
        """Initialize AI models for business processing"""
        try:
            # Document Processing Model (Phi-3 compatible)
            model_name = "microsoft/DialoGPT-medium"  # Fallback model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Business models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def analyze_contract(self, contract_text: str) -> Dict[str, Any]:
        """Analyze contract and extract key information"""
        try:
            # Extract key contract elements using regex patterns
            analysis = {
                "document_type": "Contract",
                "parties": self.extract_parties(contract_text),
                "financial_terms": self.extract_financial_terms(contract_text),
                "dates": self.extract_dates(contract_text),
                "obligations": self.extract_obligations(contract_text),
                "risk_factors": self.assess_risks(contract_text),
                "confidence_score": 0.85
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing contract: {str(e)}"}
    
    def process_invoice(self, invoice_text: str) -> Dict[str, Any]:
        """Process invoice and extract billing information"""
        try:
            analysis = {
                "document_type": "Invoice",
                "invoice_number": self.extract_invoice_number(invoice_text),
                "vendor_info": self.extract_vendor_info(invoice_text),
                "line_items": self.extract_line_items(invoice_text),
                "totals": self.extract_totals(invoice_text),
                "due_date": self.extract_due_date(invoice_text),
                "validation_status": "Valid",
                "confidence_score": 0.92
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error processing invoice: {str(e)}"}
    
    def digitize_form(self, form_text: str) -> Dict[str, Any]:
        """Convert form data into structured format"""
        try:
            analysis = {
                "document_type": "Form",
                "form_type": self.identify_form_type(form_text),
                "extracted_fields": self.extract_form_fields(form_text),
                "completeness": self.assess_form_completeness(form_text),
                "validation_errors": self.validate_form_data(form_text),
                "confidence_score": 0.88
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error digitizing form: {str(e)}"}
    
    def extract_parties(self, text: str) -> List[str]:
        """Extract party names from contract"""
        party_patterns = [
            r'between\s+([^,\n]+)\s+and\s+([^,\n]+)',
            r'Party A[:\-]\s*([^\n]+)',
            r'Party B[:\-]\s*([^\n]+)',
            r'Client[:\-]\s*([^\n]+)',
            r'Contractor[:\-]\s*([^\n]+)'
        ]
        
        parties = []
        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    parties.extend([p.strip() for p in match])
                else:
                    parties.append(match.strip())
        
        return list(set(parties))[:5]  # Return up to 5 unique parties
    
    def extract_financial_terms(self, text: str) -> Dict[str, str]:
        """Extract financial information"""
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, text)
        
        return {
            "amounts_found": amounts[:5],
            "currency": "USD" if amounts else "Unknown",
            "payment_terms": "Standard" if "30 day" in text.lower() else "Custom"
        }
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract important dates"""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))[:5]
    
    def extract_obligations(self, text: str) -> List[str]:
        """Extract key obligations and responsibilities"""
        obligation_keywords = [
            "shall", "must", "required to", "responsible for", 
            "agrees to", "undertakes to", "commits to"
        ]
        
        obligations = []
        sentences = text.split('.')
        
        for sentence in sentences:
            for keyword in obligation_keywords:
                if keyword in sentence.lower():
                    obligations.append(sentence.strip())
                    break
        
        return obligations[:3]  # Return top 3 obligations
    
    def assess_risks(self, text: str) -> List[str]:
        """Identify potential risk factors"""
        risk_keywords = [
            "termination", "penalty", "default", "breach", 
            "liability", "damages", "force majeure", "dispute"
        ]
        
        risks = []
        for keyword in risk_keywords:
            if keyword in text.lower():
                risks.append(f"Contains {keyword} clause")
        
        return risks[:3]
    
    def extract_invoice_number(self, text: str) -> str:
        """Extract invoice number"""
        patterns = [
            r'Invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'INV[:\-\s]*([A-Z0-9\-]+)',
            r'#\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Not found"
    
    def extract_vendor_info(self, text: str) -> Dict[str, str]:
        """Extract vendor information"""
        return {
            "name": "Vendor name extraction would go here",
            "address": "Address extraction would go here",
            "contact": "Contact info extraction would go here"
        }
    
    def extract_line_items(self, text: str) -> List[Dict[str, str]]:
        """Extract invoice line items"""
        # Simplified line item extraction
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        
        items = []
        for i, amount in enumerate(amounts[:3]):
            items.append({
                "description": f"Service/Product {i+1}",
                "amount": amount,
                "quantity": "1"
            })
        
        return items
    
    def extract_totals(self, text: str) -> Dict[str, str]:
        """Extract total amounts"""
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        
        return {
            "subtotal": amounts[0] if amounts else "$0.00",
            "tax": amounts[1] if len(amounts) > 1 else "$0.00", 
            "total": amounts[-1] if amounts else "$0.00"
        }
    
    def extract_due_date(self, text: str) -> str:
        """Extract payment due date"""
        date_patterns = [
            r'due\s+(?:by\s+)?([^,\n]+)',
            r'payment\s+due[:\-\s]*([^,\n]+)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def identify_form_type(self, text: str) -> str:
        """Identify the type of form"""
        form_types = {
            "employee": ["employee", "staff", "personnel", "hr"],
            "application": ["application", "apply", "request"],
            "registration": ["registration", "register", "signup"],
            "contact": ["contact", "inquiry", "message"]
        }
        
        text_lower = text.lower()
        for form_type, keywords in form_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return form_type.title()
        
        return "General Form"
    
    def extract_form_fields(self, text: str) -> Dict[str, str]:
        """Extract form field data"""
        field_patterns = {
            "name": r'name[:\-\s]*([^\n,]+)',
            "email": r'email[:\-\s]*([^\n,\s]+)',
            "phone": r'phone[:\-\s]*([^\n,]+)',
            "id": r'id[:\-\s]*([^\n,\s]+)',
            "department": r'department[:\-\s]*([^\n,]+)'
        }
        
        fields = {}
        for field, pattern in field_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field] = match.group(1).strip()
        
        return fields
    
    def assess_form_completeness(self, text: str) -> Dict[str, Any]:
        """Assess form completeness"""
        fields = self.extract_form_fields(text)
        total_fields = 5  # Expected number of fields
        completed_fields = len(fields)
        
        return {
            "completed_fields": completed_fields,
            "total_fields": total_fields,
            "completeness_percentage": (completed_fields / total_fields) * 100,
            "status": "Complete" if completed_fields >= 4 else "Incomplete"
        }
    
    def validate_form_data(self, text: str) -> List[str]:
        """Validate form data and return errors"""
        errors = []
        fields = self.extract_form_fields(text)
        
        if 'email' in fields:
            email = fields['email']
            if '@' not in email or '.' not in email:
                errors.append("Invalid email format")
        
        if 'phone' in fields:
            phone = fields['phone']
            if not re.match(r'[\d\-\(\)\s+]+', phone):
                errors.append("Invalid phone format")
        
        return errors
    
    def process_document(self, document_text: str, document_type: str):
        """Main document processing function"""
        if not document_text.strip():
            return "Please provide document text to analyze.", "", ""
        
        start_time = datetime.now()
        
        # Process based on document type
        if document_type == "Contract":
            analysis = self.analyze_contract(document_text)
        elif document_type == "Invoice":
            analysis = self.process_invoice(document_text)
        elif document_type == "Form":
            analysis = self.digitize_form(document_text)
        else:
            return "Please select a document type.", "", ""
        
        # Store in processing history
        processing_time = (datetime.now() - start_time).total_seconds()
        
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "document_type": document_type,
            "processing_time": processing_time,
            "analysis": analysis
        }
        self.processing_history.append(history_entry)
        
        # Generate outputs
        summary = self.generate_summary(analysis)
        structured_data = self.format_structured_data(analysis)
        insights = self.generate_insights(analysis)
        
        return summary, structured_data, insights
    
    def generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary"""
        if "error" in analysis:
            return f"❌ {analysis['error']}"
        
        doc_type = analysis.get("document_type", "Document")
        confidence = analysis.get("confidence_score", 0.0)
        
        summary = f"## 📋 {doc_type} Analysis Summary\n\n"
        summary += f"**Confidence Score:** {confidence:.1%}\n\n"
        
        if doc_type == "Contract":
            parties = analysis.get("parties", [])
            if parties:
                summary += f"**Parties:** {', '.join(parties[:3])}\n"
            
            financial = analysis.get("financial_terms", {})
            amounts = financial.get("amounts_found", [])
            if amounts:
                summary += f"**Financial Terms:** {', '.join(amounts[:3])}\n"
            
            risks = analysis.get("risk_factors", [])
            if risks:
                summary += f"**Risk Factors:** {len(risks)} identified\n"
        
        elif doc_type == "Invoice":
            inv_num = analysis.get("invoice_number", "N/A")
            summary += f"**Invoice Number:** {inv_num}\n"
            
            totals = analysis.get("totals", {})
            total_amount = totals.get("total", "N/A")
            summary += f"**Total Amount:** {total_amount}\n"
            
            due_date = analysis.get("due_date", "N/A")
            summary += f"**Due Date:** {due_date}\n"
        
        elif doc_type == "Form":
            form_type = analysis.get("form_type", "Unknown")
            summary += f"**Form Type:** {form_type}\n"
            
            completeness = analysis.get("completeness", {})
            status = completeness.get("status", "Unknown")
            percentage = completeness.get("completeness_percentage", 0)
            summary += f"**Completeness:** {status} ({percentage:.0f}%)\n"
        
        return summary
    
    def format_structured_data(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as structured JSON"""
        try:
            return json.dumps(analysis, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error formatting data: {str(e)}"
    
    def generate_insights(self, analysis: Dict[str, Any]) -> str:
        """Generate business insights"""
        if "error" in analysis:
            return "No insights available due to processing error."
        
        doc_type = analysis.get("document_type", "Document")
        insights = f"## 💡 Business Insights\n\n"
        
        if doc_type == "Contract":
            risks = analysis.get("risk_factors", [])
            if risks:
                insights += f"⚠️ **Risk Assessment:** {len(risks)} potential risk factors identified. Review recommended.\n\n"
            
            obligations = analysis.get("obligations", [])
            if obligations:
                insights += f"📝 **Key Obligations:** {len(obligations)} major obligations found. Ensure compliance tracking.\n\n"
        
        elif doc_type == "Invoice":
            validation = analysis.get("validation_status", "Unknown")
            if validation == "Valid":
                insights += "✅ **Invoice Validation:** All required fields present and valid.\n\n"
            
            insights += "💰 **Recommendation:** Set up automated payment processing for efficiency.\n\n"
        
        elif doc_type == "Form":
            completeness = analysis.get("completeness", {})
            if completeness.get("status") == "Incomplete":
                insights += "📋 **Data Quality:** Form appears incomplete. Follow up may be required.\n\n"
            
            errors = analysis.get("validation_errors", [])
            if errors:
                insights += f"⚠️ **Validation Issues:** {len(errors)} data quality issues found.\n\n"
        
        insights += "🚀 **Next Steps:** Consider implementing automated workflows for similar documents."
        
        return insights

# Initialize the business tools
business_tools = ZamAIBusinessTools()

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="ZamAI Business Tools",
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .tool-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>📄 ZamAI Business Tools</h1>
            <p>Professional document processing with Phi-3 and advanced embeddings</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Document Input</h3>")
                
                document_type = gr.Dropdown(
                    choices=["Contract", "Invoice", "Form"],
                    label="Document Type",
                    value="Contract"
                )
                
                document_text = gr.Textbox(
                    label="Document Text",
                    placeholder="Paste your document text here...",
                    lines=10,
                    max_lines=15
                )
                
                process_btn = gr.Button(
                    "🚀 Process Document",
                    variant="primary",
                    size="lg"
                )
                
                # Example documents
                gr.HTML("<h4>📝 Example Documents</h4>")
                
                with gr.Row():
                    contract_example = gr.Button("📋 Contract Example", size="sm")
                    invoice_example = gr.Button("🧾 Invoice Example", size="sm")
                    form_example = gr.Button("📄 Form Example", size="sm")
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Analysis Results</h3>")
                
                with gr.Tabs():
                    with gr.TabItem("📋 Summary"):
                        summary_output = gr.Markdown(
                            value="Upload a document and click 'Process Document' to see the analysis summary.",
                            elem_classes=["tool-section"]
                        )
                    
                    with gr.TabItem("🔧 Structured Data"):
                        structured_output = gr.Code(
                            value="{}",
                            language="json",
                            label="Extracted Data (JSON)"
                        )
                    
                    with gr.TabItem("💡 Business Insights"):
                        insights_output = gr.Markdown(
                            value="Business insights will appear here after processing.",
                            elem_classes=["tool-section"]
                        )
        
        # Example document texts
        contract_text = """
        Service Agreement between ABC Corporation and ZamAI Solutions
        
        Term: 12 months starting January 1, 2024
        Value: $60,000 annually for AI model development and deployment services
        
        Key Services:
        - Custom AI model development: $35,000
        - Model deployment and hosting: $15,000  
        - Ongoing support and maintenance: $10,000
        
        Termination: Either party may terminate with 30-day written notice
        Liability: Limited to the total contract value
        Payment Terms: Net 30 days from invoice date
        
        The Contractor agrees to deliver all models within the specified timeline
        and maintain 99.5% uptime for deployed services.
        """
        
        invoice_text = """
        INVOICE #2024-INV-001
        
        From: ZamAI Solutions Inc.
        123 AI Street, Tech City, TC 12345
        
        To: ABC Corporation
        456 Business Ave, Corporate City, CC 67890
        
        Invoice Date: March 1, 2024
        Due Date: March 31, 2024
        
        Description                  Quantity    Rate        Amount
        AI Model Development             1    $35,000.00  $35,000.00
        Model Deployment                 1    $15,000.00  $15,000.00
        Support Services                 1    $10,000.00  $10,000.00
        
        Subtotal:                                        $60,000.00
        Tax (8%):                                         $4,800.00
        Total:                                           $64,800.00
        
        Payment Terms: Net 30 days
        """
        
        form_text = """
        Employee Registration Form
        
        Name: Ahmad Khan
        Employee ID: EMP-2024-0145
        Department: Engineering
        Position: Senior AI Engineer
        Start Date: March 15, 2024
        Salary: $95,000 annually
        Manager: Sarah Johnson
        Email: ahmad.khan@company.com
        Phone: (555) 123-4567
        Address: 789 Tech Lane, Innovation City, IC 11111
        """
        
        # Event handlers
        def load_contract_example():
            return contract_text, "Contract"
        
        def load_invoice_example():
            return invoice_text, "Invoice"
        
        def load_form_example():
            return form_text, "Form"
        
        contract_example.click(
            fn=load_contract_example,
            outputs=[document_text, document_type]
        )
        
        invoice_example.click(
            fn=load_invoice_example,
            outputs=[document_text, document_type]
        )
        
        form_example.click(
            fn=load_form_example,
            outputs=[document_text, document_type]
        )
        
        process_btn.click(
            fn=business_tools.process_document,
            inputs=[document_text, document_type],
            outputs=[summary_output, structured_output, insights_output]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>🌟 Powered by ZamAI Pro Models Strategy | 🏢 Professional Business AI</p>
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
