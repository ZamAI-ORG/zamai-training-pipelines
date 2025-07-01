"""
FastAPI backend for ZamAI Pro Models Strategy
Provides REST API endpoints for educational tutor and business automation
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import librosa
import soundfile as sf
import tempfile
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ZamAI Pro Models API",
    description="REST API for educational tutoring and business automation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hugging Face Inference Client
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    language: str = "en"  # en or ps (Pashto)

class ChatResponse(BaseModel):
    response: str
    language: str

class DocumentRequest(BaseModel):
    document_text: str

class DocumentResponse(BaseModel):
    extracted_info: str
    confidence: float = 0.95

class VoiceResponse(BaseModel):
    transcription: str
    response: str

# Educational Chat Endpoint
@app.post("/chat/education", response_model=ChatResponse)
async def educational_chat(request: ChatRequest):
    """
    Educational chatbot with Pashto support
    """
    try:
        # Construct prompt based on language
        if request.language == "ps":
            prompt = f"[د ښوونې او روزنې ملګری] {request.message}"
        else:
            prompt = f"[Educational Tutor] {request.message}"
        
        response = hf_client.text_generation(
            model=os.getenv("MISTRAL_EDU_MODEL"),
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
        
        return ChatResponse(
            response=response,
            language=request.language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Business Document Processing Endpoint
@app.post("/process/document", response_model=DocumentResponse)
async def process_business_document(request: DocumentRequest):
    """
    Extract key information from business documents
    """
    try:
        prompt = f"Extract key information from this document: {request.document_text}"
        
        response = hf_client.text_generation(
            model=os.getenv("PHI3_BUSINESS_MODEL"),
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False
        )
        
        return DocumentResponse(
            extracted_info=response,
            confidence=0.95
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Voice Processing Endpoint
@app.post("/voice/process", response_model=VoiceResponse)
async def process_voice(audio_file: UploadFile = File(...)):
    """
    Process voice input: Whisper STT -> LLaMA-3 -> Response
    """
    try:
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Step 1: Speech-to-Text using your Whisper model
        transcription = hf_client.automatic_speech_recognition(
            tmp_file_path,
            model=os.getenv("WHISPER_MODEL", "tasal9/ZamAI-Whisper-v3-Pashto")
        )
        
        # Step 2: Generate response using your LLaMA-3 model
        llama3_model = os.getenv("LLAMA3_MODEL", "tasal9/ZamAI-LIama3-Pashto")
        system_prompt = "You are an intelligent voice assistant. Provide helpful and accurate responses."
        
        # Use LLaMA-3 chat format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{transcription}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        response = hf_client.text_generation(
            model=llama3_model,
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return VoiceResponse(
            transcription=transcription,
            response=response.strip()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "education": os.getenv("MISTRAL_EDU_MODEL"),
            "business": os.getenv("PHI3_BUSINESS_MODEL")
        }
    }

# Model information endpoint
@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    return {
        "educational_model": {
            "name": os.getenv("MISTRAL_EDU_MODEL"),
            "description": "Fine-tuned Mistral-7B for educational content with Pashto support",
            "capabilities": ["chat", "Q&A", "tutoring", "pashto_support"]
        },
        "business_model": {
            "name": os.getenv("PHI3_BUSINESS_MODEL"),
            "description": "Fine-tuned Phi-3-mini for business document processing",
            "capabilities": ["document_extraction", "form_processing", "structured_data"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
