# ZamAI-Pro-Models-Strategy2

Central hub by ZamAI for storing datasets and managing the full data pipeline—gathering, cleaning, and normalizing—for efficient training and fine-tuning of AI models. Enables reproducible workflows for scalable AI and ML research.

## 🎯 Quick Start

```bash
# 1. Clone and setup
git clone <this-repo>
cd ZamAI-Pro-Models-Strategy2
./setup.sh

# 2. Configure your HF token in .env file
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 3. Run comprehensive tests
python test_suite.py

# 4. Launch demos
python main.py demo chatbot    # Educational chatbot
python main.py demo voice      # Voice assistant  
python main.py demo business   # Document processor

# 5. Start API server
python main.py api
```

## 🤖 Your Existing Models

This project is built around your **6 existing models** on Hugging Face:

| Model | Type | Description | Status |
|-------|------|-------------|---------|
| `tasal9/ZamAI-Mistral-7B-Pashto` | Text Generation | Educational tutor with Pashto support | ✅ Updated 4 days ago |
| `tasal9/ZamAI-Phi-3-Mini-Pashto` | Text Generation | Business document processor | ✅ Updated 13 hours ago |
| `tasal9/ZamAI-Whisper-v3-Pashto` | Speech-to-Text | Pashto speech recognition | ✅ Updated 1 hour ago |
| `tasal9/Multilingual-ZamAI-Embeddings` | Feature Extraction | Text embeddings | ✅ Updated 4 days ago |
| `tasal9/ZamAI-LIama3-Pashto` | Text Generation (Private) | Advanced Pashto LLM | ✅ Updated 9 days ago |
| `tasal9/pashto-base-bloom` | Text Generation (0.6B) | Lightweight Pashto model | ✅ Updated 14 days ago |

## 📁 Project Structure

```
ZamAI-Pro-Models-Strategy2/
├── 🚀 main.py                 # Main launcher for all components
├── 🔧 setup.sh               # Automated setup script
├── 🧪 test_suite.py          # Comprehensive test suite
├── 📋 requirements.txt       # Python dependencies
├── 📝 .env.example           # Environment configuration template
│
├── 📱 demos/                 # Interactive Gradio demos
│   ├── chatbot_demo.py      # Educational chatbot (port 7860)
│   ├── voice_demo.py        # Voice assistant (port 7861)
│   └── business_demo.py     # Document processor (port 7862)
│
├── 🌐 api/                   # FastAPI backend
│   └── main.py              # REST API server (port 8000)
│
├── 🔥 scripts/               # Training and fine-tuning
│   ├── fine_tune_mistral.py # Mistral-7B fine-tuning
│   └── fine_tune_phi3.py    # Phi-3-mini fine-tuning
│
├── 📄 model_cards/           # Model documentation
│   ├── mistral_model_card.md
│   └── phi3_model_card.md
│
└── ⚙️ .github/workflows/      # CI/CD automation
    └── deploy-models.yml     # Auto-deploy on push
```

## 🎪 Live Demos

The project includes **3 interactive demos** using your models:

### 🎓 Educational Chatbot
- **Model**: `tasal9/ZamAI-Mistral-7B-Pashto`
- **Features**: Bilingual tutoring (English/Pashto), adaptive learning
- **Launch**: `python main.py demo chatbot`
- **URL**: http://localhost:7860

### 🎤 Voice Assistant  
- **Models**: `tasal9/ZamAI-Whisper-v3-Pashto` → `tasal9/ZamAI-LIama3-Pashto` → TTS
- **Features**: Speech-to-text, LLaMA-3 reasoning, context awareness, performance analytics
- **Launch**: `python main.py demo voice`  
- **URL**: http://localhost:7861
- **Advanced UI**: `python demos/voice_assistant_advanced.py` (Enhanced test interface)

### 📊 Business Document Processor
- **Model**: `tasal9/ZamAI-Phi-3-Mini-Pashto`
- **Features**: Contract analysis, invoice processing, form extraction
- **Launch**: `python main.py demo business`
- **URL**: http://localhost:7862

## Use Cases and Model Applications

| **Use Case**            | **Features**                          | **Tools**                                                                                        |
| ----------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Voice Assistant**     | STT, NLU, TTS                         | `tasal9/ZamAI-Whisper-v3-Pashto`, `tasal9/ZamAI-LIama3-Pashto`, HF Inference API, Coqui-TTS |
| **Tutor Bot**           | Chat, explain, adapt to learner       | `tasal9/ZamAI-Mistral-7B-Pashto`, `tasal9/ZamAI-Phi-3-Mini-Pashto`, private fine-tuning on Pashto/edu datasets |
| **Business Automation** | Parse forms, summarize docs, generate | `tasal9/ZamAI-Phi-3-Mini-Pashto`, `tasal9/Multilingual-ZamAI-Embeddings`, HF model evals, GitHub CI/CD |

## Technical Stack

| **Component**   | **Tech**                                          |
| --------------- | ------------------------------------------------- |
| LLMs            | `LLaMA-3`, `Mistral-7B`, `Phi-3-mini`             |
| Embeddings      | `intfloat/e5-large-v2`                            |
| STT / TTS       | `whisper-large-v3`, `Coqui-TTS` (hack for Pashto) |
| Frontend        | `Gradio`, `HF Spaces`, `React Native`             |
| Backend/API     | `FastAPI`, `Hugging Face Inference API`           |
| Hosting         | Private Hugging Face repos + Endpoints            |
| Data Management | HF Pro private datasets                           |

## Setup Instructions

### Step 1: Dataset Setup

Use `datasets-cli upload` with `--private` flag to securely store datasets:

- **Pashto Speech Data**: Audio files and transcriptions for voice assistant training
- **Q&A Datasets**: Educational content for tutor bot fine-tuning  
- **Business Documents**: Forms, contracts, and documents for automation models

```bash
# Upload private datasets to Hugging Face Hub
huggingface-cli upload --repo-type dataset --private [dataset-name] [local-path]
```

*Note: Datasets will be provided and uploaded to private HF repositories for secure access.*

### Step 2: Fine-Tuning

**Fine-tune Mistral-7B for Education:**
- Target: Educational content and tutoring capabilities
- Focus: Pashto language support and adaptive learning
- Method: LoRA/QLoRA fine-tuning on educational Q&A datasets

**Fine-tune Phi-3-mini for Form/Document Workflows:**
- Target: Business document processing and form parsing
- Focus: Structured data extraction and document understanding
- Method: Task-specific fine-tuning on business document datasets

```bash
# Fine-tuning commands (examples)
# Mistral-7B for education
python fine_tune.py --model mistralai/Mistral-7B-Instruct-v0.1 --dataset education_qa --output mistral-edu-pashto

# Phi-3-mini for business automation  
python fine_tune.py --model microsoft/Phi-3-mini-4k-instruct --dataset business_docs --output phi3-business-forms
```

**Upload to Private Model Repository:**
```bash
# Upload fine-tuned models to private HF repos
huggingface-cli upload --repo-type model --private [model-name] [local-model-path]
```

### Step 3: Deploy as Endpoint

**Deploy Models to Hugging Face Endpoints:**
- Create dedicated inference endpoints for fine-tuned models
- Enable auto-scaling for production workloads
- Configure private access with token authentication

**Use InferenceClient for Model Access:**
```python
from huggingface_hub import InferenceClient

# Initialize client with your HF token
client = InferenceClient(token="your_hf_token")

# Call fine-tuned Mistral-7B for education
response = client.text_generation(
    model="your-org/mistral-edu-pashto",
    prompt="Explain quantum physics in Pashto",
    max_new_tokens=500
)

# Call fine-tuned Phi-3-mini for business automation
response = client.text_generation(
    model="your-org/phi3-business-forms",
    prompt="Extract key information from this contract: [document text]",
    max_new_tokens=300
)
```

**Full Real-time API Access (Token Secured):**
- Private endpoints with HF Pro authentication
- Real-time inference with low latency
- Secure token-based access control
- Scalable deployment for production use

```bash
# Deploy to HF Inference Endpoints
huggingface-cli endpoint create --name mistral-edu-endpoint --repository your-org/mistral-edu-pashto --private
huggingface-cli endpoint create --name phi3-business-endpoint --repository your-org/phi3-business-forms --private
```

### Step 4: Build MVP with Spaces

**Use Gradio for Interactive Demos:**
- **Chatbot Demo**: Educational tutor with Pashto support
- **Voice Demo**: Speech-to-text and text-to-speech pipeline
- **Form Demo**: Automated document processing and extraction

```python
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(token="your_hf_token")

# Educational Chatbot Interface
def chatbot_demo(message, history):
    response = client.text_generation(
        model="your-org/mistral-edu-pashto",
        prompt=message,
        max_new_tokens=300
    )
    return response

# Voice Assistant Interface  
def voice_demo(audio):
    # STT with Whisper + LLM processing + TTS with Coqui
    transcription = client.automatic_speech_recognition(audio)
    response = client.text_generation(model="your-org/mistral-edu-pashto", prompt=transcription)
    return response

# Business Form Processing Interface
def form_demo(document_text):
    response = client.text_generation(
        model="your-org/phi3-business-forms",
        prompt=f"Extract key information: {document_text}",
        max_new_tokens=200
    )
    return response

# Create Gradio interfaces
chat_interface = gr.ChatInterface(chatbot_demo, title="ZamAI Educational Tutor")
voice_interface = gr.Interface(voice_demo, inputs="audio", outputs="text", title="ZamAI Voice Assistant")
form_interface = gr.Interface(form_demo, inputs="text", outputs="text", title="ZamAI Document Processor")
```

**Hugging Face Spaces Pro Benefits:**
- **More GPU**: Enhanced compute for real-time inference
- **Usage Stats**: Analytics and monitoring for user interactions
- **Custom Domains**: Professional branding with custom URLs
- **Private Spaces**: Secure demos with controlled access
- **Persistent Storage**: Store user data and conversation history

```bash
# Deploy to HF Spaces
huggingface-cli repo create --type space --space_sdk gradio zamai-chatbot-demo
huggingface-cli repo create --type space --space_sdk gradio zamai-voice-demo  
huggingface-cli repo create --type space --space_sdk gradio zamai-business-demo
```

### Step 5: CI/CD Integration

**GitHub Actions Auto-Deploy Model:**
- Automated model training and deployment pipeline
- Push to main branch triggers HF model publishing
- Continuous integration for model updates and versioning

**Auto-Publish to Hugging Face on Push:**
```yaml
# .github/workflows/deploy-models.yml
name: Deploy Models to Hugging Face

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy-models:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install huggingface_hub transformers torch datasets
        
    - name: Login to Hugging Face
      run: |
        huggingface-cli login --token ${{ secrets.HF_TOKEN }}
        
    - name: Fine-tune and Upload Mistral-7B
      run: |
        python scripts/fine_tune_mistral.py
        huggingface-cli upload --repo-type model --private zamai-org/mistral-edu-pashto ./models/mistral-edu-pashto
        
    - name: Fine-tune and Upload Phi-3-mini
      run: |
        python scripts/fine_tune_phi3.py  
        huggingface-cli upload --repo-type model --private zamai-org/phi3-business-forms ./models/phi3-business-forms
        
    - name: Update Endpoints
      run: |
        huggingface-cli endpoint update --name mistral-edu-endpoint --repository zamai-org/mistral-edu-pashto
        huggingface-cli endpoint update --name phi3-business-endpoint --repository zamai-org/phi3-business-forms
```

**Automated Workflow Benefits:**
- **Continuous Deployment**: Push code → Auto-train → Deploy models
- **Version Control**: Track model versions with Git commits
- **Quality Assurance**: Automated testing before deployment
- **Rollback Support**: Easy model versioning and rollback capabilities
- **Monitoring**: Integration with HF model monitoring and alerts

```bash
# Setup GitHub secrets for automation
# Add these secrets to your GitHub repository:
# HF_TOKEN: Your Hugging Face API token
# HF_ORG: Your Hugging Face organization name
```

## Hidden Pro Features You Should Exploit

| **Feature**                     | **Hack**                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Model Card Metadata**         | Add use case details, sample prompts                                                              |
| **Model Evaluation Benchmarks** | Validate with Pashto-specific test cases                                                          |
| **Inference-side Training**     | Skip local training; fine-tune directly in HF Pro cloud                                           |
| **Multi-modal Expansion**       | Combine llava (image) + Mistral for image-captioning or visual education use                      |

### Advanced Pro Features Implementation

**1. Enhanced Model Card Metadata:**
```yaml
# model_card.yml - Rich metadata for better discoverability
base_model: mistralai/Mistral-7B-Instruct-v0.1
license: apache-2.0
language: 
  - en
  - ps  # Pashto
tags:
  - education
  - pashto
  - tutoring
  - zamai
datasets:
  - zamai-org/pashto-education-qa
inference: true
widget:
  - text: "د فزیک د کوانټم نظریه تشریح کړئ" # Explain quantum theory in Pashto
    example_title: "Physics in Pashto"
  - text: "What is machine learning?"
    example_title: "ML Basics"
```

**2. Pashto-Specific Evaluation Benchmarks:**
```python
# Custom evaluation for Pashto language models
evaluation_config = {
    "tasks": [
        "pashto_comprehension",
        "english_to_pashto_translation", 
        "educational_qa_pashto",
        "cultural_context_understanding"
    ],
    "metrics": ["bleu", "rouge", "bertscore", "cultural_accuracy"],
    "test_datasets": ["zamai-org/pashto-eval-suite"]
}
```

**3. Cloud-Native Fine-tuning (Skip Local Training):**
```python
# Direct HF Pro cloud fine-tuning
from huggingface_hub import create_inference_endpoint
from transformers import AutoConfig

# Configure cloud training job
training_config = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
    "dataset": "zamai-org/pashto-education-qa",
    "compute_environment": "gpu-large",
    "training_args": {
        "num_train_epochs": 3,
        "learning_rate": 2e-5, 
        "lora_config": {"r": 16, "alpha": 32}
    }
}

# Submit cloud training job
endpoint = create_inference_endpoint(
    name="mistral-pashto-trainer",
    repository="zamai-org/mistral-edu-pashto",
    framework="pytorch",
    accelerator="gpu",
    instance_size="large",
    custom_image="huggingface/transformers-pytorch-gpu:latest"
)
```

**4. Multi-modal Visual Education:**
```python
# Combine LLaVA + Mistral for visual learning
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# Visual education pipeline
def visual_tutor_pipeline(image, question_pashto):
    # Step 1: Image understanding with LLaVA
    visual_description = llava_model.generate(
        image=image,
        prompt="Describe this educational diagram in detail"
    )
    
    # Step 2: Pashto educational response with fine-tuned Mistral
    combined_prompt = f"""
    Image Description: {visual_description}
    Student Question (Pashto): {question_pashto}
    Provide educational explanation in Pashto:
    """
    
    pashto_explanation = mistral_client.text_generation(
        model="zamai-org/mistral-edu-pashto",
        prompt=combined_prompt,
        max_new_tokens=400
    )
    
    return pashto_explanation

# Example usage: Science diagrams + Pashto explanations
response = visual_tutor_pipeline(
    image="physics_diagram.jpg",
    question_pashto="دا ډیاګرام څه ښیي؟"  # What does this diagram show?
)
```
