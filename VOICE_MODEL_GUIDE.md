# Voice Model Documentation

## Overview

The ZamAI platform includes a comprehensive voice assistant system with Speech-to-Text (STT), Language Understanding, and Text-to-Speech (TTS) capabilities optimized for Pashto and multilingual support.

## Voice Models in the Repository

### 1. Whisper Model (Speech-to-Text)
- **Model**: `tasal9/ZamAI-Whisper-v3-Pashto`
- **Type**: Automatic Speech Recognition (ASR)
- **Last Update**: 1 hour ago
- **Status**: ✅ Active and deployed
- **Capabilities**:
  - Pashto speech recognition
  - Multilingual support
  - High accuracy for low-resource languages

### 2. LLaMA-3 Model (Language Understanding)
- **Model**: `tasal9/ZamAI-LIama3-Pashto`
- **Type**: Large Language Model (Private)
- **Last Update**: 9 days ago
- **Status**: ✅ Active and deployed
- **Capabilities**:
  - Natural language understanding
  - Context-aware responses
  - Bilingual (English/Pashto) support

### 3. TTS Integration
- **Technology**: Coqui-TTS (hack for Pashto)
- **Status**: Integrated in voice demos
- **Capabilities**:
  - Text-to-speech synthesis
  - Pashto voice generation

## Voice Assistant Architecture

```
Audio Input → Whisper STT → LLaMA-3 NLU → Response Generation → TTS → Audio Output
```

### Pipeline Components:

1. **Input Processing**
   - Audio file upload or microphone recording
   - Format: WAV, MP3, or other common audio formats
   - Sample rate: 16kHz recommended

2. **Speech-to-Text (Whisper)**
   - Transcribes audio to text
   - Language detection
   - Handles Pashto accent variations

3. **Language Understanding (LLaMA-3)**
   - Processes transcribed text
   - Generates contextual responses
   - Supports conversation history

4. **Text-to-Speech**
   - Converts response to audio
   - Pashto voice synthesis
   - Natural-sounding output

## Existing Voice Demos

### 1. Basic Voice Demo
- **File**: `demos/voice_demo.py`
- **Port**: 7861
- **Features**:
  - Simple STT → LLM → Response pipeline
  - Language selection (English/Pashto)
  - Basic audio processing

### 2. Advanced Voice Assistant
- **File**: `demos/voice_assistant_advanced.py`
- **Port**: 7862
- **Features**:
  - Enhanced UI with conversation history
  - Performance analytics
  - Context awareness
  - Advanced audio controls

### 3. Enhanced Voice Assistant UI
- **File**: `demos/voice_assistant_enhanced_ui.py`
- **Features**:
  - Modern interface design
  - Real-time transcription display
  - Audio visualization
  - Export conversation history

### 4. Inference API Version
- **File**: `demos/voice_assistant_inference_api.py`
- **Features**:
  - Uses HF Inference API for scalability
  - Lower latency
  - Production-ready deployment

## Deployed Spaces

### Voice Assistant Space
- **Location**: `hf_spaces/voice-assistant/`
- **App**: `hf_spaces/voice-assistant/app.py`
- **Status**: Ready for deployment
- **URL**: Deploy to `https://huggingface.co/spaces/tasal9/zamai-voice-assistant`

**Features**:
- Gradio-based interactive interface
- STT, LLM, and TTS integration
- Responsive design
- Mobile-friendly

## Voice Model Fine-tuning

### Fine-tuning Whisper for Better Pashto Recognition

While Whisper already supports Pashto, you can fine-tune it for:
- Specific accents or dialects
- Domain-specific vocabulary
- Better accuracy on your use cases

**Dataset Requirements**:
- Audio files with Pashto speech
- Corresponding transcriptions
- Format: Common Voice format or custom format

**Fine-tuning Script** (to be created):
```bash
python scripts/fine_tune_whisper.py \
  --model tasal9/ZamAI-Whisper-v3-Pashto \
  --dataset tasal9/pashto-speech-dataset \
  --epochs 5
```

### Improving LLaMA-3 for Voice Interactions

Fine-tune LLaMA-3 for:
- Conversational responses
- Voice-specific language patterns
- Context retention

**Dataset**: Conversational Q&A in Pashto/English

## Testing Voice Models

### 1. Local Testing
```bash
# Test basic voice pipeline
python test_voice_assistant.py

# Test simple voice interaction
python test_simple_voice.py

# Demo voice pipeline
python demo_voice_pipeline.py
```

### 2. Launch Voice Demos
```bash
# Basic voice demo
python main.py demo voice

# Enhanced voice assistant
python launch_voice_assistant_enhanced.py

# Advanced voice assistant
python demos/voice_assistant_advanced.py
```

## Voice Model Performance

### Current Status

| Component | Model | Performance | Notes |
|-----------|-------|-------------|-------|
| STT | Whisper-v3-Pashto | High | Optimized for Pashto |
| NLU | LLaMA-3-Pashto | High | Private model |
| TTS | Coqui-TTS | Medium | Pashto synthesis |

### Optimization Opportunities

1. **Whisper Fine-tuning**
   - Train on domain-specific Pashto speech
   - Improve accuracy for technical terms
   - Reduce latency with quantization

2. **LLaMA-3 Enhancement**
   - Fine-tune on voice interaction patterns
   - Improve context handling
   - Add memory for long conversations

3. **TTS Improvement**
   - Train custom Pashto TTS model
   - Improve voice quality
   - Add emotion and intonation control

## Deployment with HF Pro Account

### Pro Features for Voice Models

1. **Inference Endpoints**
   - Deploy Whisper with dedicated GPU
   - Low latency for real-time transcription
   - Auto-scaling for high traffic

2. **Private Endpoints**
   - Secure access to LLaMA-3 model
   - Token-based authentication
   - Usage analytics

3. **Spaces Pro**
   - GPU-accelerated inference
   - Persistent storage for conversation history
   - Custom domain for voice assistant
   - Advanced analytics

### Deployment Steps

1. **Deploy Whisper Endpoint**
```bash
huggingface-cli endpoint create \
  --name whisper-pashto \
  --repository tasal9/ZamAI-Whisper-v3-Pashto \
  --private \
  --accelerator gpu \
  --instance-size medium
```

2. **Deploy Voice Assistant Space**
```bash
./deploy_finetuning_spaces.sh
```

3. **Enable Pro Features**
   - Upgrade Space to Pro tier
   - Configure GPU settings
   - Enable analytics dashboard

## API Integration

### Using Voice Models via API

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your_hf_token")

# Speech-to-Text
transcription = client.automatic_speech_recognition(
    "audio.wav",
    model="tasal9/ZamAI-Whisper-v3-Pashto"
)

# Language Understanding
response = client.text_generation(
    model="tasal9/ZamAI-LIama3-Pashto",
    prompt=f"User: {transcription}\nAssistant:",
    max_new_tokens=300
)
```

### Voice Pipeline as API

See `api/main.py` for REST API implementation with FastAPI.

**Endpoints**:
- `POST /voice/transcribe` - Convert audio to text
- `POST /voice/respond` - Get LLM response
- `POST /voice/synthesize` - Convert text to speech
- `POST /voice/complete` - Full pipeline in one call

## Voice Model Monitoring

### Metrics to Track

1. **STT Performance**
   - Word Error Rate (WER)
   - Transcription latency
   - Language detection accuracy

2. **LLM Performance**
   - Response quality
   - Latency
   - Context retention

3. **TTS Performance**
   - Audio quality
   - Synthesis speed
   - Voice naturalness

### Using HF Pro Analytics

- Monitor usage patterns
- Track user engagement
- Identify performance bottlenecks
- A/B test model versions

## Next Steps

1. ✅ **Current State**: Voice models deployed and functional
2. 🔄 **In Progress**: Creating fine-tuning Spaces
3. ⏭️ **Next**:
   - Fine-tune Whisper on domain-specific data
   - Enhance LLaMA-3 for better voice interactions
   - Create custom Pashto TTS model
   - Deploy to HF Spaces with Pro features

## Resources

- Voice demos: `demos/voice_*.py`
- Launch scripts: `launch_voice_*.py`
- Test scripts: `test_voice_*.py`
- HF Space: `hf_spaces/voice-assistant/`
- Documentation: `VOICE_ASSISTANT_SUMMARY.md`

## Support

For issues or questions about voice models:
1. Check existing demos for examples
2. Review voice assistant documentation
3. Test with different audio formats
4. Monitor logs for errors
