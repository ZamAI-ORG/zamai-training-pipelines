# Quick Start Guide: Fine-tuning Phi-3 and MT5

## 🚀 Get Started in 5 Minutes

### Prerequisites
- ✅ Hugging Face Pro account
- ✅ HF_TOKEN set in `.env` file
- ✅ Datasets uploaded to your Hub account (tasal9)

### Step 1: Validate Setup
```bash
python test_finetuning_setup.py
```
Expected: All tests pass ✅

### Step 2: Deploy Spaces (Optional)
```bash
# Set your HF token
export HF_TOKEN=your_token_here
export HF_ORG=tasal9

# Deploy all Spaces to Hub
./deploy_finetuning_spaces.sh
```

Your Spaces will be available at:
- `https://huggingface.co/spaces/tasal9/zamai-phi3-finetuning`
- `https://huggingface.co/spaces/tasal9/zamai-mt5-finetuning`

### Step 3: Fine-tune Models

#### Option A: Via Gradio Space (Recommended for Beginners)
1. Visit your deployed Space
2. Select dataset from dropdown
3. Preview dataset samples
4. Configure hyperparameters
5. Click "Start Fine-tuning"

#### Option B: Via Command Line (Recommended for Automation)

**Fine-tune Phi-3 for Business Documents**
```bash
python scripts/fine_tune_phi3.py \
  --model tasal9/ZamAI-Phi-3-Mini-Pashto \
  --dataset tasal9/your-business-dataset \
  --output ./models/phi3-custom \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5
```

**Fine-tune MT5 for Translation**
```bash
python scripts/fine_tune_mt5.py \
  --model google/mt5-base \
  --dataset tasal9/your-translation-dataset \
  --task translation \
  --source-lang en \
  --target-lang ps \
  --output ./models/mt5-en-ps \
  --epochs 3
```

**Fine-tune MT5 for Text Generation**
```bash
python scripts/fine_tune_mt5.py \
  --model google/mt5-base \
  --dataset tasal9/your-qa-dataset \
  --task generation \
  --output ./models/mt5-qa \
  --epochs 3
```

### Step 4: Test Your Models

**Test Phi-3**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/phi3-custom")
model = AutoModelForCausalLM.from_pretrained("./models/phi3-custom")

prompt = "Extract key information from this contract: ..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

**Test MT5 (Translation)**
```python
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

tokenizer = MT5Tokenizer.from_pretrained("./models/mt5-en-ps")
model = MT5ForConditionalGeneration.from_pretrained("./models/mt5-en-ps")

text = "translate en to ps: Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 5: Deploy to Production

**Upload to Hub**
```bash
huggingface-cli upload \
  --repo-type model \
  --private \
  tasal9/ZamAI-Phi-3-Custom \
  ./models/phi3-custom
```

**Create Inference Endpoint**
```bash
huggingface-cli endpoint create \
  --name phi3-custom \
  --repository tasal9/ZamAI-Phi-3-Custom \
  --accelerator gpu \
  --instance-size medium \
  --private
```

## 📚 Documentation

For detailed information, see:
- **Fine-tuning Guide**: `FINETUNING_GUIDE.md` - Complete fine-tuning documentation
- **Voice Models**: `VOICE_MODEL_GUIDE.md` - Voice model setup and usage
- **Implementation**: `IMPLEMENTATION_SUMMARY.md` - Technical details
- **Main README**: `README.md` - Project overview

## 🎤 Voice Models

Your voice assistant infrastructure is ready:

**Launch Voice Demos**
```bash
# Basic voice assistant
python main.py demo voice

# Enhanced voice assistant
python launch_voice_assistant_enhanced.py

# Advanced features
python demos/voice_assistant_advanced.py
```

**Voice Models**:
- STT: `tasal9/ZamAI-Whisper-v3-Pashto` ✅
- NLU: `tasal9/ZamAI-LIama3-Pashto` ✅
- TTS: Coqui-TTS ✅

## 🔧 Troubleshooting

**Problem**: Out of memory during training
**Solution**: Reduce batch size or use gradient accumulation
```bash
python scripts/fine_tune_phi3.py \
  --batch-size 2 \
  --gradient-accumulation-steps 4
```

**Problem**: Dataset not found
**Solution**: Check dataset name and HF_TOKEN
```bash
# Verify dataset exists
huggingface-cli repo ls tasal9/your-dataset --repo-type dataset
```

**Problem**: Space deployment fails
**Solution**: Verify HF_TOKEN and organization name
```bash
# Test login
huggingface-cli whoami

# Test Space creation
huggingface-cli repo create test-space --type space --space_sdk gradio
```

## 💡 Pro Tips

1. **Start Small**: Test with mt5-small before using mt5-large
2. **Use LoRA**: Faster training, less memory (enabled by default)
3. **Monitor Training**: Check logs regularly for errors
4. **Validate Data**: Always preview dataset before training
5. **Save Checkpoints**: Use `--save-steps 500` to save progress
6. **GPU Selection**: Use T4 for testing, A10G for production, A100 for large models

## 🎯 Common Workflows

**Business Document Processing (Phi-3)**
```bash
# 1. Prepare dataset with documents and extracted info
# 2. Fine-tune
python scripts/fine_tune_phi3.py \
  --dataset tasal9/business-docs \
  --epochs 3

# 3. Test
# Use Space testing tab or CLI

# 4. Deploy
./deploy_to_endpoint.sh phi3-business
```

**English-Pashto Translation (MT5)**
```bash
# 1. Prepare parallel corpus
# 2. Fine-tune
python scripts/fine_tune_mt5.py \
  --dataset tasal9/en-ps-parallel \
  --task translation \
  --source-lang en \
  --target-lang ps \
  --epochs 5

# 3. Test in Space or CLI
# 4. Deploy and use
```

## 📊 Expected Training Times

**With T4 GPU (Free/Basic)**:
- Phi-3 (4k dataset): 2-3 hours
- MT5-small (4k dataset): 1-2 hours
- MT5-base (4k dataset): 3-4 hours

**With A10G GPU (Pro)**:
- Phi-3 (4k dataset): 1-2 hours
- MT5-small (4k dataset): 30-60 minutes
- MT5-base (4k dataset): 1.5-2.5 hours
- MT5-large (4k dataset): 4-6 hours

**With A100 GPU (Pro Premium)**:
- Phi-3 (4k dataset): 30-60 minutes
- MT5-base (4k dataset): 45-90 minutes
- MT5-large (4k dataset): 2-3 hours
- MT5-xl (4k dataset): 6-8 hours

## 🎉 You're Ready!

Everything is set up and ready to go. Start with:
1. Run validation tests
2. Deploy Spaces (optional)
3. Fine-tune a small model to test
4. Scale up to larger models
5. Deploy to production

For help, check the comprehensive guides in:
- `FINETUNING_GUIDE.md`
- `VOICE_MODEL_GUIDE.md`
- `IMPLEMENTATION_SUMMARY.md`

Good luck with your fine-tuning! 🚀
