# Fine-tuning Guide: Phi-3 and MT5 with HF Spaces

## Overview

This guide explains how to fine-tune Phi-3 and MT5 models on your Hub datasets using Hugging Face Spaces with Pro account features.

## Prerequisites

### 1. Hugging Face Pro Account
- ✅ Access to private datasets
- ✅ Enhanced GPU compute
- ✅ Priority job scheduling
- ✅ Advanced analytics
- ✅ Private model hosting

### 2. Datasets on Hub
Your datasets should be uploaded to Hugging Face Hub under your account (tasal9):
- `tasal9/Pashto-Dataset-Creating-Dataset` (main dataset)
- Additional private datasets for specific tasks

### 3. Environment Setup
```bash
# Clone repository
git clone https://github.com/tasal9/ZamAI-Pro-Models-Strategy2.git
cd ZamAI-Pro-Models-Strategy2

# Setup environment
./setup.sh

# Configure HF token
cp .env.example .env
# Edit .env and ensure HF_TOKEN is set
```

## Fine-tuning Phi-3

### What is Phi-3?
Phi-3 is Microsoft's small language model optimized for:
- Business document processing
- Form extraction
- Contract analysis
- Instruction following

### Your Existing Phi-3 Model
- **Model**: `tasal9/ZamAI-Phi-3-Mini-Pashto`
- **Last Update**: 13 hours ago
- **Status**: ✅ Active
- **Use Case**: Business document processor

### Fine-tuning Script

#### Local Fine-tuning
```bash
python scripts/fine_tune_phi3.py \
  --model tasal9/ZamAI-Phi-3-Mini-Pashto \
  --dataset tasal9/your-business-dataset \
  --output ./models/phi3-custom \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5
```

#### Fine-tuning via HF Space

1. **Deploy Phi-3 Fine-tuning Space**
```bash
./deploy_finetuning_spaces.sh
```

2. **Access Space**
   - URL: `https://huggingface.co/spaces/tasal9/zamai-phi3-finetuning`
   - Login with your HF account

3. **Configure Fine-tuning**
   - Select dataset from your Hub account
   - Preview dataset samples
   - Configure hyperparameters:
     - Base model: `tasal9/ZamAI-Phi-3-Mini-Pashto`
     - Epochs: 3-5
     - Batch size: 4-8 (depends on GPU)
     - Learning rate: 1e-5 to 5e-5
     - LoRA: Enabled (recommended)

4. **Launch Training**
   - Click "Start Fine-tuning"
   - Monitor progress in Space UI
   - Pro account gives priority GPU access

5. **Deploy Fine-tuned Model**
   - Model saved to Hub automatically
   - Create inference endpoint
   - Test in Space or via API

### Dataset Format for Phi-3

Your dataset should have these columns:
```python
{
    "document": "Contract text or form content...",
    "extracted_info": "Key information extracted from document..."
}
```

Or for instruction following:
```python
{
    "instruction": "Extract the contract terms...",
    "input": "Contract text...",
    "output": "Extracted terms..."
}
```

## Fine-tuning MT5

### What is MT5?
MT5 (multilingual T5) is a sequence-to-sequence model for:
- Translation (100+ languages)
- Multilingual text generation
- Cross-lingual understanding
- Question answering

### MT5 for Pashto
- **Target Model**: `tasal9/ZamAI-MT5-Pashto`
- **Base Models**: 
  - `google/mt5-small` (300M) - Fast, good for testing
  - `google/mt5-base` (580M) - Balanced
  - `google/mt5-large` (1.2B) - High quality
  - `google/mt5-xl` (3.7B) - Best quality (needs Pro GPU)

### Fine-tuning Script

#### Local Fine-tuning (Translation Task)
```bash
python scripts/fine_tune_mt5.py \
  --model google/mt5-base \
  --dataset tasal9/pashto-translation-dataset \
  --task translation \
  --source-lang en \
  --target-lang ps \
  --output ./models/mt5-pashto-translation \
  --epochs 3
```

#### Local Fine-tuning (Text Generation Task)
```bash
python scripts/fine_tune_mt5.py \
  --model google/mt5-base \
  --dataset tasal9/pashto-qa-dataset \
  --task generation \
  --output ./models/mt5-pashto-qa \
  --epochs 3
```

#### Fine-tuning via HF Space

1. **Deploy MT5 Fine-tuning Space**
```bash
./deploy_finetuning_spaces.sh
```

2. **Access Space**
   - URL: `https://huggingface.co/spaces/tasal9/zamai-mt5-finetuning`

3. **Configure Fine-tuning**
   - Select dataset from Hub
   - Choose task type (translation or generation)
   - Select base model size
   - For translation:
     - Source language: English
     - Target language: Pashto
   - Configure hyperparameters
   - Enable LoRA for large models

4. **Launch Training**
   - Start fine-tuning job
   - Monitor progress
   - Pro GPU accelerates training

5. **Test Model**
   - Use built-in testing tab
   - Test translations directly in Space
   - Compare different model versions

### Dataset Format for MT5

#### Translation Dataset
```python
{
    "source": "Hello, how are you?",
    "target": "سلام، څنګه یاست؟"
}
```

#### Text Generation Dataset
```python
{
    "input": "Translate to Pashto: Hello",
    "output": "سلام"
}
```

Or:
```python
{
    "question": "What is machine learning?",
    "answer": "Machine learning is..."
}
```

## Leveraging HF Pro Account Features

### 1. Private Datasets
- Upload datasets with `--private` flag
- Secure access via token authentication
- Share with team members only

```bash
huggingface-cli upload \
  --repo-type dataset \
  --private \
  tasal9/my-private-dataset \
  ./local-data-folder
```

### 2. Enhanced GPU Compute

**Spaces Pro Hardware**:
- T4 GPU (16GB VRAM)
- A10G GPU (24GB VRAM) 
- A100 GPU (40GB VRAM) - for large models

**Benefits**:
- Faster training
- Larger batch sizes
- Support for larger models (mt5-xl, phi-3-medium)

**Enable in Space Settings**:
1. Go to Space settings
2. Select "Hardware"
3. Choose GPU tier
4. Confirm upgrade

### 3. Inference Endpoints

Deploy fine-tuned models as endpoints:

```bash
# Create endpoint for Phi-3
huggingface-cli endpoint create \
  --name phi3-custom-endpoint \
  --repository tasal9/ZamAI-Phi-3-Finetuned \
  --accelerator gpu \
  --instance-size medium \
  --private

# Create endpoint for MT5
huggingface-cli endpoint create \
  --name mt5-pashto-endpoint \
  --repository tasal9/ZamAI-MT5-Pashto \
  --accelerator gpu \
  --instance-size large \
  --private
```

**Endpoint Benefits**:
- Low latency inference
- Auto-scaling
- Usage monitoring
- Version control
- A/B testing support

### 4. Advanced Analytics

Monitor your models:
- Usage statistics
- User engagement
- Performance metrics
- Error tracking
- Cost optimization

Access via HF Hub dashboard.

### 5. Custom Domains

Brand your Spaces:
```
zamai-ai.com/phi3-finetuning
zamai-ai.com/mt5-translator
```

Enable in Space settings → Custom Domain

## Best Practices

### 1. Dataset Preparation
- Clean and validate data
- Balance dataset (avoid bias)
- Split: 80% train, 10% validation, 10% test
- Remove duplicates
- Normalize text encoding

### 2. Hyperparameter Tuning
- Start with defaults
- Use smaller models for experimentation
- Gradual learning rate decrease
- Monitor validation loss
- Early stopping to prevent overfitting

### 3. LoRA Configuration
**Recommended for Pro GPU**:
- Phi-3: r=8, alpha=16
- MT5: r=8, alpha=16
- Larger r for complex tasks
- Lower memory usage
- Faster training

### 4. Model Selection
- **Phi-3-mini**: 3.8B params - Good for most tasks
- **Phi-3-medium**: 14B params - Better quality (needs A100)
- **MT5-base**: 580M params - Balanced for translation
- **MT5-large**: 1.2B params - High quality (needs A10G)

### 5. Monitoring Training
- Check loss curves
- Validate on held-out data
- Test on real examples
- Compare with base model
- Save checkpoints regularly

## Deployment Workflow

### Complete Pipeline

1. **Prepare Dataset**
   - Upload to Hub as private dataset
   - Verify format and quality

2. **Fine-tune Model**
   - Use HF Space for interactive setup
   - Or use script for batch training
   - Monitor with Pro analytics

3. **Evaluate Model**
   - Test in Space interface
   - Compare with baseline
   - Measure metrics (BLEU, accuracy, etc.)

4. **Deploy to Production**
   - Create inference endpoint
   - Configure auto-scaling
   - Set up monitoring

5. **Create Demo Space**
   - Gradio interface for users
   - Integrate with endpoint
   - Add analytics tracking

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use LoRA
- Try smaller model

**Slow Training**
- Upgrade to larger GPU
- Increase batch size
- Use mixed precision (fp16)
- Enable gradient checkpointing
- Optimize data loading

**Poor Results**
- Check dataset quality
- Increase training epochs
- Adjust learning rate
- Try different model size
- Add more training data

**API Errors**
- Verify HF token
- Check dataset permissions
- Ensure model exists
- Monitor rate limits

## Example: Complete Fine-tuning Run

### Phi-3 for Business Documents

```bash
# 1. Set environment
export HF_TOKEN=your_token_here

# 2. Verify dataset
python -c "from datasets import load_dataset; \
  ds = load_dataset('tasal9/business-docs', token='$HF_TOKEN'); \
  print(ds['train'][0])"

# 3. Fine-tune
python scripts/fine_tune_phi3.py \
  --model tasal9/ZamAI-Phi-3-Mini-Pashto \
  --dataset tasal9/business-docs \
  --output ./models/phi3-business-custom \
  --epochs 3 \
  --batch-size 4

# 4. Upload to Hub
huggingface-cli upload \
  --repo-type model \
  --private \
  tasal9/ZamAI-Phi-3-Business-Custom \
  ./models/phi3-business-custom

# 5. Create endpoint
huggingface-cli endpoint create \
  --name phi3-business \
  --repository tasal9/ZamAI-Phi-3-Business-Custom \
  --accelerator gpu
```

### MT5 for Pashto Translation

```bash
# 1. Fine-tune
python scripts/fine_tune_mt5.py \
  --model google/mt5-base \
  --dataset tasal9/en-ps-translation \
  --task translation \
  --source-lang en \
  --target-lang ps \
  --output ./models/mt5-en-ps \
  --epochs 5

# 2. Upload
huggingface-cli upload \
  --repo-type model \
  tasal9/ZamAI-MT5-EN-PS \
  ./models/mt5-en-ps

# 3. Test
python -c "from transformers import MT5Tokenizer, MT5ForConditionalGeneration; \
  model = MT5ForConditionalGeneration.from_pretrained('tasal9/ZamAI-MT5-EN-PS'); \
  tokenizer = MT5Tokenizer.from_pretrained('tasal9/ZamAI-MT5-EN-PS'); \
  text = 'translate en to ps: Hello'; \
  inputs = tokenizer(text, return_tensors='pt'); \
  outputs = model.generate(**inputs); \
  print(tokenizer.decode(outputs[0]))"
```

## Next Steps

1. ✅ Deploy fine-tuning Spaces
2. ✅ Upload your datasets to Hub
3. 🔄 Fine-tune Phi-3 on business documents
4. 🔄 Fine-tune MT5 for Pashto translation
5. ⏭️ Create production endpoints
6. ⏭️ Build user-facing demo Spaces
7. ⏭️ Set up monitoring and analytics

## Resources

- Fine-tuning scripts: `scripts/fine_tune_*.py`
- HF Spaces: `hf_spaces/*/`
- Deployment script: `deploy_finetuning_spaces.sh`
- Environment config: `.env.example`
- README: Main documentation

## Support

For help with fine-tuning:
1. Check Space logs for errors
2. Review HF Hub model cards
3. Test with smaller datasets first
4. Monitor GPU usage in Space settings
5. Use HF Community forums for questions
