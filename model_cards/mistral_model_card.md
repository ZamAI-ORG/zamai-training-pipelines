---
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
  - multilingual
  - instruction-following
datasets:
  - zamai-org/pashto-education-qa
inference: true
widget:
  - text: "د فزیک د کوانټم نظریه تشریح کړئ"
    example_title: "Physics in Pashto"
  - text: "What is machine learning and how does it work?"
    example_title: "ML Basics"
  - text: "د ریاضیاتو د الجبرا اساسات تشریح کړئ"
    example_title: "Math in Pashto"
model-index:
  - name: ZamAI-Mistral-7B-Pashto
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          type: pashto-education-qa
          name: Pashto Educational Q&A
---

# ZamAI-Mistral-7B-Pashto

## Model Description

ZamAI-Mistral-7B-Pashto is a fine-tuned version of Mistral-7B-Instruct specifically optimized for educational content delivery in both English and Pashto languages. This model is part of the ZamAI Pro Models Strategy, designed to provide high-quality educational assistance with strong Pashto language capabilities.

## Key Features

- **Bilingual Support**: Fluent in both English and Pashto
- **Educational Focus**: Specialized for tutoring and educational content
- **Cultural Awareness**: Understanding of Afghan/Pashtun cultural context
- **Instruction Following**: Enhanced ability to follow educational prompts
- **Adaptive Learning**: Can adjust explanations based on learner level

## Use Cases

- 📚 **Educational Tutoring**: Personalized learning assistance
- 🗣️ **Language Learning**: Pashto language instruction and practice  
- 🏫 **Classroom Support**: Teacher assistance and curriculum support
- 📖 **Content Creation**: Educational material generation
- 🌍 **Cultural Education**: Teaching about Afghan/Pashtun culture

## Usage

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your_hf_token")

# Educational query in English
response = client.text_generation(
    model="tasal9/ZamAI-Mistral-7B-Pashto",
    prompt="Explain photosynthesis in simple terms for a 10-year-old student",
    max_new_tokens=300,
    temperature=0.7
)

# Educational query in Pashto
response_pashto = client.text_generation(
    model="tasal9/ZamAI-Mistral-7B-Pashto",
    prompt="د فزیک د کوانټم نظریه ساده ډول تشریح کړئ",
    max_new_tokens=300,
    temperature=0.7
)
```

## Performance

- **Languages**: English (native), Pashto (fine-tuned)
- **Context Length**: 4,096 tokens
- **Model Size**: 7 billion parameters
- **Precision**: 16-bit floating point

## Training Details

### Training Data
- Educational Q&A datasets in English and Pashto
- Curated instructional content
- Cultural context examples
- Academic subject matter across multiple domains

### Training Procedure
- **Base Model**: mistralai/Mistral-7B-Instruct-v0.1
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Transformers + PEFT
- **Hardware**: GPU-accelerated training

## Evaluation

The model has been evaluated on:
- Pashto language comprehension tasks
- Educational content accuracy
- Instruction following capability
- Cultural sensitivity assessments

## Demo Applications

Try the model in these interactive demos:

- 🎓 [Educational Chatbot](https://huggingface.co/spaces/tasal9/zamai-chatbot-demo)
- 🎤 [Voice Assistant](https://huggingface.co/spaces/tasal9/zamai-voice-demo)

## Limitations

- May occasionally generate content that requires fact-checking
- Pashto responses may vary in dialect preference
- Performance may degrade with very specialized technical topics
- Requires careful prompt engineering for optimal results

## Ethical Considerations

- Designed to promote educational access in underserved communities
- Respects cultural values and sensitivities
- Aims to preserve and promote Pashto language
- Should be used responsibly for educational purposes

## Citation

```bibtex
@misc{zamai-mistral-pashto-2024,
  title={ZamAI-Mistral-7B-Pashto: Educational AI with Pashto Language Support},
  author={ZamAI Team},
  year={2024},
  url={https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto}
}
```

## License

This model is released under the Apache 2.0 License, following the base model's licensing terms.

## Contact

For questions, issues, or collaboration opportunities, please reach out through the Hugging Face model page or the ZamAI Pro Models Strategy repository.

---

*Part of the ZamAI Pro Models Strategy - Empowering AI for Educational Excellence*
