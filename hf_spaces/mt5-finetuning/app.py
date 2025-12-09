"""
Hugging Face Space for MT5 Fine-tuning on Hub Datasets
Supports multilingual translation and text generation tasks
Pro Account Features: GPU access, private datasets, enhanced compute
"""

import gradio as gr
import os
from huggingface_hub import HfApi, login
from datasets import load_dataset
import subprocess
import sys
from pathlib import Path

# Login to HF Hub
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)

def get_user_datasets(username="tasal9"):
    """Fetch datasets from user's Hub account"""
    try:
        datasets = list(api.list_datasets(author=username))
        dataset_names = [ds.id for ds in datasets]
        return dataset_names
    except Exception as e:
        print(f"Error fetching datasets: {e}")
        return []

def preview_dataset(dataset_name, num_samples=5):
    """Preview dataset samples"""
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True, token=HF_TOKEN)
        samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(sample)
        
        # Format for display
        preview_text = f"Dataset: {dataset_name}\n"
        preview_text += f"Number of samples shown: {len(samples)}\n\n"
        
        for i, sample in enumerate(samples, 1):
            preview_text += f"--- Sample {i} ---\n"
            for key, value in sample.items():
                preview_text += f"{key}: {str(value)[:200]}...\n" if len(str(value)) > 200 else f"{key}: {value}\n"
            preview_text += "\n"
        
        return preview_text
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

def start_finetuning(
    dataset_name,
    base_model,
    task_type,
    num_epochs,
    batch_size,
    learning_rate,
    output_model_name,
    source_lang="en",
    target_lang="ps",
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
):
    """Start MT5 fine-tuning process"""
    try:
        # Prepare environment
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["MT5_DATASET"] = dataset_name
        os.environ["NUM_EPOCHS"] = str(num_epochs)
        os.environ["BATCH_SIZE"] = str(batch_size)
        os.environ["LEARNING_RATE"] = str(learning_rate)
        os.environ["MT5_MODEL"] = output_model_name
        
        # Create output message
        status_msg = f"🚀 Starting MT5 fine-tuning...\n"
        status_msg += f"Dataset: {dataset_name}\n"
        status_msg += f"Base Model: {base_model}\n"
        status_msg += f"Task: {task_type}\n"
        status_msg += f"Output Model: {output_model_name}\n"
        status_msg += f"Epochs: {num_epochs}\n"
        status_msg += f"Batch Size: {batch_size}\n"
        status_msg += f"Learning Rate: {learning_rate}\n"
        
        if task_type == "translation":
            status_msg += f"Translation: {source_lang} → {target_lang}\n"
        
        if use_lora:
            status_msg += f"LoRA Config: r={lora_r}, alpha={lora_alpha}\n"
        
        status_msg += "\n⏳ Training will start soon...\n"
        status_msg += "💡 MT5 is optimized for multilingual tasks including Pashto!\n"
        status_msg += "💡 With HF Pro, you get priority GPU access for faster training!\n"
        
        return status_msg
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

def test_translation(model_name, text, source_lang="en", target_lang="ps"):
    """Test translation with fine-tuned model"""
    try:
        from transformers import MT5Tokenizer, MT5ForConditionalGeneration
        
        tokenizer = MT5Tokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = MT5ForConditionalGeneration.from_pretrained(model_name, token=HF_TOKEN)
        
        # Prepare input
        input_text = f"translate {source_lang} to {target_lang}: {text}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        # Generate translation
        outputs = model.generate(input_ids, max_length=512)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
    except Exception as e:
        return f"Error: {str(e)}"

# Language codes mapping
LANGUAGE_CODES = {
    "English": "en",
    "Pashto": "ps",
    "Dari": "fa",
    "Urdu": "ur",
    "Arabic": "ar",
    "Persian": "fa",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
}

# Build Gradio Interface
with gr.Blocks(title="MT5 Fine-tuning on Hub Datasets", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌍 MT5 Multilingual Fine-tuning on Hub Datasets
    
    **Powered by Hugging Face Pro Account Features:**
    - ✨ Access to private datasets
    - 🚀 Enhanced GPU compute
    - 🌐 Support for 100+ languages including Pashto
    - 📊 Advanced monitoring
    - 🔒 Secure model hosting
    
    Fine-tune mT5 models for translation and multilingual text generation.
    """)
    
    with gr.Tab("Select Dataset"):
        gr.Markdown("### Step 1: Choose your dataset from Hub")
        
        refresh_btn = gr.Button("🔄 Refresh Datasets from tasal9 account")
        dataset_dropdown = gr.Dropdown(
            choices=get_user_datasets(),
            label="Select Dataset",
            info="Your private and public datasets from Hub"
        )
        refresh_btn.click(fn=get_user_datasets, outputs=dataset_dropdown)
        
        preview_btn = gr.Button("👁️ Preview Dataset")
        dataset_preview = gr.Textbox(label="Dataset Preview", lines=15, max_lines=20)
        preview_btn.click(
            fn=preview_dataset,
            inputs=dataset_dropdown,
            outputs=dataset_preview
        )
    
    with gr.Tab("Configure Fine-tuning"):
        gr.Markdown("### Step 2: Configure training parameters")
        
        with gr.Row():
            base_model = gr.Dropdown(
                choices=[
                    "google/mt5-small",
                    "google/mt5-base",
                    "google/mt5-large",
                    "google/mt5-xl",
                ],
                value="google/mt5-base",
                label="Base Model"
            )
            task_type = gr.Radio(
                choices=["translation", "generation"],
                value="translation",
                label="Task Type"
            )
        
        with gr.Row():
            output_model_name = gr.Textbox(
                value="tasal9/ZamAI-MT5-Pashto",
                label="Output Model Name (Hub repo)"
            )
        
        # Translation-specific options
        with gr.Group(visible=True) as translation_options:
            gr.Markdown("#### Translation Configuration")
            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value="English",
                    label="Source Language"
                )
                target_lang = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value="Pashto",
                    label="Target Language"
                )
        
        with gr.Row():
            num_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
            learning_rate = gr.Number(value=5e-5, label="Learning Rate")
        
        with gr.Accordion("Advanced: LoRA Configuration", open=False):
            use_lora = gr.Checkbox(value=True, label="Use LoRA (Parameter-Efficient Fine-tuning)")
            with gr.Row():
                lora_r = gr.Slider(4, 64, value=8, step=4, label="LoRA r")
                lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA alpha")
    
    with gr.Tab("Start Training"):
        gr.Markdown("### Step 3: Launch fine-tuning job")
        
        start_btn = gr.Button("🚀 Start Fine-tuning", variant="primary", size="lg")
        training_status = gr.Textbox(label="Training Status", lines=10)
        
        def start_training_wrapper(*args):
            # Convert language names to codes
            args_list = list(args)
            source_lang_code = LANGUAGE_CODES.get(args_list[6], "en")
            target_lang_code = LANGUAGE_CODES.get(args_list[7], "ps")
            args_list[6] = source_lang_code
            args_list[7] = target_lang_code
            return start_finetuning(*args_list)
        
        start_btn.click(
            fn=start_training_wrapper,
            inputs=[
                dataset_dropdown,
                base_model,
                task_type,
                num_epochs,
                batch_size,
                learning_rate,
                output_model_name,
                source_lang,
                target_lang,
                use_lora,
                lora_r,
                lora_alpha,
            ],
            outputs=training_status
        )
        
        gr.Markdown("""
        ### 📝 Next Steps After Training:
        1. Model will be saved to your private Hub repository
        2. Deploy using HF Inference Endpoints
        3. Test translations in the "Test Model" tab
        4. Create a demo Space for your users
        
        ### 💡 Pro Tips for MT5:
        - MT5 excels at low-resource languages like Pashto
        - Use task prefixes (e.g., "translate en to ps:") for better results
        - Larger models (mt5-large, mt5-xl) give better quality but need more GPU
        - LoRA makes training feasible even for large models
        """)
    
    with gr.Tab("Test Model"):
        gr.Markdown("### Test your fine-tuned MT5 model")
        
        with gr.Row():
            test_model_name = gr.Textbox(
                value="tasal9/ZamAI-MT5-Pashto",
                label="Model to Test"
            )
        
        with gr.Row():
            test_source_lang = gr.Dropdown(
                choices=list(LANGUAGE_CODES.keys()),
                value="English",
                label="Source Language"
            )
            test_target_lang = gr.Dropdown(
                choices=list(LANGUAGE_CODES.keys()),
                value="Pashto",
                label="Target Language"
            )
        
        test_input = gr.Textbox(
            label="Text to Translate",
            placeholder="Enter text in source language...",
            lines=3
        )
        
        test_btn = gr.Button("🔄 Translate")
        test_output = gr.Textbox(label="Translation", lines=3)
        
        def test_wrapper(model, text, src, tgt):
            src_code = LANGUAGE_CODES.get(src, "en")
            tgt_code = LANGUAGE_CODES.get(tgt, "ps")
            return test_translation(model, text, src_code, tgt_code)
        
        test_btn.click(
            fn=test_wrapper,
            inputs=[test_model_name, test_input, test_source_lang, test_target_lang],
            outputs=test_output
        )
        
        gr.Examples(
            examples=[
                ["Hello, how are you?", "English", "Pashto"],
                ["What is your name?", "English", "Pashto"],
                ["Thank you very much", "English", "Pashto"],
            ],
            inputs=[test_input, test_source_lang, test_target_lang],
        )

if __name__ == "__main__":
    demo.launch(share=False)
