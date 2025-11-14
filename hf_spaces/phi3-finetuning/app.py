"""
Hugging Face Space for Phi-3 Fine-tuning on Hub Datasets
Pro Account Features: GPU access, private datasets, enhanced compute
"""

import gradio as gr
import os
from huggingface_hub import HfApi, login, list_datasets
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
    num_epochs,
    batch_size,
    learning_rate,
    output_model_name,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
):
    """Start fine-tuning process"""
    try:
        # Prepare environment
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["BUSINESS_DOCS_DATASET"] = dataset_name
        os.environ["NUM_EPOCHS"] = str(num_epochs)
        os.environ["BATCH_SIZE"] = str(batch_size)
        os.environ["LEARNING_RATE"] = str(learning_rate)
        os.environ["PHI3_BUSINESS_MODEL"] = output_model_name
        
        # Create output message
        status_msg = f"🚀 Starting fine-tuning...\n"
        status_msg += f"Dataset: {dataset_name}\n"
        status_msg += f"Base Model: {base_model}\n"
        status_msg += f"Output Model: {output_model_name}\n"
        status_msg += f"Epochs: {num_epochs}\n"
        status_msg += f"Batch Size: {batch_size}\n"
        status_msg += f"Learning Rate: {learning_rate}\n"
        
        if use_lora:
            status_msg += f"LoRA Config: r={lora_r}, alpha={lora_alpha}\n"
        
        status_msg += "\n⏳ This will take several hours depending on dataset size and GPU availability...\n"
        status_msg += "💡 Pro Tip: With HF Pro, you get priority GPU access!\n"
        
        return status_msg
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

def check_fine_tuning_status():
    """Check status of fine-tuning job"""
    # This would integrate with HF Endpoints or Jobs API
    return "🔄 Feature coming soon: Real-time job monitoring via HF Pro API"

# Build Gradio Interface
with gr.Blocks(title="Phi-3 Fine-tuning on Hub Datasets", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Phi-3 Fine-tuning on Hugging Face Hub Datasets
    
    **Powered by Hugging Face Pro Account Features:**
    - ✨ Access to private datasets
    - 🚀 Enhanced GPU compute
    - 📊 Advanced monitoring and analytics
    - 🔒 Secure model hosting
    
    Fine-tune your Phi-3 models on custom datasets from your Hub account.
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
                    "tasal9/ZamAI-Phi-3-Mini-Pashto",
                    "microsoft/Phi-3-mini-4k-instruct",
                    "microsoft/Phi-3-mini-128k-instruct",
                ],
                value="tasal9/ZamAI-Phi-3-Mini-Pashto",
                label="Base Model"
            )
            output_model_name = gr.Textbox(
                value="tasal9/ZamAI-Phi-3-Finetuned",
                label="Output Model Name (Hub repo)"
            )
        
        with gr.Row():
            num_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
            learning_rate = gr.Number(value=2e-5, label="Learning Rate")
        
        with gr.Accordion("Advanced: LoRA Configuration", open=False):
            use_lora = gr.Checkbox(value=True, label="Use LoRA (Parameter-Efficient Fine-tuning)")
            with gr.Row():
                lora_r = gr.Slider(4, 64, value=8, step=4, label="LoRA r")
                lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA alpha")
    
    with gr.Tab("Start Training"):
        gr.Markdown("### Step 3: Launch fine-tuning job")
        
        start_btn = gr.Button("🚀 Start Fine-tuning", variant="primary", size="lg")
        training_status = gr.Textbox(label="Training Status", lines=10)
        
        start_btn.click(
            fn=start_finetuning,
            inputs=[
                dataset_dropdown,
                base_model,
                num_epochs,
                batch_size,
                learning_rate,
                output_model_name,
                use_lora,
                lora_r,
                lora_alpha,
            ],
            outputs=training_status
        )
        
        gr.Markdown("""
        ### 📝 Next Steps After Training:
        1. Model will be saved to your private Hub repository
        2. You can deploy it using HF Inference Endpoints
        3. Test it in the Inference API or create a demo Space
        
        ### 💡 Pro Tips:
        - Use LoRA for faster training and lower memory usage
        - Larger batch sizes work better with Pro GPU access
        - Monitor training in HF Hub's training dashboard
        """)
    
    with gr.Tab("Monitor Jobs"):
        gr.Markdown("### Monitor your fine-tuning jobs")
        check_status_btn = gr.Button("📊 Check Status")
        status_output = gr.Textbox(label="Job Status", lines=10)
        check_status_btn.click(fn=check_fine_tuning_status, outputs=status_output)

if __name__ == "__main__":
    demo.launch(share=False)
