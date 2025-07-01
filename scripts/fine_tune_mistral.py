"""
Fine-tuning script for Mistral-7B model for educational purposes with Pashto support
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi
import argparse
from dotenv import load_dotenv

load_dotenv()

def setup_model_and_tokenizer(model_name):
    """Setup model and tokenizer with LoRA configuration"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def preprocess_dataset(dataset, tokenizer, max_length=512):
    """Preprocess the dataset for fine-tuning"""
    def tokenize_function(examples):
        # Combine instruction and response for educational Q&A
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"<s>[INST] {instruction} [/INST] {response} </s>"
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B for education")
    parser.add_argument("--model", default="tasal9/ZamAI-Mistral-7B-Pashto", help="Base model")
    parser.add_argument("--dataset", default=os.getenv("EDUCATION_QA_DATASET"), help="Dataset name")
    parser.add_argument("--output", default="./models/mistral-edu-pashto", help="Output directory")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 3)), help="Number of epochs")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting fine-tuning of {args.model}")
    print(f"📚 Using dataset: {args.dataset}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model)
    
    # Load and preprocess dataset
    dataset = load_dataset(args.dataset, use_auth_token=os.getenv("HF_TOKEN"))
    train_dataset = preprocess_dataset(dataset['train'], tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", 4)),
        learning_rate=float(os.getenv("LEARNING_RATE", 2e-5)),
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        push_to_hub=False,
        report_to=None,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("🔥 Starting training...")
    trainer.train()
    
    # Save the model
    print(f"💾 Saving model to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    print("✅ Fine-tuning completed successfully!")
    print(f"📍 Model saved at: {args.output}")
    
    # Upload to Hugging Face if token is available
    if os.getenv("HF_TOKEN"):
        print("🚀 Uploading to Hugging Face Hub...")
        api = HfApi()
        try:
            api.upload_folder(
                folder_path=args.output,
                repo_id=os.getenv("MISTRAL_EDU_MODEL"),
                repo_type="model",
                token=os.getenv("HF_TOKEN"),
                private=True
            )
            print("✅ Successfully uploaded to Hugging Face Hub!")
        except Exception as e:
            print(f"❌ Failed to upload: {e}")

if __name__ == "__main__":
    main()
