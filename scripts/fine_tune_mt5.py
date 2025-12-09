"""
Fine-tuning script for mT5 (multilingual T5) model for translation and text generation
Supports Pashto and other languages with sequence-to-sequence tasks
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi
import argparse
from dotenv import load_dotenv

load_dotenv()

def setup_model_and_tokenizer(model_name):
    """Setup MT5 model and tokenizer with LoRA configuration"""
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    
    model = MT5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # LoRA Configuration for MT5 (Seq2Seq)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"]  # MT5 uses different naming
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def preprocess_translation_dataset(dataset, tokenizer, max_length=512, source_lang="en", target_lang="ps"):
    """Preprocess translation dataset for MT5"""
    def tokenize_function(examples):
        # MT5 uses task prefixes for better performance
        inputs = [f"translate {source_lang} to {target_lang}: {text}" for text in examples['source']]
        targets = examples['target']
        
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Setup target labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def preprocess_text_generation_dataset(dataset, tokenizer, max_length=512):
    """Preprocess text generation dataset (Q&A, summarization, etc.)"""
    def tokenize_function(examples):
        # For Q&A or instruction following
        inputs = examples.get('input', examples.get('question', examples.get('instruction', [])))
        targets = examples.get('output', examples.get('answer', examples.get('response', [])))
        
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
            for labels_example in model_inputs["labels"]
        ]
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune mT5 for multilingual tasks")
    parser.add_argument("--model", default="google/mt5-base", help="Base model (mt5-small, mt5-base, mt5-large)")
    parser.add_argument("--dataset", default=os.getenv("MT5_DATASET"), help="Dataset name from Hub")
    parser.add_argument("--task", default="translation", choices=["translation", "generation"], help="Task type")
    parser.add_argument("--output", default="./models/mt5-pashto", help="Output directory")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 3)), help="Number of epochs")
    parser.add_argument("--source-lang", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="ps", help="Target language code (ps=Pashto)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting fine-tuning of {args.model}")
    print(f"📚 Using dataset: {args.dataset}")
    print(f"🎯 Task: {args.task}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model)
    
    # Load dataset from Hub
    print(f"📥 Loading dataset from Hugging Face Hub...")
    dataset = load_dataset(args.dataset, token=os.getenv("HF_TOKEN"))
    
    # Preprocess dataset based on task
    if args.task == "translation":
        train_dataset = preprocess_translation_dataset(
            dataset['train'], 
            tokenizer,
            source_lang=args.source_lang,
            target_lang=args.target_lang
        )
    else:  # generation
        train_dataset = preprocess_text_generation_dataset(dataset['train'], tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", 4)),
        learning_rate=float(os.getenv("LEARNING_RATE", 5e-5)),
        warmup_steps=50,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        push_to_hub=False,
        report_to=None,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        fp16=True,  # Enable mixed precision training
    )
    
    # Data collator for Seq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
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
        
        # Determine output repo based on task
        output_repo = os.getenv("MT5_MODEL", f"tasal9/ZamAI-MT5-{args.task.title()}")
        
        try:
            api.upload_folder(
                folder_path=args.output,
                repo_id=output_repo,
                repo_type="model",
                token=os.getenv("HF_TOKEN"),
                private=True
            )
            print(f"✅ Successfully uploaded to {output_repo}!")
        except Exception as e:
            print(f"❌ Failed to upload: {e}")

if __name__ == "__main__":
    main()
