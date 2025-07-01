#!/usr/bin/env python3
"""
ZamAI Dataset Integration Script
Integration with tasal9/Pashto-Dataset-Creating-Dataset
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PashtoDatasetIntegrator:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.dataset_repo = "tasal9/Pashto-Dataset-Creating-Dataset"
        self.local_dataset_path = Path("datasets/zamai_final_dataset")
        self.processed_dataset_path = Path("datasets/processed")
        
        # Initialize HF API
        if self.hf_token:
            login(token=self.hf_token)
            self.api = HfApi()
        else:
            logger.warning("HF_TOKEN not found. Some features may not work.")
    
    def download_dataset(self):
        """Download the Pashto dataset from HuggingFace"""
        try:
            logger.info(f"Downloading dataset: {self.dataset_repo}")
            
            # Try to load the dataset
            dataset = load_dataset(
                self.dataset_repo,
                trust_remote_code=True,
                token=self.hf_token
            )
            
            logger.info(f"Dataset loaded successfully: {dataset}")
            
            # Save locally
            self.local_dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Save each split
            for split_name, split_data in dataset.items():
                output_file = self.local_dataset_path / f"{split_name}.json"
                split_data.to_json(output_file, orient="records", lines=True)
                logger.info(f"Saved {split_name} split to {output_file}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return None
    
    def process_for_tutoring(self, dataset=None):
        """Process dataset specifically for tutoring/QA tasks"""
        if dataset is None:
            # Try to load from local files
            dataset = self.load_local_dataset()
        
        if dataset is None:
            logger.error("No dataset available for processing")
            return None
        
        processed_data = []
        
        # Process each example for tutoring format
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split...")
            
            for example in split_data:
                # Convert to tutoring format
                tutoring_example = self._convert_to_tutoring_format(example)
                if tutoring_example:
                    tutoring_example['split'] = split_name
                    processed_data.append(tutoring_example)
        
        # Save processed data
        self.processed_dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        processed_file = self.processed_dataset_path / "tutoring_qa.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # Save as CSV for easy inspection
        df = pd.DataFrame(processed_data)
        csv_file = self.processed_dataset_path / "tutoring_qa.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"Processed {len(processed_data)} examples")
        logger.info(f"Saved to: {processed_file} and {csv_file}")
        
        return processed_data
    
    def _convert_to_tutoring_format(self, example):
        """Convert dataset example to tutoring Q&A format"""
        try:
            # Handle different possible formats
            if isinstance(example, dict):
                # Look for common field names
                question_fields = ['question', 'input', 'text', 'prompt', 'query']
                answer_fields = ['answer', 'output', 'response', 'target', 'label']
                
                question = None
                answer = None
                
                # Find question
                for field in question_fields:
                    if field in example and example[field]:
                        question = example[field]
                        break
                
                # Find answer
                for field in answer_fields:
                    if field in example and example[field]:
                        answer = example[field]
                        break
                
                # If we have both question and answer
                if question and answer:
                    return {
                        'question': str(question).strip(),
                        'answer': str(answer).strip(),
                        'language': 'pashto' if self._is_pashto_text(question) else 'mixed',
                        'category': example.get('category', 'general'),
                        'difficulty': example.get('difficulty', 'medium'),
                        'source': 'zamai_dataset'
                    }
                
                # If it's a single text field, try to extract Q&A
                elif 'text' in example:
                    return self._extract_qa_from_text(example['text'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            return None
    
    def _is_pashto_text(self, text):
        """Simple heuristic to detect Pashto text"""
        if not text:
            return False
        
        # Pashto Unicode ranges
        pashto_chars = 0
        total_chars = len(text.replace(' ', ''))
        
        for char in text:
            # Pashto/Arabic Unicode ranges
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                pashto_chars += 1
        
        # If more than 30% are Pashto/Arabic characters
        return (pashto_chars / max(total_chars, 1)) > 0.3
    
    def _extract_qa_from_text(self, text):
        """Extract Q&A from text if it follows a pattern"""
        try:
            # Look for common Q&A patterns
            patterns = [
                ('Q:', 'A:'),
                ('Question:', 'Answer:'),
                ('پوښتنه:', 'ځواب:'),
                ('سوال:', 'جواب:')
            ]
            
            for q_marker, a_marker in patterns:
                if q_marker in text and a_marker in text:
                    parts = text.split(q_marker, 1)
                    if len(parts) > 1:
                        qa_part = parts[1]
                        if a_marker in qa_part:
                            q_and_a = qa_part.split(a_marker, 1)
                            if len(q_and_a) == 2:
                                question = q_and_a[0].strip()
                                answer = q_and_a[1].strip()
                                
                                return {
                                    'question': question,
                                    'answer': answer,
                                    'language': 'pashto' if self._is_pashto_text(question) else 'mixed',
                                    'category': 'extracted',
                                    'difficulty': 'medium',
                                    'source': 'zamai_dataset_extracted'
                                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting Q&A from text: {e}")
            return None
    
    def load_local_dataset(self):
        """Load dataset from local files"""
        try:
            dataset = {}
            
            if self.local_dataset_path.exists():
                for json_file in self.local_dataset_path.glob("*.json"):
                    split_name = json_file.stem
                    
                    # Load JSONL format
                    data = []
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data.append(json.loads(line))
                            except:
                                # Try regular JSON
                                f.seek(0)
                                data = json.load(f)
                                break
                    
                    dataset[split_name] = data
                    logger.info(f"Loaded {len(data)} examples from {split_name}")
            
            return dataset if dataset else None
            
        except Exception as e:
            logger.error(f"Error loading local dataset: {e}")
            return None
    
    def create_evaluation_split(self, processed_data, eval_ratio=0.2):
        """Create evaluation split from processed data"""
        if not processed_data:
            return None, None
        
        # Shuffle and split
        import random
        random.shuffle(processed_data)
        
        eval_size = int(len(processed_data) * eval_ratio)
        eval_data = processed_data[:eval_size]
        train_data = processed_data[eval_size:]
        
        # Save splits
        eval_file = self.processed_dataset_path / "eval_qa.json"
        train_file = self.processed_dataset_path / "train_qa.json"
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created train split: {len(train_data)} examples")
        logger.info(f"Created eval split: {len(eval_data)} examples")
        
        return train_data, eval_data
    
    def generate_dataset_stats(self, processed_data):
        """Generate statistics about the processed dataset"""
        if not processed_data:
            return {}
        
        stats = {
            'total_examples': len(processed_data),
            'languages': {},
            'categories': {},
            'difficulties': {},
            'avg_question_length': 0,
            'avg_answer_length': 0
        }
        
        question_lengths = []
        answer_lengths = []
        
        for example in processed_data:
            # Language distribution
            lang = example.get('language', 'unknown')
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
            
            # Category distribution
            cat = example.get('category', 'unknown')
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
            
            # Difficulty distribution
            diff = example.get('difficulty', 'unknown')
            stats['difficulties'][diff] = stats['difficulties'].get(diff, 0) + 1
            
            # Length statistics
            question_lengths.append(len(example.get('question', '')))
            answer_lengths.append(len(example.get('answer', '')))
        
        # Calculate averages
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        if answer_lengths:
            stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
        
        # Save stats
        stats_file = self.processed_dataset_path / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return stats

def main():
    """Main function to run dataset integration"""
    print("🗃️ ZamAI Pashto Dataset Integration")
    print("=" * 50)
    
    integrator = PashtoDatasetIntegrator()
    
    # Step 1: Download dataset
    print("📥 Downloading dataset...")
    dataset = integrator.download_dataset()
    
    if dataset is None:
        print("❌ Failed to download dataset. Trying local files...")
        dataset = integrator.load_local_dataset()
    
    if dataset is None:
        print("❌ No dataset available. Please check the repository name and access.")
        return
    
    # Step 2: Process for tutoring
    print("🔄 Processing dataset for tutoring...")
    processed_data = integrator.process_for_tutoring(dataset)
    
    if not processed_data:
        print("❌ No data could be processed.")
        return
    
    # Step 3: Create evaluation split
    print("📊 Creating evaluation split...")
    train_data, eval_data = integrator.create_evaluation_split(processed_data)
    
    # Step 4: Generate statistics
    print("📈 Generating dataset statistics...")
    stats = integrator.generate_dataset_stats(processed_data)
    
    # Print summary
    print("\n✅ Dataset Integration Complete!")
    print(f"📊 Total Examples: {stats['total_examples']}")
    print(f"🗣️ Languages: {list(stats['languages'].keys())}")
    print(f"📂 Categories: {list(stats['categories'].keys())}")
    print(f"📝 Avg Question Length: {stats['avg_question_length']:.1f} chars")
    print(f"💬 Avg Answer Length: {stats['avg_answer_length']:.1f} chars")
    
    print(f"\n📁 Files created:")
    print(f"   - datasets/processed/tutoring_qa.json")
    print(f"   - datasets/processed/tutoring_qa.csv")
    print(f"   - datasets/processed/train_qa.json")
    print(f"   - datasets/processed/eval_qa.json")
    print(f"   - datasets/processed/dataset_stats.json")

if __name__ == "__main__":
    main()
