"""
Train a custom subword tokenizer for Universal HyperFold using HuggingFace Tokenizers.
Saves tokenizer files to scripts/tokenizer_1b/
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os

import sys

# --- CONFIG ---
DATA_FILES = [
    "../datasets/code_dataset.csv",
    "../datasets/combined_dataset.csv",
    "../datasets/creative_dataset.csv",
    "../datasets/general_dataset.csv",
    "../datasets/math_dataset.csv"
]

# Model configs for vocab sizes
MODEL_CONFIGS = {
    "350M": {"vocab_size": 16000},
    "1B": {"vocab_size": 2145}, # 32000 for 1B
    "3B": {"vocab_size": 48000},
    "6B": {"vocab_size": 64000},
    "14B": {"vocab_size": 128000}
}

# --- DATA EXTRACTION ---
def extract_texts(files):
    texts = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                # If CSV, extract prompt/output columns
                if "," in line:
                    parts = line.strip().split(",")
                    texts.extend(parts)
                else:
                    texts.append(line.strip())
    return texts

# --- TOKENIZER TRAINING ---
def train_tokenizer(texts, vocab_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=True)
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"✅ Tokenizer trained and saved to {save_dir}/tokenizer.json")

if __name__ == "__main__":
    print("🔬 Training custom tokenizers for all Universal HyperFold model sizes...")
    texts = extract_texts(DATA_FILES)
    for model_size, config in MODEL_CONFIGS.items():
        vocab_size = config["vocab_size"]
        save_dir = f"tokenizer_{model_size}"
        print(f"\n🚀 Training tokenizer for {model_size} (vocab_size={vocab_size})...")
        train_tokenizer(texts, vocab_size, save_dir)
    print("\n🎉 All tokenizers trained and saved.")
