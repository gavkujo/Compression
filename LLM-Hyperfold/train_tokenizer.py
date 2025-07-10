import os
import json
import sentencepiece as spm
from datasets import load_dataset
from shutil import copyfile

# === CONFIG ===
CORPUS_FILE = "corpus.txt"
TOKENIZER_DIR = "llama-tokenizer"
MODEL_PREFIX = "llama_tokenizer"
VOCAB_SIZE = 32000

def download_corpus():
    if not os.path.exists(CORPUS_FILE):
        print("📥 Downloading full Wikitext-2 corpus...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            for split in ["train", "validation", "test"]:
                for line in dataset[split]["text"]:
                    line = line.strip()
                    if line:
                        f.write(line + "\n")
        print("✅ Corpus saved to corpus.txt")
    else:
        print("📝 Corpus already exists!")

def train_tokenizer():
    print("🔧 Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3
    )
    print("✅ Tokenizer trained!")

def setup_hf_tokenizer_folder():
    print("🗂️ Preparing HuggingFace-compatible tokenizer folder...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # Rename model + copy
    copyfile(f"{MODEL_PREFIX}.model", f"{TOKENIZER_DIR}/tokenizer.model")

    # Write tokenizer_config.json
    tokenizer_config = {
        "model_type": "bpe",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": True,
        "tokenizer_class": "LlamaTokenizer"
    }

    with open(f"{TOKENIZER_DIR}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Special tokens map (optional but nice)
    with open(f"{TOKENIZER_DIR}/special_tokens_map.json", "w") as f:
        json.dump({
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>"
        }, f, indent=2)

    print(f"✅ Tokenizer files saved in `{TOKENIZER_DIR}/`")

if __name__ == "__main__":
    download_corpus()
    train_tokenizer()
    setup_hf_tokenizer_folder()
    print("\n🎉 You're all set! Load with:\n")
    print("   from transformers import LlamaTokenizer")
    print(f"   tokenizer = LlamaTokenizer.from_pretrained('{TOKENIZER_DIR}')")
