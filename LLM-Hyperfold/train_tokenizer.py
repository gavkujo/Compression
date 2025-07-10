import os
import sentencepiece as spm
import requests
import json

# === CONFIG ===
CORPUS_FILE = "corpus.txt"
TOKENIZER_DIR = "llama-tokenizer"
TOKENIZER_MODEL = os.path.join(TOKENIZER_DIR, "tokenizer.model")
TOKENIZER_CONFIG = os.path.join(TOKENIZER_DIR, "tokenizer_config.json")
VOCAB_SIZE = 32000

def download_corpus():
    if not os.path.exists(CORPUS_FILE):
        print("Downloading full wikitext-2 corpus...")
        urls = [
            "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/train.txt",
            "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/valid.txt",
            "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/test.txt"
        ]
        with open(CORPUS_FILE, "wb") as f:
            for url in urls:
                f.write(requests.get(url).content)
                f.write(b"\n")
        print("########## Corpus downloaded and merged!")
    else:
        print("########## Corpus already exists.")

def train_tokenizer():
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix="llama_tokenizer",
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3
    )
    print("Tokenizer trained!")

def setup_hf_compatible_folder():
    print("Setting up HuggingFace-compatible tokenizer folder...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    os.rename("llama_tokenizer.model", TOKENIZER_MODEL)

    tokenizer_config = {
        "model_type": "bpe",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": True
    }

    with open(TOKENIZER_CONFIG, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"Tokenizer files saved in `{TOKENIZER_DIR}/`")

if __name__ == "__main__":
    download_corpus()
    train_tokenizer()
    setup_hf_compatible_folder()
    print("You're all set! Load with `LlamaTokenizer.from_pretrained('llama-tokenizer')`")
