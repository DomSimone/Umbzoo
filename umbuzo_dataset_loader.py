"""
Umbuzo Dataset Downloader and Tokenizer.
Updated for ThinkingCap-Qwen3.6 (Qwen-based architecture).
Includes robustness fixes for empty dataset generation.
"""
import os
import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import login

# Set environment variables for HF
from config import HF_TOKEN, DATA_DIR, model_cfg, data_cfg
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

class UmbuzoTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.input_ids[idx], dtype=torch.long)
        }

def download_and_prepare_data():
    print("="*60)
    print("  SOURCING ENHANCED KNOWLEDGE DATASETS FROM HF HUB")
    print("="*60)
    
    try:
        login(token=HF_TOKEN, overwrite=True)
        print("✓ Logged into Hugging Face Hub.")
    except Exception as e:
        print(f"! Login warning: {e}")
    
    corpus_texts = []
    africa_keywords = ["Africa", "South Africa", "Nigeria", "Kenya", "Egypt", "Ghana", "Ethiopia"]

    # Loading clean Wikipedia subsets
    print("\n[1/2] Loading Clean Wikipedia...")
    try:
        clean_wiki = load_dataset("DragonLLM/Clean-Wikipedia-English-Articles", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for entry in clean_wiki:
            text = entry.get('text', '')
            if text and any(kw.lower() in text.lower() for kw in africa_keywords):
                corpus_texts.append(text)
                count += 1
            if count >= 2000:
                break
        print(f"  -> Extracted {count} articles.")
    except Exception as e:
        print(f"  -> Error: {e}")

    # Loading regional Finance
    print("\n[2/2] Loading regional Finance articles...")
    try:
        finance_wiki = load_dataset("DragonLLM/Wikipedia-Finance-Articles", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for entry in finance_wiki:
            text = entry.get('text', '')
            if text:
                corpus_texts.append(text)
                count += 1
            if count >= 1000:
                break
        print(f"  -> Extracted {count} articles.")
    except Exception as e:
        print(f"  -> Error: {e}")

    if not corpus_texts:
        print("\n! Warning: Online datasets returned no content. Using fallback knowledge base.")
        corpus_texts.append("Africa is a continent with a diverse history and economy. It consists of 54 countries including South Africa, Nigeria, Kenya, and Ghana. The African Union (AU) promotes integration and development across the continent. Economic trends show growth in technology and agriculture sectors.")
        corpus_texts.append("The history of Africa includes ancient civilizations such as Egypt and Great Zimbabwe. Modern challenges include urbanization, infrastructure development, and education. African cultures are rich in music, art, and oral traditions.")

    full_text = "\n\n".join(corpus_texts)
    corpus_path = os.path.join(DATA_DIR, "umbuzo_qwen_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"\nCorpus saved to {corpus_path} (Size: {len(full_text)} chars)")
    del corpus_texts
    gc.collect()
    return corpus_path

def create_dataloaders(corpus_path):
    # Double check if file is empty
    if os.path.getsize(corpus_path) == 0:
        print(f"! Error: {corpus_path} is empty. Writing fallback content.")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("Fallback knowledge content about Africa for LLM training.")

    print("\nInitializing Qwen-style Tokenizer...")
    
    model_id = "Qwen/Qwen2.5-1.5B" 
    
    try:
        print(f"Attempting to load tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=HF_TOKEN, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Primary load failed: {e}. Trying generic gpt2 as absolute fallback.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", token=HF_TOKEN) # Pass token here too

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"✓ Tokenizer {tokenizer.__class__.__name__} initialized.")

    print("\nTokenizing corpus for local LLM...")
    dataset = load_dataset("text", data_files={"train": corpus_path})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=model_cfg.max_seq_length
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    if len(tokenized_datasets["train"]) < 2:
        print("! Dataset too small for 80/20 split. Duplicating content.")
        split_dataset = {"train": tokenized_datasets["train"], "test": tokenized_datasets["train"]}
    else:
        split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2, seed=data_cfg.seed)
    
    train_loader = DataLoader(UmbuzoTextDataset(split_dataset["train"]), batch_size=model_cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(UmbuzoTextDataset(split_dataset["test"]), batch_size=model_cfg.batch_size, num_workers=0)
    
    print(f"DataLoaders ready. Vocab size: {len(tokenizer)}")
    
    try:
        sample = next(iter(train_loader))
        print("\n--- SAMPLE TOKENIZED OUTPUT ---")
        print(f"Input IDs Shape: {sample['input_ids'].shape}")
        print(f"Decoded snippet: {tokenizer.decode(sample['input_ids'][0][:20])}")
        print("--------------------------------\n")
    except StopIteration:
        print("! Warning: Train loader is empty.")
    
    return train_loader, val_loader, tokenizer # Return tokenizer as well

if __name__ == "__main__":
    path = download_and_prepare_data()
    create_dataloaders(path)
