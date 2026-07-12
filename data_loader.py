"""
Data Loader: Creates PyTorch DataLoaders with batching and 80/20 train/val split.
Handles tokenized text data for LLM training.
"""
import os
import json
import random
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from config import TOKENIZED_DIR, EXTRACTED_TEXT_DIR, model_cfg, data_cfg


class TextDataset(Dataset):
    """PyTorch Dataset for tokenized text data."""

    def __init__(self, tokenized_data: List[Dict]):
        self.data = tokenized_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["input_ids"], dtype=torch.long),
        }


class QADataset(Dataset):
    """PyTorch Dataset for Q&A training data."""

    def __init__(self, qa_pairs: List[Dict], tokenizer, max_length: int = None):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length or model_cfg.max_seq_length
        self._encoded = None

    def _encode_all(self):
        """Pre-encode all Q&A pairs."""
        self._encoded = []
        for qa in tqdm(self.qa_pairs, desc="Encoding Q&A pairs"):
            text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self._encoded.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
            })

    def __len__(self) -> int:
        if self._encoded is None:
            self._encode_all()
        return len(self._encoded)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._encoded is None:
            self._encode_all()
        return self._encoded[idx]


class DataLoaderFactory:
    """
    Creates DataLoaders with 80/20 train/val split from tokenized data.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)

    def load_tokenized_data(self, tokenized_dir: str = None) -> List[Dict]:
        """Load pre-tokenized data from files."""
        if tokenized_dir is None:
            tokenized_dir = TOKENIZED_DIR

        all_data = []
        tokenized_files = list(Path(tokenized_dir).glob("*.pt")) + \
                          list(Path(tokenized_dir).glob("*.json"))

        for fpath in tqdm(tokenized_files, desc="Loading tokenized data"):
            try:
                if fpath.suffix == ".pt":
                    data = torch.load(fpath)
                    if isinstance(data, list):
                        all_data.extend(data)
                    elif isinstance(data, dict):
                        all_data.append(data)
                elif fpath.suffix == ".json":
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        return all_data

    def load_qa_data(self, qa_dir: str = None) -> List[Dict]:
        """Load Q&A data from JSON files."""
        if qa_dir is None:
            qa_dir = os.path.join(os.path.dirname(__file__), "data", "qa_datasets")

        all_qa = []
        if os.path.isdir(qa_dir):
            for fname in os.listdir(qa_dir):
                if fname.endswith(".json"):
                    fpath = os.path.join(qa_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_qa.extend(data)
                    elif isinstance(data, dict):
                        all_qa.append(data)

        return all_qa

    def create_text_dataloaders(
        self, tokenized_data: List[Dict] = None,
        batch_size: int = None, train_split: float = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/val DataLoaders from tokenized text data."""
        if tokenized_data is None:
            tokenized_data = self.load_tokenized_data()

        if not tokenized_data:
            # Create synthetic data for testing
            print("No tokenized data found. Creating synthetic test data...")
            tokenized_data = self._create_synthetic_data()

        batch_size = batch_size or model_cfg.batch_size
        train_split = train_split or data_cfg.train_split

        dataset = TextDataset(tokenized_data)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(data_cfg.seed),
        )

        # Auto-detect workers based on system
        num_workers = min(data_cfg.num_workers, 2)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def create_qa_dataloaders(
        self, qa_pairs: List[Dict] = None,
        batch_size: int = None, train_split: float = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/val DataLoaders from Q&A data."""
        if qa_pairs is None:
            qa_pairs = self.load_qa_data()

        if not qa_pairs:
            print("No Q&A data found. Creating synthetic test data...")
            qa_pairs = self._create_synthetic_qa()

        if self.tokenizer is None:
            raise ValueError("Tokenizer required for Q&A dataset. Set tokenizer first.")

        batch_size = batch_size or model_cfg.batch_size
        train_split = train_split or data_cfg.train_split

        dataset = QADataset(qa_pairs, self.tokenizer)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(data_cfg.seed),
        )

        # Auto-detect workers based on system
        num_workers = min(data_cfg.num_workers, 2)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

        print(f"Train Q&A samples: {len(train_dataset)}, Val Q&A samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _create_synthetic_data(self, num_samples: int = 100) -> List[Dict]:
        """Create synthetic tokenized data for testing."""
        data = []
        seq_len = model_cfg.max_seq_length
        for _ in range(num_samples):
            input_ids = [random.randint(0, model_cfg.vocab_size - 1) for _ in range(seq_len)]
            attention_mask = [1] * seq_len
            data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
        return data

    def _create_synthetic_qa(self, num_pairs: int = 50) -> List[Dict]:
        """Create synthetic Q&A pairs for testing."""
        topics = ["political_civic_engagement", "economic_labor_trends",
                  "urban_community_dynamics", "demographic_trends",
                  "social_structures_lifestyles"]
        countries = ["South Africa", "Nigeria", "Kenya", "Ghana", "Ethiopia"]

        qa_pairs = []
        for i in range(num_pairs):
            topic = random.choice(topics)
            country = random.choice(countries)
            qa_pairs.append({
                "question": f"What are the key {topic.replace('_', ' ')} trends in {country}?",
                "answer": f"Analysis of {topic.replace('_', ' ')} in {country} reveals "
                         f"significant developments across multiple dimensions. "
                         f"Key factors include policy changes, demographic shifts, "
                         f"and economic conditions that shape outcomes in this area.",
                "topic": topic,
                "country": country,
                "question_type": "factual",
                "difficulty": "medium",
            })
        return qa_pairs

    def get_sample_batch(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Get a single sample batch for inspection."""
        for batch in loader:
            return batch
        return {}


def inspect_batch(batch: Dict[str, torch.Tensor]):
    """Print information about a batch."""
    print("\n" + "=" * 60)
    print("BATCH INSPECTION")
    print("=" * 60)
    for key, tensor in batch.items():
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
        if tensor.numel() < 20:
            print(f"  values: {tensor}")
    print()


if __name__ == "__main__":
    factory = DataLoaderFactory()
    train_loader, val_loader = factory.create_text_dataloaders()
    batch = factory.get_sample_batch(train_loader)
    if batch:
        inspect_batch(batch)