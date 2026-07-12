"""
Unified Trainer Script for CPU and GPU (DeepSpeed).
Handles boarding and fine-tuning across different hardware environments.
"""
import os
import sys
import json
import argparse
import math
import gc
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm

from config import MODELS_DIR, model_cfg, data_cfg, HF_TOKEN as CONFIG_HF_TOKEN

class DeepSpeedTrainer:
    """
    Boarding and training engine with support for CPU and DeepSpeed/GPU.
    """

    def __init__(self, args, model: nn.Module):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = args.epochs or model_cfg.num_epochs
        self.batch_size = args.batch_size or model_cfg.batch_size
        self.lr = args.lr or model_cfg.learning_rate
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"  -> Hardware Context: {self.device}")
        
        # Set seed
        set_seed(data_cfg.seed)

        # Board the model
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.deepspeed_engine = None
        self.use_deepspeed = args.deepspeed and torch.cuda.is_available()

        if self.use_deepspeed:
            self.setup_deepspeed()
        else:
            # For standard PyTorch (CPU/GPU), move model to device and enable gradient checkpointing
            if next(self.model.parameters()).device != self.device:
                self.model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency on CPU/single GPU
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  -> Gradient Checkpointing enabled for memory efficiency.")
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False # Incompatible with gradient checkpointing

    def setup_optimizer(self, train_loader: DataLoader):
        """Set up optimizer and scheduler for standard PyTorch (CPU/GPU)."""
        if self.optimizer is None:
            # FIX: Define no_decay list
            no_decay = ["bias", "LayerNorm.weight"]
            
            # MEMORY OPTIMIZATION: Filter for trainable parameters only
            params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                    "weight_decay": model_cfg.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = AdamW(params, lr=self.lr)
            total_steps = len(train_loader) * self.epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=model_cfg.warmup_steps,
                num_training_steps=total_steps,
            )

    def setup_deepspeed(self):
        """DeepSpeed initialization for CUDA devices."""
        ds_config = {
            "train_batch_size": self.batch_size * model_cfg.gradient_accumulation_steps,
            "gradient_accumulation_steps": model_cfg.gradient_accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {"lr": self.lr, "weight_decay": model_cfg.weight_decay}},
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 1.0,
        }

        import deepspeed
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_config,
        )
        self.deepspeed_engine = self.model

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

        # Check for BF16 support safely
        bf16_supported = False
        if self.device.type == 'cpu' and hasattr(torch.cpu, 'is_bf16_supported'):
            bf16_supported = torch.cpu.is_bf16_supported()

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_deepspeed:
                outputs = self.deepspeed_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                self.deepspeed_engine.backward(loss)
                self.deepspeed_engine.step()
            else:
                # CPU OPTIMIZATION: Use autocast for BF16 if possible, otherwise standard
                if bf16_supported:
                    with torch.cpu.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss / model_cfg.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % model_cfg.gradient_accumulation_steps == 0:
                    # Clip gradients to prevent spikes
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Periodic memory cleanup during training
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(train_loader)

    def save_final_model(self):
        print(f"Saving fine-tuned Umbuzo model to {self.output_dir}...")
        # MEMORY OPTIMIZATION during saving: set max_shard_size to prevent OOM
        if self.use_deepspeed:
            self.deepspeed_engine.save_pretrained(self.output_dir)
        else:
            self.model.save_pretrained(self.output_dir, max_shard_size="500MB")

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        if not self.use_deepspeed:
            self.setup_optimizer(train_loader)

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch+1} finished. Avg Loss: {train_loss:.4f}")
            # Epoch cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        self.save_final_model()
