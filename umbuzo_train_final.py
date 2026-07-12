"""
Umbuzo LLM: Training and Fine-tuning script.
Integrates DeepSpeed, Mixed Precision, and Gradient Checkpointing.
"""
import os
import torch
import deepspeed
from torch.utils.data import DataLoader
from umbuzo_model import UmbuzoLLM
from umbuzo_dataset_loader import download_and_prepare_data, create_dataloaders
from config import model_cfg, data_cfg, HF_TOKEN

def train():
    print("Starting Umbuzo LLM Training System...")
    
    # 1. Prepare Data
    corpus_path = download_and_prepare_data()
    train_loader, val_loader = create_dataloaders(corpus_path)
    
    # 2. Initialize Model
    model = UmbuzoLLM(model_cfg)
    
    # 3. DeepSpeed Configuration
    ds_config = {
        "train_batch_size": model_cfg.batch_size * model_cfg.gradient_accumulation_steps,
        "gradient_accumulation_steps": model_cfg.gradient_accumulation_steps,
        "fp16": {
            "enabled": True,
            "auto_cast": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": model_cfg.learning_rate,
                "weight_decay": model_cfg.weight_decay
            }
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10
    }

    # 4. Initialize DeepSpeed Engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # 5. Training Loop
    print("\nCommencing Training Loop...")
    model_engine.train()
    for epoch in range(model_cfg.num_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            outputs, loss = model_engine(input_ids, labels=labels)
            
            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")

    print("Training Complete. Saving Umbuzo LLM...")
    model_engine.save_checkpoint(os.path.join(os.getcwd(), "models/umbuzo_final"))

if __name__ == "__main__":
    # Ensure HF Token is in env
    os.environ["HF_TOKEN"] = HF_TOKEN
    train()
