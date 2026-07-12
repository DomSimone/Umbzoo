import gc
import os
import torch
import random
import sys
from config import HF_TOKEN, MODELS_DIR
from train import DeepSpeedTrainer
from umbuzo_dataset_loader import download_and_prepare_data, create_dataloaders
from transformers import AutoModelForCausalLM, AutoConfig

# Mocking args for the trainer
class TrainingArgs:
    def __init__(self, output_dir):
        self.deepspeed = torch.cuda.is_available() 
        self.local_rank = -1
        self.epochs = 3 
        self.batch_size = 1 
        self.lr = 5e-5 # Slightly higher LR for partial fine-tuning
        self.output_dir = output_dir
        self.resume_from = None
        self.train_only = False 

def start_training():
    print("="*60)
    print("  UMBUZO LOCAL LLM TRAINING: OOM-RESISTANT PIPELINE")
    print("="*60)
    
    # Environment Setup
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HTTP_TIMEOUT"] = "86400" 
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Prepare Data
    print("\n[1/3] Sourcing and Tokenizing Data...")
    try:
        corpus_path = download_and_prepare_data()
        train_loader, val_loader, tokenizer = create_dataloaders(corpus_path)
    except Exception as e:
        print(f"! Data preparation failed: {e}")
        return

    # 2. Boarding base model with RAM Optimization
    print("\n[2/3] Boarding base model (Freezing layers to prevent shutdown)...")
    output_dir = os.path.join(MODELS_DIR, "africa_umbuzo_qwen_finetuned")
    os.makedirs(output_dir, exist_ok=True)
    
    # Priority on 0.5B for stability on VM CPUs, fallback from 1.5B
    model_ids = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
    model = None

    for model_id in model_ids:
        try:
            print(f"  -> Attempting to board: {model_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=HF_TOKEN,
                trust_remote_code=True,
                torch_dtype=torch.float32, # float32 is safer for CPU training
                low_cpu_mem_usage=True,
                device_map=None 
            )
            
            # --- MEMORY RECOVERY: LAYER FREEZING ---
            print("  -> Freezing 90% of model to save RAM...")
            model.requires_grad_(False) # Freeze everything first
            
            # Unfreeze the last 2 layers and the LM head for specialized training
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                num_layers = len(model.model.layers)
                for i in range(num_layers - 2, num_layers):
                    model.model.layers[i].requires_grad_(True)
                print(f"  -> Unfroze last 2 layers of {num_layers} total blocks.")
            
            if hasattr(model, 'lm_head'):
                model.lm_head.requires_grad_(True)
            
            # ---------------------------------------
            
            print(f"✓ {model_id} boarded successfully.")
            break 
        except Exception as e:
            print(f"! Boarding {model_id} failed: {e}")
            if model_id == model_ids[-1]: return
            gc.collect()

    # 3. Training
    print("\n[3/3] Commencing Fine-tuning (Optimized RAM Path)...")
    args = TrainingArgs(output_dir)
    
    try:
        trainer = DeepSpeedTrainer(args, model=model)
        print(f"✓ Boarding complete. Starting loop...")
        
        trainer.train(train_loader, val_loader)
        
        print("\n" + "="*60)
        print(f"  TRAINING COMPLETE: Model saved to {output_dir}")
        print("="*60)
        tokenizer.save_pretrained(output_dir)
        
    except Exception as e:
        print(f"\nTraining/Boarding interrupted: {e}")
        if 'trainer' in locals() and hasattr(trainer, 'save_final_model'):
            trainer.save_final_model()

if __name__ == "__main__":
    start_training()
