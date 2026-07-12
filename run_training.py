import sys
import os

# Set environment variables for minimal resource usage if needed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from main import run_pipeline

if __name__ == "__main__":
    print("Starting Umbuzo LLM Training Pipeline...")
    
    # We run tokenizer training first, then the model training
    # 'train' step in main.py calls train.py with minimal settings
    steps = ["tokenizer", "train"]
    
    try:
        results = run_pipeline(steps)
        print("\nTraining Pipeline Completed Successfully!")
    except Exception as e:
        print(f"\nTraining Pipeline Failed: {e}")
        sys.exit(1)
