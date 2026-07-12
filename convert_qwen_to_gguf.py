import os
import subprocess
import sys
import json
import urllib.request
import zipfile
import shutil

def convert_qwen_to_gguf():
    print("="*60)
    print("  UMBUZO QWEN LLM: GGUF CONVERSION PIPELINE (TOKENIZER FIX)")
    print("="*60)

    model_path = "/media/domsimone/A2643A6C643A42F9/PythonProject/models/africa_umbuzo_qwen_finetuned"
    output_dir = "/media/domsimone/A2643A6C643A42F9/PythonProject/models/gguf"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- STEP 0: FIX TOKENIZER CLASS ---
    # Qwen2Tokenizer often causes issues in llama.cpp conversion if not mapped to Qwen2TokenizerFast
    # We update tokenizer_config.json to use the standard 'Qwen2TokenizerFast'
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        print("\n[0/3] Patching tokenizer_config.json for Qwen2 compatibility...")
        with open(config_path, "r") as f:
            t_config = json.load(f)
        
        # Change class to Fast to ensure the JSON-based vocab is used instead of looking for .model file
        t_config["tokenizer_class"] = "Qwen2TokenizerFast"
        
        with open(config_path, "w") as f:
            json.dump(t_config, f, indent=2)
        print("  -> tokenizer_class updated to Qwen2TokenizerFast.")

    # 1. Get conversion tools
    llama_cpp_path = os.path.join(os.getcwd(), "llama.cpp")
    if not os.path.exists(llama_cpp_path):
        print("\n[1/3] Sourcing llama.cpp conversion tools...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
        except:
            zip_url = "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip"
            urllib.request.urlretrieve(zip_url, "llama_cpp.zip")
            with zipfile.ZipFile("llama_cpp.zip", 'r') as z: z.extractall(".")
            extracted = [f for f in os.listdir(".") if f.startswith("llama.cpp-")][0]
            os.rename(extracted, "llama.cpp")
            os.remove("llama_cpp.zip")

    # Install/Update requirements
    print("  -> Ensuring dependencies (transformers>=4.40.0)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers>=4.40.0", "gguf", "tiktoken"], check=True)

    # 2. Convert to GGUF
    print("\n[2/3] Converting Qwen2 model to GGUF...")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    output_file = os.path.join(output_dir, "umbuzo_qwen.gguf")
    
    # Using f32 to match your model's current float32 status
    convert_cmd = [
        sys.executable, convert_script,
        model_path,
        "--outfile", output_file,
        "--outtype", "f32"
    ]
    
    print(f"  -> Running conversion...")
    try:
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"! ERROR:\n{result.stderr}")
            # If it still fails, try without outtype
            print("\n  -> Retrying without explicit outtype...")
            subprocess.run([sys.executable, convert_script, model_path, "--outfile", output_file], check=True)
        else:
            print(f"\nSUCCESS: GGUF model created at {output_file}")

        return output_file
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    convert_qwen_to_gguf()
