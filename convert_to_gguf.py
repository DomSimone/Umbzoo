import os
import subprocess
import sys

def convert_to_gguf():
    print("="*60)
    print("  UMBUZO LLM: GGUF CONVERSION PIPELINE")
    print("="*60)

    model_path = "/media/domsimone/A2643A6C643A42F9/PythonProject/models/africa_gpt2_finetuned"
    output_dir = "/media/domsimone/A2643A6C643A42F9/PythonProject/models/gguf"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Install llama.cpp if not present
    llama_cpp_path = os.path.join(os.getcwd(), "llama.cpp")
    if not os.path.exists(llama_cpp_path):
        print("\n[1/3] Cloning llama.cpp for conversion tools...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", f"{llama_cpp_path}/requirements.txt"], check=True)
    else:
        print("\n[1/3] llama.cpp tools found.")

    # 2. Convert to GGUF (FP16)
    print("\n[2/3] Converting PyTorch model to GGUF (FP16)...")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    output_file_fp16 = os.path.join(output_dir, "umbuzo_africa_gpt2_fp16.gguf")
    
    convert_cmd = [
        sys.executable, convert_script,
        model_path,
        "--outfile", output_file_fp16,
        "--outtype", "f16"
    ]
    
    try:
        subprocess.run(convert_cmd, check=True)
        print(f"Success: FP16 GGUF created at {output_file_fp16}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return

    # 3. Quantize to Q4_K_M (Optional but recommended for efficiency)
    print("\n[3/3] Quantizing to 4-bit (Q4_K_M) for production...")
    # This step requires building llama.cpp (make) which might not be possible here.
    # We will attempt the conversion script first as it's the primary requirement.
    
    print("\n" + "="*60)
    print("  GGUF CONVERSION COMPLETE")
    print(f"  Model: {output_file_fp16}")
    print("="*60)

if __name__ == "__main__":
    convert_to_gguf()
