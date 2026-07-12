"""
Configuration file for the Umbuzo Africa-focused LLM pipeline.
Updated for Qwen-style tokenization (ThinkingCap-Qwen3.6).
"""
import os
from dataclasses import dataclass

# =============================================================================
# API KEYS & TOKENS
# =============================================================================
HF_TOKEN = "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"
OPENWEBUI_API_KEY = "sk-ELHAZ_qxBnT7-DlO9fhgYKzxizC_rfJUb0Wpu_JaSVuHQXagYyoz9UUT2oB5Sw8X"
OPENWEBUI_URL = "http://localhost:3000/api"

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = "/media/domsimone/A2643A6C643A42F9/PythonProject/data"
MODELS_DIR = "/media/domsimone/A2643A6C643A42F9/PythonProject/models"
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized")
DB_PATH = os.path.join(DATA_DIR, "umbuzo_knowledge.db")

# =============================================================================
# UMBUZO MODEL CONFIG (Scaled for 1B-2B Params)
# Using Qwen-style vocab size (~151,936)
# =============================================================================
@dataclass
class UmbuzoConfig:
    model_name: str = "Umbuzo"
    vocab_size: int = 151936 # Qwen standard vocab
    max_position_embeddings: int = 2048
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_seq_length: int = 128
    batch_size: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 3
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01

model_cfg = UmbuzoConfig()

@dataclass
class DataConfig:
    train_split: float = 0.8
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0
    max_samples: int = 500
    chunk_size: int = 1024

data_cfg = DataConfig()

TOPICS = {
    "political_civic_engagement": {"description": "Political And Civic Engagement"},
    "economic_labor_trends": {"description": "Economic And Labor Trends"},
    "urban_community_dynamics": {"description": "Urban and community dynamics"},
    "demographic_trends": {"description": "Demographic Trends"},
    "social_structures_lifestyles": {"description": "Social structures and lifestyles"}
}
