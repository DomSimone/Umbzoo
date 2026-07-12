"""
Umbuzo - Africa-focused LLM Chat Interface Backend
FastAPI server with user auth, chat management, and remote GGUF integration from Hugging Face.
"""
import os
import json
import sqlite3
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import httpx
from huggingface_hub import hf_hub_download

# GGUF Inference support
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

from config import (
    DB_PATH, DATA_DIR, MODELS_DIR,
    OPENWEBUI_API_KEY as CONFIG_OPENWEBUI_API_KEY,
    OPENWEBUI_URL as CONFIG_OPENWEBUI_URL,
    HF_TOKEN
)

# =============================================================================
# Configuration
# =============================================================================
SECRET_KEY = "umbuzo-secret-key-change-in-production-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Guest / anonymous user config
GUEST_USERNAME = "_guest"
GUEST_SESSION_ID = "anonymous"

# Hugging Face Model Config
HF_REPO_ID = "DomSimone/mbuzo2"
HF_FILENAME = "umbuzo_qwen.gguf"

# =============================================================================
# GGUF Model Management
# =============================================================================
local_llm = None
chat_histories = {}

def download_and_load_model():
    """Downloads the GGUF model from HF if not present and mounts it."""
    global local_llm
    if not HAS_LLAMA_CPP:
        print("! llama-cpp-python not installed. Skipping local LLM mount.")
        return

    try:
        print(f"Checking for Umbuzo model in Hugging Face: {HF_REPO_ID}...")
        # Download from HF Hub using the provided token
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN,
            cache_dir=os.path.join(MODELS_DIR, "hf_cache")
        )
        
        print(f"✓ Model ready at {model_path}. Mounting engine...")
        local_llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=max(1, os.cpu_count() // 2),
            verbose=False
        )
        print("✓ Umbuzo-Qwen GGUF mounted successfully.")
    except Exception as e:
        print(f"! Failed to download or mount GGUF from Hugging Face: {e}")

def generate_gguf_response(chat_id: int, user_message: str) -> str:
    if local_llm is None:
        return None
        
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
        
    history = chat_histories[chat_id][-3:]
    prompt = "You are Umbuzo, a specialized AI assistant with deep knowledge of African history, politics, and culture. Use context and be precise.\n\n"
    for turn in history:
        prompt += f"Human: {turn['q']}\nUmbuzo: {turn['a']}\n"
    
    prompt += f"Human: {user_message}\nUmbuzo:"
    
    try:
        response = local_llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["Human:", "\n\n"],
            echo=False
        )
        
        ai_text = response["choices"][0]["text"].strip()
        chat_histories[chat_id].append({"q": user_message, "a": ai_text})
        return ai_text
    except Exception as e:
        print(f"! GGUF Inference Error: {e}")
        return None

# =============================================================================
# Security & Database Setup
# =============================================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)
DB_PATH_APP = os.path.join(DATA_DIR, "umbuzo_app.db")

def init_app_database():
    conn = sqlite3.connect(DB_PATH_APP)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE, hashed_password TEXT, full_name TEXT, is_guest INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login TIMESTAMP)")
    cursor.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, title TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, is_archived INTEGER DEFAULT 0)")
    cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()

# Initialize on import
init_app_database()

# =============================================================================
# FastAPI Models & Routes
# =============================================================================

class ChatCreate(BaseModel):
    title: Optional[str] = None

class MessageCreate(BaseModel):
    content: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_and_load_model()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/chats")
async def create_chat(chat: ChatCreate):
    conn = sqlite3.connect(DB_PATH_APP)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_id, title) VALUES (1, ?)", (chat.title or "New Chat",))
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": chat_id, "title": chat.title or "New Chat"}

@app.post("/api/chats/{chat_id}/messages")
async def create_message(chat_id: int, message: MessageCreate):
    ai_response = generate_gguf_response(chat_id, message.content)
    
    if not ai_response:
        return JSONResponse(
            status_code=503,
            content={"detail": "Umbuzo engine is downloading from Hugging Face. Please refresh in 60 seconds."}
        )

    return {
        "role": "assistant",
        "content": ai_response,
        "created_at": datetime.utcnow().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
