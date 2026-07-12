# Umbuzo - Africa-Focused LLM & Knowledge Base

Umbuzo is a comprehensive platform for processing, training, and interacting with African-focused knowledge. It combines a data preprocessing pipeline, a custom LLM training workflow, and a modern chat interface.

## 🌟 Features

- **Data Extraction**: Extracts text from PDFs and scrapes high-quality web sources for African-specific content.
- **Q&A Generation**: Uses complex reasoning templates to generate synthetic datasets focused on African politics, economics, urban dynamics, and demographics.
- **Custom Pipeline**: End-to-end orchestration from raw data to tokenization and dataloaders.
- **Knowledge Base**: SQLite-backed storage for structured storage of topics, countries, and Q&A pairs.
- **Chat Interface**: A FastAPI backend providing a web UI for interacting with the knowledge base and integrated LLMs.
- **OpenWebUI Integration**: Seamless connection to OpenWebUI for advanced LLM capabilities.

---

## 🛠️ Data Training Workflow

The project follows a multi-step pipeline to transform raw information into a trainable dataset:

1.  **Extraction**: PDF documents in the `TEXTS_DIR` are processed to extract text.
2.  **Scraping**: High-quality African news and historical sites are scraped for the latest information.
3.  **Tokenization**: A GPT-2 style BPE tokenizer is trained on the entire corpus to handle African languages and terminology.
4.  **Q&A Generation**: A reasoning engine generates diverse questions (factual, comparative, causal, etc.) based on the corpus.
5.  **Database Sync**: All processed data, including topics and country-specific mappings, are synced to an SQLite database.
6.  **DataLoaders**: Preparation of training and validation sets with an 80/20 split for model training.

To run the full pipeline:
```bash
python main.py --steps extract scrape tokenizer qa database dataloaders
```

---

## 🚀 Web User Interface

The Umbuzo web interface allows users to chat with the system, explore the knowledge base, and manage chat history.

### Accessing the UI
1. Start the backend server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:8000`.

### Features in UI
- **Anonymous Chat**: Start chatting immediately as a guest.
- **User Accounts**: Register/Login to save chat history across sessions.
- **GPT Tools**: Access specialized assistants like "Data Analyst" or "Policy Advisor."
- **Search**: Search the local knowledge base directly for verified facts.

---

## 🔑 API Activation & Configuration

Umbuzo can be configured to use local knowledge or external LLMs via environment variables or the `config.py` file.

### Environment Variables
Set these variables in your environment to override defaults:
- `OPENWEBUI_URL`: URL of your OpenWebUI instance (default: `http://localhost:3000/api`).
- `OPENWEBUI_API_KEY`: Your OpenWebUI API Key.
- `HF_TOKEN`: HuggingFace Token for model access.
- `LLM_API_URL`: OpenAI-compatible endpoint for LLM completions.
- `LLM_API_KEY`: API key for the LLM service.

### Project Config (`config.py`)
The `config.py` file contains the central configuration for API keys and paths.

```python
# Keys are pre-configured for the project workflow
HF_TOKEN = "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"
OPENWEBUI_API_KEY = "sk-ELHAZ_qxBnT7-DlO9fhgYKzxizC_rfJUb0Wpu_JaSVuHQXagYyoz9UUT2oB5Sw8X"
```

---

## 📂 Project Structure

- `app.py`: FastAPI backend and web server.
- `main.py`: Pipeline orchestration script.
- `config.py`: Global configuration and hyperparameters.
- `scraper.py`: Web scraping logic for African sources.
- `qa_generator.py`: Synthetic Q&A data generation engine.
- `sql_database.py`: Database interface and semantic search.
- `static/`: Frontend assets (HTML, CSS, JS).
- `data/`: Local storage for datasets, database, and tokenizers.

---

## 📝 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

*Umbuzo - Empowering African Knowledge through AI.*