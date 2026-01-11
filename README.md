# Local-RAG Document Assistant

A privacy-focused Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents locally. No data ever leaves your machine.

## Key Features
- **100% Private:** Uses local LLMs via Ollama.
- **Semantic Search:** Powered by ChromaDB and Sentence-Transformers.
- **Interactive UI:** Built with Gradio for a seamless user experience.

## Tech Stack
- **Framework:** LangChain
- **LLM:** Meta Llama 3 (via Ollama)
- **Vector DB:** ChromaDB
- **Embeddings:** Hugging Face (all-MiniLM-L6-v2)


## Why This Stack?

### Llama 3 (Large Language Model)
* **Privacy & Security:** By using **Llama 3** via **Ollama**, all data processing remains local. Sensitive PDF content never leaves your machine.
* **State-of-the-Art Performance:** Llama 3 provides high-reasoning capabilities that rival proprietary models, ensuring accurate, context-aware answers and high-quality text generation.

### ChromaDB (Vector Database)
* **Speed & Efficiency:** **ChromaDB** is optimized for high-speed similarity searches. It allows the system to find relevant text chunks in milliseconds, even as the document library grows.
* **Persistence:** It stores embeddings locally in the `chroma_db/` directory, meaning you only need to process and index your PDF once. Subsequent queries are near-instant.



## Demo
<div align="center">
  <img src="./assets/demo.png" width="800">
</div>


## System Requirements
| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **Storage** | 10 GB Free SSD Space | 20 GB Free SSD Space |
| **GPU** | Optional | NVIDIA RTX 30+ or Apple M-Series |

## Setup Instructions

1. **Install Ollama:** Download from [ollama.com](https://ollama.com)
2. **Download the Model:**
```bash
   ollama pull llama3
```
3. **Environment Setup:**
```bash
    python -m venv venv
    source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
    pip install -r requirements.txt
```

4. **Run the App:**
```bash
    python app.py
```