# Snug-Project: Retrieval-Augmented Generation (RAG) Policy Bot

This project implements a Retrieval-Augmented Generation (RAG) LLM-based application designed to answer user questions exclusively about a corpus of company policies and procedures. It uses a local vector store (ChromaDB) and the OpenAI API for generation.

---

## Project Status and Architecture

| Component | Status | Details |
| :--- | :--- | :--- |
| **Data Corpus** | Complete | 10 policy documents in various formats (PDF, TXT, MD). |
| **Ingestion** | Complete | Documents parsed, chunked, embedded with `all-MiniLM-L6-v2`, and indexed in ChromaDB. |
| **RAG Pipeline** | Complete | Implements Top-k retrieval, conditional prompting, source citation, and a guardrail against out-of-corpus questions. |
| **Web Application** | Complete | Built with Flask, providing a web UI (`/`) and an API endpoint (`/chat`). |
| **Deployment** | Pending | To be deployed using a free-tier host (e.g., Render/Railway). |
| **CI/CD** | Pending | To be implemented via GitHub Actions. |

---

## 1. Local Setup and Execution

To run this application locally, follow these steps:

### A. Environment Setup (First Time Only)

1.  **Clone the repository:**
    ```bash
    git clone [YOUR GITHUB REPO URL HERE]
    cd snug-project
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Mac/Linux
    # .venv\Scripts\activate   # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # NOTE: If you encounter issues with PDF parsing, run: pip install "unstructured[pdf]"
    ```

### B. API Key Configuration

1.  Create a file named **`.env`** in the root directory.
2.  Add your OpenAI API key for the LLM connection (GPT-3.5-turbo):
    ```
    OPENAI_API_KEY="sk-YOUR_OPENAI_KEY_HERE"
    ```

### C. Data Ingestion (Indexing)

The vector database must be built before the app can run.

1.  Place your policy documents (Policy Document 1-9 and `remote_work_policy.md`) inside the **`policies/`** directory.
2.  Run the ingestion script. This will create the searchable `chroma_data/` folder.
    ```bash
    python ingest.py
    ```

### D. Run the Web Application

Start the Flask server:

```bash
python app.py