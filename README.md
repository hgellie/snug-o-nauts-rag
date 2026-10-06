# Snug-Project: Retrieval-Augmented Generation (RAG) Policy Bot

This project implements a Retrieval-Augmented Generation (RAG) LLM-based application designed to answer user questions exclusively about a corpus of company policies and procedures. It uses a local vector store (ChromaDB) and the OpenAI API for both embeddings (`text-embedding-3-small`) and generation (`gpt-3.5-turbo`).

## Features

- 🔍 **Smart Document Retrieval**: Uses semantic search to find relevant policy information
- 🤖 **AI-Powered Answers**: Leverages OpenAI's GPT models for natural language understanding
- 📚 **Source Citations**: Every answer includes references to source documents
- 🛡️ **Built-in Guardrails**: Prevents answers from outside the policy corpus
- 🌐 **User-Friendly Interface**: Clean web UI for easy interaction
- 🔗 **API Access**: RESTful endpoints for programmatic access

## Project Architecture

| Component | Status | Details |
| :--- | :--- | :--- |
| **Data Corpus** | ✅ | Policy documents in various formats (PDF, TXT, MD) |
| **Ingestion** | ✅ | Documents chunked & embedded using OpenAI `text-embedding-3-small` |
| **Vector Store** | ✅ | ChromaDB for efficient similarity search |
| **RAG Pipeline** | ✅ | Top-k retrieval with source citation |
| **Web Application** | ✅ | Flask-based UI and API endpoints |
| **CI/CD** | ✅ | GitHub Actions for automated testing |

## Quick Start

### 1. Prerequisites

- Python 3.12+
- OpenAI API key
- Git

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/hgellie/snug-o-nauts-rag.git
cd snug-project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY="your-api-key-here"
```

2. Place your policy documents in the `policies/` directory

### 4. Data Ingestion

```bash
python ingest.py
```

### 5. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## API Endpoints

### Web Interface
- `GET /` - Web UI for interactive Q&A
- `GET /health` - Health check endpoint

### API
- `POST /chat`
  ```json
  {
    "question": "What is the remote work policy?"
  }
  ```
  Returns:
  ```json
  {
    "answer": "...",
    "citations": ["policy_doc.md"],
    "question": "What is the remote work policy?"
  }
  ```

## CI/CD Pipeline

The project includes automated checks via GitHub Actions:

- ✅ Python syntax validation
- ✅ Code quality checks (flake8)
- ✅ Dependency installation verification
- ✅ File structure validation

The workflow runs on:
- Push to main branch
- Pull requests to main branch

## Project Structure

```
snug-project/
├── .github/
│   └── workflows/
│       └── python-ci.yml    # CI/CD configuration
├── chroma_data/            # Vector database (generated)
├── policies/              # Source policy documents
├── app.py                # Flask web application
├── evaluate.py           # Evaluation scripts
├── ingest.py            # Document ingestion
├── rag_pipeline.py      # Core RAG implementation
└── requirements.txt     # Python dependencies
```

## Development

### Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test locally

3. Create a pull request to main

The CI pipeline will automatically run checks on your PR.

### Running Tests Locally

```bash
# Install test dependencies
pip install flake8 pytest

# Run linting
flake8 .

# Validate syntax
python -m py_compile app.py rag_pipeline.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details