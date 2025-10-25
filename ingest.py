import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma


# --- Configuration ---

# 1. Load Environment Variables (for API keys, if you use a paid model later)
load_dotenv()

# 2. Define data and storage paths
CHROMA_PATH = "chroma_data"
DATA_PATH = "policies"

# --- Ingestion Functions ---

def load_documents():
    """Load all policy documents from the 'policies' directory."""
    
    # Use DirectoryLoader and specify a glob pattern to load all files
    # The 'loader_cls=UnstructuredFileLoader' ensures complex files like PDFs are handled correctly.
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages/splits.")
    return documents

def split_text(documents):
    """Split documents into smaller, more precise, overlapping chunks."""

    # CRITICAL FIX: Smaller chunk size to avoid context contamination
    # A smaller chunk size forces the retriever to be more precise.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # NEW: Down from 500
        chunk_overlap=30,      # NEW: Down from 50
        # NEW: Specify separators to prioritize splitting at double newlines (paragraphs/sections)
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_embeddings(chunks):
    """Create embeddings and store them in the vector database."""
    
    # Project Requirement: "Use free or zero-cost options when possible."
    # HuggingFaceEmbeddings with a local model is a great free-tier option.
    # The 'all-MiniLM-L6-v2' model is fast and highly effective.
    # Use OpenAI's text-embedding-3-small model (API-based)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Clear out any existing database contents for a clean run
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print(f"Removed old database at {CHROMA_PATH}")

    # Project Requirement: "Store the embedded document chunks in a local or lightweight vector database (e.g. Chroma)."
    # Create the vector database and add the chunks
    print(f"Creating new vector store at {CHROMA_PATH}...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print("Ingestion complete!")

# --- Main Execution ---

if __name__ == "__main__":
    policy_documents = load_documents()
    text_chunks = split_text(policy_documents)
    create_embeddings(text_chunks)