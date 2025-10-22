import os
from dotenv import load_dotenv
# Corrected imports for modular LangChain packages
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (to get the OPENAI_API_KEY)
load_dotenv()

# Configuration constants (must match what was used in ingest.py)
CHROMA_PATH = "chroma_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Guardrail constant
MAX_OUTPUT_LENGTH = 500
# Template for answering when context is available (no guardrail needed)
ANSWER_TEMPLATE = """
You are an expert Q&A system for company policies. Your goal is to answer the user's question based ONLY on the provided context. Do not use outside knowledge.

Always include citations by listing the source document ID/title for every piece of information you provide. Limit your output length to {max_length} tokens.

--- CONTEXT ---
{context}
---

--- QUESTION ---
{question}
"""

# Template for the RAG prompt
# Project Requirement: "Add basic guardrails: Refuse to answer outside the corpus"
# Project Requirement: "Limit output length"
# Project Requirement: "Always cite source doc IDs/titles for answers."
PROMPT_TEMPLATE = """
You are an expert Q&A system for company policies. Your goal is to answer the user's question based ONLY on the provided context.

If the provided context is empty or does not contain the information needed to answer the question, you MUST respond with "I can only answer questions about our company policies and procedures." Otherwise, answer based ONLY on the context.

Always include citations by listing the source document ID/title for every piece of information you provide. Limit your output length to {max_length} tokens.

--- CONTEXT ---
{context}
---

--- QUESTION ---
{question}
"""

# --------------------------------------------------------------------------------------
# RAG Component Functions
# --------------------------------------------------------------------------------------

def setup_rag_components():
    """Sets up the LLM and loads the vector store."""
    
    # 1. Initialize the free Embedding Model (must match ingest.py)
    # Note: If this errors, you need to ensure 'sentence-transformers' is installed.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. Load the Vector Database
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Vector store not found at {CHROMA_PATH}. Did you run 'python ingest.py'?")
        return None, None
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 3. Initialize the LLM (using your OpenAI key from .env)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    return db, llm

def answer_question(question: str):
    """
    Retrieves relevant context and generates an answer using the LLM.
    """
    db, llm = setup_rag_components()
    if db is None:
        return "Setup failed: Vector store not available."

    # --- Retrieval (Top-k) ---
    results = db.similarity_search_with_score(question, k=4)
    
    # 4. Extract context and source document IDs/titles
    context_text = ""
    source_docs = set()
    for doc, score in results:
        context_text += f"{doc.page_content}\n\n"
        source_docs.add(doc.metadata.get("source", "Unknown Source"))

    # --- Generation ---
    # 5. Build the prompt with retrieved context
    if len(context_text) > 100:
        template_to_use = ANSWER_TEMPLATE
    else:
        template_to_use = PROMPT_TEMPLATE

    formatted_prompt = template_to_use.format(
        context=context_text,
        question=question,
        max_length=MAX_OUTPUT_LENGTH
    )
    prompt = ChatPromptTemplate.from_template(formatted_prompt)

    # 6. Call the LLM
    rag_chain = prompt | llm | StrOutputParser()
    response_content = rag_chain.invoke({
        "context": context_text,
        "question": question,
        "max_length": MAX_OUTPUT_LENGTH
    })

    # 7. Append citations to the final response
    citation_text = "\n\n**Sources:** " + "; ".join(sorted(list(source_docs)))
    final_answer = response_content + citation_text

    return final_answer

# --------------------------------------------------------------------------------------
# Main Execution for Testing
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Testing RAG Pipeline ---")
    
    # Test Question 1: Should be answerable from the corpus
    test_question_1 = "What is the policy regarding working remotely?"
    print(f"\n--- Question: {test_question_1} ---")
    print(answer_question(test_question_1))
    
    print("\n" + "="*50 + "\n")

    # Test Question 2: Should trigger the guardrail
    test_question_2 = "What is the capital of France?"
    print(f"--- Question: {test_question_2} ---")
    print(answer_question(test_question_2))