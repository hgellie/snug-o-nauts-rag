import os
from dotenv import load_dotenv
# Corrected imports for modular LangChain packages
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Use the simple string output from the model; no parser required here

# Load environment variables (to get the OPENAI_API_KEY)
load_dotenv()

# Configuration constants (must match what was used in ingest.py)
CHROMA_PATH = "chroma_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Guardrail constant
MAX_OUTPUT_LENGTH = 500
# Template for answering when context is available (no guardrail needed)
ANSWER_TEMPLATE = """
You are an expert Q&A system for Snug-O-Nauts company policies. Your goal is to answer the user's question based **ONLY** on the provided context.

**RULE 1: FAITHFULNESS & DISAMBIGUATION.** Answer the user's question concisely, directly, and using ONLY the provided text in the 'CONTEXT' section. **DO NOT invent information, combine details from different roles, mix protocols, or assume numerical ranges.** If the context does not fully support the answer, do not guess.
**RULE 2: CITATION.** You MUST cite all sources used at the end of the answer. A citation is only valid if the document directly supports a fact used in the answer. Do not include irrelevant documents in the citations.


Limit your output length to **two sentences or a concise list**.

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
You are an expert Q&A system for Snug-O-Nauts company policies. Your goal is to answer the user's question based **ONLY** on the provided context.

**RULE 1: FAITHFULNESS & DISAMBIGUATION.** Answer the user's question concisely, directly, and using ONLY the provided text in the 'CONTEXT' section. **DO NOT invent information, combine details from different roles, mix protocols, or assume numerical ranges.** If the context does not fully support the answer, do not guess.
**RULE 2: CITATION.** You MUST cite all sources used at the end of the answer. A citation is only valid if the document directly supports a fact used in the answer. Do not include irrelevant documents in the citations.
**RULE 3: GUARDRAIL.** If the CONTEXT is empty or does not contain the answer, you MUST respond with "I can only answer questions about our company policies and procedures."

Limit your output length to **two sentences or a concise list**.

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
    """Sets up the LLM, vector store, and the ContextualCompressionRetriever."""
    
    # 1. Initialize the Embedding Model (must match ingest.py)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Load the Vector Database
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Vector store not found at {CHROMA_PATH}. Did you run 'python ingest.py'?")
        return None, None, None # Must return the same number of Nones as successful outputs
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 3. Initialize the LLM (using your OpenAI key from .env)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # 4. Create a simple retriever from the vector DB.
    # Note: the environment for this project may not include specialized
    # rerankers or compressors; use the standard retriever which works
    # with the Chroma vector store created by `ingest.py`.
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # FINAL RETURN: Return the DB, LLM and retriever
    return db, llm, retriever

def answer_question(question: str):
    """
    Retrieves relevant context and generates an answer using the LLM.
    """
    db, llm, compression_retriever = setup_rag_components()
    if db is None:
        return "Setup failed: Vector store not available."

    # --- Retrieval (Top-k) ---
    # Use the retriever returned by setup_rag_components to fetch relevant docs
    results = compression_retriever.get_relevant_documents(question)
    
    # 4. Extract context and source document IDs/titles
    context_text = ""
    source_docs = set()
    for doc in results: # Results are now just Document objects, not (Document, score) tuples
        context_text += f"{doc.page_content}\n\n"
        source_docs.add(doc.metadata.get("source", "Unknown Source"))

    # --- Generation ---
    # 5. Build the prompt with retrieved context
    if len(context_text) > 100:
        template_to_use = ANSWER_TEMPLATE
    else:
        template_to_use = PROMPT_TEMPLATE

    # Create a ChatPromptTemplate and format it into a PromptValue that the LLM can consume
    prompt = ChatPromptTemplate.from_template(template_to_use)
    prompt_value = prompt.format_prompt(context=context_text, question=question)

    # 6. Call the LLM. `invoke` returns a BaseMessage; use its `.content` attribute.
    llm_response = llm.invoke(prompt_value)
    response_content = getattr(llm_response, "content", str(llm_response))

    # 7. Append citations to the final response
    citation_text = "\n\n**Sources:** " + "; ".join(sorted(list(source_docs))) if source_docs else ""
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