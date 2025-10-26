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

**RULE 1: EXACT MATCH PRIORITY.** First, look for exact phrases or statements in the context that directly answer the question. Prefer using these exact matches in your answer, preserving the original wording where possible.

**RULE 2: COMPLETENESS & PRECISION.** If no exact match exists, ensure your answer includes ALL key details from the context that are relevant to the question. Be precise and specific, using the same terminology as found in the source document.

**RULE 3: SINGLE SOURCE FIDELITY.** Use information from exactly ONE source document. If you cannot find a complete answer in the provided context, state "I need more context to fully answer this question."

**RULE 4: NO INFERENCE.** Do not make assumptions or combine information from different sections. Stick strictly to what is explicitly stated in the context.


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
"""

# --- Function definition moved below ---
def answer_question(
    question: str,
    use_mmr: bool = True,
    use_ngrams: bool = True,
    k: int = 8,
    use_weighted_scoring: bool = True
) -> str:
    """
    Retrieves relevant context and generates an answer using the LLM.
    Supports ablation study by toggling retrieval and scoring components.
    """
    db, llm, _ = setup_rag_components()
    if db is None:
        return "Setup failed: Vector store not available."
    search_kwargs = {"k": k}
    # Chroma retriever no longer supports 'search_type', 'lambda_mult', 'fetch_k'. Only pass 'k'.
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    results = retriever.get_relevant_documents(question)
    if not results:
        return "I couldn't find any relevant information in our policy documents. Could you rephrase your question?"
    def get_normalized_tokens(text: str) -> set:
        text = text.replace('**', '').replace('"', '"').replace('"', '"')
        tokens = set()
        for word in text.lower().replace('(', ' ').replace(')', ' ').split():
            word = word.strip('.,;:?!')
            if word and len(word) > 1 and not word.isnumeric():
                tokens.add(word)
        return tokens
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    scores = []
    question_tokens = get_normalized_tokens(question)
    if use_ngrams:
        question_bigrams = get_ngrams(question, 2)
        question_trigrams = get_ngrams(question, 3)
    else:
        question_bigrams = set()
        question_trigrams = set()
    for doc in results:
        doc_tokens = get_normalized_tokens(doc.page_content)
        doc_content = doc.page_content.lower()
        shared_tokens = question_tokens & doc_tokens
        token_overlap = len(shared_tokens) / len(question_tokens) if question_tokens else 0
        if use_ngrams:
            doc_bigrams = get_ngrams(doc_content, 2)
            doc_trigrams = get_ngrams(doc_content, 3)
            bigram_matches = len(question_bigrams & doc_bigrams) / len(question_bigrams) if question_bigrams else 0
            trigram_matches = len(question_trigrams & doc_trigrams) / len(question_trigrams) if question_trigrams else 0
            phrase_score = (bigram_matches + trigram_matches) / 2
        else:
            phrase_score = 0
        key_terms = set(word.lower() for word in question.split() if len(word) > 3)
        term_matches = sum(1 for term in key_terms if term in doc_content)
        term_score = term_matches / len(key_terms) if key_terms else 0
        completeness = min(len(doc_tokens) / 100, 1.0)
        if use_weighted_scoring:
            final_score = (
                token_overlap * 0.35 +
                phrase_score * 0.25 +
                term_score * 0.20 +
                completeness * 0.20
            )
        else:
            final_score = token_overlap
        scores.append({
            'doc': doc,
            'score': final_score,
            'token_overlap': token_overlap,
            'phrase_score': phrase_score,
            'term_score': term_score,
            'completeness': completeness
        })
    scores.sort(key=lambda x: x['score'], reverse=True)
    best_score = scores[0]
    if len(scores) > 1:
        candidates = scores[:min(3, len(scores))]
        strong_matches = [
            s for s in candidates
            if s['token_overlap'] >= 0.4 or s['phrase_score'] >= 0.5
        ]
        if strong_matches:
            best_score = max(
                strong_matches,
                key=lambda x: (x['token_overlap'] * 0.6 + x['phrase_score'] * 0.4)
            )
        else:
            for candidate in candidates[1:]:
                if (candidate['token_overlap'] > best_score['token_overlap'] * 1.4 or
                    candidate['phrase_score'] > best_score['phrase_score'] * 1.3 or
                    (candidate['score'] > best_score['score'] * 0.9 and 
                     candidate['completeness'] > best_score['completeness'] * 1.2)):
                    best_score = candidate
    best_doc = best_score['doc']
    context_text = best_doc.page_content
    source_doc = best_doc.metadata.get("source", "Unknown Source")
    if len(context_text) > 100:
        template_to_use = ANSWER_TEMPLATE
    else:
        template_to_use = PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template_to_use)
    prompt_value = prompt.format_prompt(context=context_text, question=question)
    llm_response = llm.invoke(prompt_value)
    response_content = getattr(llm_response, "content", str(llm_response))
    citation_text = f"\n\n**Sources:** {source_doc}"
    final_answer = response_content + citation_text
    return final_answer

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
    # Use MMR with optimized parameters for better groundedness
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,           # Get more candidates
            "fetch_k": 20,    # Larger candidate pool
            "lambda_mult": 0.85  # Stronger emphasis on relevance vs diversity
        }
    )

    # FINAL RETURN: Return the DB, LLM and retriever
    return db, llm, retriever
