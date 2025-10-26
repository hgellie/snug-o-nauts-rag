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
    
    # 4. Process retrieved documents
    if not results:
        return "I can only answer questions about our company policies and procedures."
    
    def get_normalized_tokens(text: str) -> set:
        """Extract normalized tokens from text."""
        text = text.replace('**', '').replace('"', '"').replace('"', '"')
        tokens = set()
        for word in text.lower().replace('(', ' ').replace(')', ' ').split():
            word = word.strip('.,;:?!')
            if word and len(word) > 1 and not word.isnumeric():
                tokens.add(word)
        return tokens

    # Enhanced document scoring with n-gram matching and semantic relevance
    scores = []
    question_tokens = get_normalized_tokens(question)
    
    # Extract important phrases from question (2-3 word sequences)
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    question_bigrams = get_ngrams(question, 2)
    question_trigrams = get_ngrams(question, 3)
    
    for doc in results:
        # Basic token analysis
        doc_tokens = get_normalized_tokens(doc.page_content)
        doc_content = doc.page_content.lower()
        
        # 1. Token Overlap Score (35%)
        shared_tokens = question_tokens & doc_tokens
        token_overlap = len(shared_tokens) / len(question_tokens) if question_tokens else 0
        
        # 2. Phrase Matching Score (25%)
        doc_bigrams = get_ngrams(doc_content, 2)
        doc_trigrams = get_ngrams(doc_content, 3)
        bigram_matches = len(question_bigrams & doc_bigrams) / len(question_bigrams) if question_bigrams else 0
        trigram_matches = len(question_trigrams & doc_trigrams) / len(question_trigrams) if question_trigrams else 0
        phrase_score = (bigram_matches + trigram_matches) / 2
        
        # 3. Key Terms Presence (20%)
        key_terms = set(word.lower() for word in question.split() if len(word) > 3)
        term_matches = sum(1 for term in key_terms if term in doc_content)
        term_score = term_matches / len(key_terms) if key_terms else 0
        
        # 4. Answer Completeness (20%)
        completeness = min(len(doc_tokens) / 100, 1.0)
        
        # Calculate final weighted score
        final_score = (
            token_overlap * 0.35 +
            phrase_score * 0.25 +
            term_score * 0.20 +
            completeness * 0.20
        )
        
        # Store detailed scoring for analysis
        scores.append({
            'doc': doc,
            'score': final_score,
            'token_overlap': token_overlap,
            'phrase_score': phrase_score,
            'term_score': term_score,
            'completeness': completeness
        })
    
    # Sort by final score
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Enhanced document selection with multi-factor analysis
    best_score = scores[0]
    if len(scores) > 1:
        # Consider top 3 candidates if available
        candidates = scores[:min(3, len(scores))]
        
        # Find documents with strong token overlap or phrase matching
        strong_matches = [
            s for s in candidates
            if s['token_overlap'] >= 0.4 or s['phrase_score'] >= 0.5
        ]
        
        if strong_matches:
            # Among strong matches, prefer the one with the best balance
            best_score = max(
                strong_matches,
                key=lambda x: (x['token_overlap'] * 0.6 + x['phrase_score'] * 0.4)
            )
        else:
            # If no strong matches, look for best potential match
            for candidate in candidates[1:]:
                # Check if significantly better in any metric
                if (candidate['token_overlap'] > best_score['token_overlap'] * 1.4 or
                    candidate['phrase_score'] > best_score['phrase_score'] * 1.3 or
                    (candidate['score'] > best_score['score'] * 0.9 and 
                     candidate['completeness'] > best_score['completeness'] * 1.2)):
                    best_score = candidate
    
    best_doc = best_score['doc']
    
    # Include full context from the best document
    context_text = best_doc.page_content
    source_doc = best_doc.metadata.get("source", "Unknown Source")

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

    # 7. Append citation to the final response
    citation_text = f"\n\n**Sources:** {source_doc}"
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