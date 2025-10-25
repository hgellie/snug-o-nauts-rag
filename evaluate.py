import time
import json
import numpy as np
import os
from rag_pipeline import answer_question # Your core RAG logic

EVAL_DATA_PATH = "evaluation_data.json"
RESULTS_FILE_PATH = "evaluation_results_raw.json"

def load_evaluation_data():
    """Loads the test questions and gold answers."""
    if not os.path.exists(EVAL_DATA_PATH):
        raise FileNotFoundError(f"Evaluation data not found at: {EVAL_DATA_PATH}. Please create the file.")
    with open(EVAL_DATA_PATH, 'r') as f:
        return json.load(f)

def run_evaluation():
    data = load_evaluation_data()
    latency_times = []
    results = []

    print(f"Starting evaluation on {len(data)} questions...")

    for i, item in enumerate(data):
        start_time = time.time()
        
        # 1. Run the RAG query
        rag_answer = answer_question(item['query']) # Using 'query' key from your JSON
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latency_times.append(latency)

        # 2. Collect result for manual quality check
        result = {
            "id": item['id'],
            "query": item['query'],
            "ground_truth": item['ground_truth'],
            "expected_source": item['source_document'],
            "rag_answer": rag_answer,
            "latency_ms": f"{latency:.2f}",
            "query_type": item['query_type'],
            # Placeholders for manual scoring (REQUIRED BY RUBRIC):
            "manual_score_groundedness": None, # (e.g., True/False or 1/0)
            "manual_score_citation_accuracy": None, # (e.g., True/False or 1/0)
            "manual_score_exact_match": None, # (Optional: True/False or 1/0)
        }
        results.append(result)
        print(f"Completed query {i+1}/{len(data)}. Latency: {latency:.2f}ms")

    # 3. Calculate System Metrics
    p50_latency = np.percentile(latency_times, 50)
    p95_latency = np.percentile(latency_times, 95)

    print("\n--- System Metrics ---")
    print(f"P50 Latency (median): {p50_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")

    # 4. Save results for manual review
    with open(RESULTS_FILE_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nRaw results saved to '{RESULTS_FILE_PATH}' for manual scoring.")
    print("Next Steps: Manually score the Groundedness and Citation Accuracy columns, then update your documentation.")

if __name__ == "__main__":
    run_evaluation()