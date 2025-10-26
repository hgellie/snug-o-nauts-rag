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

def run_evaluation(answer_func=None):
    data = load_evaluation_data()
    latency_times = []
    results = []

    print(f"Starting evaluation on {len(data)} questions...")

    if answer_func is None:
        answer_func = lambda q: answer_question(q)

    for i, item in enumerate(data):
        start_time = time.time()
        rag_answer = answer_func(item['query'])
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        latency_times.append(latency)
        result = {
            "id": item['id'],
            "query": item['query'],
            "ground_truth": item['ground_truth'],
            "expected_source": item['source_document'],
            "rag_answer": rag_answer,
            "latency_ms": f"{latency:.2f}",
            "query_type": item['query_type'],
            "manual_score_groundedness": None,
            "manual_score_citation_accuracy": None,
            "manual_score_exact_match": None,
        }
        results.append(result)
        print(f"Completed query {i+1}/{len(data)}. Latency: {latency:.2f}ms")

    p50_latency = np.percentile(latency_times, 50)
    p95_latency = np.percentile(latency_times, 95)

    print("\n--- System Metrics ---")
    print(f"P50 Latency (median): {p50_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")

    with open(RESULTS_FILE_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nRaw results saved to '{RESULTS_FILE_PATH}' for manual scoring.")
    print("Next Steps: Manually score the Groundedness and Citation Accuracy columns, then update your documentation.")
    return results

if __name__ == "__main__":
    run_evaluation()