import json
from evaluate import run_evaluation
from rag_pipeline import answer_question

def run_ablation_configuration(config):
    """Run evaluation with specific configuration"""
    # Use a lambda to inject config into answer_question
    def custom_answer(q):
        return answer_question(q, **config)
    results = run_evaluation(answer_func=custom_answer)
    return results

def main():
    configs = {
        "baseline": {
            "use_mmr": True,
            "use_ngrams": True,
            "k": 20,
            "use_weighted_scoring": True
        },
        "no_mmr": {
            "use_mmr": False,
            "use_ngrams": True,
            "k": 20,
            "use_weighted_scoring": True
        },
        "no_ngrams": {
            "use_mmr": True,
            "use_ngrams": False,
            "k": 20,
            "use_weighted_scoring": True
        },
        "small_k": {
            "use_mmr": True,
            "use_ngrams": True,
            "k": 5,
            "use_weighted_scoring": True
        },
        "no_weights": {
            "use_mmr": True,
            "use_ngrams": True,
            "k": 20,
            "use_weighted_scoring": False
        }
    }
    results = {}
    for name, config in configs.items():
        print(f"Running ablation for: {name}")
        results[name] = run_ablation_configuration(config)
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
