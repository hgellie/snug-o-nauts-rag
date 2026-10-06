import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(path="ablation_results.json"):
    with open(path, "r") as f:
        return json.load(f)

def mean_of_scored(config_results, field):
    """Average a manual score field, skipping unscored (None) items."""
    values = [float(r[field]) for r in config_results if r.get(field) is not None]
    return sum(values) / len(values) if values else 0

def extract_metrics(results):
    configs = []
    grounded = []
    citation = []
    exact = []
    for name, config_results in results.items():
        configs.append(name)
        grounded.append(mean_of_scored(config_results, "manual_score_groundedness"))
        citation.append(mean_of_scored(config_results, "manual_score_citation_accuracy"))
        exact.append(mean_of_scored(config_results, "manual_score_exact_match"))
    return configs, grounded, citation, exact

def plot_metrics(configs, grounded, citation, exact):
    x = np.arange(len(configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, grounded, width, label="Groundedness")
    ax.bar(x, citation, width, label="Citation Accuracy")
    ax.bar(x + width, exact, width, label="Exact Match")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: RAG Pipeline Metrics by Configuration")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ablation_metrics.png")
    plt.show()

def main():
    results = load_results()
    configs, grounded, citation, exact = extract_metrics(results)
    plot_metrics(configs, grounded, citation, exact)
    print("Ablation metrics plot saved as ablation_metrics.png")

if __name__ == "__main__":
    main()
