import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(path="ablation_results.json"):
    with open(path, "r") as f:
        return json.load(f)

def extract_metrics(results):
    configs = []
    grounded = []
    citation = []
    exact = []
    for name, config_results in results.items():
        total = len(config_results)
        g = sum(r.get("auto_score_groundedness", 0) for r in config_results)
        c = sum(r.get("auto_score_citation_accuracy", 0) for r in config_results)
        e = sum(r.get("auto_score_exact_match", 0) for r in config_results)
        configs.append(name)
        grounded.append(g / total if total else 0)
        citation.append(c / total if total else 0)
        exact.append(e / total if total else 0)
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
