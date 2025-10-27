import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    with open('ablation_results.json', 'r') as f:
        return json.load(f)

def calculate_metrics(config_results):
    metrics = {
        'groundedness': [],
        'citation_accuracy': [],
        'exact_match': []
    }
    
    for result in config_results:
        # Only include numeric (non-null) scores
        if result.get('manual_score_groundedness') is not None:
            metrics['groundedness'].append(float(result['manual_score_groundedness']))
        if result.get('manual_score_citation_accuracy') is not None:
            metrics['citation_accuracy'].append(float(result['manual_score_citation_accuracy']))
        if result.get('manual_score_exact_match') is not None:
            metrics['exact_match'].append(float(result['manual_score_exact_match']))
    
    # compute latency list
    latency_list = [float(r['latency_ms']) for r in config_results if r.get('latency_ms') is not None]
    return {
        'avg_groundedness': np.mean(metrics['groundedness']) if metrics['groundedness'] else 0.0,
        'n_grounded': len(metrics['groundedness']),
        'avg_citation': np.mean(metrics['citation_accuracy']) if metrics['citation_accuracy'] else 0.0,
        'n_citation': len(metrics['citation_accuracy']),
        'avg_exact_match': np.mean(metrics['exact_match']) if metrics['exact_match'] else 0.0,
        'n_exact': len(metrics['exact_match']),
        'med_latency': np.median(latency_list) if latency_list else 0.0,
        'n_latency': len(latency_list)
    }

def analyze_ablation():
    results = load_results()
    configs = ['baseline', 'no_mmr', 'no_ngrams', 'small_k', 'no_weights']
    metrics = {}
    
    for config in configs:
        metrics[config] = calculate_metrics(results[config])
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    width = 0.35
    x = np.arange(len(configs))
    
    # Groundedness
    groundedness = [metrics[c]['avg_groundedness'] for c in configs]
    ax1.bar(x, groundedness)
    ax1.set_title('Average Groundedness')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    
    # Citation Accuracy
    citation = [metrics[c]['avg_citation'] for c in configs]
    ax2.bar(x, citation)
    ax2.set_title('Average Citation Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)
    
    # Exact Match
    exact = [metrics[c]['avg_exact_match'] for c in configs]
    ax3.bar(x, exact)
    ax3.set_title('Average Exact Match')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=45)
    
    # Latency
    latency = [metrics[c]['med_latency'] for c in configs]
    ax4.bar(x, latency)
    ax4.set_title('Median Latency (ms)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(configs, rotation=45)
    
    plt.tight_layout()
    plt.savefig('ablation_analysis.png')

    # Save numeric summary to JSON and CSV
    summary = {}
    csv_path = os.path.join(os.getcwd(), 'ablation_summary.csv')
    json_path = os.path.join(os.getcwd(), 'ablation_summary.json')

    # build summary dict for JSON and rows for CSV
    csv_rows = []
    for config in configs:
        m = metrics[config]
        summary[config] = {
            'avg_groundedness': m['avg_groundedness'],
            'n_grounded': m['n_grounded'],
            'avg_citation': m['avg_citation'],
            'n_citation': m['n_citation'],
            'avg_exact_match': m['avg_exact_match'],
            'n_exact': m['n_exact'],
            'med_latency_ms': m['med_latency'],
            'n_latency': m['n_latency']
        }
        csv_rows.append([
            config,
            m['avg_groundedness'], m['n_grounded'],
            m['avg_citation'], m['n_citation'],
            m['avg_exact_match'], m['n_exact'],
            m['med_latency'], m['n_latency']
        ])

    # write JSON
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2)

    # write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['config', 'avg_groundedness', 'n_grounded', 'avg_citation', 'n_citation', 'avg_exact_match', 'n_exact', 'med_latency_ms', 'n_latency'])
        writer.writerows(csv_rows)
    
    # Print summary
    print("\nAblation Study Results Summary:")
    print("=" * 50)
    for config in configs:
        print(f"\n{config.upper()}:")
        print(f"Groundedness: {metrics[config]['avg_groundedness']:.2f}")
        print(f"Citation Accuracy: {metrics[config]['avg_citation']:.2f}")
        print(f"Exact Match: {metrics[config]['avg_exact_match']:.2f}")
        print(f"Median Latency: {metrics[config]['med_latency']:.1f}ms")
    
    # Determine best configuration for each metric
    best_groundedness = max(configs, key=lambda c: metrics[c]['avg_groundedness'])
    best_citation = max(configs, key=lambda c: metrics[c]['avg_citation'])
    best_exact = max(configs, key=lambda c: metrics[c]['avg_exact_match'])
    fastest = min(configs, key=lambda c: metrics[c]['med_latency'])
    
    print("\nBest Configurations:")
    print("=" * 50)
    print(f"Best Groundedness: {best_groundedness}")
    print(f"Best Citation Accuracy: {best_citation}")
    print(f"Best Exact Match: {best_exact}")
    print(f"Fastest Response: {fastest}")

if __name__ == "__main__":
    analyze_ablation()