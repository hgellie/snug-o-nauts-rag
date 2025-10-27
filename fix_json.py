import json

def fix_scores(results):
    """Fix any missing or malformed scores in the results."""
    for config in results.values():
        for item in config:
            # Ensure all manual scores are numbers
            for score_field in ['manual_score_groundedness', 'manual_score_citation_accuracy', 'manual_score_exact_match']:
                if not isinstance(item.get(score_field), (int, float)) or item[score_field] is None:
                    item[score_field] = 0

def main():
    # Read the current file
    with open('ablation_results.json', 'r') as f:
        results = json.load(f)
    
    # Fix any formatting issues
    fix_scores(results)
    
    # Write back the cleaned data
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()