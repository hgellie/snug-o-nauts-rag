import json

def normalize_score(value):
    """Return the score as a number, or None if it is missing or malformed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None

def fix_scores(results):
    """Fix any missing or malformed scores in the results."""
    for config in results.values():
        for item in config:
            # Keep numeric scores; mark anything else as unscored (None) so
            # analyze_ablation.py excludes it from averages instead of counting it as 0
            for score_field in ['manual_score_groundedness', 'manual_score_citation_accuracy', 'manual_score_exact_match']:
                item[score_field] = normalize_score(item.get(score_field))

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