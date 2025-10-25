import json
import difflib
import os
from typing import List

RAW_PATH = "evaluation_results_raw.json"
SCORED_PATH = "evaluation_results_scored.json"
SUMMARY_CSV = "evaluation_results_summary.csv"


def load_raw_results(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw results not found at: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def normalize_source(source: str) -> str:
    """Convert a source filename to its human-readable policy document title."""
    # Handle common filename patterns like 'policies/Policy Document X.pdf'
    if source.startswith('policies/Policy Document '):
        # Extract the number and strip .pdf
        num = source.split('Document ')[-1].replace('.pdf', '')
        titles = {
            '1': 'Policy Document 1: The Snug-O-Nauts Celestial Comfort & Plush Integrity Act of 2025',
            '2': 'Policy Document 2: The Snug-O-Nauts Orbital De-Stressing & Re-Entry Protocol',
            '3': 'Policy Document 3: The Snug-O-Nauts Inter-Species & Interspecies Communication Guidelines',
            '4': 'Policy Document 4: The Great Cosmic Cuddler\'s Code of Conduct',
            '5': 'Policy Document 5: The Snug-O-Nauts Space Junk & Plush Debris Containment Protocol',
            '6': 'Policy Document 6: The Snug-O-Nauts Fiscal & Ethical Responsibility Act',
            '7': 'Policy Document 7: The Snug-O-Nauts Material & Safety Protocols Act',
            '8': 'Policy Document 8: The Snug-O-Nauts Employee Wellness & Sentient Support Protocol',
            '9': 'Policy Document 9: The Snug-O-Nauts Foundational Principles & Charter',
        }
        return titles.get(num, source)
    return source


def parse_answer_and_sources(rag_answer: str):
    # If the answer contains the marker we used earlier, split it out
    if "**Sources:**" in rag_answer:
        answer_part, sources_part = rag_answer.split("**Sources:**", 1)
        sources = [s.strip() for s in sources_part.split(";") if s.strip()]
        # Map filenames to human-readable titles
        sources = [normalize_source(s) for s in sources]
    else:
        answer_part = rag_answer
        sources = []
    return answer_part.strip(), sources


def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_item(item: dict, fuzzy_threshold: float = 0.40) -> dict:
    answer_raw = item.get('rag_answer', '') or ''
    ground_truth = item.get('ground_truth', '') or ''
    expected_source = (item.get('expected_source') or '').strip()

    answer_text, sources = parse_answer_and_sources(answer_raw)

    # Groundedness: treat as positive when either the ground_truth is a substring
    # of the answer OR fuzzy similarity exceeds threshold.
    gt_lower = ground_truth.lower().strip()
    ans_lower = answer_text.lower().strip()
    substring_hit = bool(gt_lower) and (gt_lower in ans_lower)
    similarity = fuzzy_ratio(gt_lower, ans_lower) if gt_lower and ans_lower else 0.0
    auto_grounded = 1 if (substring_hit or similarity >= fuzzy_threshold) else 0

    # Citation accuracy: check if expected_source appears in the recorded sources
    sources_joined = "; ".join(sources).lower()
    citation_hit = bool(expected_source) and (expected_source.lower() in sources_joined)
    auto_citation = 1 if citation_hit else 0

    # Exact match: strict string equality (trimmed)
    auto_exact = 1 if ans_lower == gt_lower and gt_lower != "" else 0

    scored = dict(item)  # copy
    scored.update({
        'auto_score_groundedness': auto_grounded,
        'auto_score_citation_accuracy': auto_citation,
        'auto_score_exact_match': auto_exact,
        'groundedness_similarity': round(similarity, 4),
        'parsed_answer': answer_text,
        'parsed_sources': sources,
    })

    return scored


def run_autoscore():
    raw = load_raw_results(RAW_PATH)
    scored = []
    for item in raw:
        scored_item = score_item(item)
        scored.append(scored_item)

    # Save detailed scored results
    with open(SCORED_PATH, 'w') as f:
        json.dump(scored, f, indent=2)

    # Produce a compact CSV summary
    import csv
    headers = [
        'id', 'query', 'auto_score_groundedness', 'auto_score_citation_accuracy',
        'auto_score_exact_match', 'groundedness_similarity', 'latency_ms'
    ]
    with open(SUMMARY_CSV, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=headers)
        writer.writeheader()
        for r in scored:
            writer.writerow({
                'id': r.get('id'),
                'query': r.get('query'),
                'auto_score_groundedness': r.get('auto_score_groundedness'),
                'auto_score_citation_accuracy': r.get('auto_score_citation_accuracy'),
                'auto_score_exact_match': r.get('auto_score_exact_match'),
                'groundedness_similarity': r.get('groundedness_similarity'),
                'latency_ms': r.get('latency_ms'),
            })

    # Print summary metrics
    total = len(scored)
    if total == 0:
        print("No items to score.")
        return
    sum_grounded = sum(s['auto_score_groundedness'] for s in scored)
    sum_citation = sum(s['auto_score_citation_accuracy'] for s in scored)
    sum_exact = sum(s['auto_score_exact_match'] for s in scored)

    print(f"Autoscore complete for {total} items")
    print(f"Groundedness (auto): {sum_grounded}/{total} = {sum_grounded/total:.2%}")
    print(f"Citation accuracy (auto): {sum_citation}/{total} = {sum_citation/total:.2%}")
    print(f"Exact match (auto): {sum_exact}/{total} = {sum_exact/total:.2%}")
    print(f"Detailed results written to: {SCORED_PATH}")
    print(f"Summary CSV written to: {SUMMARY_CSV}")


if __name__ == '__main__':
    run_autoscore()
