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


def get_tokens(text: str) -> set:
    """Extract normalized tokens from text, handling markdown and quotes."""
    # Remove markdown bold markers and normalize quotes
    text = text.replace('**', '').replace('"', '"').replace('"', '"')
    # Split on common delimiters and normalize
    tokens = set()
    for word in text.lower().replace('(', ' ').replace(')', ' ').split():
        # Remove punctuation from word edges
        word = word.strip('.,;:?!')
        # Keep meaningful tokens (not too short, not just numbers)
        if word and len(word) > 1 and not word.isnumeric():
            tokens.add(word)
    return tokens

def token_overlap_score(ground_truth: str, answer: str) -> tuple[float, set, set]:
    """Compute token overlap metrics between ground truth and answer."""
    gt_tokens = get_tokens(ground_truth)
    ans_tokens = get_tokens(answer)
    
    if not gt_tokens or not ans_tokens:
        return 0.0, set(), set()

    # Find shared and missing important tokens
    shared = gt_tokens & ans_tokens
    missing = gt_tokens - ans_tokens

    # Calculate overlap score (% of ground truth tokens present in answer)
    overlap = len(shared) / len(gt_tokens)
    return overlap, shared, missing

def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def score_item(item: dict, fuzzy_threshold: float = 0.30, token_threshold: float = 0.35) -> dict:
    answer_raw = item.get('rag_answer', '') or ''
    ground_truth = item.get('ground_truth', '') or ''
    expected_source = (item.get('expected_source') or '').strip()

    answer_text, sources = parse_answer_and_sources(answer_raw)

    # 1. Check token overlap (key terms and concepts)
    token_score, shared_tokens, missing_tokens = token_overlap_score(ground_truth, answer_text)
    token_hit = token_score >= token_threshold

    # 2. Check fuzzy string similarity
    gt_lower = ground_truth.lower().strip()
    ans_lower = answer_text.lower().strip()
    substring_hit = bool(gt_lower) and (gt_lower in ans_lower)
    similarity = fuzzy_ratio(gt_lower, ans_lower) if gt_lower and ans_lower else 0.0
    fuzzy_hit = similarity >= fuzzy_threshold

    # Consider an answer grounded if it meets any of our criteria
    # - Good token overlap (threshold lowered)
    # - High fuzzy similarity
    # - Contains ground truth as substring
    # - Has at least 35% token overlap and some fuzzy similarity
    hybrid_hit = token_score >= 0.35 and similarity >= 0.2
    auto_grounded = 1 if (token_hit or fuzzy_hit or substring_hit or hybrid_hit) else 0

    # Citation accuracy: check if ONLY the expected source is cited
    if not expected_source:
        auto_citation = 0
    else:
        # Convert both lists to lowercase for comparison
        expected = expected_source.lower()
        actual = [s.lower() for s in sources]
        # Citation is accurate if exactly one source is cited and it matches the expected source
        citation_hit = len(sources) == 1 and expected in actual[0]
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

    # Save detailed scored results with token info
    for item in scored:
        token_score, shared, missing = token_overlap_score(
            item.get('ground_truth', ''),
            item.get('parsed_answer', '')
        )
        item['token_overlap_score'] = round(token_score, 4)
        item['shared_tokens'] = sorted(list(shared))
        item['missing_tokens'] = sorted(list(missing))

    with open(SCORED_PATH, 'w') as f:
        json.dump(scored, f, indent=2)

    # Produce a compact CSV summary
    import csv
    headers = [
        'id', 'query', 'auto_score_groundedness', 'auto_score_citation_accuracy',
        'auto_score_exact_match', 'groundedness_similarity', 'token_overlap_score',
        'latency_ms'
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
                'token_overlap_score': r.get('token_overlap_score'),
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
