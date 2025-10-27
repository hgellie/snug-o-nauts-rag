import csv
import json
import sys
import os
import statistics

ROOT = os.getcwd()
EVAL_RAW = os.path.join(ROOT, 'evaluation_results_raw.json')
CSV_PATH = os.path.join(ROOT, 'evaluation_to_score.csv')
APPLIED_JSON = os.path.join(ROOT, 'evaluation_results_scored.json')
SUMMARY_JSON = os.path.join(ROOT, 'evaluation_summary.json')
SUMMARY_CSV = os.path.join(ROOT, 'evaluation_summary.csv')


def prepare_csv():
    with open(EVAL_RAW, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['id', 'query', 'rag_answer', 'latency_ms', 'manual_score_groundedness', 'manual_score_citation_accuracy', 'manual_score_exact_match'])
        for item in data:
            writer.writerow([
                item.get('id'),
                item.get('query').replace('\n', ' '),
                item.get('rag_answer').replace('\n', ' '),
                item.get('latency_ms'),
                '' if item.get('manual_score_groundedness') is None else item.get('manual_score_groundedness'),
                '' if item.get('manual_score_citation_accuracy') is None else item.get('manual_score_citation_accuracy'),
                '' if item.get('manual_score_exact_match') is None else item.get('manual_score_exact_match'),
            ])
    print(f'Wrote {CSV_PATH} — open this file in a spreadsheet editor, fill manual score columns (numbers or blank), then run:')
    print(f'  python score_evaluation.py apply')


def _as_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except:
        return None


def apply_csv():
    # load raw JSON
    with open(EVAL_RAW, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # load CSV
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as cf:
        reader = csv.DictReader(cf)
        for r in reader:
            rows.append(r)

    # map by id and update
    id_map = {int(item['id']): item for item in data}
    for r in rows:
        try:
            idx = int(r['id'])
        except:
            continue
        if idx not in id_map:
            continue
        item = id_map[idx]
        gs = _as_float(r.get('manual_score_groundedness'))
        cs = _as_float(r.get('manual_score_citation_accuracy'))
        ex = _as_float(r.get('manual_score_exact_match'))
        item['manual_score_groundedness'] = gs
        item['manual_score_citation_accuracy'] = cs
        item['manual_score_exact_match'] = ex

    # write applied JSON
    with open(APPLIED_JSON, 'w', encoding='utf-8') as af:
        json.dump(list(id_map.values()), af, indent=2)
    print(f'Wrote scored JSON to {APPLIED_JSON}')

    # compute metrics
    g_vals = [it['manual_score_groundedness'] for it in id_map.values() if it.get('manual_score_groundedness') is not None]
    c_vals = [it['manual_score_citation_accuracy'] for it in id_map.values() if it.get('manual_score_citation_accuracy') is not None]
    ex_vals = [it['manual_score_exact_match'] for it in id_map.values() if it.get('manual_score_exact_match') is not None]
    lat_vals = [float(it['latency_ms']) for it in id_map.values() if it.get('latency_ms') is not None]

    summary = {
        'n_grounded': len(g_vals),
        'mean_grounded': statistics.mean(g_vals) if g_vals else None,
        'n_citation': len(c_vals),
        'mean_citation': statistics.mean(c_vals) if c_vals else None,
        'n_exact': len(ex_vals),
        'mean_exact': statistics.mean(ex_vals) if ex_vals else None,
        'n_latency': len(lat_vals),
        'median_latency_ms': statistics.median(lat_vals) if lat_vals else None
    }

    with open(SUMMARY_JSON, 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, indent=2)
    print(f'Wrote evaluation summary to {SUMMARY_JSON}')

    # write CSV summary
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as scf:
        writer = csv.writer(scf)
        writer.writerow(['metric','n','mean_or_median'])
        writer.writerow(['groundedness', summary['n_grounded'], summary['mean_grounded']])
        writer.writerow(['citation_accuracy', summary['n_citation'], summary['mean_citation']])
        writer.writerow(['exact_match', summary['n_exact'], summary['mean_exact']])
        writer.writerow(['median_latency_ms', summary['n_latency'], summary['median_latency_ms']])
    print(f'Wrote evaluation summary CSV to {SUMMARY_CSV}')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'prepare':
        prepare_csv()
    elif sys.argv[1] == 'apply':
        apply_csv()
    else:
        print('Usage: python score_evaluation.py [prepare|apply]')
