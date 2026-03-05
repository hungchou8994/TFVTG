"""
Hybrid: Gemini answer + BLIP-2 grounding
=========================================
Kết hợp:
  - predicted_answer/predicted_answer_idx từ gemini_baseline_outputs (Acc@QA cao)
  - best_proposal từ verifier_outputs (BLIP-2 grounding)

Mode --mode:
  blip2      : dùng thẳng best_proposal từ BLIP-2 verifier
  intersect  : giao đoạn Gemini segment ∩ BLIP-2 proposal → tighter boundary
  snap       : chọn top-5 BLIP-2 proposal có IoU cao nhất với Gemini segment

Usage:
  python -m nextgqa.hybrid_eval --split test --mode intersect
  python -m nextgqa.eval_qa --source baseline --input dataset/nextgqa/hybrid_outputs_test.json
"""

import json
import argparse
import os


def iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',    default='test', choices=['test', 'val'])
    parser.add_argument('--baseline', default=None)
    parser.add_argument('--grounding',default=None)
    parser.add_argument('--output',   default=None)
    parser.add_argument('--mode',     default='intersect',
                        choices=['blip2', 'intersect', 'snap'],
                        help='blip2=BLIP-2 only, intersect=Gemini∩BLIP-2, snap=pick closest BLIP-2')
    args = parser.parse_args()

    baseline_path  = args.baseline  or f'dataset/nextgqa/gemini_baseline_outputs_{args.split}.json'
    grounding_path = args.grounding or f'dataset/nextgqa/verifier_outputs_{args.split}.json'
    output_path    = args.output    or f'dataset/nextgqa/hybrid_outputs_{args.split}.json'

    with open(baseline_path,  encoding='utf-8') as f:
        baseline = json.load(f)
    with open(grounding_path, encoding='utf-8') as f:
        grounding = json.load(f)

    # Build grounding lookup
    grounding_map = {
        (str(e['video_id']), str(e['qid'])): e
        for e in grounding
    }

    OUTPUT_KEYS = ('video_id', 'qid', 'question', 'type', 'choices',
                   'answer', 'answer_idx', 'gt_segments', 'duration',
                   'predicted_answer', 'predicted_answer_idx',
                   'answer_reasoning', 'best_proposal')

    results = []
    n_grounding_found = 0
    for entry in baseline:
        if 'predicted_answer' not in entry:
            continue
        key = (str(entry['video_id']), str(entry['qid']))
        gentry = grounding_map.get(key)
        entry = dict(entry)

        if gentry is not None:
            n_grounding_found += 1
            bp = gentry.get('best_proposal')          # [s, e, score]
            gs, ge = bp[0], bp[1]
            duration = float(entry.get('duration', 0))

            if args.mode == 'blip2':
                entry['best_proposal'] = bp

            elif args.mode == 'intersect':
                # Gemini segment
                gem_bp = entry.get('best_proposal', bp)
                gs2, ge2 = gem_bp[0], gem_bp[1]
                ns = max(gs, gs2)
                ne = min(ge, ge2)
                if ne - ns < 0.5:          # no real overlap → fall back to BLIP-2
                    ns, ne = gs, ge
                entry['best_proposal'] = [ns, ne, 1.0]

            elif args.mode == 'snap':
                # pick top-5 proposal with highest IoU to Gemini segment
                gem_bp = entry.get('best_proposal', bp)
                gs2, ge2 = gem_bp[0], gem_bp[1]
                top5 = gentry.get('top5_proposals', [bp])
                best = max(top5, key=lambda p: iou(gs2, ge2, p[0], p[1]))
                entry['best_proposal'] = best

        out = {k: entry[k] for k in OUTPUT_KEYS if k in entry}
        results.append(out)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'Mode              : {args.mode}')
    print(f'Baseline entries  : {len(baseline)}')
    print(f'With BLIP-2 info  : {n_grounding_found}/{len(results)}')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
