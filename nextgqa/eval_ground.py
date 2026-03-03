"""
NExT-GQA Grounding Evaluation
==============================
Tính metrics grounding trên grounder_outputs_{split}.json.

Metrics:
  - mIoU      : mean Intersection over Union
  - mIoP      : mean Intersection over Prediction (metric chính của NExT-GQA)
  - IoU@0.3/0.5/0.7
  - IoP@0.3/0.5/0.7
  - Breakdown by question type: CW, CH, TN, TC, TP

Notes:
  - Câu hỏi không có gt_segments → skip (không tính vào metrics)
  - Câu hỏi có nhiều GT segments → lấy max IoU/IoP over all GT segments
  - Dùng top-1 proposal để tính (prediction = proposal[0])

Usage:
  python -m nextgqa.eval_ground --split test
  python -m nextgqa.eval_ground --input dataset/nextgqa/grounder_outputs_test.json
  python -m nextgqa.eval_ground --split test --topk 1   # default
  python -m nextgqa.eval_ground --split test --topk 5   # oracle: best of top-5
"""

import json
import argparse
import numpy as np
from collections import defaultdict


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_iou(pred_start, pred_end, gt_start, gt_end):
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def calc_iop(pred_start, pred_end, gt_start, gt_end):
    """Intersection over Prediction length — metric chính của NExT-GQA."""
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    pred_len = pred_end - pred_start
    return inter / pred_len if pred_len > 0 else 0.0


def best_iou_iop(pred_start, pred_end, gt_segments):
    """Max IoU và IoP over all GT segments (vì 1 question có thể có nhiều segments)."""
    best_iou = 0.0
    best_iop = 0.0
    for gs, ge in gt_segments:
        best_iou = max(best_iou, calc_iou(pred_start, pred_end, gs, ge))
        best_iop = max(best_iop, calc_iop(pred_start, pred_end, gs, ge))
    return best_iou, best_iop


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results, topk=1):
    """
    Evaluate grounding results.
    topk=1: dùng proposal #1 (thực tế)
    topk>1: oracle — lấy proposal tốt nhất trong top-k (upper bound)
    """
    thresholds = [0.3, 0.5, 0.7]

    # Overall
    all_ious = []
    all_iops = []
    recall_iou = np.zeros(len(thresholds))
    recall_iop = np.zeros(len(thresholds))

    # Per type
    type_ious   = defaultdict(list)
    type_iops   = defaultdict(list)
    type_rec_iou = defaultdict(lambda: np.zeros(len(thresholds)))
    type_rec_iop = defaultdict(lambda: np.zeros(len(thresholds)))

    skipped = 0

    for entry in results:
        gt_segs = entry.get('gt_segments')
        if not gt_segs:
            skipped += 1
            continue

        proposals = entry.get('top5_proposals', [])
        if not proposals:
            skipped += 1
            continue

        qtype = entry.get('type', 'UNK')

        if topk == 1:
            # Dùng proposal tốt nhất sau khi re-rank (top-1)
            p = proposals[0]
            iou, iop = best_iou_iop(p[0], p[1], gt_segs)
        else:
            # Oracle: lấy proposal cho IoP cao nhất trong top-k
            best_iou_v, best_iop_v = 0.0, 0.0
            for p in proposals[:topk]:
                u, op = best_iou_iop(p[0], p[1], gt_segs)
                best_iou_v = max(best_iou_v, u)
                best_iop_v = max(best_iop_v, op)
            iou, iop = best_iou_v, best_iop_v

        all_ious.append(iou)
        all_iops.append(iop)
        type_ious[qtype].append(iou)
        type_iops[qtype].append(iop)

        for k, th in enumerate(thresholds):
            if iou >= th:
                recall_iou[k] += 1
                type_rec_iou[qtype][k] += 1
            if iop >= th:
                recall_iop[k] += 1
                type_rec_iop[qtype][k] += 1

    n = len(all_ious)
    if n == 0:
        print('No grounded questions found!')
        return

    print(f'\n{"="*55}')
    label = f'Top-{topk} Oracle' if topk > 1 else 'Top-1'
    print(f'NExT-GQA Grounding Eval ({label})')
    print(f'  Evaluated : {n} questions')
    print(f'  Skipped   : {skipped} (no GT segments)')
    print(f'{"="*55}')

    miou = np.mean(all_ious)
    miop = np.mean(all_iops)
    print(f'\n  {"Metric":<20} {"Value":>8}')
    print(f'  {"-"*30}')
    print(f'  {"mIoU":<20} {miou*100:>7.2f}%')
    print(f'  {"mIoP":<20} {miop*100:>7.2f}%  <- main metric')
    for k, th in enumerate(thresholds):
        print(f'  {"IoU@"+str(th):<20} {recall_iou[k]/n*100:>7.2f}%')
    for k, th in enumerate(thresholds):
        print(f'  {"IoP@"+str(th):<20} {recall_iop[k]/n*100:>7.2f}%')

    # Per-type breakdown
    print(f'\n  {"Type":<8} {"N":>5} {"mIoU":>8} {"mIoP":>8} {"IoP@0.5":>9}')
    print(f'  {"-"*45}')
    for qtype in sorted(type_ious.keys()):
        nt = len(type_ious[qtype])
        t_miou = np.mean(type_ious[qtype]) * 100
        t_miop = np.mean(type_iops[qtype]) * 100
        t_iop5 = type_rec_iop[qtype][1] / nt * 100
        print(f'  {qtype:<8} {nt:>5} {t_miou:>7.2f}% {t_miop:>7.2f}% {t_iop5:>8.2f}%')

    print(f'{"="*55}\n')

    return {
        'mIoU':    miou,
        'mIoP':    miop,
        'IoU@0.3': recall_iou[0] / n,
        'IoU@0.5': recall_iou[1] / n,
        'IoU@0.7': recall_iou[2] / n,
        'IoP@0.3': recall_iop[0] / n,
        'IoP@0.5': recall_iop[1] / n,
        'IoP@0.7': recall_iop[2] / n,
        'n': n,
    }


# ── Verifier-aware evaluation ─────────────────────────────────────────────────

def evaluate_verifier(results):
    """Evaluate grounding using best_proposal selected by Gemini Verifier."""
    thresholds = [0.3, 0.5, 0.7]
    all_ious, all_iops = [], []
    recall_iou = np.zeros(len(thresholds))
    recall_iop = np.zeros(len(thresholds))
    type_ious   = defaultdict(list)
    type_iops   = defaultdict(list)
    type_rec_iou = defaultdict(lambda: np.zeros(len(thresholds)))
    type_rec_iop = defaultdict(lambda: np.zeros(len(thresholds)))
    skipped = 0

    for entry in results:
        if 'best_proposal' not in entry:
            skipped += 1
            continue
        gt_segs = entry.get('gt_segments')
        if not gt_segs:
            skipped += 1
            continue

        p = entry['best_proposal']
        iou, iop = best_iou_iop(p[0], p[1], gt_segs)
        qtype = entry.get('type', 'UNK')

        all_ious.append(iou)
        all_iops.append(iop)
        type_ious[qtype].append(iou)
        type_iops[qtype].append(iop)

        for k, th in enumerate(thresholds):
            if iou >= th:
                recall_iou[k] += 1
                type_rec_iou[qtype][k] += 1
            if iop >= th:
                recall_iop[k] += 1
                type_rec_iop[qtype][k] += 1

    n = len(all_ious)
    if n == 0:
        print('No verified entries found.')
        return

    print(f'\n{"="*55}')
    print(f'NExT-GQA Grounding Eval (Verifier best_proposal)')
    print(f'  Evaluated : {n} questions')
    print(f'  Skipped   : {skipped} (no best_proposal or no GT)')
    print(f'{"="*55}')
    miou = np.mean(all_ious)
    miop = np.mean(all_iops)
    print(f'\n  {"Metric":<20} {"Value":>8}')
    print(f'  {"-"*30}')
    print(f'  {"mIoU":<20} {miou*100:>7.2f}%')
    print(f'  {"mIoP":<20} {miop*100:>7.2f}%  <- main metric')
    for k, th in enumerate(thresholds):
        print(f'  {"IoU@"+str(th):<20} {recall_iou[k]/n*100:>7.2f}%')
    for k, th in enumerate(thresholds):
        print(f'  {"IoP@"+str(th):<20} {recall_iop[k]/n*100:>7.2f}%')

    print(f'\n  {"Type":<8} {"N":>5} {"mIoU":>8} {"mIoP":>8} {"IoP@0.5":>9}')
    print(f'  {"-"*45}')
    for qtype in sorted(type_ious.keys()):
        nt = len(type_ious[qtype])
        t_miou = np.mean(type_ious[qtype]) * 100
        t_miop = np.mean(type_iops[qtype]) * 100
        t_iop5 = type_rec_iop[qtype][1] / nt * 100
        print(f'  {qtype:<8} {nt:>5} {t_miou:>7.2f}% {t_miop:>7.2f}% {t_iop5:>8.2f}%')

    # Grounder rank distribution
    ranks = [r.get('grounder_rank', 1) for r in results if 'grounder_rank' in r]
    if ranks:
        from collections import Counter
        rc = Counter(ranks)
        print(f'\n  Grounder rank of Verifier pick:')
        for r in sorted(rc.keys()):
            print(f'    Rank #{r}: {rc[r]} ({rc[r]/len(ranks)*100:.1f}%)')

    print(f'{"="*55}\n')
    return {'mIoU': miou, 'mIoP': miop, 'n': n}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate NExT-GQA grounding results')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--input', default=None, help='Path to grounder or verifier JSON')
    parser.add_argument('--source', default='grounder', choices=['grounder', 'verifier'],
                        help='grounder: use top5_proposals[0] + oracle | verifier: use best_proposal')
    parser.add_argument('--topk', type=int, default=1,
                        help='(grounder only) 1=top-1, >1=oracle upper bound')
    args = parser.parse_args()

    if args.source == 'verifier':
        input_path = args.input or f'dataset/nextgqa/verifier_outputs_{args.split}.json'
        print(f'Loading verifier outputs: {input_path}')
        with open(input_path, encoding='utf-8') as f:
            results = json.load(f)
        print(f'Total entries: {len(results)}')
        evaluate_verifier(results)
    else:
        input_path = args.input or f'dataset/nextgqa/grounder_outputs_{args.split}.json'
        print(f'Loading grounder outputs: {input_path}')
        with open(input_path, encoding='utf-8') as f:
            results = json.load(f)
        print(f'Total entries: {len(results)}')
        evaluate(results, topk=1)
        evaluate(results, topk=5)


if __name__ == '__main__':
    main()
