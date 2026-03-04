"""
NExT-GQA QA Evaluation
=======================
Tính metrics QA trên answerer_outputs_{split}.json.

Metrics:
  - Acc@QA      : % predicted_answer_idx == answer_idx  (trả lời đúng)
  - Acc@GQA_IoU : % correct AND IoU(best_proposal, gt_segments) >= threshold
  - Acc@GQA_IoP : % correct AND IoP(best_proposal, gt_segments) >= threshold
  - Breakdown by question type: CW, CH, TN, TC, TP

Notes:
  - Entries không có predicted_answer → skip (chưa trả lời)
  - Entries không có best_proposal → skip (PROHIBITED_CONTENT)
  - Denominator cho Acc@GQA = số entries có cả predicted_answer lẫn gt_segments lẫn best_proposal

Usage:
  python -m nextgqa.eval_qa --split test
  python -m nextgqa.eval_qa --input dataset/nextgqa/answerer_outputs_test.json
"""

import json
import argparse
import numpy as np
from collections import defaultdict


# ── IoU / IoP helpers ─────────────────────────────────────────────────────────

def calc_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def calc_iop(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    pred_len = pe - ps
    return inter / pred_len if pred_len > 0 else 0.0


def best_iou_iop(ps, pe, gt_segments):
    best_iou, best_iop = 0.0, 0.0
    for gs, ge in gt_segments:
        best_iou = max(best_iou, calc_iou(ps, pe, gs, ge))
        best_iop = max(best_iop, calc_iop(ps, pe, gs, ge))
    return best_iou, best_iop


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_qa(results, iou_thresh=0.5, iop_thresh=0.5):
    """
    Evaluate QA + grounded QA metrics.
    """
    thresholds = [0.3, 0.5, 0.7]

    # Overall counters
    qa_correct = 0
    gqa_iou_correct = 0
    gqa_iop_correct = 0
    qa_total = 0       # entries with predicted_answer
    gqa_total = 0      # entries with predicted_answer + gt_segments + best_proposal
    no_pred = 0
    no_proposal = 0

    # Grounding accumulators (only for answered entries)
    all_ious, all_iops = [], []
    recall_iou = np.zeros(len(thresholds))
    recall_iop = np.zeros(len(thresholds))

    # Per type
    type_qa_correct  = defaultdict(int)
    type_gqa_iou_cor = defaultdict(int)
    type_gqa_iop_cor = defaultdict(int)
    type_qa_total    = defaultdict(int)
    type_gqa_total   = defaultdict(int)
    type_ious        = defaultdict(list)
    type_iops        = defaultdict(list)
    type_rec_iou     = defaultdict(lambda: np.zeros(len(thresholds)))
    type_rec_iop     = defaultdict(lambda: np.zeros(len(thresholds)))

    for entry in results:
        qtype = entry.get('type', 'UNK')

        # Must have predicted_answer
        if 'predicted_answer' not in entry:
            no_pred += 1
            continue

        qa_total += 1
        type_qa_total[qtype] += 1

        is_correct = (entry.get('predicted_answer_idx') == entry.get('answer_idx'))
        if is_correct:
            qa_correct += 1
            type_qa_correct[qtype] += 1

        # For Acc@GQA + grounding: need gt_segments and best_proposal
        gt_segs = entry.get('gt_segments')
        bp = entry.get('best_proposal')
        if not gt_segs or not bp:
            if not bp:
                no_proposal += 1
            continue

        gqa_total += 1
        type_gqa_total[qtype] += 1

        iou, iop = best_iou_iop(bp[0], bp[1], gt_segs)

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

        if is_correct and iou >= iou_thresh:
            gqa_iou_correct += 1
            type_gqa_iou_cor[qtype] += 1
        if is_correct and iop >= iop_thresh:
            gqa_iop_correct += 1
            type_gqa_iop_cor[qtype] += 1

    if qa_total == 0:
        print('No answered entries found!')
        return

    acc_qa      = qa_correct / qa_total * 100
    acc_gqa_iou = gqa_iou_correct / gqa_total * 100 if gqa_total else 0
    acc_gqa_iop = gqa_iop_correct / gqa_total * 100 if gqa_total else 0

    ng = len(all_ious)
    miou = np.mean(all_ious) * 100 if ng else 0
    miop = np.mean(all_iops) * 100 if ng else 0

    print(f'\n{"="*60}')
    print(f'NExT-GQA QA Evaluation')
    print(f'  Answered          : {qa_total}')
    print(f'  Skipped (no pred) : {no_pred}')
    print(f'  Total entries     : {qa_total + no_pred}')
    print(f'{"="*60}')

    # ── Grounding on answered entries ──
    print(f'\n  [Grounding — answered {ng} entries]')
    print(f'  {"Metric":<20} {"Value":>8}')
    print(f'  {"-"*30}')
    print(f'  {"mIoU":<20} {miou:>7.2f}%')
    print(f'  {"mIoP":<20} {miop:>7.2f}%  <- main metric')
    for k, th in enumerate(thresholds):
        print(f'  {"IoU@"+str(th):<20} {recall_iou[k]/ng*100:>7.2f}%')
    for k, th in enumerate(thresholds):
        print(f'  {"IoP@"+str(th):<20} {recall_iop[k]/ng*100:>7.2f}%')

    # ── QA metrics ──
    print(f'\n  [QA Accuracy]')
    print(f'  {"Metric":<35} {"Value":>8}')
    print(f'  {"-"*45}')
    print(f'  {"Acc@QA":<35} {acc_qa:>7.2f}%')
    print(f'  {f"Acc@GQA (IoU≥{iou_thresh})":<35} {acc_gqa_iou:>7.2f}%')
    print(f'  {f"Acc@GQA (IoP≥{iop_thresh})":<35} {acc_gqa_iop:>7.2f}%')

    # Per-type breakdown
    print(f'\n  {"Type":<6} {"N":>5} {"mIoU":>7} {"mIoP":>7} {"IoP@0.5":>8} {"Acc@QA":>8} {"GQA(IoU)":>10} {"GQA(IoP)":>10}')
    print(f'  {"-"*70}')
    for qtype in sorted(type_qa_total.keys()):
        nqa  = type_qa_total[qtype]
        ngqa = type_gqa_total.get(qtype, 0)
        a_qa  = type_qa_correct[qtype] / nqa * 100
        a_iou = type_gqa_iou_cor[qtype] / ngqa * 100 if ngqa else 0
        a_iop = type_gqa_iop_cor[qtype] / ngqa * 100 if ngqa else 0
        t_miou = np.mean(type_ious[qtype]) * 100 if type_ious[qtype] else 0
        t_miop = np.mean(type_iops[qtype]) * 100 if type_iops[qtype] else 0
        t_iop5 = type_rec_iop[qtype][1] / ngqa * 100 if ngqa else 0
        print(f'  {qtype:<6} {nqa:>5} {t_miou:>6.2f}% {t_miop:>6.2f}% {t_iop5:>7.2f}% {a_qa:>7.2f}% {a_iou:>9.2f}% {a_iop:>9.2f}%')

    print(f'{"="*60}\n')
    return {
        'mIoU': miou, 'mIoP': miop,
        'Acc@QA': acc_qa,
        f'Acc@GQA_IoU_{iou_thresh}': acc_gqa_iou,
        f'Acc@GQA_IoP_{iop_thresh}': acc_gqa_iop,
        'n_qa': qa_total,
        'n_gqa': gqa_total,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate NExT-GQA QA results')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--source', default='gemini', choices=['gemini', 'qwen'],
                        help='Which answerer output to evaluate: '
                             'gemini → answerer_outputs_{split}.json, '
                             'qwen  → answerer_outputs_qwen_{split}.json')
    parser.add_argument('--input', default=None, help='Explicit path to answerer outputs JSON '
                        '(overrides --source)')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='IoU threshold for Acc@GQA (default: 0.5)')
    parser.add_argument('--iop-thresh', type=float, default=0.5,
                        help='IoP threshold for Acc@GQA (default: 0.5)')
    args = parser.parse_args()

    if args.input:
        input_path = args.input
    elif args.source == 'qwen':
        input_path = f'dataset/nextgqa/answerer_outputs_qwen_{args.split}.json'
    else:
        input_path = f'dataset/nextgqa/answerer_outputs_{args.split}.json'
    print(f'Loading: {input_path}')
    with open(input_path, encoding='utf-8') as f:
        results = json.load(f)
    print(f'Total entries: {len(results)}')

    evaluate_qa(results, iou_thresh=args.iou_thresh, iop_thresh=args.iop_thresh)


if __name__ == '__main__':
    main()
