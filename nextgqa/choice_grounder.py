"""
NExT-GQA Choice-Aware Grounder
================================
Thay vì dùng một vague grounding_description (answer-agnostic), grounder này
chạy BLIP-2 localize() riêng cho mỗi trong 5 choice query từ file
test_rewrite_5_choices.csv (query_a … query_e).

Cách hoạt động:
  1. Với mỗi câu hỏi, lấy 5 query đã viết lại theo từng choice.
  2. Với mỗi query chạy localize() → collect proposals → select_proposal().
  3. Best proposal của mỗi choice = props[0] (top-ranked sau re-rank).
  4. Choice có confidence cao nhất → predicted_answer_idx + best_proposal.

Input :  test_rewrite_5_choices.csv   (query_a..query_e per question)
         gsub_{split}.json            (duration + gt_segments)
         blip2_features/{vid}.npy     (BLIP-2 visual features)
Output:  choice_grounder_outputs_{split}.json

Compatible với eval_qa.py:
  required fields: predicted_answer, predicted_answer_idx, best_proposal,
                   gt_segments, answer_idx

Usage:
    python -m nextgqa.choice_grounder --split test
    python -m nextgqa.choice_grounder --split test --first_n 1000
    python -m nextgqa.choice_grounder --split test --dry_run
"""

import os
import csv
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from data_configs import DATASETS
from vlm_localizer import localize
from llm_prompting import select_proposal


CHOICE_KEYS = ['query_a', 'query_b', 'query_c', 'query_d', 'query_e']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gsub(gsub_path):
    """Load grounding JSON → {vid: {duration, location: {qid_str: [[s,e],...]}}}."""
    with open(gsub_path) as f:
        return json.load(f)


def load_5choice_csv(csv_path):
    """
    Load test_rewrite_5_choices.csv.
    Returns {(video_id, qid_int): {
        question, type, answer, answer_idx,
        choices: [a0..a4],
        queries: [query_a..query_e]
    }}
    """
    data = {}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            vid = row['video_id']
            qid = int(row['qid'])
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            queries  = [row[k] for k in CHOICE_KEYS]
            answer_text = row['answer']
            answer_idx  = next(
                (i for i, c in enumerate(choices) if c.strip() == answer_text.strip()), -1
            )
            data[(vid, qid)] = {
                'question':   row['question'],
                'type':       row['type'],
                'choices':    choices,
                'queries':    queries,   # [q_a, q_b, q_c, q_d, q_e]
                'answer':     answer_text,
                'answer_idx': answer_idx,
            }
    return data


# ── Core: ground one choice query ─────────────────────────────────────────────

def ground_for_choice(video_feature, duration, query_text, stride, max_stride):
    """
    Chạy BLIP-2 localize() cho một choice query.
    Returns best proposal [start, end, conf] (as list).
    """
    query_json = [{'descriptions': [query_text]}]
    answers = localize(video_feature, duration, query_json, stride, max_stride)

    proposals = []
    for t in range(3):
        proposals += [
            [p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']]
            for p in answers if len(p['response']) > t
        ]

    if not proposals:
        return [0.0, duration, 0.0]

    ranked = select_proposal(np.array(proposals))  # IoU-weighted re-rank
    return ranked[0].tolist()                       # [start, end, conf]


# ── Main runner ───────────────────────────────────────────────────────────────

def run_choice_grounder(split, feature_path, stride, max_stride_factor,
                        csv_5choice_path, gsub_path, output_path,
                        resume=True, dry_run=False, first_n=None):
    """Run choice-aware grounder trên toàn bộ (hoặc first_n) câu hỏi."""

    print(f'Loading data for split={split}...')
    gsub     = load_gsub(gsub_path)
    csv_data = load_5choice_csv(csv_5choice_path)
    print(f'  Videos in gsub   : {len(gsub)}')
    print(f'  Questions in CSV : {len(csv_data)}')

    # Resume
    existing = {}
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            saved = json.load(f)
        for entry in saved:
            existing[(entry['video_id'], entry['qid'])] = entry
        print(f'  Resumed: {len(existing)} questions already done')

    # Build task list
    tasks = []
    for (vid, qid), meta in csv_data.items():
        if (vid, qid) in existing:
            continue  # already done
        if vid not in gsub:
            continue  # no duration/GT info

        feature_file = os.path.join(feature_path, vid + '.npy')
        if not os.path.exists(feature_file):
            continue

        duration    = float(gsub[vid]['duration'])
        gt_segments = gsub[vid].get('location', {}).get(str(qid), None)

        tasks.append({
            'vid':          vid,
            'qid':          qid,
            'question':     meta['question'],
            'type':         meta['type'],
            'choices':      meta['choices'],
            'queries':      meta['queries'],
            'answer':       meta['answer'],
            'answer_idx':   meta['answer_idx'],
            'duration':     duration,
            'gt_segments':  gt_segments,
            'feature_file': feature_file,
        })

    print(f'  Questions to ground: {len(tasks)}')

    if dry_run:
        tasks = tasks[:10]
        print(f'  DRY RUN: processing first 10 questions')
    elif first_n is not None:
        tasks = tasks[:first_n]
        print(f'  first_n={first_n}: processing first {len(tasks)} questions')

    results = list(existing.values())

    # Group by video to avoid reloading features
    vid_to_tasks = defaultdict(list)
    for t in tasks:
        vid_to_tasks[t['vid']].append(t)

    pbar = tqdm(vid_to_tasks.items(), desc='Choice Grounding')
    for vid, vtasks in pbar:
        pbar.set_postfix({'vid': vid[:14], 'done': len(results)})

        try:
            video_feature = np.load(vtasks[0]['feature_file'])
        except Exception as e:
            tqdm.write(f'  [ERROR] load feature {vid}: {e}')
            continue

        max_stride = int(video_feature.shape[0] * max_stride_factor)

        for task in vtasks:
            try:
                # ── Run BLIP-2 for each of 5 choice queries ──
                best_per_choice = []   # [[start, end, conf], ...]
                for q_text in task['queries']:
                    bp = ground_for_choice(
                        video_feature, task['duration'],
                        q_text, stride, max_stride
                    )
                    best_per_choice.append(bp)

                # ── Pick winner: choice with highest confidence ──
                confs = [bp[2] for bp in best_per_choice]
                pred_idx    = int(np.argmax(confs))
                pred_answer = task['choices'][pred_idx]
                best_prop   = best_per_choice[pred_idx]

                results.append({
                    'video_id':           vid,
                    'qid':                task['qid'],
                    'question':           task['question'],
                    'type':               task['type'],
                    'choices':            task['choices'],
                    'answer':             task['answer'],
                    'answer_idx':         task['answer_idx'],
                    'gt_segments':        task['gt_segments'],
                    'duration':           task['duration'],
                    'predicted_answer':     pred_answer,
                    'predicted_answer_idx': pred_idx,
                    'best_proposal':        best_prop,       # [start, end, conf]
                    'choice_proposals':     best_per_choice, # [[s,e,c] × 5]
                })

            except Exception as e:
                tqdm.write(f'  [ERROR] {vid} qid={task["qid"]}: {e}')

        # Checkpoint after each video
        with open(output_path, 'w') as f:
            json.dump(results, f)

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*50}')
    print(f'Choice grounding done!')
    print(f'  Total results: {len(results)}')
    print(f'  Output: {output_path}')
    print(f'{"="*50}')
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run Choice-Aware Grounder on NExT-GQA')
    parser.add_argument('--split',    default='test', choices=['val', 'test'])
    parser.add_argument('--output',   default=None,   help='Output JSON path (optional)')
    parser.add_argument('--first_n',  type=int,       default=None, help='Process first N questions only')
    parser.add_argument('--resume',   action='store_true', default=True)
    parser.add_argument('--no_resume', dest='resume', action='store_false')
    parser.add_argument('--dry_run',  action='store_true', help='Process 10 questions only')
    args = parser.parse_args()

    cfg   = DATASETS['nextgqa']
    split = cfg['splits'][args.split]

    feature_path      = cfg['feature_path']
    stride            = cfg['stride']
    max_stride_factor = cfg['max_stride_factor']

    csv_5choice_path = f'dataset/nextgqa/test_rewrite_5_choices.csv'
    gsub_path        = split['grounding_file']
    output_path      = args.output or f'dataset/nextgqa/choice_grounder_outputs_{args.split}.json'

    run_choice_grounder(
        split=args.split,
        feature_path=feature_path,
        stride=stride,
        max_stride_factor=max_stride_factor,
        csv_5choice_path=csv_5choice_path,
        gsub_path=gsub_path,
        output_path=output_path,
        resume=args.resume,
        dry_run=args.dry_run,
        first_n=args.first_n,
    )


if __name__ == '__main__':
    main()
