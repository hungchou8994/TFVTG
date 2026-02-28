"""
NExT-GQA Grounder
==================
Wraps TFVTG vlm_localizer to produce top-5 temporal proposals per question.

Tái sử dụng logic eval_with_llm() từ evaluate.py nhưng:
  - Input : llm_outputs_{split}.json          (query rewrite output từ Bước 5)
            gsub_{split}.json                 (grounding GT + duration)
            {split}.csv                       (choices, answer_idx)
            blip2_features/{video_id}.npy     (BLIP-2 visual features từ Bước 2)
  - Output: grounder_outputs_{split}.json  (top-5 proposals per question)

Cần GPU (BLIP-2 load lúc import vlm_localizer).
Chạy trên Google Colab sau khi extract features xong.

Usage:
    python -m nextgqa.grounder --split test
    python -m nextgqa.grounder --split test --dry_run
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
from llm_prompting import select_proposal, filter_and_integrate


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gsub(gsub_path):
    """Load grounding JSON. Returns {vid: {duration, location: {qid_str: [[s,e],...]}}}."""
    with open(gsub_path) as f:
        return json.load(f)


def load_csv_meta(csv_path):
    """Load CSV metadata. Returns {vid: {qid_int: {choices, answer_idx, type, question}}}."""
    meta = defaultdict(dict)
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            vid = row['video_id']
            qid = int(row['qid'])
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            answer_text = row['answer']
            answer_idx = next(
                (i for i, c in enumerate(choices) if c.strip() == answer_text.strip()), -1
            )
            meta[vid][qid] = {
                'question': row['question'],
                'type': row['type'],
                'choices': choices,
                'answer': answer_text,
                'answer_idx': answer_idx,
            }
    return dict(meta)


def load_llm_outputs(llm_path):
    """Load llm_outputs JSON from query rewrite step."""
    with open(llm_path) as f:
        return json.load(f)


# ── Core grounding logic (mirrors eval_with_llm from evaluate.py) ─────────────

def ground_one_question(video_feature, duration, sentence, response, stride, max_stride):
    """
    Run TFVTG grounding for a single question.
    Returns sorted proposals array of shape [N, 3] = [[start, end, conf], ...]
    """
    sub_query_proposals = []
    relation = 'single-query'

    if response and 'query_json' in response:
        relation = response.get('relationship', 'single-query')
        # Sub-queries (id >= 1)
        for sq in response['query_json']:
            if sq['sub_query_id'] == 0:
                continue
            query_json = [{'descriptions': sq['descriptions']}]
            answers = localize(video_feature, duration, query_json, stride, max_stride)
            proposals = []
            for t in range(3):
                proposals += [
                    [p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']]
                    for p in answers if len(p['response']) > t
                ]
            if proposals:
                sub_query_proposals.append(select_proposal(np.array(proposals))[:3])

    # Main query (id == 0) + raw sentence
    query_json = [{'descriptions': [sentence]}]
    if response and 'query_json' in response:
        main_sq = next((sq for sq in response['query_json'] if sq['sub_query_id'] == 0), None)
        if main_sq:
            query_json += [{'descriptions': main_sq['descriptions']}]

    answers = localize(video_feature, duration, query_json, stride, max_stride)
    proposals = []
    for t in range(3):
        proposals += [
            [p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']]
            for p in answers if len(p['response']) > t
        ]

    # Merge main + sub-query proposals, re-rank
    all_proposals = proposals[:7] + filter_and_integrate(sub_query_proposals, relation)
    if not all_proposals:
        return np.array([[0.0, duration, 0.5]])

    return select_proposal(np.array(all_proposals))


# ── Main runner ───────────────────────────────────────────────────────────────

def run_grounder(split, feature_path, stride, max_stride_factor,
                 llm_outputs_path, gsub_path, csv_path,
                 output_path, resume=True, dry_run=False, top_k=5):
    """
    Run grounder on all questions in the split.
    Saves results to output_path as a list of dicts.
    """
    # Load data
    print(f'Loading data for split={split}...')
    llm_outputs = load_llm_outputs(llm_outputs_path)
    gsub        = load_gsub(gsub_path)
    csv_meta    = load_csv_meta(csv_path)
    print(f'  Videos in llm_outputs: {len(llm_outputs)}')
    print(f'  Videos in gsub:        {len(gsub)}')

    # Resume: load existing results
    existing = {}  # key: (vid, qid) → result dict
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            saved = json.load(f)
        for entry in saved:
            existing[(entry['video_id'], entry['qid'])] = entry
        print(f'  Resumed: {len(existing)} questions already done')

    # Build task list
    tasks = []
    for vid, ann in llm_outputs.items():
        if vid not in gsub:
            continue
        duration = gsub[vid]['duration']
        location = gsub[vid].get('location', {})

        feature_file = os.path.join(feature_path, vid + '.npy')
        if not os.path.exists(feature_file):
            continue

        for i, sentence in enumerate(ann['sentences']):
            meta = ann['meta']['questions'][i]
            qid  = meta['qid']

            if (vid, qid) in existing:
                continue  # already done

            # GT segments (may be None if not in gsub location)
            gt_segments = location.get(str(qid), None)

            # CSV meta (choices etc.)
            csv_q = csv_meta.get(vid, {}).get(qid, {})

            tasks.append({
                'vid': vid,
                'qid': qid,
                'sentence': sentence,
                'response': ann['response'][i],
                'duration': float(duration),
                'gt_segments': gt_segments,
                'feature_file': feature_file,
                'type': meta.get('type', csv_q.get('type', '')),
                'choices': csv_q.get('choices', meta.get('choices', [])),
                'answer': meta.get('answer', csv_q.get('answer', '')),
                'answer_idx': meta.get('answer_idx', csv_q.get('answer_idx', -1)),
            })

    print(f'  Questions to ground: {len(tasks)}')

    if dry_run:
        tasks = tasks[:10]
        print(f'  DRY RUN: processing first 10 questions')

    results = list(existing.values())

    # Group tasks by video to avoid reloading features
    vid_to_tasks = defaultdict(list)
    for t in tasks:
        vid_to_tasks[t['vid']].append(t)

    dataset_cfg = DATASETS['nextgqa']
    max_stride_abs = None  # computed per video from feature shape

    pbar = tqdm(vid_to_tasks.items(), desc='Grounding')
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
                props = ground_one_question(
                    video_feature,
                    task['duration'],
                    task['sentence'],
                    task['response'],
                    stride,
                    max_stride,
                )
                top5 = props[:top_k].tolist()

                results.append({
                    'video_id':             vid,
                    'qid':                  task['qid'],
                    'question':             task['sentence'],
                    'type':                 task['type'],
                    'choices':              task['choices'],
                    'answer':               task['answer'],
                    'answer_idx':           task['answer_idx'],
                    'gt_segments':          task['gt_segments'],
                    'duration':             task['duration'],
                    'grounding_description': (task['response'] or {}).get('grounding_description', ''),
                    'relationship':         (task['response'] or {}).get('relationship', 'single-query'),
                    'top5_proposals':       top5,   # [[start, end, conf], ...]
                })
            except Exception as e:
                tqdm.write(f'  [ERROR] {vid} qid={task["qid"]}: {e}')

        # Save checkpoint after each video
        with open(output_path, 'w') as f:
            json.dump(results, f)

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*50}')
    print(f'Grounding done!')
    print(f'  Total results: {len(results)}')
    print(f'  Output: {output_path}')
    print(f'{"="*50}')
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run TFVTG Grounder on NExT-GQA')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--output', default=None, help='Output JSON path')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', dest='resume', action='store_false')
    parser.add_argument('--dry_run', action='store_true', help='Process 10 questions only')
    parser.add_argument('--top_k', type=int, default=5, help='Number of proposals to keep')
    args = parser.parse_args()

    cfg   = DATASETS['nextgqa']
    split = cfg['splits'][args.split]

    feature_path     = cfg['feature_path']
    stride           = cfg['stride']
    max_stride_factor = cfg['max_stride_factor']
    llm_outputs_path = f'dataset/nextgqa/llm_outputs_{args.split}.json'
    gsub_path        = split['grounding_file']
    csv_path         = split['qa_file']
    output_path      = args.output or f'dataset/nextgqa/grounder_outputs_{args.split}.json'

    run_grounder(
        split=args.split,
        feature_path=feature_path,
        stride=stride,
        max_stride_factor=max_stride_factor,
        llm_outputs_path=llm_outputs_path,
        gsub_path=gsub_path,
        csv_path=csv_path,
        output_path=output_path,
        resume=args.resume,
        dry_run=args.dry_run,
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
