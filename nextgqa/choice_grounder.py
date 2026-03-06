"""
NExT-GQA Choice-Aware Grounder
================================
Thay vì dùng một vague grounding_description (answer-agnostic), grounder này
chạy BLIP-2 localize() riêng cho mỗi trong 5 choice query từ file
test_rewrite_5_choices.csv (query_a … query_e).

Cách hoạt động:
  1. Với mỗi câu hỏi, lấy 5 query đã viết lại theo từng choice.
  2. Với mỗi query gọi generate_proposal() → tính 3 thành phần score:
     - S_temp  = mean(top-3 static scores)  [foreground-background contrast]
     - S_sem   = cosine(mean_video, query)  [global semantic alignment]
     - S_local = cosine(mean_moment, query) [local moment semantic]
  3. Composite score:
     S_i = 0.6*S_temp_norm + 0.25*S_sem + 0.15*S_local
     S_i *= (1 + 0.3 * sharpness)     [bonus choice grounding rõ nét]
     S_final = S_i - max(S_j, j≠i)   [contrastive: tăng margin giữa choices]
  4. argmax(S_final) → blip2_pred_idx (BLIP-2 standalone pred — ablation)

LÝ DO không dùng localize():
  localize() normalize scores / scores.max() → top proposal CỦA MỌI CHOICE đều = 1.0
  → argmax luôn chọn choice 0, không có sự phân biệt.
  ori_scores là cosine similarity thô, có thể so sánh trực tiếp giữa các choice.

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
from vlm_localizer import generate_proposal, calc_scores_per_query
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

def ground_for_choice(video_feature, duration, query_text, stride, max_stride,
                      _precomputed_scores=None):
    """
    Chạy BLIP-2 generate_proposal() cho một choice query.
    _precomputed_scores: [1, T] tensor — nếu đã tính trước từ calc_scores_per_query (batched).
    Returns (all_props, s_temp, s_sem, s_local, sharpness).
    """
    proposals, filt_scores, pre_proposals, ori_scores = generate_proposal(
        video_feature, [query_text], stride, max_stride,
        _precomputed_scores=_precomputed_scores
    )

    # ori_scores: [1, T] raw cosine sim mỗi frame window với query
    T = ori_scores.shape[-1]

    # S_sem: global semantic — mean cosine sim toàn video (normalize [-1,1] → [0,1])
    s_sem = float(((ori_scores.mean() + 1.0) / 2.0).cpu())

    # Fallback nếu không có proposal
    if len(proposals[0]) == 0:
        return np.array([[0.0, duration, 0.0]]), 0.0, s_sem, s_sem, 0.0

    static_pred  = (proposals[0][:10] * duration).cpu().numpy()   # [N, 2]
    dynamic_pred = (pre_proposals[0][:10] * duration).cpu().numpy()  # [N]
    scores_raw   = filt_scores[0][:10].cpu().numpy()               # [N] static scores

    # S_temp = mean of top-3 static scores (foreground-background contrast)
    top3 = sorted(scores_raw, reverse=True)[:3]
    s_temp = float(np.mean(top3))

    # Sharpness = relative gap top1 vs top3 (clamped [0,1])
    if len(top3) >= 3:
        sharpness = float(np.clip((top3[0] - top3[-1]) / (abs(top3[0]) + 1e-8), 0.0, 1.0))
    else:
        sharpness = 0.0

    # Build ALL proposals (dynamic_start, static_end, norm_conf)
    max_s = scores_raw.max()
    norm_scores = scores_raw / max_s if max_s > 0 else scores_raw
    all_props = np.array([
        [float(dynamic_pred[i]), float(static_pred[i][1]), float(norm_scores[i])]
        for i in range(len(static_pred))
    ])  # [N, 3]

    # S_local: use best (top-1) proposal window for local semantic
    best_prop = all_props[0]  # already sorted by static score (desc)
    start_f = max(0, int(best_prop[0] / duration * T))
    end_f   = min(T, max(start_f + 1, int(best_prop[1] / duration * T)))
    s_local_raw = float(ori_scores[0, start_f:end_f].mean().cpu())
    s_local = (s_local_raw + 1.0) / 2.0

    return all_props, s_temp, s_sem, s_local, sharpness


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

        # ── Batch encode ALL queries for ALL tasks on this video in 1 GPU call ──
        # vtasks may have N questions × 5 queries = 5N queries total
        all_queries_flat = [q for task in vtasks for q in task['queries']]  # [N*5]
        try:
            all_scores_flat = calc_scores_per_query(
                video_feature, all_queries_flat
            )  # [N*5, T]
        except Exception as e:
            tqdm.write(f'  [WARN] batch encode failed for {vid}, falling back: {e}')
            all_scores_flat = None

        for t_idx, task in enumerate(vtasks):
            try:
                # Slice precomputed scores for this task [5, T]
                if all_scores_flat is not None:
                    all_scores_batch = all_scores_flat[t_idx*5:(t_idx+1)*5]
                else:
                    all_scores_batch = calc_scores_per_query(video_feature, task['queries'])

                # ── Run BLIP-2 for each of 5 choice queries ──
                best_per_choice = []   # [[start, end, norm_conf] × 5] — top-1 per choice (debug)
                all_pool = []          # all proposals from all 5 queries for global top-5
                s_temps, s_sems, s_locals, sharpnesses = [], [], [], []

                for i, q_text in enumerate(task['queries']):
                    scores_i = all_scores_batch[i:i+1]  # [1, T] for this query
                    all_props, s_temp, s_sem, s_local, sharp = ground_for_choice(
                        video_feature, task['duration'],
                        q_text, stride, max_stride,
                        _precomputed_scores=scores_i
                    )
                    best_per_choice.append(all_props[0].tolist())  # top-1 per choice
                    all_pool.extend(all_props.tolist())            # all proposals pooled
                    s_temps.append(s_temp)
                    s_sems.append(s_sem)
                    s_locals.append(s_local)
                    sharpnesses.append(sharp)

                # ── Pool all proposals → global top-5 by conf (NMS already applied per-query) ──
                all_pool_arr = np.array(all_pool)  # [N_total, 3]
                # Sort by conf descending, deduplicate overlapping by simple IoU NMS
                sorted_idx = np.argsort(-all_pool_arr[:, 2])
                all_pool_arr = all_pool_arr[sorted_idx]
                # Greedy NMS across pool
                def pool_iou(a, b):
                    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
                    union = max(a[1], b[1]) - min(a[0], b[0])
                    return inter / union if union > 0 else 0.0
                kept = []
                for prop in all_pool_arr:
                    if all(pool_iou(prop, k) < 0.5 for k in kept):
                        kept.append(prop)
                    if len(kept) == 5:
                        break
                # Pad if fewer than 5
                while len(kept) < 5:
                    kept.append(kept[-1] if kept else [0.0, task['duration'], 0.0])
                top5_global = [p.tolist() for p in kept]

                # ── Composite scoring ──────────────────────────────────────
                # Normalize s_temp across 5 choices → [0,1] (min-max)
                # (s_temp là static score chưa chuẩn hóa, cần scale)
                t_arr = np.array(s_temps)
                t_min, t_max = t_arr.min(), t_arr.max()
                if t_max > t_min:
                    t_norm = (t_arr - t_min) / (t_max - t_min)
                else:
                    t_norm = np.full(5, 0.5)

                s_arr = np.array(s_sems)
                l_arr = np.array(s_locals)
                sh_arr = np.array(sharpnesses)

                # S_i = 0.6*S_temp + 0.25*S_sem + 0.15*S_local
                composite = 0.6 * t_norm + 0.25 * s_arr + 0.15 * l_arr

                # Sharpness bonus: S_i *= (1 + 0.3 * sharpness)
                composite = composite * (1.0 + 0.3 * sh_arr)

                # Contrastive: S_i -= max(S_j for j != i)
                contrastive = np.array([
                    composite[i] - np.max(np.delete(composite, i))
                    for i in range(5)
                ])

                blip2_pred_idx = int(np.argmax(contrastive))

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
                    # ── Fields cho verifier pipeline ──
                    'top5_proposals':        top5_global,              # global top-5 pooled — verifier cần
                    'grounding_description': task['question'],
                    # ── Composite scoring debug ──
                    'choice_proposals':      best_per_choice,          # top-1 per choice (debug)
                    'choice_s_temp':         t_norm.tolist(),           # normalized temporal
                    'choice_s_sem':          s_arr.tolist(),            # global semantic
                    'choice_s_local':        l_arr.tolist(),            # local semantic
                    'choice_sharpness':      sh_arr.tolist(),           # sharpness per choice
                    'choice_composite':      composite.tolist(),        # trước contrastive
                    'choice_contrastive':    contrastive.tolist(),      # score cuối
                    'blip2_pred_idx':        blip2_pred_idx,            # BLIP-2 standalone pred
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
