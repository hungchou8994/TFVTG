"""
NExT-GQA Data Loader
====================
Converts NExT-GQA CSV + grounding JSON into a unified format
for the TFVTG pipeline.

NExT-GQA CSV columns:
  video_id, frame_count, width, height, question, answer, qid, type, a0, a1, a2, a3, a4

NExT-GQA grounding JSON (gsub_val.json / gsub_test.json):
  { video_id: { duration, location: { qid: [[start, end], ...] }, fps } }

Output format (per video):
  {
    video_id: {
      "duration": float,
      "questions": [
        {
          "question": str,
          "answer": str,          # correct answer text
          "answer_idx": int,      # index in a0-a4 (0-based)
          "qid": int,
          "type": str,            # CW, CH, TN, TC, TP
          "choices": [str, str, str, str, str],  # a0-a4
          "grounding": [[start, end], ...] or None
        },
        ...
      ]
    }
  }
"""

import csv
import json
import argparse
from collections import defaultdict


def load_nextgqa(qa_csv_path, grounding_json_path):
    """Load and merge NExT-GQA QA + grounding data."""
    
    # Load grounding annotations
    with open(grounding_json_path, 'r') as f:
        gsub = json.load(f)
    
    # Load QA annotations from CSV
    data = defaultdict(lambda: {"duration": None, "questions": []})
    
    with open(qa_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['video_id']
            qid = row['qid']
            
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            answer_text = row['answer']
            
            # Find the correct answer index
            answer_idx = -1
            for i, c in enumerate(choices):
                if c.strip() == answer_text.strip():
                    answer_idx = i
                    break
            
            # Get grounding if available
            grounding = None
            if vid in gsub and qid in gsub[vid].get('location', {}):
                grounding = gsub[vid]['location'][qid]
            
            # Get duration from grounding json
            duration = gsub[vid]['duration'] if vid in gsub else None
            if duration is not None:
                data[vid]['duration'] = duration
            
            data[vid]['questions'].append({
                'question': row['question'],
                'answer': answer_text,
                'answer_idx': answer_idx,
                'qid': int(qid),
                'type': row['type'],
                'choices': choices,
                'grounding': grounding,
            })
    
    return dict(data)


def print_stats(data, split_name):
    """Print dataset statistics."""
    total_q = sum(len(v['questions']) for v in data.values())
    grounded_q = sum(
        1 for v in data.values() 
        for q in v['questions'] 
        if q['grounding'] is not None
    )
    
    type_counts = defaultdict(int)
    for v in data.values():
        for q in v['questions']:
            type_counts[q['type']] += 1
    
    print(f"\n{'='*50}")
    print(f"NExT-GQA {split_name} split statistics:")
    print(f"{'='*50}")
    print(f"  Videos:              {len(data)}")
    print(f"  Total questions:     {total_q}")
    print(f"  Grounded questions:  {grounded_q} ({grounded_q/total_q*100:.1f}%)")
    print(f"  Question types:")
    for t in ['CW', 'CH', 'TN', 'TC', 'TP']:
        print(f"    {t}: {type_counts.get(t, 0)}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and inspect NExT-GQA dataset')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--save', action='store_true', help='Save merged data as JSON')
    args = parser.parse_args()
    
    qa_file = f'dataset/nextgqa/{args.split}.csv'
    grounding_file = f'dataset/nextgqa/gsub_{args.split}.json'
    
    data = load_nextgqa(qa_file, grounding_file)
    print_stats(data, args.split)
    
    # Show a sample
    sample_vid = next(iter(data))
    sample = data[sample_vid]
    print(f"Sample video: {sample_vid}")
    print(f"  Duration: {sample['duration']}s")
    print(f"  Questions: {len(sample['questions'])}")
    q = sample['questions'][0]
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answer']} (idx={q['answer_idx']})")
    print(f"  Choices: {q['choices']}")
    print(f"  Grounding: {q['grounding']}")
    print(f"  Type: {q['type']}")
    
    if args.save:
        out_path = f'dataset/nextgqa/{args.split}_merged.json'
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {out_path}")
