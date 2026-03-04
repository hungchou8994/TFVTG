"""
NExT-GQA Answerer — Qwen2-VL
==============================
Answers MCQ questions using Qwen2-VL-7B-Instruct (base model, no fine-tuning
needed for the answerer role per VideoMind design).

Given `best_proposal` from verifier_outputs, sends the trimmed video window
[start, end] directly to Qwen2-VL via the video_start/video_end fields in the
message content (no ffmpeg pre-trim needed).

Input : dataset/nextgqa/verifier_outputs_{split}.json
Output: dataset/nextgqa/answerer_outputs_qwen_{split}.json

Usage (local — CPU only, slow, for testing):
  python -m nextgqa.answerer_qwen --split test --model_path Qwen/Qwen2-VL-7B-Instruct

Usage (Colab with GPU):
  See run_answerer_qwen.ipynb
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import Optional

# ── VideoMind path ────────────────────────────────────────────────────────────
# Add VideoMind/ to sys.path so we can use its process_vision_info
# which supports video_start / video_end natively via decord.
_VIDEOMIND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VideoMind'))
if os.path.isdir(_VIDEOMIND_DIR) and _VIDEOMIND_DIR not in sys.path:
    sys.path.insert(0, _VIDEOMIND_DIR)

# ── Constants ─────────────────────────────────────────────────────────────────
ANSWER_LETTERS     = ['A', 'B', 'C', 'D', 'E']
MAP_VID_VIDOR_PATH = 'dataset/nextgqa/map_vid_vidorID.json'
VIDEO_ROOT         = 'dataset/nextgqa/videos'


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_video_path(video_root: str, vidor_rel: str) -> Optional[str]:
    if not vidor_rel:
        return None
    folder, vid_file = vidor_rel.split('/')
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        p = os.path.join(video_root, folder, vid_file + ext)
        if os.path.exists(p):
            return p
    return None


def _atomic_save(data: list, path: str) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def parse_response(response: str, num_options: int = 5) -> tuple:
    """Parse Qwen response → (letter A-E, 0-based index) or (None, None)."""
    valid = [chr(ord('A') + i) for i in range(num_options)]
    for ch in response.strip():
        if ch.upper() in valid:
            letter = ch.upper()
            return letter, ord(letter) - ord('A')
    return None, None


def build_prompt(question: str, choices: list) -> str:
    prompt = question.strip()
    if not prompt.endswith('?'):
        prompt += '?'
    prompt += '\nOptions:'
    for i, opt in enumerate(choices):
        label = chr(ord('A') + i)
        cap_opt = opt[0].upper() + opt[1:]
        prompt += f'\n({label}) {cap_opt}'
    prompt += '\nPlease only give the best option.'
    return prompt


# ── Main logic ────────────────────────────────────────────────────────────────

def run(args):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # Import process_vision_info — prefer VideoMind's version (supports
    # video_start/video_end via decord); fall back to qwen_vl_utils if missing.
    try:
        from videomind.dataset.utils import process_vision_info
        print('Vision processor: VideoMind (decord)')
    except ImportError:
        try:
            from qwen_vl_utils import process_vision_info
            print('Vision processor: qwen_vl_utils')
        except ImportError:
            raise ImportError(
                'Neither VideoMind nor qwen_vl_utils found. '
                'Install with: pip install qwen-vl-utils  or install VideoMind deps.'
            )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'\nLoading model: {args.model_path}')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map='auto',
        attn_implementation='sdpa',
    )
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs.pop('torch_dtype', None)
        print('  Quantization: int4 (BitsAndBytes)')

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, **load_kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, do_resize=False)
    device = next(model.parameters()).device
    print(f'  Loaded. device={device}  dtype={dtype}\n')

    # ── Load data ─────────────────────────────────────────────────────────────
    input_path  = args.input  or f'dataset/nextgqa/verifier_outputs_{args.split}.json'
    output_path = args.output or f'dataset/nextgqa/answerer_outputs_qwen_{args.split}.json'

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} entries from {input_path}')

    # Resume from existing output
    if os.path.exists(output_path) and not args.no_resume:
        with open(output_path, encoding='utf-8') as f:
            existing = json.load(f)
        existing_map = {(str(e.get('video_id')), str(e.get('qid'))): e for e in existing}
        resumed = 0
        for entry in data:
            key = (str(entry.get('video_id')), str(entry.get('qid')))
            ex  = existing_map.get(key)
            if ex and 'predicted_answer' in ex:
                entry['predicted_answer']     = ex['predicted_answer']
                entry['predicted_answer_idx'] = ex['predicted_answer_idx']
                entry['answer_reasoning']     = ex.get('answer_reasoning', '')
                resumed += 1
        print(f'  Resumed: {resumed} already answered')
    elif args.no_resume:
        for entry in data:
            entry.pop('predicted_answer', None)
            entry.pop('predicted_answer_idx', None)
            entry.pop('answer_reasoning', None)
        print('  --no_resume: cleared all previous predictions')

    # Load video ID map
    with open(MAP_VID_VIDOR_PATH, encoding='utf-8') as f:
        vid_map = json.load(f)

    # Build list of entries to process
    to_process     = []
    n_done         = 0
    n_no_proposal  = 0
    n_no_video     = 0
    for entry in data:
        if 'predicted_answer' in entry:
            n_done += 1
            continue
        if 'best_proposal' not in entry:
            n_no_proposal += 1
            continue
        vid_id    = str(entry.get('video_id', ''))
        vidor_rel = vid_map.get(vid_id)
        vpath     = find_video_path(VIDEO_ROOT, vidor_rel) if vidor_rel else None
        if vpath is None:
            n_no_video += 1
            continue
        to_process.append((entry, vpath))

    if args.first_n:
        to_process = to_process[:args.first_n]

    print(f'  Already done     : {n_done}')
    print(f'  No best_proposal : {n_no_proposal}')
    print(f'  Video not found  : {n_no_video}')
    print(f'  To answer        : {len(to_process)}')

    if not to_process:
        print('Nothing to do.')
        return

    # ── Inference loop ────────────────────────────────────────────────────────
    ok = fail = 0
    pbar = tqdm(to_process, desc='Answering (Qwen2-VL)')

    for i, (entry, vpath) in enumerate(pbar):
        bp    = entry['best_proposal']
        start = float(bp[0])
        end   = float(bp[1])
        dur   = float(entry.get('duration', end + 1))

        # Clamp to valid range, ensure minimum window
        start = max(0.0, min(start, dur))
        end   = min(end, dur)
        end   = max(start + 0.5, end)

        choices  = entry.get('choices', entry.get('options', []))
        question = entry.get('question', '')
        prompt   = build_prompt(question, choices)

        messages = [{
            'role': 'user',
            'content': [
                {
                    'type':        'video',
                    'video':       vpath,
                    'video_start': start,
                    'video_end':   end,
                    'min_pixels':  128 * 28 * 28,
                    'max_pixels':  256 * 28 * 28,
                    'max_frames':  args.max_frames,
                    'fps':         2.0,
                },
                {'type': 'text', 'text': prompt},
            ],
        }]

        try:
            text        = processor.apply_chat_template(messages, add_generation_prompt=True)
            text       += 'Best Option: ('   # force MCQ-style prefix
            images, videos = process_vision_info(messages)
            inputs      = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            inputs      = inputs.to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=None,
                    max_new_tokens=64,
                )

            generated = output_ids[0, inputs.input_ids.size(1):]
            if len(generated) and generated[-1] == processor.tokenizer.eos_token_id:
                generated = generated[:-1]
            response = processor.decode(generated, clean_up_tokenization_spaces=False)

            letter, idx = parse_response(response, num_options=len(choices))
            if letter is None:
                print(f"\n  [WARN] qid={entry.get('qid')}: unparseable: {repr(response)}")
                fail += 1
            else:
                entry['predicted_answer']     = letter
                entry['predicted_answer_idx'] = idx
                entry['answer_reasoning']     = response.strip()
                ok += 1

        except Exception as exc:
            print(f"\n  [ERROR] qid={entry.get('qid')}: {type(exc).__name__}: {exc}")
            fail += 1

        pbar.set_postfix(ok=ok, fail=fail)

        if (i + 1) % args.save_every == 0:
            _atomic_save(data, output_path)

    _atomic_save(data, output_path)

    total_answered = sum(1 for e in data if 'predicted_answer' in e)
    print('\n' + '=' * 58)
    print('Answerer (Qwen2-VL) done!')
    print(f'  Success  : {ok}')
    print(f'  Failed   : {fail}')
    print(f'  Total    : {total_answered}/{len(data)}')
    print(f'  Output   : {output_path}')
    print('=' * 58)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NExT-GQA Answerer using Qwen2-VL')
    parser.add_argument('--split',      default='test', choices=['test', 'val'],
                        help='Dataset split (default: test)')
    parser.add_argument('--input',      default=None,
                        help='Override input JSON path')
    parser.add_argument('--output',     default=None,
                        help='Override output JSON path')
    parser.add_argument('--model_path', default='Qwen/Qwen2-VL-7B-Instruct',
                        help='HuggingFace model ID or local path (default: Qwen/Qwen2-VL-7B-Instruct)')
    parser.add_argument('--max_frames', type=int, default=32,
                        help='Max video frames to sample (default: 32)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N questions (default: 50)')
    parser.add_argument('--first_n',   type=int, default=None,
                        help='Process only the first N unanswered entries (for testing)')
    parser.add_argument('--no_resume', action='store_true',
                        help='Ignore existing output and restart from scratch')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in int4 (BitsAndBytes) — for GPUs with <16GB VRAM')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
