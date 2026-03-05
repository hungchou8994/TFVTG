"""
NExT-GQA Gemini Baseline (Ablation)
=====================================
Answers MCQ questions by sending the **full video** directly to Gemini 2.5 Flash
with no temporal grounding whatsoever — no BLIP-2 proposals, no verifier re-ranking,
no clip trimming.

Purpose: ablation study baseline to measure the contribution of the
  BLIP-2 grounder + Gemini verifier in the full pipeline.

Input : dataset/nextgqa/verifier_outputs_{split}.json  (only needs video_id,
         question, choices, type, answer_idx — ignores best_proposal completely)
Output: dataset/nextgqa/gemini_baseline_outputs_{split}.json
         (same format as answerer_outputs: predicted_answer, predicted_answer_idx)

Reuses video_upload_cache.json (full-video uploads shared with verifier).

Usage:
  python -m nextgqa.gemini_baseline --split test --workers 10 --rpm 100
  python -m nextgqa.gemini_baseline --split test --first_n 200
"""

import os
import json
import time
import argparse
import asyncio
from datetime import datetime, timezone
from tqdm import tqdm
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

VIDEO_UPLOAD_CACHE_PATH = 'dataset/nextgqa/video_upload_cache.json'
UPLOAD_TTL_SECONDS      = 47 * 3600
MAP_VID_VIDOR_PATH      = 'dataset/nextgqa/map_vid_vidorID.json'
VIDEO_ROOT              = 'dataset/nextgqa/videos'

ANSWER_LETTERS = ['A', 'B', 'C', 'D', 'E']

MIME_MAP = {'.mp4': 'video/mp4', '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime', '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'}

SYSTEM_INSTRUCTION = (
    "You are a video question answering assistant. "
    "You will be given a video and a multiple-choice question with 5 options (A–E). "
    "Watch the entire video carefully, then: "
    "(1) select the single best answer (A/B/C/D/E), and "
    "(2) identify the temporal segment [start_sec, end_sec] in the video "
    "that is most relevant to answering the question. "
    "Fill all fields in the JSON response."
)


# ── Pydantic schema for structured output ─────────────────────────────────────

class BaselineResponse(BaseModel):
    answer: str = Field(description="Best option letter: A, B, C, D, or E")
    start_sec: float = Field(description="Start time in seconds of the most relevant segment")
    end_sec: float = Field(description="End time in seconds of the most relevant segment")
    reasoning: str = Field(description="One sentence explanation")


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm: int):
        self._interval = 60.0 / max(1, rpm)
        self._lock = asyncio.Lock()
        self._last_time = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_time = asyncio.get_event_loop().time()


# ── Upload cache ──────────────────────────────────────────────────────────────

def load_upload_cache() -> dict:
    if os.path.exists(VIDEO_UPLOAD_CACHE_PATH):
        try:
            with open(VIDEO_UPLOAD_CACHE_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_upload_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(VIDEO_UPLOAD_CACHE_PATH), exist_ok=True)
    tmp = VIDEO_UPLOAD_CACHE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, VIDEO_UPLOAD_CACHE_PATH)


def _cache_valid(entry: dict) -> bool:
    try:
        uploaded_at = datetime.fromisoformat(entry['uploaded_at'])
        age = (datetime.now(timezone.utc) - uploaded_at).total_seconds()
        return age < UPLOAD_TTL_SECONDS
    except Exception:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_video_path(vidor_rel: str) -> Optional[str]:
    if not vidor_rel:
        return None
    folder, vid_id = vidor_rel.split('/')
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        p = os.path.join(VIDEO_ROOT, folder, vid_id + ext)
        if os.path.exists(p):
            return p
    return None


def _atomic_save(data: list, path: str) -> None:
    OUTPUT_KEYS = ('video_id', 'qid', 'question', 'type', 'choices',
                   'answer', 'answer_idx', 'gt_segments', 'duration',
                   'predicted_answer', 'predicted_answer_idx',
                   'answer_reasoning', 'best_proposal')
    clean = [{k: e[k] for k in OUTPUT_KEYS if k in e}
             for e in data if 'predicted_answer' in e]
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return ('429' in msg or 'RESOURCE_EXHAUSTED' in msg) and not _is_daily_quota_error(exc)


def _is_daily_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return 'per_day' in msg.lower() or 'PerDay' in msg or 'per_model_per_day' in msg


# ── Video upload ──────────────────────────────────────────────────────────────

async def upload_video_async(sync_client, video_path: str,
                              cache: dict, cache_lock: asyncio.Lock) -> Optional[str]:
    abs_path = os.path.abspath(video_path)
    async with cache_lock:
        entry = cache.get(abs_path)
        if entry and _cache_valid(entry):
            return entry['file_uri']

    ext  = os.path.splitext(video_path)[1].lower()
    mime = MIME_MAP.get(ext, 'video/mp4')
    try:
        uploaded = await asyncio.to_thread(
            sync_client.files.upload, file=video_path,
            config={'mime_type': mime})
    except Exception as exc:
        print(f'  [upload] FAILED {os.path.basename(video_path)}: {exc}')
        return None

    file_name = uploaded.name
    info      = uploaded
    deadline  = time.monotonic() + 300.0
    try:
        while True:
            state = getattr(info, 'state', None)
            if state is not None:
                s = state.name if hasattr(state, 'name') else str(state)
                if s == 'ACTIVE':
                    break
                if s == 'FAILED':
                    print(f'  [upload] FAILED (processing): {file_name}')
                    return None
            if time.monotonic() > deadline:
                return None
            await asyncio.sleep(3.0)
            info = await asyncio.to_thread(sync_client.files.get, name=file_name)
    except Exception as exc:
        print(f'  [upload] Poll error {file_name}: {exc}')
        return None

    file_uri = info.uri
    async with cache_lock:
        cache[abs_path] = {
            'file_uri': file_uri, 'file_name': file_name,
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
        }
    return file_uri


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_contents(entry: dict, file_uri: str) -> list:
    from google.genai import types

    question     = entry['question']
    choices      = entry.get('choices', entry.get('options', []))
    choice_lines = '\n'.join(
        f'  {ANSWER_LETTERS[i]}. {c}' for i, c in enumerate(choices)
    )
    duration = entry.get('duration', 0)
    prompt = (
        f'Video duration: {duration:.1f} seconds.\n\n'
        f'Question: {question}\n\n'
        f'Choices:\n{choice_lines}\n\n'
        f'Select the best answer and identify the most relevant temporal segment.'
    )
    return [
        types.Part(file_data=types.FileData(file_uri=file_uri)),
        types.Part(text=prompt),
    ]


# ── Async runner ──────────────────────────────────────────────────────────────

async def _run_async(data: list, todo: list, vpaths: dict, output_path: str, args):
    from google import genai
    from google.genai import types

    api_key      = os.environ.get('GOOGLE_API_KEY', '')
    sync_client  = genai.Client(api_key=api_key)
    async_client = sync_client.aio

    rate_limiter  = RateLimiter(args.rpm)
    semaphore     = asyncio.Semaphore(args.workers)
    up_semaphore  = asyncio.Semaphore(args.upload_workers)
    cache         = load_upload_cache()
    cache_lock    = asyncio.Lock()

    # unique videos to upload
    unique_vpaths = sorted(set(vpaths.values()), key=lambda x: x or '')
    unique_vpaths = [p for p in unique_vpaths if p]
    video_uri_map: dict[str, Optional[str]] = {}

    print(f'Phase 1 — Uploading {len(unique_vpaths)} unique videos ({args.upload_workers} concurrent)...')

    async def upload_one(vpath):
        async with up_semaphore:
            uri = await upload_video_async(sync_client, vpath, cache, cache_lock)
        video_uri_map[vpath] = uri

    await asyncio.gather(*(upload_one(p) for p in unique_vpaths), return_exceptions=True)
    save_upload_cache(cache)

    ok_uploads = sum(1 for v in video_uri_map.values() if v)
    print(f'  Uploaded: {ok_uploads}/{len(unique_vpaths)}\n')

    # Filter only entries that have a video URI
    tasks = [(entry, video_uri_map.get(vpaths.get(id(entry)))) for entry in todo
             if 'predicted_answer' not in entry]
    tasks = [(e, u) for e, u in tasks if u]

    print(f'Phase 2 — Answering {len(tasks)} questions ({args.workers} workers)...')

    model_name = args.model
    ok = fail = completed = 0
    pbar = tqdm(total=len(tasks), desc=f'Answering ({args.workers} workers)')

    async def answer_one(entry, file_uri):
        nonlocal ok, fail
        contents = build_contents(entry, file_uri)
        duration = float(entry.get('duration', 0))
        for attempt in range(args.max_retry + 1):
            await rate_limiter.acquire()
            async with semaphore:
                try:
                    resp = await async_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            response_mime_type='application/json',
                            response_schema=BaselineResponse,
                            temperature=0,
                        ),
                    )
                    if not resp.text:
                        fr = getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else 'NO_CANDIDATES'
                        raise ValueError(f'empty resp.text, finish_reason={fr}')

                    obj = json.loads(resp.text)
                    letter = str(obj.get('answer', '')).strip().upper()
                    if not letter or letter[0] not in ANSWER_LETTERS:
                        raise ValueError(f'invalid answer letter: {repr(letter)}')
                    letter = letter[0]

                    entry['predicted_answer']     = letter
                    entry['predicted_answer_idx'] = ANSWER_LETTERS.index(letter)
                    entry['answer_reasoning']     = str(obj.get('reasoning', '')).strip()

                    # Store Gemini-predicted segment as best_proposal for grounding eval
                    start = float(obj.get('start_sec', 0))
                    end   = float(obj.get('end_sec', 0))
                    if duration > 0:
                        start = max(0.0, min(start, duration))
                        end   = max(start + 0.5, min(end, duration))
                    entry['best_proposal'] = [start, end, 1.0]

                    ok += 1
                    return
                except Exception as exc:
                    if _is_daily_quota_error(exc):
                        print(f'\n[FATAL] Daily quota exhausted: {exc}')
                        raise
                    if _is_rate_limit_error(exc) and attempt < args.max_retry:
                        await asyncio.sleep(min(2 ** attempt * 5, 60))
                        continue
                    if attempt == args.max_retry:
                        print(f"\n  [FAIL] qid={entry.get('qid')}: {type(exc).__name__}: {exc}")
                        fail += 1
                    else:
                        continue

    async def answer_with_save(entry, file_uri):
        nonlocal completed
        await answer_one(entry, file_uri)
        completed += 1
        pbar.update(1)
        pbar.set_postfix(ok=ok, fail=fail)
        if completed % args.save_every == 0:
            _atomic_save(data, output_path)

    await asyncio.gather(*(answer_with_save(e, u) for e, u in tasks),
                         return_exceptions=True)
    pbar.close()
    _atomic_save(data, output_path)
    return ok, fail


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NExT-GQA Gemini Baseline — full video → MCQ (no grounding)')
    parser.add_argument('--split',          default='test', choices=['test', 'val'])
    parser.add_argument('--input',          default=None)
    parser.add_argument('--output',         default=None)
    parser.add_argument('--model',          default='gemini-2.5-flash',
                        help='Gemini model name (default: gemini-2.5-flash)')
    parser.add_argument('--workers',        type=int, default=10)
    parser.add_argument('--upload_workers', type=int, default=5)
    parser.add_argument('--rpm',            type=int, default=100)
    parser.add_argument('--max_retry',      type=int, default=5)
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--first_n',        type=int, default=None)
    parser.add_argument('--no_resume',      action='store_true')
    args = parser.parse_args()

    input_path  = args.input  or f'dataset/nextgqa/baseline_input_{args.split}.json'
    output_path = args.output or f'dataset/nextgqa/gemini_baseline_outputs_{args.split}.json'

    print(f'Loading: {input_path}')
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)
    print(f'  Total entries: {len(data)}')

    # Resume
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
                entry['best_proposal']        = ex.get('best_proposal')
                resumed += 1
        print(f'  Resumed: {resumed} already answered')
    elif args.no_resume:
        for entry in data:
            entry.pop('predicted_answer', None)
            entry.pop('predicted_answer_idx', None)
            entry.pop('answer_reasoning', None)

    # Load video paths
    with open(MAP_VID_VIDOR_PATH, encoding='utf-8') as f:
        vid_map = json.load(f)

    n_done = sum(1 for e in data if 'predicted_answer' in e)
    print(f'  Already done: {n_done}')

    # Build id(entry) → vpath mapping
    todo = [e for e in data if 'predicted_answer' not in e]
    if args.first_n:
        todo = todo[:args.first_n]

    vpaths: dict[int, Optional[str]] = {}
    n_no_video = 0
    for entry in todo:
        vid_id    = str(entry.get('video_id', ''))
        vidor_rel = vid_map.get(vid_id)
        vpath     = find_video_path(vidor_rel) if vidor_rel else None
        vpaths[id(entry)] = vpath
        if vpath is None:
            n_no_video += 1

    print(f'  To answer   : {len(todo)}')
    print(f'  No video    : {n_no_video}')
    print(f'Model: {args.model} | Workers: {args.workers} | RPM: {args.rpm}\n')

    if not todo:
        print('Nothing to do.')
        return

    ok, fail = asyncio.run(_run_async(data, todo, vpaths, output_path, args))

    total_answered = sum(1 for e in data if 'predicted_answer' in e)
    print('\n' + '=' * 58)
    print('Gemini Baseline done!')
    print(f'  Success  : {ok}')
    print(f'  Failed   : {fail}')
    print(f'  Total    : {total_answered}/{len(data)}')
    print(f'  Output   : {output_path}')
    print('=' * 58)


if __name__ == '__main__':
    main()
