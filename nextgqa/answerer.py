"""
NExT-GQA Answerer
==================
Answers MCQ questions using Gemini multimodal, given the best temporal proposal
from the Verifier (or Grounder top-1 as fallback).

Strategy: Upload full video via Files API → send 1 Gemini call with the full
  video + best_proposal timestamps + question + 5 choices → pick best answer.
  - Same upload cache as Verifier (video_upload_cache.json)
  - Resume-safe: skips entries that already have `predicted_answer`
  - Fail after max_retry → skip entry (resume will retry next run)

Input : verifier_outputs_{split}.json  (needs best_proposal or top5_proposals)
Output: answerer_outputs_{split}.json  (adds predicted_answer, predicted_answer_idx)

Usage:
  python -m nextgqa.answerer --split test --workers 10
  python -m nextgqa.answerer --split test --rpm 100
  python -m nextgqa.answerer --input dataset/nextgqa/friend_verifier_outputs.json
                             --output dataset/nextgqa/friend_answerer_outputs.json
"""

import os
import json
import time
import argparse
import asyncio
import re
from datetime import datetime, timezone
from tqdm import tqdm
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

VIDEO_UPLOAD_CACHE_PATH = 'dataset/nextgqa/video_upload_cache.json'
CLIP_CACHE_PATH      = 'dataset/nextgqa/clip_upload_cache.json'
CLIPS_TMP_DIR        = 'dataset/nextgqa/.clips_tmp'
UPLOAD_TTL_SECONDS   = 47 * 3600

ANSWER_LETTERS = ['A', 'B', 'C', 'D', 'E']

SYSTEM_INSTRUCTION = (
    "You are a video question answering assistant. "
    "You will be given a short video clip and a multiple-choice question with 5 options (A–E). "
    "The clip shows the most relevant segment for answering the question. "
    "Select the single best answer based on what you observe in the clip."
)


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

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


# ── Upload cache helpers (shared with verifier) ───────────────────────────────

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


def load_clip_cache() -> dict:
    if os.path.exists(CLIP_CACHE_PATH):
        try:
            with open(CLIP_CACHE_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_clip_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CLIP_CACHE_PATH), exist_ok=True)
    tmp = CLIP_CACHE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CLIP_CACHE_PATH)


def _cache_entry_valid(entry: dict) -> bool:
    try:
        uploaded_at = datetime.fromisoformat(entry['uploaded_at'])
        age = (datetime.now(timezone.utc) - uploaded_at).total_seconds()
        return age < UPLOAD_TTL_SECONDS
    except Exception:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

MIME_MAP = {'.mp4': 'video/mp4', '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime', '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'}


# ── Video trimming ────────────────────────────────────────────────────────────

def trim_video(video_path: str, start: float, end: float, out_dir: str,
               video_id: str = '', qid: str = '') -> Optional[str]:
    """Trim video to [start, end] using ffmpeg. Returns output clip path or None."""
    import subprocess
    import imageio_ffmpeg
    os.makedirs(out_dir, exist_ok=True)
    tag = f'{video_id}_q{qid}' if video_id else os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f'{tag}_{start:.3f}_{end:.3f}.mp4')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    duration = max(0.5, end - start)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def _run(extra_args):
        cmd = [
            ffmpeg_exe, '-y',
            '-ss', str(start),
            '-i', video_path,
            '-t', str(duration),
        ] + extra_args + [out_path]
        return subprocess.run(cmd, capture_output=True, timeout=120)

    try:
        # Try fast stream copy first
        result = _run(['-c', 'copy', '-avoid_negative_ts', '1'])
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        # Fallback: re-encode with pad to ensure even dimensions (libx264 requirement)
        if os.path.exists(out_path):
            os.remove(out_path)
        result = _run(['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                       '-c:v', 'libx264', '-c:a', 'aac', '-crf', '23', '-preset', 'fast'])
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        stderr_msg = result.stderr.decode(errors='replace').strip()
        # Print last 600 chars of stderr (skip ffmpeg banner at top)
        print(f'  [trim] FAILED {os.path.basename(video_path)} [{start:.1f}-{end:.1f}]:\n'
              f'    {stderr_msg[-600:]}')
        return None
    except Exception as exc:
        print(f'  [trim] error: {exc}')
        return None


def find_video_path(video_root: str, vidor_rel: str) -> Optional[str]:
    if not vidor_rel:
        return None
    folder, vid_file = vidor_rel.split('/')
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        p = os.path.join(video_root, folder, vid_file + ext)
        if os.path.exists(p):
            return p
    return None


def seconds_to_mmss(s: float) -> str:
    s = max(0.0, float(s))
    m = int(s) // 60
    sec = int(s) % 60
    return f'{m:02d}:{sec:02d}'


def _atomic_save(data: list, path: str) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return ('429' in msg or 'RESOURCE_EXHAUSTED' in msg) and not _is_daily_quota_error(exc)


def _is_daily_quota_error(exc: Exception) -> bool:
    """Daily RPD quota — cannot retry until tomorrow."""
    msg = str(exc)
    return 'per_day' in msg.lower() or 'PerDay' in msg or 'per_model_per_day' in msg


# ── Video upload ──────────────────────────────────────────────────────────────

async def upload_video_async(sync_client, video_path: str,
                              cache: dict, cache_lock: asyncio.Lock,
                              poll_interval: float = 3.0,
                              poll_timeout: float = 300.0) -> Optional[str]:
    abs_path = os.path.abspath(video_path)
    async with cache_lock:
        entry = cache.get(abs_path)
        if entry and _cache_entry_valid(entry):
            return entry['file_uri']

    ext = os.path.splitext(video_path)[1].lower()
    mime = MIME_MAP.get(ext, 'video/mp4')
    try:
        uploaded = await asyncio.to_thread(
            sync_client.files.upload,
            file=video_path,
            config={'mime_type': mime},
        )
    except Exception as exc:
        print(f'  [upload] FAILED {os.path.basename(video_path)}: {exc}')
        return None

    deadline = time.monotonic() + poll_timeout
    file_name = uploaded.name
    info = uploaded
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
                print(f'  [upload] Timeout: {file_name}')
                return None
            await asyncio.sleep(poll_interval)
            info = await asyncio.to_thread(sync_client.files.get, name=file_name)
    except Exception as exc:
        print(f'  [upload] Poll error {file_name}: {exc}')
        return None

    file_uri = info.uri
    async with cache_lock:
        cache[abs_path] = {
            'file_uri': file_uri,
            'file_name': file_name,
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
        }
    return file_uri


# ── Clip upload ───────────────────────────────────────────────────────────────

async def upload_clip_async(sync_client, video_path: str, start: float, end: float,
                             clip_cache: dict, clip_cache_lock: asyncio.Lock,
                             video_id: str = '', qid: str = '',
                             poll_interval: float = 3.0,
                             poll_timeout: float = 300.0) -> Optional[str]:
    """Trim video to [start,end] then upload the clip. Cache by (video_id, qid)."""
    cache_key = f'{video_id}::q{qid}::{start:.3f}::{end:.3f}' if video_id else \
                f'{os.path.abspath(video_path)}::{start:.3f}::{end:.3f}'

    async with clip_cache_lock:
        entry = clip_cache.get(cache_key)
        if entry and _cache_entry_valid(entry):
            return entry['file_uri']

    # Trim
    clip_path = await asyncio.to_thread(
        trim_video, video_path, start, end, CLIPS_TMP_DIR, video_id, qid)
    if not clip_path:
        return None

    # Upload
    ext  = os.path.splitext(clip_path)[1].lower()
    mime = MIME_MAP.get(ext, 'video/mp4')
    try:
        uploaded = await asyncio.to_thread(
            sync_client.files.upload,
            file=clip_path,
            config={'mime_type': mime},
        )
    except Exception as exc:
        print(f'  [upload_clip] FAILED {os.path.basename(clip_path)}: {exc}')
        return None

    # Poll for ACTIVE
    deadline  = time.monotonic() + poll_timeout
    file_name = uploaded.name
    info      = uploaded
    try:
        while True:
            state = getattr(info, 'state', None)
            if state is not None:
                s = state.name if hasattr(state, 'name') else str(state)
                if s == 'ACTIVE':
                    break
                if s == 'FAILED':
                    print(f'  [upload_clip] FAILED (processing): {file_name}')
                    return None
            if time.monotonic() > deadline:
                print(f'  [upload_clip] Timeout: {file_name}')
                return None
            await asyncio.sleep(poll_interval)
            info = await asyncio.to_thread(sync_client.files.get, name=file_name)
    except asyncio.CancelledError:
        print(f'  [upload_clip] Cancelled during poll: {file_name}')
        return None
    except Exception as exc:
        print(f'  [upload_clip] Poll error {file_name}: {exc}')
        return None

    file_uri = info.uri
    async with clip_cache_lock:
        clip_cache[cache_key] = {
            'file_uri': file_uri,
            'file_name': file_name,
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
        }
    return file_uri


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_contents_answer(entry: dict, file_uri: str) -> list:
    from google.genai import types

    question     = entry['question']
    choices      = entry['choices']   # list of 5 strings
    choice_lines = '\n'.join(
        f'  {ANSWER_LETTERS[i]}. {c}' for i, c in enumerate(choices)
    )

    prompt = (
        f'Question: {question}\n\n'
        f'Choices:\n{choice_lines}\n\n'
        f'Select the single best answer based on what you observe in the clip. '
        f'Respond in JSON: {{"answer": "A", "reasoning": "one sentence"}}'
    )

    return [
        types.Part(file_data=types.FileData(file_uri=file_uri)),
        types.Part(text=prompt),
    ]


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_answer(text: str) -> Optional[dict]:
    """Parse Gemini response → {answer_letter, answer_idx}. Returns None if unparseable."""
    if not text:
        return None
    text = text.strip()

    # Try JSON
    def _from_json(s):
        try:
            obj = json.loads(s)
        except Exception:
            m = re.search(r'\{[\s\S]*\}', s)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None
        a = str(obj.get('answer', '')).strip().upper()
        if a and a[0] in ANSWER_LETTERS:
            letter = a[0]
            return {'answer_letter': letter,
                    'answer_idx': ANSWER_LETTERS.index(letter),
                    'reasoning': str(obj.get('reasoning', '')).strip()}
        return None

    result = _from_json(text)
    if result:
        return result

    # Fallback: scan for first A/B/C/D/E in text
    m = re.search(r'\b([ABCDE])\b', text)
    if m:
        letter = m.group(1)
        return {'answer_letter': letter,
                'answer_idx': ANSWER_LETTERS.index(letter),
                'reasoning': ''}
    return None


# ── Answer one entry ──────────────────────────────────────────────────────────

async def answer_one(async_client, semaphore, rate_limiter: RateLimiter,
                     model_name: str, entry: dict, file_uri: str,
                     max_retry: int = 5) -> Optional[dict]:
    from google.genai import types

    contents = build_contents_answer(entry, file_uri)

    for attempt in range(max_retry):
        try:
            await rate_limiter.acquire()
            async with semaphore:
                resp = await async_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type='application/json',
                        temperature=0.1,
                    ),
                )
            if not resp.text:
                fr = getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else 'NO_CANDIDATES'
                raise ValueError(f'resp.text is None (finish_reason={fr})')
            result = _parse_answer(resp.text)
            if result:
                return result
            raise ValueError(f'unparseable response: {resp.text[:100]}')
        except Exception as exc:
            if _is_daily_quota_error(exc):
                print(f'\n  [FATAL] Daily RPD quota exhausted. Switch API key and retry.\n  {exc}\n')
                raise
            if _is_rate_limit_error(exc):
                wait = min(60 * (attempt + 1), 300)
                await asyncio.sleep(wait)
                continue
            print(f'  [answer] attempt={attempt} FAIL qid={entry.get("qid","?")}: {type(exc).__name__}: {exc}')
            if attempt < max_retry - 1:
                await asyncio.sleep(min(2 ** attempt, 10))

    return None


# ── Async runner ──────────────────────────────────────────────────────────────

async def _run_async(tasks, results, sync_client, api_key, model_name,
                     workers, upload_workers, save_every, output_file,
                     max_retry, chunk_size, clip_cache, clip_cache_lock, rpm):
    from google import genai

    async_client = genai.Client(api_key=api_key).aio
    semaphore    = asyncio.Semaphore(workers)
    rate_limiter = RateLimiter(rpm)

    print(f'  Rate limiter : {rpm} RPM ({60/rpm:.2f}s/req minimum)\n')

    # ── Phase 1: Trim + upload one clip per task ─────────────────────────────
    up_semaphore  = asyncio.Semaphore(upload_workers)
    task_clip_uri = {}   # task idx → file_uri

    # Build list of (task_idx, entry, vpath, start, end)
    task_clips = []
    for idx, entry, vpath in tasks:
        if vpath:
            bp    = entry['best_proposal']
            start = float(bp[0])
            end   = float(bp[1])
            task_clips.append((idx, entry, vpath, start, end))

    print(f'Phase 1 — Trimming & uploading {len(task_clips)} clips ({upload_workers} concurrent)...')

    async def upload_one_clip(task_idx, entry, vpath, start, end):
        try:
            async with up_semaphore:
                uri = await upload_clip_async(
                    sync_client, vpath, start, end, clip_cache, clip_cache_lock,
                    video_id=str(entry.get('video_id', '')),
                    qid=str(entry.get('qid', '')))
            task_clip_uri[task_idx] = uri
        except BaseException as exc:
            print(f'  [upload_one_clip] ERROR idx={task_idx}: {type(exc).__name__}: {exc}')
            task_clip_uri[task_idx] = None

    results_upload = await asyncio.gather(
        *(upload_one_clip(idx, entry, vpath, start, end)
          for idx, entry, vpath, start, end in task_clips),
        return_exceptions=True,
    )
    # Log any remaining exceptions from gather itself
    for r in results_upload:
        if isinstance(r, Exception):
            print(f'  [upload gather] unhandled: {r}')
    save_clip_cache(clip_cache)

    uploaded_ok = sum(1 for v in task_clip_uri.values() if v)
    print(f'  Upload complete: {uploaded_ok}/{len(task_clips)} succeeded\n')

    # ── Phase 2: Answer ───────────────────────────────────────────────────────
    print(f'Phase 2 — Answering {len(tasks)} questions ({workers} workers)...')

    ok = fail = skip = completed = 0
    pbar = tqdm(total=len(tasks), desc=f'Answering ({workers} workers)')

    for chunk_start in range(0, len(tasks), chunk_size):
        chunk = tasks[chunk_start:chunk_start + chunk_size]

        async def run_one(task):
            idx, entry, vpath = task
            file_uri = task_clip_uri.get(idx) if vpath else None
            if not file_uri:
                return idx, None, 'no_video'
            res = await answer_one(async_client, semaphore, rate_limiter,
                                   model_name, entry, file_uri, max_retry)
            return idx, res, 'ok' if res else 'fail'

        outs = await asyncio.gather(*(run_one(t) for t in chunk), return_exceptions=True)

        for out in outs:
            completed += 1
            if isinstance(out, Exception):
                fail += 1
                pbar.update(1)
                pbar.set_postfix({'ok': ok, 'fail': fail, 'skip': skip})
                continue

            idx, res, status = out
            if res:
                results[idx]['predicted_answer']     = res['answer_letter']
                results[idx]['predicted_answer_idx'] = res['answer_idx']
                results[idx]['answer_reasoning']     = res['reasoning']
                ok += 1
            else:
                if status == 'no_video':
                    skip += 1
                else:
                    fail += 1

            pbar.update(1)
            pbar.set_postfix({'ok': ok, 'fail': fail, 'skip': skip})

            if completed % save_every == 0:
                _atomic_save(results, output_file)

        _atomic_save(results, output_file)

    pbar.close()
    return ok, fail, skip


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Answer NExT-GQA MCQ with Gemini (Files API + best proposal)')
    parser.add_argument('--split',          default='test', choices=['val', 'test'])
    parser.add_argument('--input',          default=None,
                        help='Path to verifier output JSON (default: verifier_outputs_{split}.json)')
    parser.add_argument('--output',         default=None,
                        help='Path for answerer output JSON (default: answerer_outputs_{split}.json)')
    parser.add_argument('--model',          default='gemini-2.5-flash')
    parser.add_argument('--workers',        type=int, default=10)
    parser.add_argument('--upload_workers', type=int, default=5)
    parser.add_argument('--max_retry',      type=int, default=5)
    parser.add_argument('--rpm',            type=int, default=100)
    parser.add_argument('--save_every',     type=int, default=50)
    parser.add_argument('--chunk_size',     type=int, default=100)
    parser.add_argument('--first_n',        type=int, default=None)
    parser.add_argument('--dry_run',        action='store_true')
    parser.add_argument('--no_resume',      action='store_true')
    args = parser.parse_args()

    input_path  = args.input  or f'dataset/nextgqa/verifier_outputs_{args.split}.json'
    output_path = args.output or f'dataset/nextgqa/answerer_outputs_{args.split}.json'

    print(f'Loading: {input_path}')
    with open(input_path, encoding='utf-8') as f:
        results = json.load(f)
    print(f'  Total entries: {len(results)}')

    # Load video mapping
    with open('dataset/nextgqa/map_vid_vidorID.json', encoding='utf-8') as f:
        vid_to_vidor = json.load(f)
    video_root = 'dataset/nextgqa/videos'

    # Resume from existing output
    if not args.no_resume and os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
        existing_map = {(e['video_id'], e['qid']): e for e in existing}
        for entry in results:
            key = (entry['video_id'], entry['qid'])
            if key in existing_map and 'predicted_answer' in existing_map[key]:
                entry['predicted_answer']     = existing_map[key]['predicted_answer']
                entry['predicted_answer_idx'] = existing_map[key]['predicted_answer_idx']
                entry['answer_reasoning']     = existing_map[key].get('answer_reasoning', '')

    resumed   = sum(1 for e in results if 'predicted_answer' in e)
    print(f'  Resumed: {resumed} already answered')

    # Only entries with best_proposal are answerable
    tasks = []
    no_proposal = 0
    for idx, entry in enumerate(results):
        if 'predicted_answer' in entry:
            continue
        if not entry.get('best_proposal'):
            no_proposal += 1
            continue
        vidor_rel = vid_to_vidor.get(entry['video_id'])
        vpath = find_video_path(video_root, vidor_rel) if vidor_rel else None
        tasks.append((idx, entry, vpath))

    if args.dry_run:
        tasks = tasks[:5]
        print('DRY RUN: 5 questions')
    elif args.first_n:
        tasks = tasks[:args.first_n]
        print(f'FIRST_N: {args.first_n} questions')

    print(f'  To answer: {len(tasks)}')
    print(f'  Skipped (no best_proposal): {no_proposal}')

    if not tasks:
        print('Nothing to do.')
        _atomic_save(results, output_path)
        return

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY not set in .env')

    clip_cache      = load_clip_cache()
    print(f'  Clip upload cache: {len(clip_cache)} entries loaded')

    print(f'Model: {args.model} | Workers: {args.workers} | '
          f'Upload workers: {args.upload_workers} | Max retry: {args.max_retry} | RPM: {args.rpm}')

    from google import genai
    sync_client     = genai.Client(api_key=api_key)
    clip_cache_lock = asyncio.Lock()

    ok, fail, skip = asyncio.run(_run_async(
        tasks=tasks,
        results=results,
        sync_client=sync_client,
        api_key=api_key,
        model_name=args.model,
        workers=args.workers,
        upload_workers=args.upload_workers,
        save_every=args.save_every,
        output_file=output_path,
        max_retry=args.max_retry,
        chunk_size=args.chunk_size,
        clip_cache=clip_cache,
        clip_cache_lock=clip_cache_lock,
        rpm=args.rpm,
    ))

    _atomic_save(results, output_path)
    save_clip_cache(clip_cache)

    total_answered = sum(1 for r in results if 'predicted_answer' in r)
    print(f'\n{"=" * 50}')
    print(f'Answerer done!')
    print(f'  Success     : {ok}')
    print(f'  Failed/retry: {fail}')
    print(f'  Skipped     : {skip}  (no video file found)')
    print(f'  Total done  : {total_answered}/{len(results)}')
    print(f'  Output      : {output_path}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
