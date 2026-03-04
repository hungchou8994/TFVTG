"""
NExT-GQA Verifier
==================
Re-ranks top-5 Grounder proposals using Gemini multimodal video understanding.

Strategy: Upload full video once via Files API → send 1 Gemini call with the full
  video + proposals described as MM:SS timestamps → ask for ranking.
  - No decord / frame sampling required
  - Videos cached in video_upload_cache.json (TTL 47h) for reuse across runs
  - Fail after 3 retries → skip entry (resume will retry next run)

Input : grounder_outputs_{split}.json + video files
Output: verifier_outputs_{split}.json  (adds best_proposal, gemini_ranking)

Usage:
  python -m nextgqa.verifier --split test --workers 10
  python -m nextgqa.verifier --split test --first_n 500
  python -m nextgqa.verifier --split test --dry_run
  python -m nextgqa.verifier --split test --upload_workers 5 --workers 15 --rpm 900

Tier 1 limits: RPM=1000, RPD=10000, TPM=1M
  → Use --rpm 900 --workers 15 for safe full run
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
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

VIDEO_UPLOAD_CACHE_PATH = 'dataset/nextgqa/video_upload_cache.json'
UPLOAD_TTL_SECONDS = 47 * 3600  # 47 hours (Files API TTL is 48h)


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter: allows at most `rpm` requests per 60 seconds."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / max(1, rpm)  # seconds per token
        self._lock = asyncio.Lock()
        self._last_time = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_time = asyncio.get_event_loop().time()


# ── Pydantic schema for structured output ─────────────────────────────────────

class VerifierResponse(BaseModel):
    ranking: list[int] = Field(
        description="Segment numbers (1-5) ranked from most to least relevant. Include ALL segments."
    )
    reasoning: str = Field(description="One sentence explanation")


# ── Upload cache helpers ───────────────────────────────────────────────────────

def load_upload_cache() -> dict:
    """Load upload cache from disk. Returns {abs_path: {file_uri, file_name, uploaded_at}}."""
    if os.path.exists(VIDEO_UPLOAD_CACHE_PATH):
        try:
            with open(VIDEO_UPLOAD_CACHE_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_upload_cache(cache: dict) -> None:
    """Persist upload cache to disk."""
    os.makedirs(os.path.dirname(VIDEO_UPLOAD_CACHE_PATH), exist_ok=True)
    with open(VIDEO_UPLOAD_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _cache_entry_valid(entry: dict) -> bool:
    """Return True if the cached upload is still within TTL."""
    try:
        uploaded_at = datetime.fromisoformat(entry['uploaded_at'])
        age = (datetime.now(timezone.utc) - uploaded_at).total_seconds()
        return age < UPLOAD_TTL_SECONDS
    except Exception:
        return False


# ── Video path resolution ─────────────────────────────────────────────────────

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.webm']

MIME_MAP = {
    '.mp4':  'video/mp4',
    '.avi':  'video/avi',
    '.mkv':  'video/webm',
    '.mov':  'video/mov',
    '.webm': 'video/webm',
}


def find_video_path(video_root: str, vidor_rel: str) -> Optional[str]:
    folder, vid_id = vidor_rel.split('/')
    # Flat: {folder}/{vid_id}.ext
    for ext in VIDEO_EXTENSIONS:
        p = os.path.join(video_root, folder, vid_id + ext)
        if os.path.isfile(p):
            return p
    # Nested: {folder}/{vid_id}/{vid_id}.ext
    nested = os.path.join(video_root, folder, vid_id)
    if os.path.isdir(nested):
        for ext in VIDEO_EXTENSIONS:
            p = os.path.join(nested, vid_id + ext)
            if os.path.isfile(p):
                return p
    return None


# ── Timestamp helper ──────────────────────────────────────────────────────────

def seconds_to_mmss(s: float) -> str:
    """Convert seconds (float) to 'MM:SS' string for Gemini timestamp prompting."""
    s = max(0.0, float(s))
    m = int(s) // 60
    sec = int(s) % 60
    return f'{m:02d}:{sec:02d}'


# ── Video upload ──────────────────────────────────────────────────────────────

async def upload_video_async(sync_client,
                              video_path: str,
                              cache: dict,
                              cache_lock: asyncio.Lock,
                              poll_interval: float = 3.0,
                              poll_timeout: float = 300.0) -> Optional[str]:
    """
    Upload video to Gemini Files API (if not already cached).
    Returns file_uri string on success, None on failure.
    """
    abs_path = os.path.abspath(video_path)

    # Check cache first (under lock)
    async with cache_lock:
        entry = cache.get(abs_path)
        if entry and _cache_entry_valid(entry):
            return entry['file_uri']

    # Upload in thread pool (sync API)
    ext = os.path.splitext(video_path)[1].lower()
    mime = MIME_MAP.get(ext, 'video/mp4')

    try:
        uploaded = await asyncio.to_thread(
            sync_client.files.upload,
            file=video_path,
            config={'mime_type': mime},
        )
    except Exception as exc:
        print(f'  [upload] FAILED for {os.path.basename(video_path)}: {exc}')
        return None

    # Poll until state == ACTIVE
    deadline = time.monotonic() + poll_timeout
    file_name = uploaded.name
    info = uploaded
    try:
        while True:
            state = getattr(info, 'state', None)
            if state is not None:
                state_str = state.name if hasattr(state, 'name') else str(state)
                if state_str == 'ACTIVE':
                    break
                if state_str == 'FAILED':
                    print(f'  [upload] File processing FAILED: {file_name}')
                    return None
            if time.monotonic() > deadline:
                print(f'  [upload] Timeout waiting for ACTIVE: {file_name}')
                return None
            await asyncio.sleep(poll_interval)
            info = await asyncio.to_thread(sync_client.files.get, name=file_name)
    except Exception as exc:
        print(f'  [upload] Poll error for {file_name}: {exc}')
        return None

    file_uri = info.uri
    now_iso = datetime.now(timezone.utc).isoformat()
    async with cache_lock:
        cache[abs_path] = {
            'file_uri': file_uri,
            'file_name': file_name,
            'uploaded_at': now_iso,
        }

    return file_uri


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = (
    "You are a temporal video verifier for video question answering. "
    "You will be given a full video and a list of candidate time segments described by MM:SS timestamps. "
    "Watch the relevant parts of the video and rank the segments from most to least relevant: "
    "which segment best shows the visual event needed to answer the question."
)


def build_contents_video(entry: dict, file_uri: str) -> list:
    """Build Gemini contents: full video (Files API URI) + text with proposals as MM:SS timestamps."""
    from google.genai import types

    proposals = entry['top5_proposals']
    n = len(proposals)

    proposals_text = '\n'.join(
        f'  Segment {i + 1}: {seconds_to_mmss(p[0])} to {seconds_to_mmss(p[1])}'
        f'  ({float(p[0]):.1f}s – {float(p[1]):.1f}s)'
        for i, p in enumerate(proposals)
    )

    prompt = (
        f'Video duration: {entry["duration"]:.1f}s\n'
        f'Question: "{entry["question"]}"\n'
        f'Grounding description: "{entry.get("grounding_description", "")}"\n\n'
        f'Below are {n} candidate time segments from this video:\n'
        f'{proposals_text}\n\n'
        f'Watch the video at each of the listed timestamps and rank ALL {n} segments (1-{n}) '
        f'from MOST to LEAST relevant for answering the question.\n'
        f'Respond in JSON: {{"ranking": [best, ..., worst], "reasoning": "one sentence"}}'
    )

    return [
        types.Part(file_data=types.FileData(file_uri=file_uri)),
        types.Part(text=prompt),
    ]


# ── Response normalization ────────────────────────────────────────────────────

def _try_parse_json(text) -> dict:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def _atomic_save(data: list, path: str) -> None:
    """Write JSON atomically: dump to .tmp then rename over the target."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def normalize_ranking(raw: dict, n: int = 5) -> Optional[dict]:
    ranking = raw.get('ranking')
    if not isinstance(ranking, list):
        return None
    valid = [x for x in ranking if isinstance(x, int) and 1 <= x <= n]
    if not valid:
        return None
    # Fill any missing segments at the end
    missing = [i for i in range(1, n + 1) if i not in valid]
    full = valid + missing
    return {
        'ranking': full[:n],
        'reasoning': str(raw.get('reasoning', '')).strip(),
    }


# ── Single question async verify ──────────────────────────────────────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc)
    return '429' in s or 'RESOURCE_EXHAUSTED' in s or 'quota' in s.lower()


async def verify_one(async_client, semaphore, rate_limiter: 'RateLimiter',
                     model_name: str, entry: dict, file_uri: str,
                     max_retry: int = 5) -> Optional[dict]:
    """
    Send 1 Gemini call with the full video + timestamp-based proposals.
    Returns normalized ranking dict or None (caller will skip and resume later).
    """
    n = len(entry['top5_proposals'])
    if n == 0:
        return None

    contents = build_contents_video(entry, file_uri)

    from google.genai import types

    for attempt in range(max_retry):
        # Try structured output first
        try:
            await rate_limiter.acquire()
            async with semaphore:
                resp = await async_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type='application/json',
                        response_schema=VerifierResponse,
                    ),
                )
            raw = _try_parse_json(resp.text)
            if not raw:
                fr = getattr(resp.candidates[0], 'finish_reason', '?') if (resp.candidates) else 'NO_CANDIDATES'
                raise ValueError(f'empty/None resp.text, finish_reason={fr}')
            normalized = normalize_ranking(raw, n)
            if normalized:
                return normalized
            raise ValueError('bad schema response')
        except Exception as exc:
            if _is_rate_limit_error(exc):
                wait = min(60 * (attempt + 1), 300)
                await asyncio.sleep(wait)
                continue
            print(f'  [verify] attempt={attempt} structured FAIL qid={entry.get("qid","?")}: {type(exc).__name__}: {exc}')


        # Fallback: raw JSON (no schema constraint)
        try:
            await rate_limiter.acquire()
            async with semaphore:
                resp = await async_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type='application/json',
                    ),
                )
            raw = _try_parse_json(resp.text)
            if not raw:
                fr = getattr(resp.candidates[0], 'finish_reason', '?') if (resp.candidates) else 'NO_CANDIDATES'
                raise ValueError(f'empty/None resp.text, finish_reason={fr}')
            normalized = normalize_ranking(raw, n)
            if normalized:
                return normalized
        except Exception as exc:
            if _is_rate_limit_error(exc):
                wait = min(60 * (attempt + 1), 300)
                await asyncio.sleep(wait)
                continue
            print(f'  [verify] attempt={attempt} fallback FAIL qid={entry.get("qid","?")}: {type(exc).__name__}: {exc}')

        if attempt < max_retry - 1:
            await asyncio.sleep(min(2 ** attempt, 10))

    return None  # Skip; resume will retry next run


# ── Async runner ──────────────────────────────────────────────────────────────

async def _run_async(tasks, results, sync_client, api_key, model_name,
                     workers, upload_workers, save_every, output_file,
                     max_retry, chunk_size, upload_cache, cache_lock, rpm):
    from google import genai

    async_client = genai.Client(api_key=api_key).aio
    semaphore = asyncio.Semaphore(max(1, workers))
    upload_semaphore = asyncio.Semaphore(max(1, upload_workers))
    rate_limiter = RateLimiter(rpm)
    print(f'  Rate limiter : {rpm} RPM ({60/rpm:.2f}s/req minimum)')

    # ── Phase 1: Pre-upload unique videos ─────────────────────────────────────
    unique_paths = list({vpath for _, _, vpath in tasks if vpath})
    print(f'\nPhase 1 — Uploading {len(unique_paths)} unique videos '
          f'({upload_workers} concurrent)...')

    path_to_uri: dict[str, Optional[str]] = {}

    async def upload_one(vpath: str):
        async with upload_semaphore:
            uri = await upload_video_async(sync_client, vpath, upload_cache, cache_lock)
        path_to_uri[vpath] = uri

    await asyncio.gather(*(upload_one(p) for p in unique_paths))

    # Persist cache after uploads
    save_upload_cache(upload_cache)

    uploaded_ok = sum(1 for v in path_to_uri.values() if v)
    print(f'  Upload complete: {uploaded_ok}/{len(unique_paths)} succeeded\n')

    # ── Phase 2: Verify with Gemini ────────────────────────────────────────────
    print(f'Phase 2 — Verifying {len(tasks)} questions ({workers} workers)...')

    ok = fail = skip = completed = 0
    pbar = tqdm(total=len(tasks), desc=f'Verifying ({workers} workers)')

    for chunk_start in range(0, len(tasks), chunk_size):
        chunk = tasks[chunk_start:chunk_start + chunk_size]

        async def run_one(task):
            idx, entry, vpath = task
            file_uri = path_to_uri.get(vpath) if vpath else None
            if not file_uri:
                return idx, None, 'no_video'
            res = await verify_one(async_client, semaphore, rate_limiter,
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
                best_0 = res['ranking'][0] - 1  # 0-based
                results[idx]['gemini_ranking']     = res['ranking']
                results[idx]['verifier_reasoning'] = res['reasoning']
                results[idx]['best_proposal']      = results[idx]['top5_proposals'][best_0]
                results[idx]['grounder_rank']      = best_0 + 1
                ok += 1
            else:
                # Do NOT write best_proposal — resume will retry
                if status == 'no_video':
                    skip += 1
                else:
                    fail += 1

            pbar.update(1)
            pbar.set_postfix({'ok': ok, 'fail': fail, 'skip': skip})

            if completed % save_every == 0:
                _atomic_save(results, output_file)

        # Save after each chunk
        _atomic_save(results, output_file)

    pbar.close()
    return ok, fail, skip


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Verify NExT-GQA proposals with Gemini (Files API + timestamps)')
    parser.add_argument('--split',          default='test', choices=['val', 'test'])
    parser.add_argument('--input',          default=None)
    parser.add_argument('--output',         default=None)
    parser.add_argument('--model',          default='gemini-2.5-flash')
    parser.add_argument('--workers',        type=int, default=10,
                        help='Concurrent Gemini verify calls')
    parser.add_argument('--upload_workers', type=int, default=5,
                        help='Concurrent Files API uploads (Phase 1)')
    parser.add_argument('--max_retry',      type=int, default=5)
    parser.add_argument('--rpm',            type=int, default=300,
                        help='Max requests/minute to Gemini (Tier1=1000, conservative default=300)')
    parser.add_argument('--save_every',     type=int, default=50)
    parser.add_argument('--chunk_size',     type=int, default=100)
    parser.add_argument('--first_n',        type=int, default=None,
                        help='Process only first N pending questions')
    parser.add_argument('--dry_run',        action='store_true', help='Process 5 questions only')
    parser.add_argument('--resume',         action='store_true', default=True)
    parser.add_argument('--no_resume',      dest='resume', action='store_false')
    args = parser.parse_args()

    input_path  = args.input  or f'dataset/nextgqa/grounder_outputs_{args.split}.json'
    output_path = args.output or f'dataset/nextgqa/verifier_outputs_{args.split}.json'

    print(f'Loading: {input_path}')
    with open(input_path, encoding='utf-8') as f:
        results = json.load(f)
    print(f'  Total entries: {len(results)}')

    # Load video mapping
    with open('dataset/nextgqa/map_vid_vidorID.json') as f:
        vid_to_vidor = json.load(f)
    video_root = 'dataset/nextgqa/videos'

    # Resume
    if args.resume and os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
        if existing:
            existing_map = {(e['video_id'], e['qid']): e for e in existing}
            for i, entry in enumerate(results):
                key = (entry['video_id'], entry['qid'])
                if key in existing_map and 'best_proposal' in existing_map[key]:
                    results[i] = existing_map[key]
            already_done = sum(1 for r in results if 'best_proposal' in r)
            print(f'  Resumed: {already_done} already verified')

    # Build task list (skip entries that already have best_proposal)
    tasks = []
    for idx, entry in enumerate(results):
        if 'best_proposal' in entry:
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

    print(f'  To verify: {len(tasks)}')

    if not tasks:
        print('Nothing to do.')
        _atomic_save(results, output_path)
        return

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY not set in .env')

    # Load upload cache
    upload_cache = load_upload_cache()
    print(f'  Upload cache: {len(upload_cache)} entries loaded')

    print(f'Model: {args.model} | Verify workers: {args.workers} | '
          f'Upload workers: {args.upload_workers} | Max retry: {args.max_retry} | RPM: {args.rpm}')

    from google import genai
    sync_client = genai.Client(api_key=api_key)
    cache_lock  = asyncio.Lock()

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
        upload_cache=upload_cache,
        cache_lock=cache_lock,
        rpm=args.rpm,
    ))

    # Final save
    _atomic_save(results, output_path)

    # Persist upload cache
    save_upload_cache(upload_cache)

    total_verified = sum(1 for r in results if 'best_proposal' in r)
    print(f'\n{"=" * 50}')
    print(f'Verifier done!')
    print(f'  Success     : {ok}')
    print(f'  Failed/retry: {fail}')
    print(f'  Skipped     : {skip}  (no video file found)')
    print(f'  Total done  : {total_verified}/{len(results)}')
    print(f'  Output      : {output_path}')
    print(f'  Cache       : {VIDEO_UPLOAD_CACHE_PATH}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
