"""
NExT-GQA Query Rewriter
========================
Converts NExT-GQA questions into grounding queries using Gemini API.
Only requires test.csv (no video, no grounding JSON needed at this step).

Uses google-genai SDK with structured output (Pydantic schema) for
reliable JSON parsing.

Usage:
  python nextgqa_query_rewrite.py --split test
  python nextgqa_query_rewrite.py --split test --resume
  python nextgqa_query_rewrite.py --split test --dry_run
  python nextgqa_query_rewrite.py --split test --sample 10

Outputs:
  - llm_outputs_{split}.json   (readable, indented JSON)
  - llm_outputs_{split}.csv    (readable CSV)

Modes:
    - Realtime mode: direct generate_content calls (supports --workers)
    - Batch mode: upload JSONL requests to Gemini Batch API for async processing
"""

import json
import csv
import os
import time
import argparse
import re
import threading
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal
from tqdm import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

_thread_local = threading.local()


# ── Pydantic schema for Gemini structured output ──────────────────────────

class SubQuery(BaseModel):
    sub_query_id: int = Field(description="0 = full description, >=1 = sub-queries")
    descriptions: list[str] = Field(description="3 paraphrases of this sub-query")


class QueryRewriteResponse(BaseModel):
    reasoning: str = Field(description="Brief analysis of what the question asks about visually")
    grounding_description: str = Field(description="The converted declarative description")
    relationship: Literal["single-query", "simultaneously", "sequentially"] = Field(
        description="How sub-queries relate temporally"
    )
    query_json: list[SubQuery] = Field(description="Sub-query decomposition with paraphrases")


# ── Data loading ──────────────────────────────────────────────────────────

def load_questions_from_csv(qa_csv_path):
    """Load NExT-GQA questions from CSV only (no video/grounding needed)."""
    
    videos = defaultdict(lambda: {
        'sentences': [],
        'response': [],
        'meta': {'questions': []}
    })
    
    with open(qa_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['video_id']
            qid = row['qid']
            
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            answer_text = row['answer']
            answer_idx = next(
                (i for i, c in enumerate(choices) if c.strip() == answer_text.strip()), -1
            )
            
            videos[vid]['sentences'].append(row['question'])
            videos[vid]['response'].append(None)
            videos[vid]['meta']['questions'].append({
                'qid': int(qid),
                'type': row['type'],
                'answer': answer_text,
                'answer_idx': answer_idx,
                'choices': choices,
            })
    
    return dict(videos)


def _try_extract_json_object(text):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _normalize_response(parsed):
    if not isinstance(parsed, dict):
        return None

    relationship = parsed.get('relationship', 'single-query')
    if relationship not in {'single-query', 'simultaneously', 'sequentially'}:
        relationship = 'single-query'

    query_json = parsed.get('query_json', [])
    if not isinstance(query_json, list):
        query_json = []

    normalized_qj = []
    for item in query_json:
        if not isinstance(item, dict):
            continue
        sid = item.get('sub_query_id', None)
        if not isinstance(sid, int):
            continue
        desc = item.get('descriptions', [])
        if not isinstance(desc, list):
            continue
        desc = [str(d).strip() for d in desc if str(d).strip()]
        if not desc:
            continue
        normalized_qj.append({'sub_query_id': sid, 'descriptions': desc[:1]})

    if len(normalized_qj) == 0:
        gd = str(parsed.get('grounding_description', '')).strip()
        if gd:
            normalized_qj = [{'sub_query_id': 0, 'descriptions': [gd]}]
        else:
            return None

    if all(item['sub_query_id'] != 0 for item in normalized_qj):
        gd = str(parsed.get('grounding_description', '')).strip()
        if gd:
            normalized_qj.insert(0, {'sub_query_id': 0, 'descriptions': [gd]})
        else:
            normalized_qj.insert(0, {
                'sub_query_id': 0,
                'descriptions': [normalized_qj[0]['descriptions'][0]],
            })

    normalized_qj = sorted(normalized_qj, key=lambda x: x['sub_query_id'])
    grounding_description = str(parsed.get('grounding_description', '')).strip()
    if not grounding_description:
        grounding_description = normalized_qj[0]['descriptions'][0]

    reasoning = str(parsed.get('reasoning', '')).strip()
    if not reasoning:
        reasoning = 'Converted from question to grounding description.'

    return {
        'reasoning': reasoning,
        'grounding_description': grounding_description,
        'relationship': relationship,
        'query_json': normalized_qj,
    }


def _build_user_input(question, question_type, choices):
    choice_lines = []
    labels = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else f'OPT{i}'
        choice_lines.append(f"- {label}: {choice}")
    joined_choices = "\n".join(choice_lines)

    return (
        "Input:\n"
        f"question_type: {question_type}\n"
        f"question: {question}\n"
        "choices:\n"
        f"{joined_choices}\n"
    )


def _build_prompt(question, question_type, choices):
    from chat_bots.prompts_nextgqa import QUESTION_TO_QUERY_PROMPT
    return QUESTION_TO_QUERY_PROMPT + _build_user_input(question, question_type, choices)


def _get_thread_client(api_key):
    if not hasattr(_thread_local, 'client'):
        from google import genai
        _thread_local.client = genai.Client(api_key=api_key)
    return _thread_local.client


def _extract_retry_after_seconds(error_message):
    if not error_message:
        return None
    match = re.search(r'[Rr]etry[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*s', error_message)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _is_rate_limit_error(exc):
    code = getattr(exc, 'code', None)
    if code == 429:
        return True
    txt = str(exc).lower()
    return (
        '429' in txt
        or 'resource_exhausted' in txt
        or 'quota' in txt
        or 'rate limit' in txt
    )


# ── Gemini query rewrite ──────────────────────────────────────────────────

def rewrite_question(client, model_name, question, question_type, choices, max_retry=5):
    """Call Gemini with structured output to convert question → grounding query."""
    prompt = _build_prompt(question, question_type, choices)
    
    for attempt in range(max_retry):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': QueryRewriteResponse,
                }
            )
            parsed = _try_extract_json_object(response.text)
            normalized = _normalize_response(parsed)
            if normalized is not None:
                return normalized
            raise ValueError('Failed to normalize schema response')
        except Exception as e:
            print(f"  Attempt {attempt+1} failed (schema): {e}")

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
                parsed = _try_extract_json_object(response.text)
                normalized = _normalize_response(parsed)
                if normalized is not None:
                    return normalized
                raise ValueError('Failed to normalize raw-json response')
            except Exception as e2:
                print(f"  Attempt {attempt+1} failed (raw-json): {e2}")

            wait = min(2 ** (attempt + 1), 45)
            if attempt < max_retry - 1:
                time.sleep(wait)
    
    return None


async def rewrite_question_async(async_client, semaphore, model_name, question, question_type, choices, max_retry=8):
    """Async Gemini rewrite with robust retry/backoff for high-throughput realtime mode."""
    prompt = _build_prompt(question, question_type, choices)

    for attempt in range(max_retry):
        try:
            async with semaphore:
                response = await async_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': QueryRewriteResponse,
                    }
                )
            parsed = _try_extract_json_object(response.text)
            normalized = _normalize_response(parsed)
            if normalized is not None:
                return normalized
            raise ValueError('Failed to normalize schema response')
        except Exception as e:
            try:
                async with semaphore:
                    response = await async_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                parsed = _try_extract_json_object(response.text)
                normalized = _normalize_response(parsed)
                if normalized is not None:
                    return normalized
                raise ValueError('Failed to normalize raw-json response')
            except Exception as e2:
                if attempt >= max_retry - 1:
                    return None

                err_for_backoff = e2 if _is_rate_limit_error(e2) else e
                retry_after = _extract_retry_after_seconds(str(err_for_backoff))
                if retry_after is not None:
                    wait = retry_after + 0.5
                else:
                    wait = min((2 ** (attempt + 1)) * 0.75, 45)
                await asyncio.sleep(wait)

    return None


async def _run_realtime_async(
    all_questions,
    data,
    api_key,
    model_name,
    max_retry,
    workers,
    save_every,
    output_file,
    chunk_size,
):
    from google import genai

    client = genai.Client(api_key=api_key)
    async_client = client.aio
    semaphore = asyncio.Semaphore(max(1, workers))

    total_success = 0
    total_completed = 0
    total_failed = 0
    failed_items = []

    pbar = tqdm(total=len(all_questions), desc=f"Rewriting questions async ({workers} workers)")

    for chunk_start in range(0, len(all_questions), chunk_size):
        chunk = all_questions[chunk_start:chunk_start + chunk_size]

        async def run_one(task):
            vid, i, question, question_type, choices, qid = task
            result = await rewrite_question_async(
                async_client,
                semaphore,
                model_name,
                question,
                question_type,
                choices,
                max_retry=max_retry,
            )
            return {
                'video_id': vid,
                'question_index': i,
                'qid': qid,
                'question': question,
                'type': question_type,
                'result': result,
            }

        results = await asyncio.gather(*(run_one(task) for task in chunk), return_exceptions=True)

        for out in results:
            total_completed += 1

            if isinstance(out, Exception):
                total_failed += 1
                failed_items.append({
                    'video_id': '',
                    'qid': '',
                    'type': '',
                    'question': '',
                    'error': str(out),
                })
                pbar.update(1)
                pbar.set_postfix({'done': total_success, 'fail': total_failed})
                continue

            vid = out['video_id']
            i = out['question_index']
            if out['result']:
                data[vid]['response'][i] = out['result']
                total_success += 1
            else:
                total_failed += 1
                failed_items.append({
                    'video_id': vid,
                    'qid': out['qid'],
                    'type': out['type'],
                    'question': out['question'],
                    'error': 'No normalized response',
                })

            pbar.update(1)
            pbar.set_postfix({
                'done': total_success,
                'fail': total_failed,
                'vid': vid[:12],
            })

            if total_completed % save_every == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    pbar.close()
    return total_success, total_completed, total_failed, failed_items


def _process_one_question(task, api_key, model_name, max_retry, sleep_sec):
    vid, i, question, question_type, choices, qid = task
    client = _get_thread_client(api_key)
    result = rewrite_question(
        client,
        model_name,
        question,
        question_type,
        choices,
        max_retry=max_retry,
    )
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return {
        'video_id': vid,
        'question_index': i,
        'qid': qid,
        'question': question,
        'type': question_type,
        'result': result,
    }


def _parse_batch_key(key):
    try:
        vid, idx, qid = key.split('|', 2)
        return vid, int(idx), qid
    except Exception:
        return None


def _extract_text_from_batch_response(response_obj):
    if response_obj is None:
        return None

    if isinstance(response_obj, dict):
        if isinstance(response_obj.get('text'), str):
            return response_obj['text']

        candidates = response_obj.get('candidates', [])
        if isinstance(candidates, list) and len(candidates) > 0:
            first = candidates[0] if isinstance(candidates[0], dict) else {}
            content = first.get('content', {}) if isinstance(first, dict) else {}
            parts = content.get('parts', []) if isinstance(content, dict) else []
            texts = []
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get('text'), str):
                    texts.append(part['text'])
            if texts:
                return '\n'.join(texts)

    return None


def _build_batch_requests(tasks):
    requests = []
    for vid, i, question, question_type, choices, qid in tasks:
        prompt = _build_prompt(question, question_type, choices)
        requests.append({
            'key': f'{vid}|{i}|{qid}',
            'request': {
                'contents': [
                    {
                        'parts': [{'text': prompt}],
                        'role': 'user',
                    }
                ],
            },
        })
    return requests


def _save_batch_requests_jsonl(batch_requests, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in batch_requests:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def _poll_batch_job(client, job_name, poll_interval):
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }
    while True:
        job = client.batches.get(name=job_name)
        state_name = job.state.name
        print(f'  Batch state: {state_name}')
        if state_name in completed_states:
            return job
        time.sleep(poll_interval)


# ── Export functions ──────────────────────────────────────────────────────

def export_csv(data, output_path, qa_csv_path, csv_desc_mode='all'):
    """Export results to CSV, preserving the exact row order of the original CSV."""
    with open(qa_csv_path, 'r', encoding='utf-8') as f_in:
        original_rows = list(csv.DictReader(f_in))
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'video_id', 'qid', 'type',
            'question', 'answer', 'answer_idx', 'a0', 'a1', 'a2', 'a3', 'a4',
            'grounding_description', 'relationship', 'reasoning',
            'sub_0', 'sub_1', 'sub_2', 'sub_3',
        ])
        
        # Track per-video question index to match response array
        vid_qi = defaultdict(int)
        
        for row in original_rows:
            vid = row['video_id']
            qi = vid_qi[vid]
            vid_qi[vid] += 1
            
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            answer_text = row['answer']
            answer_idx = next(
                (i for i, c in enumerate(choices) if c.strip() == answer_text.strip()), -1
            )
            
            r = None
            if vid in data and qi < len(data[vid]['response']):
                r = data[vid]['response'][qi]
            
            # Extract sub-queries into separate columns
            subs = ['', '', '', '']
            if r and 'query_json' in r:
                for sq in r['query_json']:
                    sid = sq['sub_query_id']
                    if sid < len(subs):
                        descriptions = sq.get('descriptions', [])
                        if csv_desc_mode == 'first':
                            subs[sid] = descriptions[0] if descriptions else ''
                        else:
                            subs[sid] = ' | '.join(descriptions)
            
            writer.writerow([
                vid, row['qid'], row['type'],
                row['question'], answer_text, answer_idx,
                row['a0'], row['a1'], row['a2'], row['a3'], row['a4'],
                r.get('grounding_description', '') if r else '',
                r.get('relationship', '') if r else '',
                r.get('reasoning', '') if r else '',
                subs[0], subs[1], subs[2], subs[3],
            ])
    
    print(f"  CSV exported: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Rewrite NExT-GQA questions to grounding queries')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--output', default=None, help='Output JSON path (auto-generated if not specified)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    parser.add_argument('--dry_run', action='store_true', help='Test with 2 videos only')
    parser.add_argument('--sample', type=int, default=None, help='Process N random videos')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Gemini model name')
    parser.add_argument('--max_retry', type=int, default=5, help='Max retries per question')
    parser.add_argument('--sleep', type=float, default=0.0, help='Sleep time (seconds) between requests')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel requests (threads)')
    parser.add_argument('--async_chunk_size', type=int, default=100, help='Async mode: number of questions per gather chunk')
    parser.add_argument('--save_every', type=int, default=25, help='Save checkpoint every N completed questions')
    parser.add_argument('--start_index', type=int, default=0, help='Skip the first N pending questions before processing')
    parser.add_argument('--first_n', type=int, default=None, help='Process first N pending questions in original order')
    parser.add_argument('--mode', default='realtime_async', choices=['realtime', 'realtime_async', 'batch'], help='Rewrite mode')
    parser.add_argument('--submit_only', action='store_true', help='Batch mode: submit job and exit without polling')
    parser.add_argument('--batch_job_name', default=None, help='Batch mode: existing job name to collect, e.g. batches/123')
    parser.add_argument('--poll_interval', type=int, default=30, help='Batch mode: polling interval in seconds')
    parser.add_argument('--csv_desc_mode', default='all', choices=['all', 'first'], help='CSV sub-query export mode')
    args = parser.parse_args()
    
    # Setup paths — only need CSV
    qa_file = f'dataset/nextgqa/{args.split}.csv'
    output_file = args.output or f'dataset/nextgqa/llm_outputs_{args.split}.json'
    
    # Load questions from CSV only
    print(f"Loading NExT-GQA {args.split} split...")
    data = load_questions_from_csv(qa_file)
    total_questions = sum(len(v['sentences']) for v in data.values())
    print(f"  {len(data)} videos, {total_questions} questions")
    
    # Resume if exists
    if args.resume and os.path.exists(output_file):
        print(f"Resuming from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        for vid in existing:
            if vid in data:
                for i, resp in enumerate(existing[vid].get('response', [])):
                    if i < len(data[vid]['response']) and resp:
                        data[vid]['response'][i] = resp
        already_done = sum(
            1 for v in data.values()
            for r in v['response'] if r
        )
        print(f"  {already_done} questions already processed")
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Set it in .env file.")

    if args.workers < 1:
        raise ValueError('--workers must be >= 1')
    if args.async_chunk_size < 1:
        raise ValueError('--async_chunk_size must be >= 1')
    if args.save_every < 1:
        raise ValueError('--save_every must be >= 1')
    if args.poll_interval < 5:
        raise ValueError('--poll_interval should be >= 5 seconds')

    client = _get_thread_client(api_key)
    print(f"Using model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.mode in {'realtime', 'realtime_async'}:
        print(f"Workers: {args.workers}")
    
    # Determine which videos to process
    video_ids = list(data.keys())
    if args.dry_run:
        video_ids = video_ids[:2]
        print("DRY RUN: Processing 2 videos only")
    elif args.sample:
        import random
        random.seed(42)
        random.shuffle(video_ids)
        video_ids = video_ids[:args.sample]
        print(f"SAMPLE: Processing {args.sample} videos")
    
    # Build flat list of tasks
    all_questions = []
    for vid in video_ids:
        for i, question in enumerate(data[vid]['sentences']):
            if not data[vid]['response'][i]:
                qmeta = data[vid]['meta']['questions'][i]
                all_questions.append((
                    vid,
                    i,
                    question,
                    qmeta['type'],
                    qmeta['choices'],
                    qmeta['qid'],
                ))
    
    total_skipped = sum(
        1 for vid in video_ids
        for r in data[vid]['response'] if r
    )
    print(f"  {len(all_questions)} questions to rewrite, {total_skipped} already done")

    if args.start_index < 0:
        raise ValueError('--start_index must be >= 0')
    if args.start_index > 0:
        all_questions = all_questions[args.start_index:]
        print(f"START_INDEX: Skipped first {args.start_index} pending questions")

    if args.first_n is not None:
        if args.first_n < 1:
            raise ValueError('--first_n must be >= 1')
        all_questions = all_questions[:args.first_n]
        print(f"FIRST_N: Processing first {len(all_questions)} pending questions")
    
    total_success = 0
    total_completed = 0
    total_failed = 0
    last_saved_vid = None
    failed_items = []

    if args.mode == 'batch':
        from google.genai import types

        if args.batch_job_name:
            job_name = args.batch_job_name
            print(f"Collecting existing batch job: {job_name}")
        else:
            if not all_questions:
                print('No pending questions to submit in batch mode.')
                job_name = None
            else:
                batch_requests = _build_batch_requests(all_questions)
                batch_jsonl_file = output_file.replace('.json', '_batch_requests.jsonl')
                _save_batch_requests_jsonl(batch_requests, batch_jsonl_file)
                print(f"Batch requests saved: {batch_jsonl_file}")

                uploaded_file = client.files.upload(
                    file=batch_jsonl_file,
                    config=types.UploadFileConfig(
                        display_name=f'nextgqa-{args.split}-rewrite-requests',
                        mime_type='jsonl',
                    )
                )
                print(f"Uploaded batch input file: {uploaded_file.name}")

                batch_job = client.batches.create(
                    model=args.model,
                    src=uploaded_file.name,
                    config={'display_name': f'nextgqa-{args.split}-rewrite-batch'},
                )
                job_name = batch_job.name
                print(f"Created batch job: {job_name}")

                job_meta_file = output_file.replace('.json', '_batch_job.json')
                with open(job_meta_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'job_name': job_name,
                        'model': args.model,
                        'split': args.split,
                        'output_file': output_file,
                        'request_file': batch_jsonl_file,
                    }, f, indent=2, ensure_ascii=False)
                print(f"Saved batch job metadata: {job_meta_file}")

        if args.submit_only:
            print('Submit-only mode finished.')
            return

        if not job_name:
            print('Nothing to collect.')
            return

        print(f"Polling batch job every {args.poll_interval}s...")
        final_job = _poll_batch_job(client, job_name, args.poll_interval)
        final_state = final_job.state.name
        print(f"Final batch state: {final_state}")

        if final_state != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(f'Batch job did not succeed: {final_state}')

        result_file_name = getattr(final_job.dest, 'file_name', None)
        if not result_file_name:
            raise RuntimeError('Batch succeeded but no result file found in destination.')

        file_content = client.files.download(file=result_file_name).decode('utf-8')
        result_lines = [ln for ln in file_content.splitlines() if ln.strip()]
        print(f"Downloaded {len(result_lines)} batch result lines")

        for line in result_lines:
            try:
                item = json.loads(line)
            except Exception:
                continue

            key = item.get('key', '')
            parsed_key = _parse_batch_key(key)
            if not parsed_key:
                continue
            vid, i, qid = parsed_key

            total_completed += 1
            response_text = _extract_text_from_batch_response(item.get('response'))
            parsed = _try_extract_json_object(response_text)
            normalized = _normalize_response(parsed)

            if normalized is not None:
                if vid in data and 0 <= i < len(data[vid]['response']):
                    data[vid]['response'][i] = normalized
                    total_success += 1
                else:
                    total_failed += 1
                    failed_items.append({
                        'video_id': vid,
                        'qid': qid,
                        'type': '',
                        'question': '',
                        'error': 'Invalid key mapping in batch result',
                    })
            else:
                total_failed += 1
                qmeta = None
                question_text = ''
                question_type = ''
                if vid in data and 0 <= i < len(data[vid]['meta']['questions']):
                    qmeta = data[vid]['meta']['questions'][i]
                    question_text = data[vid]['sentences'][i]
                    question_type = qmeta.get('type', '')
                failed_items.append({
                    'video_id': vid,
                    'qid': qid,
                    'type': question_type,
                    'question': question_text,
                    'error': item.get('error', 'No normalized response'),
                })

        pending_after_collect = sum(
            1 for vid in video_ids
            for r in data[vid]['response'] if not r
        )
        print(f"Batch collected: success={total_success}, failed={total_failed}, pending={pending_after_collect}")

    elif args.mode == 'realtime_async':
        total_success, total_completed, total_failed, failed_items = asyncio.run(
            _run_realtime_async(
                all_questions=all_questions,
                data=data,
                api_key=api_key,
                model_name=args.model,
                max_retry=args.max_retry,
                workers=args.workers,
                save_every=args.save_every,
                output_file=output_file,
                chunk_size=args.async_chunk_size,
            )
        )
    elif args.workers == 1:
        pbar = tqdm(all_questions, desc="Rewriting questions")
        for vid, i, question, question_type, choices, qid in pbar:
            result = rewrite_question(
                client,
                args.model,
                question,
                question_type,
                choices,
                max_retry=args.max_retry,
            )

            total_completed += 1
            if result:
                data[vid]['response'][i] = result
                total_success += 1
            else:
                total_failed += 1
                failed_items.append({
                    'video_id': vid,
                    'qid': qid,
                    'type': question_type,
                    'question': question,
                })
                tqdm.write(f"  FAILED: {vid} q{i}: {question[:60]}...")

            pbar.set_postfix({
                'done': total_success,
                'fail': total_failed,
                'vid': vid[:12],
            })

            if vid != last_saved_vid or total_completed % args.save_every == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                last_saved_vid = vid

            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        pbar = tqdm(total=len(all_questions), desc=f"Rewriting questions ({args.workers} workers)")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _process_one_question,
                    task,
                    api_key,
                    args.model,
                    args.max_retry,
                    args.sleep,
                )
                for task in all_questions
            ]

            for fut in as_completed(futures):
                out = fut.result()
                vid = out['video_id']
                i = out['question_index']

                total_completed += 1
                if out['result']:
                    data[vid]['response'][i] = out['result']
                    total_success += 1
                else:
                    total_failed += 1
                    failed_items.append({
                        'video_id': vid,
                        'qid': out['qid'],
                        'type': out['type'],
                        'question': out['question'],
                    })
                    tqdm.write(f"  FAILED: {vid} q{i}: {out['question'][:60]}...")

                pbar.update(1)
                pbar.set_postfix({
                    'done': total_success,
                    'fail': total_failed,
                    'vid': vid[:12],
                })

                if total_completed % args.save_every == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Final save — readable JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Export readable CSV (same order as original test.csv)
    csv_file = output_file.replace('.json', '.csv')
    export_csv(data, csv_file, qa_file, csv_desc_mode=args.csv_desc_mode)

    if failed_items:
        failed_file = output_file.replace('.json', '_failed.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"  Failed list: {failed_file}")
    
    print(f"\n{'='*50}")
    print(f"Done! Outputs:")
    print(f"  JSON: {output_file}")
    print(f"  CSV:  {csv_file}")
    print(f"  Rewritten: {total_success}")
    print(f"  Skipped:   {total_skipped}")
    print(f"  Failed:    {total_failed}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
