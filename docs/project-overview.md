# Project Overview — Training-Free Temporal Grounded VideoQA on NExT-GQA

> **Dành cho agent session mới:** Đọc file này để hiểu toàn bộ dự án, trạng thái hiện tại, và việc cần làm tiếp theo. Sau đó đọc `docs/project-status.md` để nắm chi tiết bugs và thứ tự ưu tiên.

---

## 1. Tóm tắt dự án

**Tên:** Training-Free Video Temporal Grounding (TFVTG) trên NExT-GQA  
**Mục tiêu:** Trả lời câu hỏi MCQ về video (NExT-GQA) mà **không cần training**, đồng thời định vị được đoạn video liên quan (temporal grounding).  
**Ngữ cảnh:** KLTN (Khóa Luận Tốt Nghiệp) — Đại học KHTN TP.HCM.

**Ý tưởng cốt lõi:** Kết hợp:
1. **BLIP-2 + TFVTG sliding window** → sinh top-5 temporal proposals (không cần training)
2. **Gemini 2.5 Flash (Files API + full video)** → re-rank proposals (Verifier) và trả lời MCQ (Answerer)

---

## 2. Dataset — NExT-GQA

| Thuộc tính | Giá trị |
|---|---|
| Split dùng | **test** (5553 câu hỏi) |
| Số video | 990 videos từ VidOR dataset |
| Câu hỏi | MCQ 5 choices (A/B/C/D/E) |
| Annotation | GT temporal segments + correct answer index |
| Question types | CW, CH, TN, TC, TP |

**File dữ liệu:**
- `dataset/nextgqa/test.csv` — danh sách câu hỏi
- `dataset/nextgqa/gsub_test.json` — GT grounding segments + video duration
- `dataset/nextgqa/map_vid_vidorID.json` — map video_id → VidOR subfolder
- `dataset/nextgqa/videos/` — video files tổ chức theo subfolder 4 chữ số

**Cấu trúc một entry trong gsub_test.json:**
```json
{
  "video_key": {
    "q_id": {
      "question": "...",
      "answer": "A",          // correct answer letter
      "answer_idx": 0,       // 0-indexed
      "type": "CW",
      "a0": "...", "a1": "...", "a2": "...", "a3": "...", "a4": "...",
      "qid_start": 5.0,      // GT start seconds
      "qid_end": 12.0,       // GT end seconds
      "vid_dur": 30.5        // video duration
    }
  }
}
```

---

## 3. Pipeline

```
Query Rewrite → Grounder → Verifier → Answerer → Eval
     ✅             ✅          ⚠️         ❌        ❌
```

### Bước 1: Query Rewrite ✅

**Script:** `nextgqa_query_rewrite.py`  
**Input:** `dataset/nextgqa/test.csv`  
**Output:** `dataset/nextgqa/llm_outputs_test.json`  
**Mô tả:** Dùng Gemini để viết lại câu hỏi thành dạng declarative phrase (query) phù hợp cho BLIP-2 text-visual matching.

**Chạy lại (nếu cần):**
```bash
python nextgqa_query_rewrite.py --split test
```

---

### Bước 2: Feature Extraction ⚠️ (GPU/Colab)

**Script:** `feature_extraction_nextgqa.py`  
**Output:** `dataset/nextgqa/blip2_features/` (nằm trên Google Drive, không có local)  
**Mô tả:** Extract BLIP-2 visual features từng frame của video. Cần GPU.

**Lưu ý:** Features đã extract xong, nằm trên Drive. Chỉ cần re-run nếu thêm video mới.

---

### Bước 3: Grounder ✅

**Script:** `nextgqa/grounder.py`  
**Input:** `dataset/nextgqa/llm_outputs_test.json` + `blip2_features/`  
**Output:** `dataset/nextgqa/grounder_outputs_test.json`  
**Mô tả:** BLIP-2 scoring + sliding window → 5 temporal proposals mỗi câu hỏi.  
**Cần GPU** → chạy trên Colab.

**Format output:**
```json
[
  {
    "qid": "...", "video_id": "...", "question": "...",
    "answer_idx": 0, "type": "CW",
    "a0": "...", ..., "a4": "...",
    "top5_proposals": [[start, end, score], [start, end, score], ...],
    "query": "...",
    "gt_start": 5.0, "gt_end": 12.0, "vid_dur": 30.5
  },
  ...
]
```

**Kết quả hiện tại (đã eval):**
- Top-1 Grounder: mIoU=25.08%, mIoP=31.42%, IoP@0.5=26.26%
- Oracle Top-5:   mIoU=39.25%, mIoP=46.77%, IoP@0.5=43.96%

---

### Bước 4: Verifier ⚠️ (2950/5553 — đang chạy)

**Script:** `nextgqa/verifier.py`  
**Input:** `grounder_outputs_test.json`  
**Output:** `verifier_outputs_test.json`  
**Cache:** `dataset/nextgqa/video_upload_cache.json` (989 videos đã upload, TTL 47h)  
**Mô tả:** Upload video lên Gemini Files API → hỏi Gemini re-rank 5 proposals → chọn `best_proposal`.

**Kiến trúc verifier.py:**
- Phase 1: Upload unique videos lên Files API (async, `upload_workers=5`)
- Phase 2: Verify từng entry (async, `workers=15`, `rpm=300`)
- Rate limiter: token bucket RPM=300
- Retry 429: wait `60s × attempt`, max_retry=5
- Cache: `video_upload_cache.json` (key = absolute path, value = `{file_uri, file_name, uploaded_at}`)
- Timestamps: seconds → `MM:SS` format
- Model: `gemini-2.5-flash`

**Format output (mỗi entry verified thêm):**
```json
{
  "gemini_ranking": [4, 2, 1, 3, 5],
  "verifier_reasoning": "...",
  "best_proposal": [18.0, 27.3],
  "grounder_rank": 5
}
```

**Resume command:**
```bash
python -m nextgqa.verifier --split test --workers 15 --rpm 300
```
(Code tự động skip entries đã có `best_proposal`, resume-safe)

**Kết quả trên 2950/5553:**
- mIoU=29.03%, mIoP=35.92%, IoP@0.5=30.81%
- **Vượt VideoMind-2B** (28.6/36.4) mà không cần training

---

### Bước 5: Answerer ❌ (CHƯA IMPLEMENT)

**Script cần tạo:** `nextgqa/answerer.py`  
**Input:** `verifier_outputs_test.json`  
**Output:** `answerer_outputs_test.json`

**Thiết kế dự kiến (tương tự verifier.py):**
- Reuse `video_upload_cache.json` để lấy `file_uri`
- Dùng `videoMetadata(start_offset, end_offset)` để clip tại `best_proposal`
- Prompt: video clip + câu hỏi + 5 choices (A/B/C/D/E)
- Parse output: `predicted_answer` (letter), `predicted_answer_idx` (0-indexed)
- Same rate limiter pattern: `rpm=300`, `max_retry=5`

**Format output:**
```json
[
  {
    ...all verifier_outputs_test.json fields...,
    "predicted_answer": "B",
    "predicted_answer_idx": 1
  },
  ...
]
```

**Chạy:**
```bash
python -m nextgqa.answerer --split test --workers 15 --rpm 300
```

---

### Bước 6: Eval QA ❌ (CHƯA IMPLEMENT)

**Script cần tạo:** `nextgqa/eval_qa.py`  
**Input:** `answerer_outputs_test.json` + `gsub_test.json`  
**Metrics:**
- **Acc@QA** = % `predicted_answer_idx == answer_idx`
- **Acc@GQA** = % correct AND IoP(best_proposal, gt_segments) >= 0.5
- Breakdown by type: CW, CH, TN, TC, TP

---

## 4. Eval Grounding (đã có)

**Script:** `nextgqa/eval_ground.py`  
```bash
# Eval grounder
python -m nextgqa.eval_ground --split test --source grounder

# Eval verifier
python -m nextgqa.eval_ground --split test --source verifier
```

Metrics: mIoU, mIoP (main), IoP@0.3, IoP@0.5, IoP@0.7

---

## 5. Baseline Comparison

| Model | Training | mIoU | mIoP | IoP@0.5 | Acc@GQA |
|---|---|---|---|---|---|
| VideoMind-2B | Yes | 28.6 | 36.4 | — | 56.4 |
| VideoMind-7B | Yes | 31.4 | 38.7 | — | 59.4 |
| **Ours Grounder Top-1** | **No** | **25.08** | **31.42** | **26.26** | — |
| **Ours Verifier (partial)** | **No** | **29.03** | **35.92** | **30.81** | — |
| Target | No | ~28-31 | ~36-39 | — | ~20-25 |

---

## 6. Environment Setup

**OS:** Windows (local), Google Colab L4/A100 (GPU tasks)  
**Python:** Anaconda base env  
**API key:** `.env` file, `GOOGLE_API_KEY=...`  
**Gemini tier:** Tier 1 — RPM=1000, RPD=10000, TPM=1M  
**Model:** `gemini-2.5-flash`

**Key packages:**
```
google-genai==1.65.0   # Gemini API
aiohttp>=3.11          # async HTTP (3.10.5 có bug ClientConnectorDNSError)
python-dotenv
```

**Cài đặt:**
```bash
pip install -r requirements.txt
pip install aiohttp --upgrade  # đảm bảo >=3.11
```

---

## 7. File Index

| File | Status | Mô tả |
|---|---|---|
| `nextgqa/grounder.py` | ✅ | BLIP-2 sliding window, GPU, Colab |
| `nextgqa/verifier.py` | ✅ | Gemini Files API + timestamps, local |
| `nextgqa/eval_ground.py` | ✅ | mIoU/mIoP metrics |
| `nextgqa/answerer.py` | ❌ | Chưa implement |
| `nextgqa/eval_qa.py` | ❌ | Chưa implement |
| `dataset/nextgqa/llm_outputs_test.json` | ✅ | Query rewrite 5553/5553 |
| `dataset/nextgqa/grounder_outputs_test.json` | ✅ | BLIP-2 top-5 proposals 5553/5553 |
| `dataset/nextgqa/verifier_outputs_test.json` | ⚠️ | 2950/5553 verified, 2601 pending |
| `dataset/nextgqa/video_upload_cache.json` | ✅ | 989 videos uploaded to Files API |
| `dataset/nextgqa/blip2_features/` | ⚠️ | Trống locally, nằm trên Drive |
| `dataset/nextgqa/videos/` | ✅ | VidOR videos đã extract |

---

## 8. Thứ tự ưu tiên tiếp theo

1. **Resume verifier** (sau quota reset 0h UTC):
   ```bash
   python -m nextgqa.verifier --split test --workers 15 --rpm 300
   ```

2. **Eval verifier full** (sau khi 5553/5553 done):
   ```bash
   python -m nextgqa.eval_ground --split test --source verifier
   ```

3. **Implement `nextgqa/answerer.py`** — tham khảo `nextgqa/verifier.py` làm template

4. **Implement `nextgqa/eval_qa.py`** — Acc@QA, Acc@GQA

5. **Chạy Answerer + Eval final**

---

## 9. Ghi chú kỹ thuật quan trọng

### Gemini Files API
- Upload dùng `client.aio.files.upload(path=..., config=UploadFileConfig(mime_type="video/mp4"))`
- Poll trạng thái: `client.aio.files.get(name=file.name)` cho đến `state == "ACTIVE"`
- TTL: 48h sau khi upload → cache ghi `uploaded_at` timestamp, TTL check = 47h
- Limit: 20GB/file, tổng 20GB lưu trữ miễn phí
- Sau khi upload, truyền vào prompt qua `Part(file_data=FileData(file_uri=..., mime_type="video/mp4"))`

### Structured JSON output (Gemini SDK)
```python
from google.genai import types

response = await client.aio.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MySchema,  # Pydantic model
        temperature=0.1
    )
)
```

### Rate limiting pattern
```python
class RateLimiter:
    def __init__(self, rpm): ...
    async def acquire(self): ...  # token bucket, blocks until token available

rate_limiter = RateLimiter(rpm=300)
await rate_limiter.acquire()
response = await client.aio.models.generate_content(...)
```

### 429 retry pattern
```python
for attempt in range(max_retry):
    try:
        result = await verify_one(entry)
        break
    except Exception as e:
        if _is_rate_limit_error(e):
            await asyncio.sleep(60 * (attempt + 1))
        else:
            await asyncio.sleep(2 ** attempt)
```

### Resume pattern (verifier/answerer)
```python
# Skip đã processed
if "best_proposal" in entry:  # hoặc "predicted_answer"
    continue
```
→ Chạy lại command line là tự resume, không cần flag đặc biệt.
