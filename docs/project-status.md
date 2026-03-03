# Project Status — Training-Free Temporal Grounded VideoQA on NExT-GQA

> **Cập nhật lần cuối:** 2026-03-02

## Tổng quan

Kết hợp **VideoMind pipeline** (Grounder → Verifier → Answerer, bỏ Planner) với **TFVTG training-free Grounder** (BLIP-2) và **Gemini 2.5 Flash** (Verifier + Answerer). Benchmark trên **NExT-GQA test split** (5553 questions, 990 videos từ VidOR).

### Pipeline

```
Question + Choices
       |
       v
[Bước 5] Q→Query Rewrite (Gemini text-only)
       |
       v llm_outputs_test.json
       |
       v  ← cần BLIP-2 features + GPU (Colab)
[Bước 6] TFVTG Grounder (BLIP-2 + sliding window + NMS)
       |
       v grounder_outputs_test.json
       |
       v  ← cần video files + Gemini API (local OK)
[Bước 7] Gemini Verifier (ranking 5 proposals)
       |
       v verifier_outputs_test.json
       |
       v  ← cần video files + Gemini API (local OK)
[Bước 8] Gemini Answerer (MCQ từ best segment)
       |
       v answerer_outputs_test.json
       |
       v
[Bước 10] Evaluation (mIoU, mIoP, IoP@0.5, Acc@QA, Acc@GQA)
```

---

## Kết quả đã đạt được

### Bước 5 — Query Rewrite ✅ HOÀN THÀNH

- **File:** `dataset/nextgqa/llm_outputs_test.json` (6.9MB) + `llm_outputs_test.csv`
- **Script:** `nextgqa_query_rewrite.py`
- **Kết quả:** 5553/5553 questions đã được rewrite (100%)
- **Model:** gemini-2.5-flash, async mode
- **Format output mỗi question:**
  ```json
  {
    "reasoning": "...",
    "grounding_description": "A lady doing something with a puppy...",
    "relationship": "sequentially",
    "query_json": [
      {"sub_query_id": 0, "descriptions": ["full description"]},
      {"sub_query_id": 1, "descriptions": ["sub-event 1"]},
      {"sub_query_id": 2, "descriptions": ["sub-event 2"]}
    ]
  }
  ```

### Bước 2 — BLIP-2 Feature Extraction ⚠️ CHẠY TRÊN COLAB

- **Script:** `feature_extraction_nextgqa.py`
- **Lưu ý:** BLIP-2 cần GPU (CUDA), chạy trên Google Colab L4/A100
- **Status:** Đã chạy trên Colab (L4), có ~750+ videos extracted thành công
- **Vấn đề:** Một số videos bị OOM với batch_size=256 → cần retry với auto-retry code mới
- **Output:** `dataset/nextgqa/blip2_features/{video_id}.npy` shape `[T, 32, 256]`
- **Lưu trữ:** Features trên Google Drive: `KLTN/nextgqa_blip2_features/`
- **Notebook:** `.github/colab_feature_extraction.ipynb` — có cell cho cả feature extraction và grounder

### Bước 6 — Grounder ✅ HOÀN THÀNH

- **Script:** `nextgqa/grounder.py`
- **Status:** `grounder_outputs_test.json` hợp lệ, 5553 entries, mỗi entry có `top5_proposals`
- **Kết quả (đã tính điểm):**
  - Top-1 Grounder: mIoU=25.08%, mIoP=31.42%, IoP@0.5=26.26%
  - Top-5 Oracle:   mIoU=39.25%, mIoP=46.77%, IoP@0.5=43.96%
  - Breakdown: TN type yếu nhất (mIoP=22.84%), CW mạnh nhất (mIoP=35.46%)

---

## Cấu trúc files hiện tại

```
TFVTG/
├── dataset/nextgqa/
│   ├── val.csv                      ✅ (638KB)
│   ├── test.csv                     ✅ (1.0MB)
│   ├── gsub_val.json                ✅ (115KB) — GT grounding + duration
│   ├── gsub_test.json               ✅ (195KB) — GT grounding + duration
│   ├── map_vid_vidorID.json         ✅ (332KB) — video_id → VidOR folder
│   ├── llm_outputs_test.json        ✅ (6.9MB) — query rewrite hoàn chỉnh
│   ├── llm_outputs_test.csv         ✅ (3.2MB) — readable version
│   ├── grounder_outputs_test.json   ✅ 5553 entries — hợp lệ
│   ├── verifier_outputs_test.json   ⚠️ 2950/5553 done — đang chạy tiếp (quota hết, resume mai)
│   ├── video_upload_cache.json      ✅ 989 videos đã upload lên Files API (TTL 47h)
│   ├── blip2_features/              ⚠️ Rỗng locally — features nằm trên Drive
│   └── videos/                      ✅ Videos VidOR đã extract locally
│
├── nextgqa/                         Module chính
│   ├── __init__.py                  ✅
│   ├── grounder.py                  ✅ — TFVTG wrapper, chạy trên Colab
│   ├── verifier.py                  ✅ — Gemini Files API + timestamps, rate limiter RPM=300
│   ├── eval_ground.py               ✅ — tính mIoU/mIoP, hỗ trợ grounder + verifier source
│   └── (answerer.py)                ❌ Chưa implement
│
├── nextgqa_loader.py                ✅
├── nextgqa_query_rewrite.py         ✅
├── feature_extraction_nextgqa.py   ✅
├── data_configs.py                  ✅
├── vlm_localizer.py                 ✅
├── llm_prompting.py                 ✅
├── evaluate.py                      ✅
└── docs/
    ├── project-status.md            ← file này
    └── project-overview.md          ✅ — mô tả toàn diện cho agent session mới
```

---

## Vấn đề cần giải quyết

### 1. ⏳ Verifier chưa chạy đủ 5553 — cần resume sau khi quota reset

Đã chạy được 2950/5553. Quota `gemini-2.5-flash` Tier 1 (RPD=10000) bị hết trong ngày.
Resume bằng lệnh (quota reset lúc 0h UTC):
```bash
python -m nextgqa.verifier --split test --workers 15 --rpm 300
```
Code đã có rate limiter (RPM=300) và retry 429 tự động nên sẽ không fail hàng loạt nữa.

### 2. ❌ Chưa implement Answerer

`nextgqa/answerer.py` chưa tồn tại. Cần implement tương tự verifier:
- Input: `verifier_outputs_test.json` (dùng `best_proposal`)
- Upload video lên Files API (dùng lại `video_upload_cache.json`)
- Clip video tại `best_proposal` bằng `videoMetadata(start_offset, end_offset)`
- Prompt MCQ: question + choices (A/B/C/D/E) + video clip
- Output: `answerer_outputs_test.json` với `predicted_answer`, `predicted_answer_idx`

### 3. ❌ Chưa implement eval_qa.py

`nextgqa/eval_qa.py` chưa tồn tại. Cần tính:
- **Acc@QA** = % `predicted_answer_idx == answer_idx`
- **Acc@GQA** = % correct AND IoP(best_proposal, gt_segments) >= 0.5
- Breakdown by type: CW, CH, TN, TC, TP

---

## Thứ tự thực hiện tiếp theo

```
1. [Local] Resume verifier (sau khi quota reset)
   python -m nextgqa.verifier --split test --workers 15 --rpm 300

2. [Local] Eval verifier full
   python -m nextgqa.eval_ground --split test --source verifier

3. [Local] Implement nextgqa/answerer.py
   - Upload video Files API (reuse video_upload_cache.json)
   - Dùng videoMetadata(start_offset, end_offset) để clip tại best_proposal
   - Prompt MCQ với choices A-E, model gemini-2.5-flash
   - Output: answerer_outputs_test.json

4. [Local] Implement nextgqa/eval_qa.py
   - Acc@QA, Acc@GQA, breakdown by type CW/CH/TN/TC/TP

5. [Local] Chạy Answerer full
   python -m nextgqa.answerer --split test --workers 15 --rpm 300

6. [Local] Eval final
   python -m nextgqa.eval_ground --split test --source verifier
   python -m nextgqa.eval_qa --split test
```

---

## Kết quả tham khảo (baseline comparison)

| Method | Train? | mIoU | mIoP | IoP@0.5 | Acc@GQA |
|--------|--------|------|------|---------|---------|
| VideoMind-2B | Yes (481K SFT) | 28.6 | 36.4 | 32.6 | 25.2 |
| VideoMind-7B | Yes (481K SFT) | 31.4 | 39.0 | 35.3 | 28.2 |
| **Ours Grounder-only (Top-1)** | **No** | **25.08** | **31.42** | **26.26** | — |
| **Ours Grounder Oracle (Top-5)** | **No** | **39.25** | **46.77** | **43.96** | — |
| **Ours Verifier (2950/5553)** | **No** | **29.03** | **35.92** | **30.81** | — |
| Target | No | ~28-31 | ~36-39 | — | ~20-25 |

Verifier đã **vượt VideoMind-2B** về mIoU (29.03 vs 28.6) mà **không cần training**.
Kỳ vọng khi full 5553: mIoP ~36-38%, tiệm cận VideoMind-7B.

---

## Môi trường chạy

| Component | Môi trường | GPU cần? | Lý do |
|-----------|-----------|---------|-------|
| Query Rewrite (Bước 5) | Local Windows | Không | Chỉ Gemini API |
| Feature Extraction (Bước 2) | **Colab L4/A100** | **Có** | BLIP-2 ViT-L |
| Grounder (Bước 6) | **Colab L4/A100** | **Có** | BLIP-2 text encoder |
| Verifier (Bước 7) | Local Windows | Không | Video files + Gemini API |
| Answerer (Bước 8) | Local Windows | Không | Video files + Gemini API |
| Evaluation | Local Windows | Không | Pure Python |

**Lưu ý:** Videos VidOR đã có tại `dataset/nextgqa/videos/` (sau khi extract `videos.tar.gz`). Verifier + Answerer đọc trực tiếp từ đây.

---

## API Keys & Config

- **GOOGLE_API_KEY**: trong file `.env` (gitignored), dùng cho Query Rewrite + Verifier + Answerer
- **Model mặc định:** `gemini-2.5-flash` (default của verifier/answerer)
- **Dataset config:** `data_configs.py` → key `'nextgqa'`
  - `feature_path: 'dataset/nextgqa/blip2_features/'`
  - `video_path: 'dataset/nextgqa/videos/'`
  - `stride: 16, max_stride_factor: 0.5`
