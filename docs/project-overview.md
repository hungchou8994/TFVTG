# Project Overview - Training-Free Temporal Grounded VideoQA on NExT-GQA

> For new agent sessions: read this file first, then `docs/project-status.md`.

---

## 1. Project summary

**Goal:** Build a training-free pipeline for Temporal Grounded VideoQA on NExT-GQA.

**Core idea:**
1. Query rewrite with Gemini (text-only).
2. Temporal grounding with BLIP-2 features + TFVTG (`vlm_localizer.py`).
3. Proposal re-ranking with Gemini Verifier (Files API).
4. MCQ answering with Gemini Answerer (clip-based), with optional Qwen2-VL branch.
5. Evaluate both grounding and QA metrics.

This repo combines:
- Original TFVTG code (root-level modules)
- Adapted NExT-GQA pipeline (`nextgqa/`)
- Upstream VideoMind codebase as reference/baseline (`VideoMind/`)

---

## 2. Dataset and data files

Main split used now: **NExT-GQA test**
- Questions: 5553
- Videos: 990
- Types: CW, CH, TN, TC, TP

Key files:
- `dataset/nextgqa/test.csv`
- `dataset/nextgqa/gsub_test.json`
- `dataset/nextgqa/map_vid_vidorID.json`
- `dataset/nextgqa/videos/`

---

## 3. Pipeline (current)

```text
Query Rewrite -> Grounder -> Verifier -> Answerer -> Eval
```

### Step 1: Query Rewrite
- Script: `nextgqa_query_rewrite.py`
- Input: `test.csv`
- Output: `llm_outputs_test.json` (+ csv export)
- Model: `gemini-2.5-flash`

Run:
```bash
python nextgqa_query_rewrite.py --split test
```

### Step 2: Feature Extraction (GPU/Colab)
- Script: `feature_extraction_nextgqa.py`
- Output: `dataset/nextgqa/blip2_features/{video_id}.npy`
- Requires GPU (BLIP-2)

### Step 3: Grounder
- Script: `nextgqa/grounder.py`
- Input: `llm_outputs_test.json` + BLIP-2 features
- Output: `grounder_outputs_test.json` (top-5 proposals)

Run:
```bash
python -m nextgqa.grounder --split test
```

### Step 4: Verifier
- Script: `nextgqa/verifier.py`
- Input: `grounder_outputs_test.json`
- Output: `verifier_outputs_test.json`
- Uses Gemini Files API + upload cache

Run:
```bash
python -m nextgqa.verifier --split test --workers 15 --rpm 300
```

### Step 5: Answerer (Gemini)
- Script: `nextgqa/answerer.py`
- Input: `verifier_outputs_test.json`
- Output: `answerer_outputs_test.json`
- Uses best proposal -> trim clip -> answer A/B/C/D/E

Run:
```bash
python -m nextgqa.answerer --split test --workers 15 --rpm 300
```

### Optional: Answerer (Qwen2-VL)
- Script: `nextgqa/answerer_qwen.py`
- Input: `verifier_outputs_test.json`
- Output: `answerer_outputs_qwen_test.json`

Run:
```bash
python -m nextgqa.answerer_qwen --split test --model_path Qwen/Qwen2-VL-7B-Instruct
```

### Step 6: Evaluation
- Grounding eval: `nextgqa/eval_ground.py`
- QA eval: `nextgqa/eval_qa.py`

Run:
```bash
python -m nextgqa.eval_ground --split test --source verifier
python -m nextgqa.eval_qa --split test --source gemini
```

---

## 4. Actual current status (from repo state)

- `llm_outputs_test.json`: ready (990 video keys)
- `grounder_outputs_test.json`: 5553 entries
- `verifier_outputs_test.json`: 5553 entries
  - entries with `best_proposal`: 5522
- `answerer_outputs_test.json`: 5553 entries
  - entries with `predicted_answer_idx`: 781
- `answerer_outputs_qwen_test.json`: not generated yet

So the active bottleneck is **finishing Answerer inference** and then running final QA eval.

---

## 5. Environment notes

- Local: Windows + Gemini API steps (rewrite/verifier/answerer/eval)
- Colab GPU: BLIP-2 extraction + grounder
- API key: `.env` with `GOOGLE_API_KEY`

---

## 6. Important scripts index

- `nextgqa_query_rewrite.py`
- `feature_extraction_nextgqa.py`
- `nextgqa/grounder.py`
- `nextgqa/verifier.py`
- `nextgqa/answerer.py`
- `nextgqa/answerer_qwen.py`
- `nextgqa/eval_ground.py`
- `nextgqa/eval_qa.py`

