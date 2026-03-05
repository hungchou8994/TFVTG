# Project Status - Training-Free Temporal Grounded VideoQA on NExT-GQA

> Last updated: 2026-03-04

## 1. Snapshot

Pipeline in use:
```text
Query Rewrite -> Grounder -> Verifier -> Answerer -> Evaluation
```

Stack:
- Grounder: TFVTG + BLIP-2 features
- Verifier: Gemini 2.5 Flash (Files API)
- Answerer: Gemini 2.5 Flash (clip-based)
- Optional branch: Qwen2-VL answerer

---

## 2. Completed vs in-progress

### Completed
1. Query rewrite implementation and output generation.
2. Grounder implementation and full test split output.
3. Verifier implementation with resume-safe async flow and cache.
4. QA eval script implementation (`nextgqa/eval_qa.py`).
5. Qwen answerer script implementation (`nextgqa/answerer_qwen.py`).

### In progress
1. Gemini Answerer full-run completion on test split.
2. Final QA/GQA metrics after answerer run finishes.

---

## 3. Current dataset artifact status

### Main files
- `dataset/nextgqa/llm_outputs_test.json`: ready (990 video keys)
- `dataset/nextgqa/grounder_outputs_test.json`: 5553 entries
- `dataset/nextgqa/verifier_outputs_test.json`: 5553 entries
- `dataset/nextgqa/answerer_outputs_test.json`: 5553 entries

### Field-level progress
- `verifier_outputs_test.json`
  - `best_proposal`: 5522 / 5553
- `answerer_outputs_test.json`
  - `predicted_answer_idx`: 781 / 5553

### Side experiments
- `dataset/nextgqa/friend_verifier_outputs.json`: 1000 entries (997 with `best_proposal`)
- `dataset/nextgqa/friend_answerer_outputs.json`: not generated
- `dataset/nextgqa/answerer_outputs_qwen_test.json`: not generated

---

## 4. Grounding results (known)

From previous eval runs/documented results:
- Grounder Top-1: mIoU 25.08, mIoP 31.42, IoP@0.5 26.26
- Grounder Top-5 Oracle: mIoU 39.25, mIoP 46.77, IoP@0.5 43.96

Verifier historical partial report (older run):
- mIoU 29.03, mIoP 35.92, IoP@0.5 30.81

Note: verifier file is now full-length (5553 entries), so re-evaluation should be re-run for current final numbers.

---

## 5. Known blockers / risks

1. Answerer throughput and quota limits (Gemini API) can interrupt long runs.
2. Some entries do not have `best_proposal` (31 entries), reducing max possible Acc@GQA denominator.
3. Local environment may miss Python dependencies (for example `numpy`) when running eval scripts.

---

## 6. Recommended next commands

1. Resume/continue Gemini answerer:
```bash
python -m nextgqa.answerer --split test --workers 15 --rpm 300
```

2. Recompute grounding metrics on verifier output:
```bash
python -m nextgqa.eval_ground --split test --source verifier
```

3. Compute QA and grounded-QA metrics:
```bash
python -m nextgqa.eval_qa --split test --source gemini
```

4. Optional Qwen branch run:
```bash
python -m nextgqa.answerer_qwen --split test --model_path Qwen/Qwen2-VL-7B-Instruct
python -m nextgqa.eval_qa --split test --source qwen
```

---

## 7. Script ownership map

Core NExT-GQA scripts:
- `nextgqa_query_rewrite.py`
- `feature_extraction_nextgqa.py`
- `nextgqa/grounder.py`
- `nextgqa/verifier.py`
- `nextgqa/answerer.py`
- `nextgqa/answerer_qwen.py`
- `nextgqa/eval_ground.py`
- `nextgqa/eval_qa.py`

Support/debug:
- `_debug_verifier.py`

Reference baseline repo:
- `VideoMind/`

