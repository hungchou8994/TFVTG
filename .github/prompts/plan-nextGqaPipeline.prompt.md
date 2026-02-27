# Plan: Training-Free Temporal Grounded VideoQA on NExT-GQA & ReXTime

## Nguồn gốc & Ý tưởng

| Thành phần | Nguồn gốc | Paper |
|---|---|---|
| **Pipeline tổng thể** | **VideoMind** (bỏ Planner) | [arxiv 2503.13444](https://arxiv.org/abs/2503.13444) — ICLR 2026 |
| **Grounder** | **TFVTG** (training-free, BLIP-2) | [arxiv 2408.16219](https://arxiv.org/abs/2408.16219) — ECCV 2024 |
| **Verifier + Answerer** | Gemini Pro/Flash (multimodal) | Training-free, API-based |
| **Dataset chính** | **NExT-GQA** | [arxiv 2309.01327](https://arxiv.org/abs/2309.01327) — CVPR 2024 |
| **Dataset phụ** | **ReXTime** (val split, 921 samples) | [arxiv 2406.19392](https://arxiv.org/abs/2406.19392) — NeurIPS 2024 D&B |

---

## So sánh kiến trúc

### VideoMind (gốc) — Chain-of-LoRA, CẦN training

```
Question ──→ [Planner LoRA] ──→ [Grounder LoRA] ──→ [Verifier LoRA] ──→ [Answerer LoRA]
                  │                    │                    │                    │
               Qwen2-VL            Qwen2-VL            Qwen2-VL            Qwen2-VL
              (LoRA adapter)      (LoRA adapter)      (LoRA adapter)      (LoRA adapter)
```

- Mỗi module là 1 LoRA adapter trên cùng 1 base model (Qwen2-VL)
- Grounder: output `<|reg|>` tokens → predicted timestamps + confidence
- Verifier: nhận cropped video với `<|seg_start|>``<|seg_end|>` markers → Yes/No (logit comparison)
- Answerer: nhận best moment video crop → MCQ selection
- **CẦN 481K SFT data để train 4 LoRA adapters**

### Phương pháp đề xuất — Training-Free

```
Question ──→ [Gemini: Q→Query] ──→ [TFVTG Grounder] ──→ [Gemini: Verifier] ──→ [Gemini: Answerer]
                  │                       │                      │                       │
              Gemini API              BLIP-2                 Gemini API              Gemini API
             (text-only)      (frame-text similarity,     (multimodal:             (multimodal:
                              sliding window + NMS)       frames+question)         frames+MCQ)
```

- **Không cần training** — hoàn toàn dựa trên pre-trained models
- Grounder dùng BLIP-2 text-frame similarity (đã chứng minh hiệu quả trên Charades-STA/ActivityNet)
- Verifier/Answerer dùng Gemini multimodal xem frames trực tiếp
- Q→Query conversion thay thế vai trò Planner

### Tại sao kết hợp này hợp lý?

1. **TFVTG Grounder đã mạnh sẵn**: R@0.5 = 49.97 trên Charades-STA (zero-shot), gần bằng VideoMind-2B (51.1) mà VideoMind CẦN training
2. **VideoMind pipeline đã validated**: Grounder→Verifier→Answerer là design pattern hiệu quả cho grounded VideoQA
3. **Bỏ Planner hợp lý**: VideoMind cũng nhận ra không phải lúc nào cũng cần Planner — trong NExT-GQA mọi question đều cần grounding
4. **Gemini multimodal mạnh**: Gemini Pro/Flash xem frames trực tiếp → verify + answer tốt hơn pure text reasoning
5. **Training-free = generalizable**: Không overfit vào distribution cụ thể

### So sánh cụ thể (reference numbers)

| Method | Train? | Charades R@0.5 | NExT-GQA mIoU | NExT-GQA Acc@GQA |
|---|---|---|---|---|
| VideoMind-2B | Yes (481K SFT) | 51.1 | 28.6 | 25.2 |
| VideoMind-7B | Yes (481K SFT) | 59.1 | 31.4 | 28.2 |
| TFVTG (VLM+LLM) | **No** | 49.97 | — | — |
| **Ours (proposed)** | **No** | ~50 (reuse) | **target: 25-30** | **target: 20-25** |

---

## TL;DR

Kết hợp **VideoMind pipeline** (Grounder→Verifier→Answerer, bỏ Planner) với **TFVTG training-free Grounder** (BLIP-2) và **Gemini multimodal** (Verifier+Answerer). Benchmark trên **NExT-GQA** (primary) và **ReXTime** (secondary, val split 921 samples). Cả hai đều là Temporal Grounded VideoQA (MCQ + grounding). Toàn bộ training-free, dual features (BLIP-2 + CLIP-L). Target: tiệm cận hoặc vượt VideoMind-2B trên NExT-GQA mà không cần training.

---

## Bước 1: Chuẩn bị dữ liệu NExT-GQA

- Tạo thư mục `dataset/nextgqa/` chứa:
  - `val.csv`, `test.csv` (QA annotations — format: `video_id, qid, width, height, question, answer, answer_id, type, a0, a1, a2, a3, a4`)
  - `gsub_val.json`, `gsub_test.json` (grounding annotations — format: `{video_id: {duration, location: {qid: [[start, end], ...]}, fps}}`)
  - `map_vid_vidorID.json` (mapping video_id → video filename trong VidOR)
- Tải raw videos từ VidOR dataset, decode ở 3fps cho BLIP-2 features
- Cấu trúc features: `features/nextgqa/blip2/{video_id}.npy` (shape `[T, 32, D]`)

---

## Bước 1b: Chuẩn bị dữ liệu ReXTime (secondary benchmark)

> **ReXTime** (Reasoning-Across-Time, NeurIPS 2024 D&B): Video temporal reasoning — cause-and-effect across different video segments. Dùng **val split (921 samples)** làm benchmark, videos từ **ActivityNet + QVHighlights** (TFVTG đã hỗ trợ sẵn 2 dataset này).

### So sánh NExT-GQA vs ReXTime

| Aspect | NExT-GQA | ReXTime |
|---|---|---|
| Benchmark split | val (~5K) + test (~5K) | **val (921)** |
| Video source | VidOR | ActivityNet + QVHighlights |
| MCQ choices | 5 (A-E) | 4 (A-D) |
| Grounding metric | **IoP** (intersection/prediction) | **IoU** (standard) |
| Question format | Raw question ("What did X do after Y?") | Also question, but more descriptive/causal |
| Question types | CW, CH, TN, TC | sequential, cause_effect, means_end |
| Annotation format | CSV + JSON tách rời | JSON duy nhất |
| Q→Query cần convert? | **Bắt buộc** (BLIP-2 cần description) | **Vẫn cần** (questions, không phải descriptions) |
| Answer template | Chỉ answer_id | "From `<s0>` to `<e0>`, `<option>`." |
| Multi-GT segments | ~10% questions | Không (1 segment/question) |

### Data format (HuggingFace `rextime_val.json`)

```json
{
    "qid": "qvh_val241",           // prefix cho biết video source: qvh_ hoặc anet_
    "vid": "2I9-kvemtSU_510.0_660.0",  // video ID (QVH format: ytid_start_end)
    "duration": 150,                // video duration (seconds)
    "question": "What were the woman and man doing before they showed reactions...?",
    "answer": "From <s0> to <e0>, <option>.",  // template — <s0>,<e0> = timestamps, <option> = answer text
    "source": "qvhighlights_val",   // hoặc "ActivityNet_val_2"
    "category": "sequential",       // sequential | cause_effect | means_end
    "options": [                     // 4 lựa chọn (A-D)
        "a woman and a man are eating spicy food",
        "a woman and a man are running a marathon",
        "a woman and a man are participating in a hot desert tour",
        "a woman and a man are singing at a karaoke night"
    ],
    "span": [0, 28],               // ground truth temporal span [start, end] in seconds
    "ans": "A"                      // correct answer letter
}
```

### Setup

- Tải `rextime_val.json` từ HuggingFace (`ReXTime/ReXTime`) → `dataset/rextime/rextime_val.json`
- Videos: **KHÔNG cần tải riêng!** ReXTime dùng videos từ ActivityNet và QVHighlights — tận dụng features đã extract cho TFVTG gốc nếu có, hoặc extract mới
- Features: `features/rextime/blip2/{vid}.npy` — cần map `vid` → actual video path:
  - `anet_*` → `vid` field là ActivityNet video ID (e.g., `v_ZMTi498qnPc`)
  - `qvh_*` → `vid` field là QVHighlights format (e.g., `2I9-kvemtSU_510.0_660.0`, cần crop YouTube video)
- Tải `rextime_eval.py` evaluation script từ [GitHub ReXTime](https://github.com/ReXTime/ReXTime)

---

## Bước 2: Trích xuất features

- **BLIP-2 features**: Chạy `feature_extraction.py` hiện tại trên videos NExT-GQA (3fps). Cần map `video_id` → actual video path qua `map_vid_vidorID.json`
  
  ```bash
  python feature_extraction.py --input_root /path/to/vidor_videos --save_root features/nextgqa/blip2 --fps 3
  ```

- **CLIP-L features**: Viết script mới `feature_extraction_clip.py` dùng `openai/clip-vit-large-patch14` (6fps) để so sánh. Output shape sẽ khác: `[T, D]` thay vì `[T, 32, D]` → cần wrapper trong Grounder

---

## Bước 3: Thêm config NExT-GQA

- Thêm entry `'nextgqa'` vào `data_configs.py`:
  ```python
  'nextgqa': {
      'feature_path': 'features/nextgqa/blip2/',
      'stride': 15,            # videos ~40s, 3fps ≈ 120 frames
      'max_stride_factor': 0.5,
      'splits': {
          'val':  { 'annotation_file': '...', 'gsub_file': '...', 'pad_sec': 0.0 },
          'test': { 'annotation_file': '...', 'gsub_file': '...', 'pad_sec': 0.0 },
      }
  }
  ```
- `stride` và `max_stride_factor` cần tuning — video NExT-GQA trung bình ~40s, segments chiếm ~20% video (~8s), nên stride ~15 frames (≈5s at 3fps) là hợp lý ban đầu

- Thêm entry `'rextime'` vào `data_configs.py`:
  ```python
  'rextime': {
      'feature_path': 'features/rextime/blip2/',
      'stride': 30,            # videos dài hơn NExT-GQA (trung bình ~100-150s)
      'max_stride_factor': 0.5,
      'annotation_file': 'dataset/rextime/rextime_val.json',
  }
  ```
- Video ReXTime dài hơn NExT-GQA (nhiều video 150s từ QVHighlights) → stride lớn hơn

---

## Bước 4: Data Loader (multi-dataset)

- Tạo file `nextgqa/data_loader.py` — hỗ trợ cả NExT-GQA và ReXTime:

### NExT-GQA loader:
  - `load_qa_data(csv_path)` → đọc CSV, trả dict `{video_id: {questions: [{qid, question, answer, answer_id, type, choices: [a0..a4]}, ...]}}`
  - `load_grounding_data(json_path)` → đọc gsub JSON, trả dict `{video_id: {duration, qid_to_segments}}`
  - `merge_data(qa_data, ground_data)` → kết hợp thành format thống nhất

### ReXTime loader:
  - `load_rextime_data(json_path)` → đọc JSON, trả dict thống nhất. ReXTime đơn giản hơn vì tất cả trong 1 file JSON
  - Map `span` → `relevant_windows` format, `options` → `choices`, `ans` → `answer_id`

### Unified format (output chung cho cả 2 dataset):
  ```python
  {
      "video_id": str,
      "qid": str,
      "question": str,
      "choices": [str, ...],     # 5 cho NExT-GQA, 4 cho ReXTime
      "answer_id": int,          # index of correct answer (0-based)
      "gt_segments": [[s, e], ...],  # ground truth temporal segments
      "duration": float,
      "question_type": str,      # CW/CH/TN/TC hoặc sequential/cause_effect/means_end
      "dataset": str,            # "nextgqa" hoặc "rextime"
  }
  ```

- Lý do cần data loader riêng: NExT-GQA dùng CSV + JSON riêng biệt, khác hoàn toàn format dict-keyed JSON của Charades-STA. ReXTime dùng single JSON format khác nữa. Unified format cho phép pipeline chạy dataset-agnostic.

---

## Bước 5: Question → Grounding Query Converter (thay thế VideoMind Planner)

Trong VideoMind, **Planner** phân tích câu hỏi và quyết định pipeline (grounder → verifier → answerer hay chỉ answerer). Vì ta luôn ground trước rồi answer, ta thay Planner bằng **Gemini text-only prompt** chuyển question → grounding query phù hợp cho BLIP-2.

- Tạo file `nextgqa/question_converter.py`
- **Vì sao cần convert?** BLIP-2 matcher trong TFVTG expect descriptive text ("a person grabbing a puppy"), không phải questions ("What did the lady do after grabbing the puppy?"). VideoMind bypass vấn đề này vì Grounder LoRA đã được train để hiểu cả 2.
- **ReXTime cũng cần convert**: Dù ReXTime questions mô tả hơn ("What were the woman and man doing before they showed reactions...?"), chúng vẫn là dạng question, không phải description thuần → vẫn cần Q→Query conversion, nhưng có thể dễ hơn NExT-GQA vì questions ReXTime đã chứa nhiều context hơn.

**Thiết kế prompt — học hỏi từ cả hai nguồn:**

```
Prompt kết hợp:
1. Từ TFVTG (prompts.py v3): decompose thành sub-queries + relationship (sequentially/simultaneously)
2. Từ VideoMind (Planner): chuyển question → grounding query text
3. MỚI cho VideoQA: kết hợp answer choices vào sub-query descriptions

Input:  Question = "What did the lady do after grabbing the puppy?"
        Type = "TN" (Temporal-After)
        Choices = ["dress the puppy", "pet the puppy", "put it down", "feed it", "play with it"]

Output: {
    "reasoning": "Question asks about an action AFTER grabbing puppy. Need to find: (1) grabbing event, (2) next action. Choices suggest dressing/petting/putting down.",
    "grounding_query": "A lady doing something with a puppy after grabbing it",
    "relationship": "sequentially",
    "query_json": [
        {"sub_query_id": 0, "descriptions": [
            "A lady handling a puppy after picking it up",
            "Someone doing something with a puppy they just grabbed",
            "A woman performing an action on a puppy she holds"
        ]},
        {"sub_query_id": 1, "descriptions": [
            "A lady grabbing a puppy",
            "Someone picking up a small dog",
            "A woman reaching for and taking a puppy"
        ]},
        {"sub_query_id": 2, "descriptions": [
            "A lady dressing a puppy",
            "Someone putting clothes on a puppy",
            "A woman petting a small dog"
        ]}
    ]
}
```

**Xử lý theo question type (quan trọng!):**

**NExT-GQA types:**

| Type | Ý nghĩa | Chiến lược chuyển đổi |
|---|---|---|
| CW (Causal-Why) | "Why did X do Y?" | Ground sự kiện Y + context trước/sau |
| CH (Causal-How) | "How did X do Y?" | Ground sự kiện Y chi tiết |
| TN (Temporal-Before/After) | "What before/after X?" | Decompose thành 2 sub-queries sequentially |
| TC (Temporal-When) | "When did X happen?" | Ground sự kiện X trực tiếp |

**ReXTime categories:**

| Category | Ý nghĩa | Chiến lược chuyển đổi |
|---|---|---|
| sequential | "What before/after X?" | Decompose thành 2 events sequentially (tương tự TN) |
| cause_effect | "What happens when/because X?" | Ground cả cause + effect, liên kết nhân quả |
| means_end | "Why/How does X do Y?" | Ground hành động + mục đích (tương tự CW/CH) |

- Script `get_gemini_outputs.py` — tương tự `get_llm_outputs.py` nhưng đọc CSV input, hỗ trợ resume, lưu JSON output

---

## Bước 6: Grounder Module (TFVTG — thay thế VideoMind Grounder LoRA)

### So sánh với VideoMind Grounder

| Aspect | VideoMind Grounder | TFVTG Grounder (ours) |
|---|---|---|
| Model | Qwen2-VL + LoRA | BLIP-2 (frozen, no training) |
| Input | Video pixels + text query | Pre-extracted features + text |
| Output | `<\|reg\|>` tokens → timestamps | Multi-scale proposals + confidence |
| Training | 210K grounding SFT data | **None** |
| Top-k | Top-5 candidates → Verifier | Top-k proposals → Verifier |

### Implementation

- Tạo file `nextgqa/grounder.py`:
  - Wrapper quanh `vlm_localizer.localize()` hiện tại — **không cần sửa vlm_localizer.py**
  - Input: `video_features (.npy)` + `grounding_queries (từ step 5)`
  - Output: top-k `moments = [(start, end, confidence), ...]`
  - Xử lý đặc biệt cho NExT-GQA:
    - Một question có thể cần ground **nhiều segments disjoint** → giữ nguyên, không merge
    - Output top-5 proposals (giống VideoMind truyền top-5 cho Verifier)

### Pipeline Grounder chi tiết (tái sử dụng TFVTG evaluate.py::eval_with_llm logic)

```
Gemini output (query_json) ──→ Với mỗi sub-query (id >= 1):
                                    │
                                    ├── localize() → top-3 proposals
                                    │
                                    └── filter_and_integrate(sub_proposals, relation)
                                                │
                                                ├── search_combination() (sequentially/simultaneously)
                                                └── select_proposal() (weighted IoU voting)
                               ──→ Với query chính (id = 0):
                                    │
                                    └── localize() → top-7 proposals
                               ──→ Ghép tất cả → select_proposal() → top-5 candidates
```

- **Với CLIP-L features**: cần viết `calc_scores_clip()` riêng vì shape khác (`[T, D]` vs `[T, 32, D]`), scoring mechanism khác (CLIP text encoder thay vì BLIP-2 Qformer)

---

## Bước 7: Verifier Module (Gemini Multimodal — thay thế VideoMind Verifier LoRA)

### So sánh với VideoMind Verifier

| Aspect | VideoMind Verifier | Gemini Verifier (ours) |
|---|---|---|
| Model | Qwen2-VL + LoRA | Gemini Pro/Flash API |
| Input | Cropped video segment + `<\|seg_start\|>` `<\|seg_end\|>` markers | Keyframes (images) + question text |
| Score | `sigmoid(logit[Yes] - logit[No])` | Gemini-generated score (0-1) |
| Training | 232K verify SFT data | **None** |
| Cost | GPU inference | API cost ($) |

### Implementation

- Tạo file `nextgqa/verifier.py`
- Logic (mô phỏng VideoMind `infer_auto.py` Verifier section):
  1. Nhận **top-5** moments từ Grounder (giống VideoMind dùng top-5)
  2. Với mỗi moment `[start, end]`:
     - Mở rộng context: `offset = (end - start) / 2`, crop `[start - offset, end + offset]` (giống VideoMind)
     - Sample 4-6 keyframes đều đặn từ expanded window
  3. Gọi Gemini multimodal API: gửi keyframes + question + marker cho vùng [start, end]
  4. Prompt (dựa trên `VERIFIER_PROMPT` của VideoMind):
     ```
     "You are acting as the verifier now.
      You will be presented a text query describing a moment in a video.
      These frames are from {expanded_start}s to {expanded_end}s.
      The candidate moment is from {start}s to {end}s (frames marked with ★).
      
      Query: '{question}'
      
      Does this video segment between {start}s and {end}s cover the moment
      relevant to answering the question?
      
      Respond in JSON: {verified: true/false, confidence: 0.0-1.0, reasoning: str}"
     ```
  5. Final score = `Gemini_confidence × Grounder_confidence`
  6. Re-rank moments theo final score

### Optimization

- **Batch Gemini calls**: Có thể gửi tất cả 5 candidates trong 1 prompt (tiết kiệm API calls)
- **Early stopping**: Nếu candidate #1 có confidence > 0.9, skip verify phần còn lại

---

## Bước 8: Answerer Module (Gemini Multimodal — thay thế VideoMind Answerer LoRA)

### So sánh với VideoMind Answerer

| Aspect | VideoMind Answerer | Gemini Answerer (ours) |
|---|---|---|
| Model | Qwen2-VL + LoRA | Gemini Pro/Flash API |
| Input | Cropped video (best moment) + MCQ prompt | Keyframes (best moment) + MCQ prompt |
| Output | Generated text → parse option | JSON with answer_id + reasoning |
| Training | Part of 481K SFT | **None** |
| Min segment | `MIN_LEN = 32` frames or `MIN_RATIO` of duration | Configurable |

### Implementation

- Tạo file `nextgqa/answerer.py`
- Logic (mô phỏng VideoMind `infer_auto.py` Answerer section):
  1. Lấy **top-1 moment** đã verified (giống VideoMind: `selected = pred[0]`)
  2. Đảm bảo segment đủ dài: `min_len = max(duration * 0.15, 8s)` — tránh quá ngắn
  3. Sample 6-10 keyframes đều đặn từ selected moment
  4. Gọi Gemini multimodal (giống VideoMind Answerer format):
     ```
     "Based on these video frames (from {start}s to {end}s of a {duration}s video),
      answer the following question.
      
      Question: {question}
      Options:
      (A) {a0}
      (B) {a1}
      (C) {a2}
      (D) {a3}
      [(E) {a4}]  ← chỉ có với NExT-GQA (5 choices), ReXTime chỉ có 4 (A-D)
      
      Please only give the best option.
      Respond in JSON: {answer: 'A'/'B'/'C'/'D'[/'E'], reasoning: str}"
     ```
  5. Output: `{answer_id: int, answer: str, grounded_moment: [start, end]}`

### Merged Verifier+Answerer option

Có thể merge V+A thành **1 lần gọi Gemini** để giảm cost:
```
"Given these frames from {start}s to {end}s:
 1. Does this moment contain visual evidence to answer: '{question}'? (Yes/No, confidence 0-1)
 2. If yes, answer: Options: (A)... (B)... (C)... (D)... (E)...
 Respond in JSON: {relevant: bool, confidence: float, answer: 'A'-'E', reasoning: str}"
```
→ Tiết kiệm 50% API calls, nhưng mất modularity cho ablation

---

## Bước 9: Pipeline Orchestration

- Tạo file `nextgqa/pipeline.py` — orchestrate toàn bộ:
  ```
  load data → [optional: convert questions] → grounder → verifier → answerer → save results
  ```
- Tạo file `run_nextgqa.py` — entry point chính:
  ```bash
  python run_nextgqa.py \
      --dataset nextgqa \         # hoặc --dataset rextime
      --split val \
      --feature_type blip2 \
      --gemini_key YOUR_KEY \
      --skip_convert  # nếu đã có gemini_outputs.json
  ```
- Hỗ trợ chạy từng bước riêng lẻ (modular):
  ```bash
  python run_nextgqa.py --dataset nextgqa --step convert   # chỉ chạy Q→query
  python run_nextgqa.py --dataset nextgqa --step ground    # chỉ chạy Grounder
  python run_nextgqa.py --dataset nextgqa --step verify    # chỉ chạy Verifier
  python run_nextgqa.py --dataset nextgqa --step answer    # chỉ chạy Answerer
  python run_nextgqa.py --dataset nextgqa --step eval      # chỉ chạy evaluation
  python run_nextgqa.py --dataset rextime --step eval      # ReXTime evaluation
  ```

---

## Bước 10: Evaluation (multi-dataset)

### NExT-GQA Evaluation

- Tạo file `nextgqa/eval_ground.py` — đánh giá grounding:
  ```python
  def eval_grounding(predictions, gsub_data):
      # Với mỗi (video_id, qid):
      #   pred = [start, end]
      #   gt_segments = [[s1,e1], [s2,e2], ...]  (có thể nhiều segments)
      #   IoU = max over gt_segments of (intersection / union)
      #   IoP = max over gt_segments of (intersection / prediction_length)
      # Metrics: mIoU, mIoP, IoU@0.3, IoU@0.5, IoP@0.3, IoP@0.5
  ```
- Tạo file `nextgqa/eval_qa.py` — đánh giá QA:
  ```python
  def eval_qa(predictions, qa_data):
      # Acc@QA = % correct answers
      # Acc@GQA = % correct answers AND IoP ≥ 0.5
      # Breakdown by type: CW, CH, TN, TC
  ```
- NExT-GQA **dùng IoP khác IoU** — IoP = intersection / prediction_length. Đây là metric quan trọng vì NExT-GQA cho phép ground chỉ 1 phần nhỏ miễn nằm trong GT

### ReXTime Evaluation

- Tái sử dụng `rextime_eval.py` từ repo gốc hoặc tạo file `nextgqa/eval_rextime.py`:
  ```python
  def eval_rextime(predictions, gt_data):
      # Moment Retrieval metrics:
      #   MR-mIoU:    average IoU (standard, KHÔNG phải IoP)
      #   MR-R1@0.3:  % predictions with IoU ≥ 0.3
      #   MR-R1@0.5:  % predictions with IoU ≥ 0.5
      # VQA metrics:
      #   VQA:        % correct answers (A/B/C/D match)
      # Grounded VQA metrics (QUAN TRỌNG NHẤT):
      #   VQA,mIoU@0.5: % correct answers AND IoU ≥ 0.5
      # Breakdown by category: sequential, cause_effect, means_end
  ```
- **Khác biệt quan trọng với NExT-GQA**: ReXTime dùng **IoU** (không phải IoP), và chỉ có **1 GT segment per question** (không cần max over multiple GTs)
- Submission format: `{"qid": "...", "pred_relevant_windows": [[start, end]], "ans": "A"}`

---

## Bước 11: CLIP-L Feature Support (benchmark thứ 2)

- Tạo file `feature_extraction_clip.py`:
  - Dùng `openai/clip-vit-large-patch14`, 6fps
  - Output shape: `[T, D]` (768-dim) — khác BLIP-2 `[T, 32, 256]`
- Tạo `vlm_localizer_clip.py` hoặc thêm adapter trong `grounder.py`:
  - `calc_scores_clip()`: CLIP text encoder + cosine similarity, không cần Qformer
  - `generate_proposal` vẫn tái sử dụng logic sliding window + NMS hiện có

---

## Cấu trúc thư mục mới

```
TFVTG/
├── (existing files giữ nguyên)\
│
├── nextgqa/                          # NEW: module chính (dùng cho cả NExT-GQA & ReXTime)
│   ├── __init__.py
│   ├── data_loader.py                # Load data: NExT-GQA (CSV+JSON) & ReXTime (JSON)
│   ├── question_converter.py         # Q+choices → grounding queries (Gemini)
│   ├── grounder.py                   # Wrapper vlm_localizer (dataset-agnostic)
│   ├── verifier.py                   # Gemini multimodal verification
│   ├── answerer.py                   # Gemini multimodal QA
│   ├── pipeline.py                   # Orchestration toàn bộ (dataset-agnostic)
│   ├── eval_ground.py                # NExT-GQA grounding metrics (mIoU, mIoP, ...)
│   ├── eval_qa.py                    # NExT-GQA QA metrics (Acc@QA, Acc@GQA)
│   ├── eval_rextime.py              # ReXTime metrics (MR-mIoU, VQA, VQA,mIoU@0.5)
│   └── prompts.py                    # Prompt templates cho Gemini
│
├── feature_extraction_clip.py        # NEW: CLIP-L feature extraction
├── run_nextgqa.py                    # NEW: main entry point (--dataset nextgqa|rextime)
│
├── dataset/
│   ├── nextgqa/                      # NEW: NExT-GQA data
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── gsub_val.json
│   │   ├── gsub_test.json
│   │   ├── map_vid_vidorID.json
│   │   ├── gemini_outputs_val.json   # (generated: Q→query results)
│   │   └── gemini_outputs_test.json
│   │
│   └── rextime/                      # NEW: ReXTime data
│       ├── rextime_val.json          # 921 samples (từ HuggingFace ReXTime/ReXTime)
│       ├── rextime_eval.py           # Official eval script
│       └── gemini_outputs_val.json   # (generated: Q→query results)
│
└── features/
    ├── nextgqa/
    │   ├── blip2/                    # BLIP-2 features (.npy per video)
    │   └── clip/                     # CLIP-L features (.npy per video)
    │
    └── rextime/
        ├── blip2/                    # BLIP-2 features (anet + qvh videos)
        └── clip/                     # CLIP-L features
```

---

## Verification Checklist

### NExT-GQA
1. **Grounder-only ablation**: Chạy Grounder trên val set → kiểm tra mIoU, IoP (target: ~25-28 mIoU, so với VideoMind-2B 28.6)
2. **Grounder + Verifier ablation**: Thêm Verifier → kiểm tra mIoU cải thiện bao nhiêu (VideoMind Verifier tăng ~3-5 mIoU)
3. **Full pipeline test**: Chạy toàn bộ → kiểm tra cả 7 metrics (mIoU, mIoP, IoU@0.3/0.5, IoP@0.3/0.5, Acc@GQA)
4. **QA-only ablation**: Answerer nhận toàn bộ video (không ground) → Acc@QA baseline
5. **Unit test**: 10 samples, verify output format mỗi module
6. **BLIP-2 vs CLIP-L comparison**: Cùng pipeline, đổi feature type
7. **Gemini Flash vs Pro comparison**: Trade-off cost vs quality

### ReXTime
8. **ReXTime data loading**: Load `rextime_val.json` → verify 921 samples, đúng format
9. **ReXTime Grounder test**: 50 samples → kiểm tra MR-mIoU, MR-R1@0.5
10. **ReXTime full pipeline**: Chạy toàn bộ 921 samples → report MR-mIoU, MR-R1@0.3, MR-R1@0.5, VQA, VQA,mIoU@0.5
11. **Cross-dataset comparison**: So sánh pipeline behavior trên 2 datasets (NExT-GQA vs ReXTime)

---

## Ablation Study Plan

### NExT-GQA (primary)

| Exp | Q→Query | Grounder | Verifier | Answerer | Expect |
|---|---|---|---|---|---|
| A1 | ✗ (raw Q) | BLIP-2 | ✗ | ✗ | Baseline grounding (thấp vì Q ≠ description) |
| A2 | ✓ Gemini | BLIP-2 | ✗ | ✗ | Main grounding (target ~25 mIoU) |
| A3 | ✓ Gemini | BLIP-2 | ✓ Gemini | ✗ | Improved grounding (+3-5 mIoU) |
| A4 | ✓ Gemini | BLIP-2 | ✓ Gemini | ✓ Gemini | Full pipeline (target ~25 Acc@GQA) |
| A5 | ✗ | ✗ | ✗ | ✓ Gemini (full video) | QA-only baseline |
| A6 | ✓ Gemini | CLIP-L | ✓ Gemini | ✓ Gemini | Feature comparison |

### ReXTime (secondary — chạy sau khi NExT-GQA pipeline ổn định)

| Exp | Q→Query | Grounder | Verifier | Answerer | Expect |
|---|---|---|---|---|---|
| R1 | ✓ Gemini | BLIP-2 | ✗ | ✗ | ReXTime grounding-only (MR-mIoU, MR-R1@0.5) |
| R2 | ✓ Gemini | BLIP-2 | ✓ Gemini | ✓ Gemini | Full pipeline ReXTime (VQA, VQA,mIoU@0.5) |
| R3 | ✓ Gemini | CLIP-L | ✓ Gemini | ✓ Gemini | Feature comparison trên ReXTime |
| R4 | ✗ | ✗ | ✗ | ✓ Gemini (full video) | QA-only baseline ReXTime |

> **Lưu ý**: ReXTime val chỉ 921 samples (vs ~5K NExT-GQA), nên chạy nhanh hơn nhiều và tiết kiệm API cost. Dùng ReXTime để cross-validate pipeline trên domain khác (ActivityNet + QVHighlights videos vs VidOR).

---

## Rủi ro & Giải pháp

| Rủi ro | Mức độ | Giải pháp |
|---|---|---|
| BLIP-2 frame-text similarity yếu với causal Q | Cao | Q→Query conversion phải tốt; fallback: dùng CLIP-L |
| Gemini API cost lớn (5 verify calls × 3K+ questions) | Trung bình | Dùng Flash ($0.075/1M tokens); batch candidates trong 1 prompt |
| Q→Query conversion kém → grounding sai | Cao | Few-shot examples; validate trên 50 samples trước |
| NExT-GQA segments chỉ ~20% video → BLIP-2 miss | Trung bình | Giảm stride (15→10), tăng max_stride_factor |
| Multi-segment GT (1 question → nhiều disjoint segments) | Thấp | Chỉ 10% questions; eval lấy max IoU/IoP over GT segments |
| ReXTime videos dài (150s QVH) → nhiều proposals | Trung bình | Stride lớn hơn (30); NMS aggressive hơn |
| ReXTime dùng IoU (không phải IoP) → metric khác nhau | Thấp | Implement eval riêng; không nhầm lẫn 2 metric systems |
| ReXTime videos overlap với TFVTG existing → duplicate features | Thấp | Check existing features trước khi extract; symlink nếu có |

---

## Key Decisions

- **Gemini Pro/Flash**: Dùng Flash cho batch lớn (rẻ, nhanh), Pro cho quality test
- **BLIP-2 features dùng trước**: Tận dụng `vlm_localizer.py` không cần sửa, CLIP-L đi sau
- **IoP metric (NExT-GQA)**: Khác hoàn toàn IoU — cần implement riêng, không tái dùng `calc_iou` hiện tại
- **IoU metric (ReXTime)**: Standard IoU — tái dùng logic IoU từ TFVTG
- **Verifier + Answerer tách riêng**: Dù Gemini gọi 2 lần tốn hơn, nhưng modular hơn, dễ debug và ablation
- **Question → Query conversion bắt buộc cho cả 2 datasets**: Câu hỏi dạng "Why did X do Y?" không thể đưa thẳng vào BLIP-2 matcher
- **Stride=15 (NExT-GQA), Stride=30 (ReXTime)**: ReXTime videos dài hơn → cần stride lớn hơn
- **Top-5 candidates cho Verifier**: Giống VideoMind, balance giữa recall và API cost
- **Bỏ Planner**: Không cần vì cả NExT-GQA và ReXTime mọi Q đều cần grounding
- **Unified data format**: Pipeline chạy dataset-agnostic, chỉ khác ở data loader và eval metrics
- **NExT-GQA first, ReXTime after**: Develop + debug trên NExT-GQA, transfer sang ReXTime khi pipeline ổn

---

## Timeline ước tính

| Phase | Tasks | Thời gian |
|---|---|---|
| **P0** | Setup: data download (NExT-GQA + ReXTime), feature extraction, data loader (unified) | 3-4 ngày |
| **P1** | Q→Query converter + Grounder wrapper (A1, A2) | 3-4 ngày |
| **P2** | Verifier module (A3) + NExT-GQA evaluation | 2-3 ngày |
| **P3** | Answerer module (A4, A5) + full NExT-GQA eval | 2-3 ngày |
| **P4** | CLIP-L feature support (A6) + NExT-GQA ablation | 3-4 ngày |
| **P5** | ReXTime benchmark (R1-R4) + cross-dataset analysis | 2-3 ngày |
| **P6** | Tuning, final benchmark, results compilation | 2-3 ngày |
| **Total** | | **~3.5 tuần** |

> **Lưu ý**: P5 (ReXTime) có thể bắt đầu ngay khi P3 xong vì pipeline đã dataset-agnostic. ReXTime chỉ cần: (1) data loader, (2) feature extraction cho ReXTime videos, (3) eval script riêng. Phần còn lại (Q→Query, Grounder, Verifier, Answerer) tái sử dụng hoàn toàn.
