# Training-free Zero-Shot Video Temporal Grounding using Large-scale Pre-trained Models

> **Bài báo được chấp nhận tại ECCV-2024**

Dự án này đề xuất một phương pháp **Video Temporal Grounding không cần huấn luyện (training-free), zero-shot** bằng cách kết hợp sức mạnh của các mô hình lớn được pre-trained sẵn — cụ thể là **BLIP-2** (VLM) và **GPT-4 / Gemini / LLaMA** (LLM). Phương pháp đạt hiệu suất tốt nhất trên Charades-STA và ActivityNet Captions trong bối cảnh zero-shot, đồng thời thể hiện khả năng tổng quát hóa cao trên các thiết lập cross-dataset và OOD.

![pipeline](imgs/pipeline.png)

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Giải thích chi tiết từng thành phần](#3-giải-thích-chi-tiết-từng-thành-phần)
   - [data_configs.py](#31-data_configspy)
   - [feature_extraction.py](#32-feature_extractionpy)
   - [vlm_localizer.py](#33-vlm_localizerpy)
   - [llm_prompting.py](#34-llm_promptingpy)
   - [get_llm_outputs.py](#35-get_llm_outputspy)
   - [chat_bots/](#36-chat_bots)
   - [evaluate.py](#37-evaluatepy)
   - [qvhhighlight_eval.py](#38-qvhhighlight_evalpy)
4. [Luồng thực thi (Pipeline)](#4-luồng-thực-thi-pipeline)
5. [Cài đặt và chạy thử](#5-cài-đặt-và-chạy-thử)
6. [Kết quả chính](#6-kết-quả-chính)
7. [Kiểm thử trên dataset tùy chỉnh](#7-kiểm-thử-trên-dataset-tùy-chỉnh)

---

## 1. Tổng quan kiến trúc

**Video Temporal Grounding (VTG)** là bài toán: cho trước một đoạn văn bản truy vấn (ví dụ: `"a person sits down on a chair"`), hệ thống cần xác định khoảng thời gian `[start, end]` trong video mà sự kiện đó xảy ra.

Phương pháp của dự án này **không cần huấn luyện bất kỳ mô hình nào**. Thay vào đó, nó khai thác hai loại mô hình lớn:

| Thành phần | Mô hình sử dụng | Vai trò |
|---|---|---|
| **VLM** (Vision-Language Model) | BLIP-2 (blip2_image_text_matching, coco) | Tính điểm tương đồng giữa frame video và văn bản, sinh ra các đề xuất đoạn thời gian |
| **LLM** (Large Language Model) | GPT-4-turbo / Gemini / LLaMA-3-70b | Phân tích truy vấn phức tạp thành các sub-query đơn giản, phân tích quan hệ thời gian |

**Ý tưởng cốt lõi:**
1. Dùng LLM để hiểu truy vấn phức tạp → tách thành các sub-query đơn giản + quan hệ thời gian (tuần tự / đồng thời).
2. Dùng VLM (BLIP-2) để tính điểm tương đồng văn bản-khung hình, từ đó sinh ra các đoạn đề xuất.
3. Kết hợp kết quả từ các sub-query theo quan hệ thời gian để ra đáp án cuối.

---

## 2. Cấu trúc thư mục

```
TFVTG/
│
├── data_configs.py          # Cấu hình đường dẫn và tham số cho từng dataset
├── feature_extraction.py    # Trích xuất đặc trưng BLIP-2 từ video
├── vlm_localizer.py         # Module định vị dùng VLM (BLIP-2)
├── llm_prompting.py         # Kết hợp kết quả từ sub-queries của LLM
├── get_llm_outputs.py       # Script gọi LLM API để phân tích truy vấn
├── evaluate.py              # Script đánh giá chính
├── qvhhighlight_eval.py     # Đánh giá cho dataset QVHighlight
│
├── chat_bots/               # Module giao tiếp với LLM
│   ├── __init__.py          # Factory function get_chat_model()
│   ├── base_bot.py          # Lớp cơ sở ChatBot (với cơ chế retry)
│   ├── chat_with_gemini.py  # Giao tiếp với Google Gemini
│   ├── chat_with_gpt.py     # Giao tiếp với OpenAI GPT
│   ├── chat_with_groq.py    # Giao tiếp với Groq (LLaMA)
│   └── prompts.py           # 3 phiên bản prompt (v1, v2, v3)
│
└── dataset/
    ├── charades-sta/        # Annotations và LLM outputs cho Charades-STA
    ├── activitynet/         # Annotations và LLM outputs cho ActivityNet
    └── qvhighlight/         # Annotations cho QVHighlight
```

---

## 3. Giải thích chi tiết từng thành phần

### 3.1 `data_configs.py`

File này định nghĩa một dict `DATASETS` chứa cấu hình cho từng dataset:

```python
DATASETS = {
    'charades': {
        'feature_path': '...',      # Đường dẫn đến thư mục chứa file .npy đặc trưng
        'stride': 20,               # Bước trượt cửa sổ khi sinh proposal
        'max_stride_factor': 0.5,   # Tỉ lệ tối đa chiều dài cửa sổ / tổng số frame
        'splits': {
            'default': {...},       # Split mặc định (test chuẩn)
            'OOD-1': {...},         # OOD: pad thêm 10 giây noise vào đầu video
            'OOD-2': {...},         # OOD: pad thêm 15 giây noise vào đầu video
            'test-ood': {...},      # Cross-distribution test split
            'novel-composition': {...},  # Split từ Charades-CG
            'novel-word': {...},         # Split từ Charades-CG
        }
    },
    'activitynet': { 'stride': 40, 'max_stride_factor': 1, ... },
    'qvhighlight':  { 'stride': 50, 'max_stride_factor': 0.5, ... },
}
```

**Tham số quan trọng:**
- `stride`: số frame mỗi bước trượt khi tạo proposal, ảnh hưởng trực tiếp đến độ phân giải thời gian.
- `max_stride_factor`: giới hạn kích thước cửa sổ tối đa = `max_stride_factor × total_frames`. Kiểm soát độ dài tối đa của segment được đề xuất.
- `pad_sec`: số giây noise ngẫu nhiên được thêm vào đầu video để mô phỏng OOD (out-of-distribution).

---

### 3.2 `feature_extraction.py`

Script trích xuất đặc trưng thị giác từ video thô bằng **BLIP-2**.

**Quy trình hoạt động:**

```
Video (.mp4) → Decode frames → Resize/Normalize → BLIP-2 Visual Encoder
→ Qformer → vision_proj → Đặc trưng [T, 32, D] → Lưu .npy
```

**Các hàm chính:**

| Hàm | Mô tả |
|---|---|
| `loadvideo(fname, fps, stride)` | Load video bằng `decord`, lấy mẫu frame theo fps (mặc định 3fps) hoặc stride cố định |
| `get_visual_features(video_path, ...)` | Chạy BLIP-2 để trích xuất đặc trưng, xử lý theo batch để tránh OOM |

**Định dạng đặc trưng đầu ra:**
- Shape: `(T, 32, D)` — T frame, mỗi frame có 32 query token từ Qformer, chiều không gian D.
- Lưu dưới dạng file `.npy` (half precision).

**Cách chạy:**

```bash
python feature_extraction.py --input_root /path/to/videos --save_root /path/to/features --fps 3
```

---

### 3.3 `vlm_localizer.py`

Đây là **module trung tâm** của toàn bộ hệ thống. Module này dùng BLIP-2 để tính điểm tương đồng văn bản-video và sinh ra các đoạn thời gian đề xuất.

#### Hàm `calc_scores(video_features, sentences)`

Tính ma trận điểm tương đồng giữa văn bản và từng frame video:

```
sentences → BLIP-2 Qformer (text branch) → text_proj → text_feat [M, D]
video_features [T, 32, D] → normalize
scores = einsum('md, npd -> mnp') → max over tokens → mean over sentences → [1, T]
```

- Dùng **cosine similarity** (normalize + einsum).
- Lấy `max` over 32 token query → đại diện điểm cao nhất cho mỗi frame.
- Lấy `mean` over các descriptions (nhiều cách diễn đạt cùng 1 query) → điểm frame tổng hợp.

#### Hàm `get_dynamic_scores(scores, stride, masks)`

Tính **điểm động** để phát hiện **điểm bắt đầu** (dynamic start) của sự kiện:
- Áp dụng **Gaussian smoothing** lên chuỗi điểm frame để giảm nhiễu.
- Tính **đạo hàm (diff)** của chuỗi điểm đã làm mượt.
- Tích lũy điểm khi gradient tăng mạnh liên tục (dùng hàm `nchk`) → phát hiện thời điểm bắt đầu sự kiện.
- Trả về `dynamic_idxs` (index của điểm bắt đầu) và `dynamic_scores` (điểm của điểm bắt đầu).

#### Hàm `generate_proposal(video_features, sentences, stride, max_stride)`

Sinh ra các đoạn thời gian đề xuất bằng **multi-scale sliding window**:

1. **Tính static scores**: Với mỗi kích thước cửa sổ từ `stride` đến `max_stride` (bước nhảy `stride`):
   - Dùng `conv1d` với kernel toàn 1 để tính tổng điểm bên trong cửa sổ (`inner_sum`).
   - **Static score** = `inner_sum / kernel_size - outer_sum / outer_num` (điểm trong - điểm ngoài → foreground/background contrast).
2. **Kết hợp dynamic scores**: Mỗi proposal cũng mang theo `dynamic_idx` — vị trí bắt đầu động (dùng làm thời điểm start thực sự).
3. **NMS** (`nms`): Lọc các proposal trùng lặp theo ngưỡng IoU 0.3.

#### Hàm `localize(video_feature, duration, query_json, stride, max_stride)`

Interface chính. Với mỗi query trong `query_json`:
- Gọi `generate_proposal` → lấy top-10 proposals.
- Chuyển đổi chỉ số frame → giây thực.
- Mỗi proposal trả về: `start` (dynamic start), `static_start`, `end`, `confidence`.

---

### 3.4 `llm_prompting.py`

Module này kết hợp các đề xuất từ nhiều sub-query (do LLM sinh ra) thành một kết quả cuối.

#### Hàm `select_proposal(inputs, gamma=0.6)`

Re-rank các proposal bằng phương pháp **weighted IoU voting**:

$$\text{score}[j] = \sum_{k} \text{IoU}(p_j, p_k)^{\gamma} \cdot w_k$$

- Với mỗi proposal $j$, tính tổng IoU có trọng số với tất cả proposal khác.
- Proposal nào có nhiều proposal khác "ủng hộ" (IoU cao, confidence cao) sẽ được xếp hạng cao.
- Điều này giúp chọn ra đoạn thời gian được đồng thuận nhất.

#### Hàm `search_combination(cands, idx, cur, relation)`

Duyệt đệ quy sinh tất cả tổ hợp proposal từ các sub-query khác nhau:
- **`sequentially`**: proposal của sub-query sau phải bắt đầu sau khi sub-query trước kết thúc.
- **`simultaneously`**: tất cả proposals phải overlap nhau (điều kiện: max(start) < min(end)).
- Score của một tổ hợp = tích confidence của từng proposal thành phần.

#### Hàm `filter_and_integrate(sub_query_proposals, relation)`

Wrapper gọi `search_combination`, rồi dùng `select_proposal` để chọn top-2 tổ hợp tốt nhất.

---

### 3.5 `get_llm_outputs.py`

Script tự động gọi LLM API để phân tích từng truy vấn trong dataset và lưu kết quả ra file JSON.

**Quy trình:**
1. Đọc annotation file (dict `{video_id: {sentences: [...], ...}}`).
2. Với mỗi câu truy vấn, gọi `bot.ask(query)` → nhận JSON phân tích gồm sub-queries + relationship.
3. Hỗ trợ **resume**: nếu file output đã tồn tại thì bỏ qua các truy vấn đã xử lý.
4. Lưu kết quả vào file JSON sau mỗi video (checkpoint định kỳ).

**Cách chạy:**

```bash
python get_llm_outputs.py \
    --api_key YOUR_API_KEY \
    --input_file dataset/charades-sta/charades_test.json \
    --output_file dataset/charades-sta/llm_outputs.json \
    --model_type OpenAI \
    --model_name gpt-4-turbo
```

---

### 3.6 `chat_bots/`

Module này chứa các lớp giao tiếp với LLM, được thiết kế theo mẫu **Strategy pattern**.

#### `base_bot.py` — Lớp cơ sở `ChatBot`

```python
class ChatBot:
    def __init__(self, api_key, max_retry=3):
        self.prompt = prompts.v3   # Dùng prompt phiên bản v3 mặc định

    def ask(self, query) -> (response_json, raw):
        # Thử gọi API tối đa max_retry lần
        # Parse kết quả bằng json5.loads() (linh hoạt hơn json.loads)
```

#### `__init__.py` — Factory `get_chat_model()`

```python
def get_chat_model(model_type, model_name, api_key):
    # model_type: 'OpenAI' | 'Google' | 'Groq'
    # Trả về instance của lớp chat bot tương ứng
```

#### Các lớp chat bot

| File | Lớp | Model mặc định | API |
|---|---|---|---|
| `chat_with_gpt.py` | `GPTChatBot` | `gpt-4-turbo` | OpenAI Chat Completions |
| `chat_with_gemini.py` | `GeminiChatBot` | `gemini-pro` | Google GenerativeAI |
| `chat_with_groq.py` | `GroqChatBot` | `llama3-70b-8192` | Groq Chat Completions |

#### `prompts.py` — Ba phiên bản prompt

| Phiên bản | Mô tả |
|---|---|
| `v1` | Prompt agent đa bước — LLM có thể gọi "Temporal Localizer API" rồi ra quyết định (framework agent) |
| `v2` | Prompt đơn giản hơn — LLM chỉ phân tích query thành sub-queries + relationship JSON |
| `v3` | Phiên bản hiện tại được dùng — Giống v2 nhưng thêm `sub_query_id=0` để rewrite câu truy vấn gốc |

**Cấu trúc JSON output của LLM (prompt v3):**

```json
{
    "reasoning": "...",
    "sub-queries": "1. ... 2. ...",
    "relationship": "simultaneously | sequentially | single-query",
    "query_json": [
        {
            "sub_query_id": 0,
            "descriptions": ["Rewritten original query", "..."]
        },
        {
            "sub_query_id": 1,
            "descriptions": ["Sub-query 1 description 1", "..."]
        }
    ]
}
```

---

### 3.7 `evaluate.py`

Script đánh giá chính, hỗ trợ 3 chế độ:

#### Chế độ 1: `eval()` — Chỉ dùng VLM (không có LLM)

```
Annotation file → query_json →  localize() → IoU với ground truth
```

- Hỗ trợ **OOD testing** bằng cách pad noise ngẫu nhiên vào đầu video (`pad_sec`).
- Metric: `mIoU`, `R@0.3`, `R@0.5`, `R@0.7`.

#### Chế độ 2: `eval_with_llm()` — Dùng cả VLM + LLM

```
LLM output JSON → với mỗi query:
  1. Với mỗi sub-query (sub_query_id >= 1): localize() → top-3 proposals
  2. filter_and_integrate(sub_proposals, relation) → top-2 combined proposals
  3. localize() với query gốc (sub_query_id = 0) → top-7 proposals
  4. Ghép proposals → select_proposal() → chọn proposal tốt nhất
```

Điểm mấu chốt: proposals từ **query gốc** và **tổ hợp sub-query** được kết hợp rồi re-rank bằng `select_proposal`.

#### Chế độ 3: `eval_qvhighlight()` — Dataset QVHighlight

- Format annotation khác (`.jsonl`, mỗi dòng là 1 query riêng lẻ).
- Trả về `pred_relevant_windows` (top-7 proposals).
- Gọi `eval_submission()` từ `qvhhighlight_eval.py` để tính các metric đặc thù của QVHighlight (MR/HD).

#### Hàm `calc_iou(candidates, gt)`

Tính IoU giữa tập các proposals `[N, 2]` và một ground truth `[start, end]`:

$$\text{IoU} = \frac{\min(\text{end}, e) - \max(\text{start}, s)}{\max(\text{end}, e) - \min(\text{start}, s)}$$

---

### 3.8 `qvhhighlight_eval.py`

Module đánh giá chuyên biệt cho **QVHighlight** (hỗ trợ cả Moment Retrieval và Highlight Detection). Tính các metric chuẩn của dataset này: `MR-IoU@0.5`, `MR-IoU@0.7`, `MR-mAP`, `HD-mAP`.

---

## 4. Luồng thực thi (Pipeline)

```
                    ┌─────────────────────────────────────────────────────┐
                    │                VIDEO DATABASE                        │
                    └───────────────────────┬─────────────────────────────┘
                                            │
                                  feature_extraction.py
                                  (BLIP-2, 3fps, .npy)
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────────┐
                    │              VISUAL FEATURES [T, 32, D]             │
                    └───────────────────────┬─────────────────────────────┘
                                            │
           ┌────────────────────────────────┼──────────────────────────────┐
           │                                │                              │
   [có LLM output]                          │                    [QVHighlight]
           │                                │
   get_llm_outputs.py             [chỉ VLM]
   (GPT/Gemini/LLaMA)                       │
           │                                │
           ▼                                ▼
   LLM JSON output            evaluate.py::eval()
   (sub-queries,                            │
    relationship)                           │
           │                                │
           ▼                                │
   evaluate.py::eval_with_llm()             │
           │                                │
           ├─ sub-queries ──→ vlm_localizer.py::localize()
           │                       ↓
           │               generate_proposal()
           │                  ├── calc_scores()      (BLIP-2 text-frame similarity)
           │                  ├── get_dynamic_scores() (Gaussian diff → dynamic start)
           │                  └── NMS
           │
           ├─ filter_and_integrate()  (sequentially / simultaneously)
           │
           └─ select_proposal()  (weighted IoU voting)
                    │
                    ▼
             Kết quả: [start, end] (giây)
                    │
                    ▼
             Đánh giá: mIoU, R@0.3, R@0.5, R@0.7
```

---

## 5. Cài đặt và chạy thử

### Yêu cầu

```bash
pip install torch torchvision tqdm salesforce-lavis scikit-learn json5 decord
# Tùy LLM backend:
pip install openai          # GPT
pip install google-generativeai  # Gemini
pip install groq            # Groq/LLaMA
```

### Chuẩn bị dữ liệu

1. Tải pre-extracted features từ [link này](https://disk.pku.edu.cn/link/AA3641EABF29EE483F8AE89E1C149DD496).
2. Cập nhật `feature_path` trong [`data_configs.py`](data_configs.py) cho từng dataset.
3. LLM outputs đã được cung cấp sẵn tại [`dataset/charades-sta/llm_outputs.json`](dataset/charades-sta/llm_outputs.json) và [`dataset/activitynet/llm_outputs.json`](dataset/activitynet/llm_outputs.json).

### Tạo LLM outputs (tùy chọn)

```bash
# Dùng GPT-4
python get_llm_outputs.py \
    --api_key YOUR_OPENAI_KEY \
    --input_file dataset/charades-sta/charades_test.json \
    --output_file dataset/charades-sta/llm_outputs.json \
    --model_type OpenAI

# Dùng Gemini
python get_llm_outputs.py \
    --api_key YOUR_GOOGLE_KEY \
    --model_type Google

# Dùng LLaMA-3 qua Groq
python get_llm_outputs.py \
    --api_key YOUR_GROQ_KEY \
    --model_type Groq
```

---

## 6. Kết quả chính

### Standard Split

```bash
# Charades-STA dataset
python evaluate.py --dataset charades --llm_output dataset/charades-sta/llm_outputs.json

# ActivityNet dataset
python evaluate.py --dataset activitynet --llm_output dataset/activitynet/llm_outputs.json
```

| Dataset        | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----         | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA  |  67.04  |  49.97  |  24.32  |  44.51  |
|  ActivityNet   |  49.34  |  27.02  |  13.39  |  34.10  |


### OOD Splits

```bash
# Charades-STA OOD-1
python evaluate.py --dataset charades --split OOD-1

# Charades-STA OOD-2
python evaluate.py --dataset charades --split OOD-2

# ActivityNet OOD-1
python evaluate.py --dataset activitynet --split OOD-1

# ActivityNet OOD-2
python evaluate.py --dataset activitynet --split OOD-2
```

| Dataset              | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----               | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA OOD-1  |  66.05  |  45.91  |  20.78  |  43.05  |
|  Charades-STA OOD-2  |  65.75  |  43.79  |  19.95  |  42.62  |
|  ActivityNet OOD-1   |  43.87  |  20.41  |  11.25  |  31.72  |
|  ActivityNet OOD-2   |  40.97  |  18.54  |  10.03  |  30.33  |


```bash
# Charades-CD test-ood
python evaluate.py --dataset charades --split test-ood

# Charades-CG novel-composition
python evaluate.py --dataset charades --split novel-composition

# Charades-CG novel-word
python evaluate.py --dataset charades --split novel-word
```

| Dataset                           | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----                            | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA test-ood            |  65.07  |  49.24  |  23.05  |  44.01  |
|  Charades-STA novel-composition   |  61.53  |  43.84  |  18.68  |  40.19  |
|  Charades-STA novel-word          |  68.49  |  56.26  |  28.49  |  46.90  |

---

## 7. Kiểm thử trên dataset tùy chỉnh

### Bước 1: Trích xuất đặc trưng

```bash
python feature_extraction.py --input_root VIDEO_PATH --save_root FEATURE_SAVE_PATH
```

### Bước 2: Cấu hình dataset

Thêm dataset của bạn vào [`data_configs.py`](data_configs.py). Định dạng annotation tham khảo [`dataset/charades-sta/test_trivial.json`](dataset/charades-sta/test_trivial.json).

Điều chỉnh `stride` và `max_stride_factor` cho phù hợp với độ dài video của bạn.

### Bước 3: Chạy đánh giá (chỉ VLM)

```bash
python evaluate.py --dataset YOUR_DATASET
```

### Bước 4: Chạy đánh giá (VLM + LLM)

```bash
python get_llm_outputs.py --api_key YOUR_KEY --input_file YOUR_ANNOTATION --output_file YOUR_LLM_OUTPUT
python evaluate.py --dataset YOUR_DATASET --llm_output YOUR_LLM_OUTPUT
```

## Quick Start

### Requiments
- pytorch
- torchvision
- tqdm
- salesforce-lavis
- sklearn
- json5

### Data Preparation

To reproduce the results in the paper, we provide the pre-extracted features of the VLM in [this link](https://disk.pku.edu.cn/link/AA3641EABF29EE483F8AE89E1C149DD496) and the outputs of the LLM in [`dataset/charades-sta/llm_outputs.json`](dataset/charades-sta/llm_outputs.json) and [`dataset/activitynet/llm_outputs.json`](dataset/activitynet/llm_outputs.json). Please download the pre-extracted features and configure the path for these features in [`data_configs.py`](data_configs.py) file.

## Main Results

### Standard Split

```bash
# Charades-STA dataset
python evaluate.py --dataset charades --llm_output dataset/charades-sta/llm_outputs.json

# ActivityNet dataset
python evaluate.py --dataset activitynet --llm_output dataset/activitynet/llm_outputs.json
```

| Dataset        | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----         | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA  |  67.04  |  49.97  |  24.32  |  44.51  |
|  ActivityNet   |  49.34  |  27.02  |  13.39  |  34.10  |


### OOD Splits

```bash
# Charades-STA OOD-1
python evaluate.py --dataset charades --split OOD-1

# Charades-STA OOD-2
python evaluate.py --dataset charades --split OOD-2

# ActivityNet OOD-1
python evaluate.py --dataset activitynet --split OOD-1

# ActivityNet OOD-2
python evaluate.py --dataset activitynet --split OOD-2
```

| Dataset              | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----               | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA OOD-1  |  66.05  |  45.91  |  20.78  |  43.05  |
|  Charades-STA OOD-2  |  65.75  |  43.79  |  19.95  |  42.62  |
|  ActivityNet OOD-1   |  43.87  |  20.41  |  11.25  |  31.72  |
|  ActivityNet OOD-2   |  40.97  |  18.54  |  10.03  |  30.33  |


```bash
# Charades-CD test-ood
python evaluate.py --dataset charades --split test-ood

# Charades-CG novel-composition
python evaluate.py --dataset charades --split novel-composition

# Charades-CG novel-word
python evaluate.py --dataset charades --split novel-word
```

| Dataset                           | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----                            | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA test-ood            |  65.07  |  49.24  |  23.05  |  44.01  |
|  Charades-STA novel-composition   |  61.53  |  43.84  |  18.68  |  40.19  |
|  Charades-STA novel-word          |  68.49  |  56.26  |  28.49  |  46.90  |

## Test on Custom Datasets

### Feature Extraction

Please run `feature_extraction.py` to obtain the video features of your datasets.

```bash
python feature_extraction.py --input_root VIDEO_PATH --save_root FEATURE_SAVE_PATH
```

### Data Configuration

Please add your dataset in the `data_configs.py`. You may need to adjust the stride and max_stride_factor to achieve better performance.

The format of the annotation file can refer to `dataset/charades-sta/test_trivial.json`.

### Test without LLM

To test the performance with only VLM, please run:

```bash
python evaluate.py --dataset DATASET --split SPLIT
```

`DATASET` and `SPLIT` are the dataset name and split that you add in the `data_configs.py`.

### Test with LLM

To obtain the outputs of LLM, please run:

```bash
python get_llm_outputs.py --api_key API_KEY --input_file ANNOTATION_FILE --output_file LLM_OUTPUT_FILE
```

We have implemented models from OpenAI, Google, and Groq. You can specify the model using `--model_type` and select a specific model with `--model_name`. You will need to apply for the corresponding model's API key and install the necessary dependencies, such as `openai`, `google-generativeai`, or `groq`.

To test the performance, please run:

```bash
python evaluate.py --dataset DATASET --split SPLIT --llm_output LLM_OUTPUT_FILE
```