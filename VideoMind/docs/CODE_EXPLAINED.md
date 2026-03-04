# VideoMind — Giải thích chi tiết toàn bộ Source Code

> **Paper**: [VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning](https://arxiv.org/abs/2503.13444) (ICLR 2026)

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Constants & Prompts (`constants.py`)](#3-constants--prompts)
4. [Model chính (`model/model.py`)](#4-model-chính)
   - 4.1 [Kiến trúc class hierarchy](#41-kiến-trúc-class-hierarchy)
   - 4.2 [Khởi tạo Detection Head](#42-khởi-tạo-detection-head)
   - 4.3 [Forward pass — 3 chế độ](#43-forward-pass--3-chế-độ)
5. [Building Blocks (`model/blocks.py`)](#5-building-blocks)
6. [Point Generator (`model/generator.py`)](#6-point-generator)
7. [Loss Functions (`model/loss.py`)](#7-loss-functions)
8. [Model Builder (`model/builder.py`)](#8-model-builder)
9. [Data Pipeline (`dataset/`)](#9-data-pipeline)
   - 9.1 [Video Processing (`dataset/utils.py`)](#91-video-processing)
   - 9.2 [Dataset Wrappers](#92-dataset-wrappers)
   - 9.3 [Hybrid Dataset & Collator](#93-hybrid-dataset--collator)
10. [Training Pipeline (`train/`)](#10-training-pipeline)
11. [Inference Pipeline (`eval/infer_auto.py`)](#11-inference-pipeline)
12. [Luồng dữ liệu end-to-end (ví dụ cụ thể)](#12-luồng-dữ-liệu-end-to-end)
13. [Tóm tắt kiến trúc](#13-tóm-tắt-kiến-trúc)

---

## 1. Tổng quan hệ thống

VideoMind là một **agentic multi-modal framework** cho bài toán **temporal-grounded video reasoning**. Ý tưởng cốt lõi:

- Dùng **một backbone duy nhất** (Qwen2-VL) làm nền
- Gắn **nhiều LoRA adapter** khác nhau, mỗi adapter đảm nhiệm một vai trò (role)
- Khi inference, model **chuyển đổi adapter** tuần tự để thực hiện pipeline → gọi là **Chain-of-LoRA**

**4 vai trò trong hệ thống:**

| Role | Nhiệm vụ | LoRA adapter | Thêm module |
|------|----------|-------------|-------------|
| **Planner** | Phân tích câu hỏi, lập kế hoạch hành động | LoRA #1 | Không |
| **Grounder** | Định vị timestamp trong video | LoRA #2 | Detection Head (CNN) |
| **Verifier** | Xác minh kết quả grounding | LoRA #3 | Không |
| **Answerer** | Trả lời câu hỏi dựa trên đoạn video đã ground | LoRA #4 | Không |

```
┌─────────────────────────────────────────────────────────┐
│                    Qwen2-VL Backbone                     │
│              (Vision Encoder + LLM, frozen)              │
├──────────┬──────────┬──────────┬────────────────────────┤
│ LoRA #1  │ LoRA #2  │ LoRA #3  │ LoRA #4               │
│ Planner  │ Grounder │ Verifier │ Answerer               │
│          │ +CNN Head│          │                         │
└──────────┴──────────┴──────────┴────────────────────────┘
```

---

## 2. Cấu trúc thư mục

```
videomind/
├── constants.py              # Prompt templates, special tokens
├── conversation.py           # Chat template (chatml format)
├── model/
│   ├── __init__.py           # Registry: MODELS dict
│   ├── model.py              # ★ Model chính: AgentQwen2VLForConditionalGeneration
│   ├── blocks.py             # CNN building blocks: ConvPyramid, ConvHead, Scale, ...
│   ├── generator.py          # PointGenerator: sinh anchor points
│   ├── loss.py               # BundleLoss = FocalLoss + L1Loss + SampledNCELoss
│   └── builder.py            # build_model(): load model + adapter
├── dataset/
│   ├── hybrid.py             # HybridDataset: gộp nhiều dataset con
│   ├── collator.py           # HybridDataCollator: batch padding
│   ├── utils.py              # Video reading, chat preprocessing
│   ├── wrappers/
│   │   ├── grounding.py      # GroundingDataset (cho Grounder)
│   │   ├── verifying.py      # VerifyingDataset (cho Verifier)
│   │   ├── answering.py      # AnsweringDataset (cho Answerer)
│   │   └── planning.py       # PlanningDataset (cho Planner)
│   └── sub_classes/          # 27 dataset cụ thể (nextgqa, rextime, ...)
├── train/
│   ├── train.py              # Entry point training
│   └── custom_trainer.py     # CustomTrainer: multi-LR, save logic
├── eval/
│   ├── infer_auto.py         # ★ Inference pipeline (agent loop)
│   ├── eval_auto.py          # Evaluation metrics
│   └── ...
└── utils/
    ├── parser.py             # parse_span, parse_query
    └── io.py                 # get_duration, load_subtitle
```

---

## 3. Constants & Prompts

**File**: `videomind/constants.py`

### 3.1 Special Tokens

```python
IGNORE_INDEX = -100           # Label padding (không tính loss)
REG_TOKEN = '<|reg|>'         # Token trigger detection head
SEG_S_TOKEN = '<|seg_start|>' # Đánh dấu đầu segment (cho Verifier)
SEG_E_TOKEN = '<|seg_end|>'   # Đánh dấu cuối segment (cho Verifier)
```

**`REG_TOKEN`** là token quan trọng nhất — khi LLM sinh ra token này trong quá trình generate, nó **kích hoạt detection head** để dự đoán timestamp. Đây là cầu nối giữa LLM reasoning và temporal detection.

### 3.2 Prompt Templates

Mỗi role có một prompt riêng hướng dẫn LLM đóng đúng vai:

#### PLANNER_PROMPT
```
"You are acting as the planner now..."
→ Hướng dẫn: phân tích câu hỏi, chọn tool (grounder/verifier/answerer)
→ Output mong đợi: JSON array, ví dụ:
  [{"type": "grounder", "value": "<query>"}, {"type": "verifier"}, {"type": "answerer"}]
```

#### GROUNDER_PROMPT
```
"You are acting as the grounder now..."
→ Hướng dẫn: nhận query, suy nghĩ rồi localize moment
→ Output mong đợi: text reasoning + token <|reg|> (trigger detection head)
  Ví dụ: "The relevant moment happens in <|reg|>."
```

#### VERIFIER_PROMPT
```
"You are acting as the verifier now..."
→ Hướng dẫn: kiểm tra đoạn video giữa <|seg_start|> và <|seg_end|> có đúng không
→ Output mong đợi: "Yes" hoặc "No"
```

---

## 4. Model chính

**File**: `videomind/model/model.py`

### 4.1 Kiến trúc class hierarchy

```
Qwen2VLForConditionalGeneration         (HuggingFace gốc)
    └── AgentQwen2VLForConditionalGeneration  (VideoMind custom)
            ├── visual: AgentQwen2VisionTransformerPretrainedModel
            │     └── Thêm gradient checkpointing support
            ├── model: AgentQwen2VLModel
            │     └── Thêm hook để cache hidden states trước LayerNorm cuối
            ├── lm_head: Linear (sinh text tokens)
            │
            └── [Nếu role == grounder] Detection Head:
                  ├── vis_proj, reg_proj        (projection 3584→256)
                  ├── vis_emb, reg_emb          (type embeddings)
                  ├── vis_fuse                  (3-layer Transformer Encoder)
                  ├── vis_pos                   (Positional Encoding)
                  ├── vis_norm                  (LayerNorm)
                  ├── pyramid (ConvPyramid)     (FPN 1D: strides 1,2,4,8)
                  ├── class_head (ConvHead)     (predict foreground score)
                  ├── coord_head (ConvHead)     (predict offset start/end)
                  ├── generator (PointGenerator) (sinh anchor points)
                  ├── coef (Scale)              (learnable scale per level)
                  └── bundle_loss (BundleLoss)  (training loss)
```

### 4.2 Khởi tạo Detection Head

Chỉ được tạo khi `config.role in ('all_in_one', 'grounder')`:

```python
# Projection: giảm chiều từ LLM hidden size xuống 256
self.vis_proj = Sequential(LayerNorm(3584), Linear(3584, 256))  # cho visual tokens
self.reg_proj = Sequential(LayerNorm(3584), Linear(3584, 256))  # cho reg token

# Type embeddings: phân biệt visual vs query
self.vis_emb = LearnableEmbedding(256)
self.reg_emb = LearnableEmbedding(256)

# Fusion: 3-layer Transformer Encoder, d_model=256
self.vis_fuse = ModuleList(
    TransformerEncoderLayer(256),
    TransformerEncoderLayer(256),
    TransformerEncoderLayer(256))

# Feature Pyramid Network 1D
self.pyramid = ConvPyramid(dims=256, strides=(1, 2, 4, 8))

# Prediction heads
self.class_head = ConvHead(256, out_dims=1)   # foreground score
self.coord_head = ConvHead(256, out_dims=2)   # (dist_to_start, dist_to_end)

# Anchor point generator
self.generator = PointGenerator(strides=(1,2,4,8), buffer_size=1024)

# Learnable scale per pyramid level
self.coef = Scale(strides=(1,2,4,8))

# Training loss
self.bundle_loss = BundleLoss(
    loss_cls=FocalLoss(weight=5.0),
    loss_reg=L1Loss(weight=1.0),
    loss_sal=SampledNCELoss(weight=0.05))
```

### 4.3 Forward pass — 3 chế độ

Forward pass có 3 chế độ hoạt động tùy ngữ cảnh:

```python
mode = 'training'    if self.training
     = 'caching'     if not training AND past_key_values is empty (first forward)
     = 'generating'  if not training AND past_key_values exists (auto-regressive)
```

#### Bước chung: Xác định vị trí visual tokens

```python
# Tìm vị trí visual tokens trong input sequence
vision_s_inds = nonzero(input_ids == vision_start_token_id)
vision_e_inds = nonzero(input_ids == vision_end_token_id)
# → cache_vision_inds[batch][video] = [start_idx, end_idx]
```

Input sequence trông như thế này:
```
[system_tokens..., <|vision_start|>, vis_token_1, vis_token_2, ..., vis_token_T, <|vision_end|>, text_tokens...]
                   ↑ start                                                        ↑ end
```

#### Bước chung: Forward qua backbone

```python
outputs = super().forward(input_ids, pixel_values_videos, ...)
# → outputs.loss = language_loss (cross-entropy cho auto-regressive LM)
# → outputs.logits = [batch, seq_len, vocab_size]
```

#### Chế độ `training` — Khi có timestamps (dữ liệu grounding)

**Mục đích**: Tính detection loss bổ sung và cộng vào language loss.

```python
# 1. Tìm vị trí <|reg|> token trong labels
inds = where(shift_labels[batch_idx] == reg_token_id)

# 2. Lấy reg_tokens = hidden state tại vị trí <|reg|>
reg_tokens = self.reg_proj(norm.state[batch_idx, inds])
# shape: [num_reg_tokens, 1, 256]

# 3. Lấy vis_tokens = hidden states của visual tokens
vis_tokens = norm.state[batch_idx, start:end]
# shape: [1, T_total_visual_tokens, 3584]

# 4. Average pooling: gộp spatial tokens thành temporal tokens
#    Qwen2-VL dùng spatial_merge_size=2 → mỗi frame có (H/2)*(W/2)/4 tokens
#    avg_pool gộp window tokens → 1 token per frame
vis_tokens = avg_pool1d(vis_tokens, kernel=window, stride=window)
# shape: [1, num_frames, 3584]

# 5. Project xuống 256 chiều
vis_tokens = self.vis_proj(vis_tokens)
# shape: [1, num_frames, 256]

# 6. Lặp lại cho mỗi reg token (nếu có nhiều <|reg|> trong 1 sample)
vis_tokens = vis_tokens.repeat(num_reg_tokens, 1, 1)
# shape: [num_reg_tokens, num_frames, 256]

# 7. Thêm type embedding + positional encoding
vis_tokens = self.vis_emb(vis_tokens)  # + learnable type vector
reg_tokens = self.reg_emb(reg_tokens)  # + learnable type vector
pe = self.vis_pos(vis_tokens)          # sinusoidal positional encoding

# 8. Concat rồi fuse qua 3-layer Transformer
joint = cat([vis_tokens + pe, reg_tokens], dim=1)  # [N, num_frames+1, 256]
# vis_fuse: 3 layers, output từ MỖI layer được concat lại
collected = [joint]
for blk in self.vis_fuse:
    collected.append(blk(collected[-1]))
joint = cat(collected[1:])    # concat output 3 layers
joint = self.vis_norm(joint)  # normalize

# 9. Tách lại
video_emb = joint[:, :-1]    # [N, num_frames, 256] — query-aware visual
query_emb = joint[:, -1:]    # [N, 1, 256]          — visual-aware query

# 10. ConvPyramid → multi-scale features
pymid = self.pyramid(video_emb, mask)
# pymid = [
#   [N, num_frames,   256],    stride 1
#   [N, num_frames/2, 256],    stride 2
#   [N, num_frames/4, 256],    stride 4
#   [N, num_frames/8, 256],    stride 8
# ]

# 11. Generate anchor points
point = self.generator(pymid)  # [total_points, 4]
# Mỗi point: [center_position, reg_range_min, reg_range_max, stride]

# 12. Prediction per level
out_class = cat([self.class_head(e) for e in pymid], dim=1)  # [N, total_points, 1]
out_coord = cat([self.coef(self.coord_head(e).exp(), i) for i, e in enumerate(pymid)], dim=1)
# out_coord: [N, total_points, 2] — (dist_start, dist_end)

# 13. Tính detection loss
losses = self.bundle_loss(data)
# losses = {'loss_cls': ..., 'loss_reg': ..., 'loss_sal': ...}

# 14. Cộng vào language loss
outputs.loss = outputs.loss + sum(losses.values()) / avg_factor
```

**Ý nghĩa**: Training loss = Language Loss (LLM học sinh text) + Detection Loss (detection head học dự đoán timestamp).

#### Chế độ `caching` — First forward pass khi inference

```python
# Cache lại hidden states để dùng cho generating sau này
self.cache_norm_state = self.model.norm.state  # [1, seq_len, 3584]
self.reg = []   # sẽ chứa predicted proposals
self.sal = []   # sẽ chứa saliency scores
```

#### Chế độ `generating` — Auto-regressive generation

Khi LLM sinh ra `<|reg|>` token (logits.argmax() == reg_token_id):

```python
# 1-12. Giống training nhưng dùng cache_norm_state
# ...

# 13. Decode proposals thay vì tính loss
sal = sigmoid(out_class)  # [1, total_points, 1]
bnd = out_coord           # [1, total_points, 2]

# Convert offset → absolute timestamp
bnd[:, 0] *= -1                          # dist_start là khoảng cách ngược
bnd *= point[:, 3]                       # × stride → absolute frame offset
bnd += point[:, 0]                       # + center → absolute frame position
bnd /= num_frames                        # normalize → [0, 1]
bnd = cat([bnd, sal], dim=-1)            # [total_points, 3] = (start, end, score)

# 14. NMS (Non-Maximum Suppression)
# Sắp xếp theo score giảm dần
# Với mỗi proposal: suppress các proposals có IoU >= 0.75
for i in range(bnd.size(0)):
    iou = temporal_iou(bnd[i, :-1], bnd[i+1:, :-1])
    bnd[i+1:, -1][iou >= 0.75] = 0  # zero out suppressed scores

# 15. Lưu top-100 proposals
self.reg.append(bnd[:100])   # [(start, end, score), ...]
self.sal.append(sal)
```

**Output cuối cùng**: `model.reg[0]` chứa top-100 proposals dạng `[start_normalized, end_normalized, confidence_score]`.

---

## 5. Building Blocks

**File**: `videomind/model/blocks.py`

### 5.1 `Permute`

```python
class Permute(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)  # [B, T, C] ↔ [B, C, T]
```

Dùng để chuyển đổi giữa format `(batch, time, channel)` (cho Transformer) và `(batch, channel, time)` (cho Conv1d).

### 5.2 `LearnableEmbedding`

```python
class LearnableEmbedding(nn.Module):
    def __init__(self, dims):
        self.weights = Parameter(1, 1, dims)   # learnable vector [1, 1, 256]
    
    def forward(self, x):
        return x + self.weights.expand_as(x)   # broadcast cộng vào mọi token
```

- **Input**: `[batch, seq_len, 256]`
- **Output**: `[batch, seq_len, 256]` (cộng thêm 1 learnable bias)
- **Mục đích**: Đánh dấu loại token (visual vs. query) — tương tự token type embedding trong BERT.

### 5.3 `ConvPyramid` — Feature Pyramid Network 1D

```python
class ConvPyramid(nn.Module):
    def __init__(self, dims=256, strides=[1, 2, 4, 8]):
        for s in strides:
            p = log2(s)  # số lần downsample
            if p == 0:
                block = ReLU()             # stride 1: chỉ activation
            else:
                block = Sequential()
                for _ in range(p):         # lặp p lần Conv1d stride=2
                    block.extend([
                        Permute(),         # [B,T,C] → [B,C,T]
                        Conv1d(256, 256, kernel_size=2, stride=2),
                        Permute(),         # [B,C,T/2] → [B,T/2,C]
                        LayerNorm(256),
                        ReLU()
                    ])
```

**Hoạt động** (ví dụ input có 96 frames):

```
Input: [B, 96, 256]
       │
       ├── Stride 1: ReLU only
       │   → [B, 96, 256]     (96 anchor points, mỗi point "nhìn" 1 frame)
       │
       ├── Stride 2: Conv1d(k=2,s=2) × 1 lần
       │   → [B, 48, 256]     (48 points, mỗi point "nhìn" 2 frames)
       │
       ├── Stride 4: Conv1d(k=2,s=2) × 2 lần
       │   → [B, 24, 256]     (24 points, mỗi point "nhìn" 4 frames)
       │
       └── Stride 8: Conv1d(k=2,s=2) × 3 lần
           → [B, 12, 256]     (12 points, mỗi point "nhìn" 8 frames)
       
       Tổng: 96 + 48 + 24 + 12 = 180 anchor points
```

**Mục đích**: Multi-scale temporal features — stride nhỏ phát hiện sự kiện ngắn, stride lớn phát hiện sự kiện dài.

**Forward cũng tạo mask tương ứng** (dùng `max_pool1d` để downsample mask):
```python
if return_mask:
    if s > 1:
        msk = F.max_pool1d(mask.float(), s, stride=s).long()
    else:
        msk = mask
```

### 5.4 `ConvHead` — Prediction Head

```python
class ConvHead(nn.Module):
    def __init__(self, dims=256, out_dims=1, kernel_size=3):
        self.module = Sequential(
            Permute(),                                          # [B,T,256] → [B,256,T]
            Conv1d(256, 256, kernel_size=3, padding=1),         # → [B,256,T]
            ReLU(),
            Conv1d(256, out_dims, kernel_size=3, padding=1),    # → [B,out,T]
            Permute()                                           # → [B,T,out]
        )
```

Hai instance:
- `class_head(out_dims=1)`: Dự đoán foreground score cho mỗi anchor point
- `coord_head(out_dims=2)`: Dự đoán `(dist_to_start, dist_to_end)` offset

### 5.5 `Scale` — Learnable Scale per Level

```python
class Scale(nn.Module):
    def __init__(self, strides):
        self.scale = nn.Parameter(torch.ones(len(strides)))  # [4] learnable
    
    def forward(self, x, i):
        return x * self.scale[i]  # nhân với scale của level i
```

**Mục đích**: Mỗi pyramid level có một hệ số scale riêng (learnable), giúp model tự điều chỉnh magnitude prediction cho từng level.

---

## 6. Point Generator

**File**: `videomind/model/generator.py`

### 6.1 Cách hoạt động

`PointGenerator` sinh ra **anchor points** cho temporal detection theo kiểu **FCOS** (Fully Convolutional One-Stage Detection).

```python
class PointGenerator(nn.Module):
    def __init__(self, strides=(1,2,4,8), buffer_size=1024, offset=False):
        # Tính regression range cho mỗi level
        # strides = [1, 2, 4, 8]
        # → reg_range = [(0, 2), (2, 4), (4, 8), (8, inf)]
        
        # Pre-cache points cho mỗi level (tránh tính lại mỗi lần forward)
        for stride, reg_range in zip(strides, reg_ranges):
            points = arange(0, buffer_size, stride)  # [0, 1, 2, ...] hoặc [0, 2, 4, ...]
            # Mỗi point = [center, reg_lo, reg_hi, stride]
            buffer.append(cat([points, reg_range, stride], dim=1))
```

### 6.2 Output format

Mỗi anchor point là vector 4 chiều: `[center, reg_range_min, reg_range_max, stride]`

```
Level 0 (stride=1): points = [0, 1, 2, ..., 95]
  → Mỗi point chỉ predict đoạn trong range [0, 2] frames

Level 1 (stride=2): points = [0, 2, 4, ..., 94]
  → Mỗi point predict đoạn trong range [2, 4] frames

Level 2 (stride=4): points = [0, 4, 8, ..., 92]
  → Mỗi point predict đoạn trong range [4, 8] frames

Level 3 (stride=8): points = [0, 8, 16, ..., 88]
  → Mỗi point predict đoạn trong range [8, ∞] frames
```

### 6.3 Cách decode proposal từ anchor point

```
Point:  center = 50, stride = 4
Head output: dist_start = 3.0, dist_end = 5.0

start = center - dist_start × stride = 50 - 3.0 × 4 = 38
end   = center + dist_end × stride   = 50 + 5.0 × 4 = 70

→ Proposal: frames [38, 70]
→ Normalized: [38/96, 70/96] = [0.396, 0.729]
→ Timestamps: [0.396 × duration, 0.729 × duration]
```

---

## 7. Loss Functions

**File**: `videomind/model/loss.py`

### 7.1 `BundleLoss` — Tổng hợp 3 loss

```python
class BundleLoss(nn.Module):
    def __init__(self, sample_radius=1.5):
        self._loss_cls = FocalLoss(loss_weight=5.0)     # Classification
        self._loss_reg = L1Loss(loss_weight=1.0)        # Regression
        self._loss_sal = SampledNCELoss(loss_weight=0.05)  # Saliency
```

#### 7.1.1 Target Assignment (`get_target_single`)

Trước khi tính loss, phải gán **mỗi anchor point** vào **ground truth segment** nào:

```python
def get_target_single(self, point, gt_bnd, gt_cls):
    # Với mỗi anchor point tại vị trí c:
    #   dist_start = c - gt_start
    #   dist_end   = gt_end - c
    
    # 1. Classification target:
    #    positive nếu point nằm trong vùng center ± sample_radius × stride
    #    của một GT segment
    if sample_radius > 0:
        center = (gt_start + gt_end) / 2
        t_min = center - stride × 1.5
        t_max = center + stride × 1.5
        positive = (c >= t_min) AND (c <= t_max)
    
    # 2. Regression range check:
    #    chỉ assign nếu max(dist_start, dist_end) nằm trong reg_range 
    #    của level tương ứng
    reg_msk = (max_dist >= reg_range_min) AND (max_dist <= reg_range_max)
    
    # 3. Nếu 1 point match nhiều GT → chọn GT ngắn nhất
    #    (ưu tiên GT cụ thể hơn)
```

**Ví dụ**: GT segment = [30, 70] trên video 96 frames
- Anchor tại center=50, stride=4: `dist_s=20, dist_e=20` → max=20 → trong range [4,8]? **Không** (20 > 8, nhưng stride 8 range [8,∞] thì **Có**)
- Anchor tại center=50, stride=8: `dist_s=20, dist_e=20` → max=20 → trong range [8,∞]? **Có** → positive

#### 7.1.2 `FocalLoss` (Classification, weight=5.0)

```
L_cls = -α × (1-p)^γ × log(p)     khi positive
      = -(1-α) × p^γ × log(1-p)   khi negative
```
- `α=0.25`, `γ=2.0` (default)
- **Mục đích**: Giải quyết extreme imbalance — đa số anchor points là background

#### 7.1.3 `L1Loss` (Regression, weight=1.0)

```
L_reg = |predicted_offset - gt_offset|
```
- Chỉ tính trên **positive anchors** (có GT gán)
- GT offset được normalize bằng stride: `gt_offset / stride`

#### 7.1.4 `SampledNCELoss` (Saliency/Contrastive, weight=0.05)

```python
class SampledNCELoss(nn.Module):
    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        # Tính cosine similarity giữa mỗi frame và query
        sim = cosine_similarity(video_emb, query_emb) × temperature_scale
        
        # Mask out frames có saliency cao hơn positive clip
        # (chỉ coi là negative nếu saliency thấp hơn)
        loss_msk = (saliency <= saliency[pos_clip]) × video_msk
        
        # NCE loss: kéo gần frame positive, đẩy xa frames negative
        loss = -log_softmax(sim)[pos_clip]
```

- `pos_clip`: frame index ngẫu nhiên trong GT segment
- **Mục đích**: Contrastive learning — visual embedding tại frame trong GT phải giống query embedding hơn các frame ngoài GT

### 7.2 Tổng loss khi training

```python
total_loss = language_loss + (loss_cls + loss_reg + loss_sal) / num_gt_segments
```

Nếu có 2 bộ predictions (từ 2 `<|reg|>` tokens), chọn bộ có loss **thấp hơn** (heuristic để stabilize training).

---

## 8. Model Builder

**File**: `videomind/model/builder.py`

### 8.1 `build_model(model_path, ...)`

Hai trường hợp load:

#### Trường hợp 1: Có adapter directory

```python
# 1. Load base model (Qwen2-VL gốc)
model = AutoModel.from_pretrained(config.base_model_path, ...)

# 2. Resize embeddings (vì thêm special tokens)
if embed_tokens.size != expected:
    model.model.embed_tokens.weight = new_empty(size)
if lm_head.size != expected:
    model.lm_head.weight = new_empty(size)

# 3. Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path, adapter_name=config.role)

# 4. Load detection head weights (nếu có)
if exists('pytorch_model.safetensors'):
    load_model(model, 'pytorch_model.safetensors', strict=False)

# 5. (Optional) Merge adapter vào base model
if merge_adapter:
    model = model.merge_and_unload()
```

#### Trường hợp 2: Full model

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, ...)
```

---

## 9. Data Pipeline

### 9.1 Video Processing

**File**: `videomind/dataset/utils.py`

#### Đọc video (`_read_video_decord`)

```python
def _read_video_decord(ele):
    # 1. Mở video bằng decord
    vr = VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    
    # 2. Xử lý video_start / video_end (crop temporal)
    s_frame = round(video_start × video_fps)
    e_frame = round(video_end × video_fps)
    
    # 3. Tính số frames cần lấy
    nframes = smart_nframes(config, total_frames, video_fps)
    # Dựa trên config fps, min_frames, max_frames
    
    # 4. Uniform sampling
    idx = linspace(s_frame, e_frame, nframes).round().long()
    
    # 5. Đọc frames
    video = vr.get_batch(idx)  # [T, H, W, C]
    video = video.permute(0, 3, 1, 2)  # → [T, C, H, W]
    
    return video
```

#### Resize video (`fetch_video`)

```python
def fetch_video(ele):
    video = _read_video_decord(ele)
    
    # Smart resize: giữ aspect ratio, nằm trong [min_pixels, max_pixels]
    resized_h, resized_w = smart_resize(H, W, min_pixels, max_pixels)
    video = resize(video, [resized_h, resized_w])
    
    return video  # [T, C, H, W] float tensor
```

#### Preprocessing labels (`preprocess_chatml`)

```python
def preprocess_chatml(input_ids, text, tokenizer):
    # Tách text thành các round (user + assistant)
    # Mask phần user (IGNORE_INDEX) → chỉ tính loss trên phần assistant
    
    labels = input_ids.clone()
    for round in rounds:
        instruction_len = tokenizer(instruction_part).length
        labels[cur:cur + instruction_len] = IGNORE_INDEX  # mask instruction
        cur += round_len
```

### 9.2 Dataset Wrappers

Mỗi role có wrapper riêng tạo format phù hợp:

#### `GroundingDataset` (cho Grounder)

```python
def __getitem__(self, idx):
    messages = [
        {'role': 'user', 'content': [
            {'type': 'video', 'video': video_path, 'max_frames': 150, 'fps': 1.0, ...},
            {'type': 'text', 'text': GROUNDER_PROMPT.format(query)}
        ]},
        {'role': 'assistant', 'content': f'The relevant moment happens in {REG_TOKEN}.'}
    ]
    meta = dict(messages=messages, span=span, duration=duration)
    return meta
```

- **Input**: Video (150 frames max, 1 fps) + query
- **Target**: Text với `<|reg|>` token + temporal span GT
- **Đặc biệt**: `span` và `duration` được truyền qua meta để tính detection loss

#### `VerifyingDataset` (cho Verifier)

```python
def __getitem__(self, idx):
    # Crop video quanh predicted span (mở rộng context 50%)
    s0, e0 = parse_span(anno['pred'], duration, min_len=2)
    offset = (e0 - s0) / 2
    s1, e1 = parse_span([s0 - offset, e0 + offset], duration)
    
    # Tính vị trí tương đối để chèn SEG tokens
    s = (s0 - s1) / (e1 - s1)  # vị trí start trong cropped video
    e = (e0 - s1) / (e1 - s1)  # vị trí end trong cropped video
    
    messages = [
        {'role': 'user', 'content': [
            {'type': 'video', 'video': video_path, 'video_start': s1, 'video_end': e1, ...},
            {'type': 'text', 'text': VERIFIER_PROMPT.format(query)}
        ]},
        {'role': 'assistant', 'content': 'Yes.' if positive else 'No.'}
    ]
    meta = dict(messages=messages, ss=s, se=e)
    return meta
```

- **Input**: Video crop + query + segment markers
- **Target**: "Yes." hoặc "No."
- **Đặc biệt**: `ss`, `se` dùng để chèn `<|seg_start|>`, `<|seg_end|>` tokens vào visual sequence

#### `AnsweringDataset` / `AnsweringCropDataset` (cho Answerer)

```python
# AnsweringDataset: dùng toàn bộ video
messages = [
    {'role': 'user', 'content': [
        {'type': 'video', 'video': video_path, 'max_frames': 32, 'fps': 2.0, ...},
        {'type': 'text', 'text': question}
    ]},
    {'role': 'assistant', 'content': answer}
]

# AnsweringCropDataset: crop video tại GT span + temporal jittering
s, e = parse_span(span, duration, min_len=16)
offset = (e - s) / 4
s = random.uniform(s - offset, s + offset)  # jitter start
e = random.uniform(e - offset, e + offset)  # jitter end
# → video crop tại [s, e]
```

- **AnsweringCropDataset** dùng **temporal jittering** (data augmentation): dịch khung hình ngẫu nhiên ±25% để model robust hơn

#### `PlanningDataset` (cho Planner)

```python
def __getitem__(self, idx):
    if route == 1:  # cần rephrasing + grounding + answering
        response = '[{"type": "grounder", "value": "<rephrased_query>"}, {"type": "verifier"}, {"type": "answerer"}]'
    elif route == 4:  # chỉ cần answering
        response = '[{"type": "answerer"}]'
    
    messages = [
        {'role': 'user', 'content': [
            {'type': 'video', ...},
            {'type': 'text', 'text': PLANNER_PROMPT.format(question)}
        ]},
        {'role': 'assistant', 'content': response}
    ]
```

### 9.3 Hybrid Dataset & Collator

#### `HybridDataset` (`dataset/hybrid.py`)

Gộp nhiều dataset con thành 1 dataset:

```python
class HybridDataset(Dataset):
    def __init__(self, processor, model_config, ...):
        # Parse danh sách datasets từ config
        for key in data_args.datasets.split(','):
            datasets.append(DATASETS.get(key)(...))
        
        # Tính index ranges cho mỗi dataset con
        # Ví dụ: [qvhighlights(0-5000), didemo(5000-12000), tacos(12000-18000)]
        
    def fetch_data(self, idx):
        # 1. Xác định dataset con nào chứa idx
        for (s, e), dataset in zip(self.idx_ranges, self.datasets):
            if s <= idx < e:
                meta = dataset[idx - s]
        
        # 2. Apply chat template
        text = processor.apply_chat_template(meta['messages'])
        
        # 3. Process video/image
        images, videos = process_vision_info(meta['messages'])
        data = processor(text=text, images=images, videos=videos)
        
        # 4. Create labels (mask instruction, only compute loss on response)
        data['labels'] = preprocess(data['input_ids'], text, tokenizer)
        
        # 5. Insert SEG tokens (nếu là Verifier data)
        if 'ss' in meta and 'se' in meta:
            # Tính vị trí frame trong visual token sequence
            pos_s = round(ss × num_frames) × window + base_idx + 1
            pos_e = round(se × num_frames) × window + base_idx + 2
            input_ids.insert(pos_s, seg_s_token_id)
            input_ids.insert(pos_e, seg_e_token_id)
        
        # 6. Compute grounding targets (nếu là Grounder data)
        if 'span' in meta:
            timestamps = [[s/duration, e/duration] for s, e in span]
            saliency = zeros(num_frames)
            for s, e in span:
                saliency[ceil(s*fps):ceil(e*fps)] = 1
            pos_clip = random_sample(saliency.nonzero())
```

#### `HybridDataCollator` (`dataset/collator.py`)

Batch nhiều samples lại:

```python
class HybridDataCollator:
    def __call__(self, batch):
        # Pad input_ids và labels tới cùng length
        input_ids = pad_sequence(batch_input_ids, padding_value=pad_token_id)
        labels = pad_sequence(batch_labels, padding_value=IGNORE_INDEX)
        
        # Concat pixel values (không cần pad vì mỗi sample số pixels khác nhau)
        pixel_values_videos = cat([d['pixel_values_videos'] for d in batch])
        
        # Collect grounding targets as lists (không pad)
        timestamps = [d['timestamps'] for d in batch]  # list of lists
        saliency = [d['saliency'] for d in batch]
        pos_clip = [d['pos_clip'] for d in batch]
```

---

## 10. Training Pipeline

**File**: `videomind/train/train.py`, `videomind/train/custom_trainer.py`

### 10.1 Luồng training (`train.py`)

```python
def train():
    # 1. Parse arguments
    model_args, data_args, training_args = parser.parse_args()
    
    # 2. Build model
    config = AgentQwen2VLConfig.from_pretrained(model_path)
    config.role = model_args.role  # 'grounder', 'verifier', ...
    model, processor = build_model(model_path, config, is_trainable=True)
    
    # 3. Apply LoRA
    if lora_enable:
        target_modules = get_target_modules(model, lora_type='qkvo')
        # → ['q_proj', 'k_proj', 'v_proj', 'o_proj'] trong LLM
        
        lora_config = LoraConfig(
            r=64, lora_alpha=64, lora_dropout=0.1,
            target_modules=target_modules,
            modules_to_save=['embed_tokens', 'lm_head']  # nếu grounder/verifier
        )
        model = get_peft_model(model, lora_config, adapter_name=role)
    
    # 4. Add special tokens
    processor.tokenizer.add_special_tokens([REG_TOKEN, SEG_S_TOKEN, SEG_E_TOKEN])
    model.resize_token_embeddings(len(processor.tokenizer))
    # Khởi tạo embedding mới = mean của embeddings cũ
    
    # 5. Set trainable parameters
    # LoRA parameters: trainable (tự động bởi peft)
    # embed_tokens, lm_head: trainable (nếu grounder/verifier — vì thêm token mới)
    # Detection head: trainable (nếu grounder)
    # Tất cả còn lại: frozen
    
    # 6. Create trainer & train
    trainer = CustomTrainer(model, train_dataset=HybridDataset(...))
    trainer.train()
    trainer.gather_and_save_model()
```

### 10.2 LoRA Target Modules

```python
def get_target_modules(model, lora_type='qkvo'):
    if lora_type == 'qkvo':
        # Chỉ apply LoRA vào attention layers của LLM
        keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']
        # Bỏ qua visual encoder (trừ khi lora_type='all')
    elif lora_type == 'linear':
        # Apply LoRA vào TẤT CẢ linear layers của LLM
```

### 10.3 Custom Trainer — Multi Learning Rate

```python
class CustomTrainer(Trainer):
    def create_optimizer(self):
        # 6 parameter groups:
        groups = [
            # Group 1-2: Non-LoRA, non-head params (embed_tokens, lm_head)
            {'params': other_decay_params, 'lr': base_lr, 'weight_decay': 0.01},
            {'params': other_no_decay_params, 'lr': base_lr, 'weight_decay': 0.0},
            
            # Group 3-4: LoRA params (có thể dùng LR riêng)
            {'params': lora_decay_params, 'lr': lora_lr, 'weight_decay': 0.01},
            {'params': lora_no_decay_params, 'lr': lora_lr, 'weight_decay': 0.0},
            
            # Group 5-6: Detection head params (có thể dùng LR riêng, thường cao hơn)
            {'params': head_decay_params, 'lr': head_lr, 'weight_decay': 0.01},
            {'params': head_no_decay_params, 'lr': head_lr, 'weight_decay': 0.0},
        ]
```

**Tại sao cần multi LR?**
- Detection head train from scratch → cần LR cao hơn
- LoRA fine-tune → LR tiêu chuẩn
- `no_decay` group: bias và LayerNorm không áp dụng weight decay

### 10.4 Saving Logic

```python
def gather_and_save_model(self):
    if save_full_model:
        # Merge LoRA → base model → save toàn bộ
        model.merge_and_unload()
        model.save_pretrained(output_dir)
    else:
        # Save LoRA adapter riêng
        model.save_pretrained(output_dir)  # → output_dir/<role>/
        
        # Save detection head weights riêng
        state_dict = {n: p for n, p in model.named_parameters() 
                      if any(k in n for k in head_keys)}
        save_file(state_dict, 'pytorch_model.safetensors')
```

### 10.5 GroupSampler — Batch theo data type

```python
class GroupSampler(Sampler):
    # Nhóm samples cùng data_type vào cùng batch
    # Ví dụ: batch toàn "multimodal" hoặc toàn "video_only"
    # → tránh conflict khi 1 batch có samples khác loại (visual vs text-only)
```

---

## 11. Inference Pipeline

**File**: `videomind/eval/infer_auto.py`

### 11.1 Khởi tạo

```python
# 1. Load grounder model (base + grounder LoRA + detection head)
model, processor = build_model(model_gnd_path)

# 2. Load thêm các LoRA adapter khác (cùng 1 model!)
model.load_adapter(planner_path, adapter_name='planner')    # optional
model.load_adapter(verifier_path, adapter_name='verifier')  # optional
model.load_adapter(answerer_path, adapter_name='answerer')  # optional
```

### 11.2 Agent Loop (cho mỗi video sample)

```
┌────────────────────────────────────────────────────────────────┐
│                     Input: Video + Question                     │
└─────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌─── STEP 1: PLANNER (optional) ────────────────────────────────┐
│                                                                │
│  model.set_adapter('planner')                                  │
│                                                                │
│  Input:                                                        │
│    - Video: max 100 frames, 1 fps, 36-64 patches/frame        │
│    - Text: PLANNER_PROMPT.format(question)                     │
│                                                                │
│  Processing:                                                   │
│    output = model.generate(video + prompt, max_tokens=256)     │
│                                                                │
│  Output:                                                       │
│    JSON: [{"type": "grounder", "value": "rephrased query"},    │
│           {"type": "verifier"},                                │
│           {"type": "answerer"}]                                │
│                                                                │
│  Decisions:                                                    │
│    - Nếu auto_rephrasing: query = action['value']              │
│    - Nếu auto_planning & action='answerer': bỏ qua grounding  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─── STEP 2: GROUNDER ──────────────────────────────────────────┐
│                                                                │
│  model.set_adapter('grounder')                                 │
│                                                                │
│  Input:                                                        │
│    - Video: max 150 frames, 1 fps, 36-64 patches/frame        │
│    - Text: GROUNDER_PROMPT.format(query)                       │
│                                                                │
│  Processing:                                                   │
│    output_ids = model.generate(video + prompt, max_tokens=256) │
│    # LLM sinh text reasoning rồi token <|reg|>                │
│    # Khi sinh <|reg|>: detection head tự động kích hoạt        │
│    # → model.reg[0] chứa top-100 proposals                    │
│                                                                │
│  Post-processing:                                              │
│    blob = model.reg[0]            # [100, 3] = (s, e, score)  │
│    pred = blob[:, :2] × duration  # → timestamps (seconds)    │
│    pred = clamp(pred, 0, duration)                             │
│    pred = round(pred / unit) × unit  # round to unit          │
│    # Sắp xếp lại nếu start > end                              │
│                                                                │
│  Output: pred = [[start, end], ...] (top-100, sorted by conf) │
│          conf = [score, ...] (confidence scores)               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─── STEP 3: VERIFIER (optional, chỉ khi có >1 proposal) ──────┐
│                                                                │
│  model.set_adapter('verifier')                                 │
│                                                                │
│  Lặp qua top-5 proposals:                                     │
│    for cand in pred[:5]:                                       │
│      # 1. Mở rộng context ±50%                                │
│      s0, e0 = cand                                             │
│      offset = (e0 - s0) / 2                                   │
│      s1, e1 = [s0 - offset, e0 + offset]  # wider window      │
│                                                                │
│      # 2. Tính vị trí tương đối                               │
│      s = (s0 - s1) / (e1 - s1)  # relative start              │
│      e = (e0 - s1) / (e1 - s1)  # relative end                │
│                                                                │
│      # 3. Crop video tại [s1, e1] — max 64 frames, 2 fps      │
│                                                                │
│      # 4. Chèn SEG tokens vào visual token sequence           │
│      #    Tìm vị trí frame tương ứng với s, e                 │
│      pos_s = round(s × num_frames)                             │
│      pos_e = round(e × num_frames)                             │
│      #    Chèn <|seg_start|> trước frame pos_s                │
│      #    Chèn <|seg_end|> sau frame pos_e                    │
│      input_ids.insert(pos_s × window + offset, seg_s_id)      │
│      input_ids.insert(pos_e × window + offset, seg_e_id)      │
│                                                                │
│      # 5. Forward pass (không generate, chỉ lấy logits)       │
│      logits = model(**data).logits[0, -1].softmax()            │
│                                                                │
│      # 6. Tính verification score                              │
│      score = sigmoid(logits[Yes_id] - logits[No_id])           │
│      #   Yes_id = 9454, No_id = 2753 trong Qwen2-VL vocab     │
│                                                                │
│  Re-rank: sắp xếp proposals theo verifier score giảm dần      │
│                                                                │
│  Output: pred (re-ranked), probs, ranks                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─── STEP 4: ANSWERER ──────────────────────────────────────────┐
│                                                                │
│  model.set_adapter('answerer')                                 │
│  # hoặc model.disable_adapter() nếu dùng base model           │
│                                                                │
│  Input:                                                        │
│    - Video: crop tại best proposal [s, e]                      │
│      (mở rộng tới min_len nếu quá ngắn)                       │
│      128-256 patches/frame, max 32 frames, 2 fps               │
│    - Text: question + options (MCQ format)                     │
│    - (Optional) Subtitles trong khoảng [s, e]                  │
│                                                                │
│  Format MCQ:                                                   │
│    "What did the man do?"                                      │
│    "Options:"                                                  │
│    "(A) Walked away"                                           │
│    "(B) Sat down"                                              │
│    "Please only give the best option."                         │
│                                                                │
│  Processing:                                                   │
│    # Với MCQ: prefix "Best Option: (" để guide generation      │
│    output = model.generate(video + prompt, max_tokens=256)     │
│                                                                │
│  Output: response text (e.g., "B" hoặc free-form answer)      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Save results to JSON
```

### 11.3 Adapter Switching

```python
# Mỗi lần chuyển role:
model.base_model.disable_adapter_layers()   # tắt tất cả adapters
model.base_model.enable_adapter_layers()    # bật lại adapter system
model.set_adapter('grounder')               # chọn adapter cụ thể

# Backbone (Qwen2-VL) luôn giữ nguyên, chỉ swap LoRA weights
# → Tiết kiện VRAM (~100MB per adapter thay vì ~14GB full model)
```

### 11.4 Xử lý fallback

```python
# Nếu grounder thất bại (không sinh <|reg|> token):
if not dump['grounder_success']:
    if adapter_state['verifier']:
        # Tạo 5 proposals đều dọc video để verifier chọn
        pred = [[i*duration/6, (i+2)*duration/6] for i in range(5)]
    else:
        # Dùng toàn bộ video
        pred = [[0, duration]]
```

---

## 12. Luồng dữ liệu end-to-end (ví dụ cụ thể)

### Ví dụ: NExT-GQA sample

**Input**:
- Video: `video_3245.mp4`, duration = 40s
- Question: "What did the man do after he ate?"
- Options: ["Walked away", "Sat down", "Talked to someone", "Left the room", "Read a book"]
- GT Answer: "B" (Sat down)
- GT Span: [15.2, 22.8] (giây)

### Step 1: Grounder

```
query = parse_query("What did the man do after he ate?")
     → "What did the man do after he ate"

Video: 40s → sample 40 frames (1 fps) → Qwen2-VL encode
     → visual tokens: [1, 640, 3584] (40 frames × 16 patches/frame)
     
LLM generate: "The man ate food on the table. The relevant moment happens in <|reg|>."
                                                                          ↑ trigger!

Detection Head kích hoạt:
  1. vis_tokens = hidden_states[vision_start:vision_end]  # [1, 640, 3584]
  2. avg_pool (window=16): [1, 640, 3584] → [1, 40, 3584]  (40 frames)
  3. vis_proj: [1, 40, 3584] → [1, 40, 256]
  4. reg_proj: [1, 1, 3584] → [1, 1, 256]
  5. Transformer Fusion: [1, 41, 256] → [1, 41, 256]
  6. ConvPyramid:
     Level 0: [1, 40, 256]  → 40 points
     Level 1: [1, 20, 256]  → 20 points
     Level 2: [1, 10, 256]  → 10 points
     Level 3: [1, 5, 256]   → 5 points
     Total: 75 anchor points
  7. class_head → [75, 1] foreground scores
  8. coord_head → [75, 2] offsets
  9. Decode: point center=15, stride=1 → start=12, end=25 → [12/40, 25/40]=[0.3, 0.625]
  10. NMS → top-100 proposals

Output: pred[0] = [0.38 × 40, 0.58 × 40] = [15.2, 23.2] (seconds)
```

### Step 2: Verifier

```
Top-5 proposals, xét proposal[0] = [15.2, 23.2]:
  s0, e0 = 15.2, 23.2
  offset = (23.2 - 15.2) / 2 = 4.0
  s1, e1 = [11.2, 27.2]  (mở rộng ±4s)
  
  Crop video [11.2, 27.2] → ~32 frames (2 fps)
  s = (15.2 - 11.2) / (27.2 - 11.2) = 0.25
  e = (23.2 - 11.2) / (27.2 - 11.2) = 0.75
  
  Chèn <|seg_start|> tại frame 8 (0.25 × 32)
  Chèn <|seg_end|> tại frame 24 (0.75 × 32)
  
  Forward → logits: P(Yes) = 0.82, P(No) = 0.12
  score = sigmoid(0.82 - 0.12) = 0.67

Lặp cho 4 proposals còn lại → re-rank
```

### Step 3: Answerer

```
Best proposal = [15.2, 23.2]
min_len = 32s? → mở rộng tới [7.2, 31.2] nếu cần

Crop video [15.2, 23.2] → 16 frames (2 fps, 128-256 patches/frame)

Prompt: "What did the man do after he ate?
Options:
(A) Walked away
(B) Sat down
(C) Talked to someone
(D) Left the room
(E) Read a book
Please only give the best option."

Prefix: "Best Option: ("

Generate → "B"
```

---

## 13. Tóm tắt kiến trúc

```
╔═══════════════════════════════════════════════════════════════════════╗
║                         VIDEO + QUESTION                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ┌──────────────────────── Qwen2-VL Backbone ────────────────────┐   ║
║   │                                                                │   ║
║   │   ViT (Vision Encoder) ──→ LLM (28/32 Transformer layers)     │   ║
║   │   600M params               1.5B/7B params                    │   ║
║   │                             + LoRA adapter (role-specific)    │   ║
║   │                                                                │   ║
║   │   Outputs:                                                     │   ║
║   │   • text tokens (language generation)                          │   ║
║   │   • hidden states (cached for detection head)                  │   ║
║   │                                                                │   ║
║   └────────────────────────────────────────────────────────────────┘   ║
║            │                              │                            ║
║            │ Khi sinh <|reg|> token        │ Text output               ║
║            ▼                              ▼                            ║
║   ┌────────────────────────────┐   ┌────────────────────────────┐     ║
║   │    Detection Head          │   │    Language Outputs         │     ║
║   │                            │   │                            │     ║
║   │ vis_proj [3584→256]        │   │  Planner: JSON plan         │     ║
║   │ reg_proj [3584→256]        │   │  Verifier: "Yes"/"No"      │     ║
║   │    ↓                       │   │  Answerer: "A"/"B"/...     │     ║
║   │ vis_emb + reg_emb          │   │                            │     ║
║   │ + Positional Encoding      │   └────────────────────────────┘     ║
║   │    ↓                       │                                       ║
║   │ Transformer Fusion (×3)    │                                       ║
║   │ [vis; reg] → self-attn     │                                       ║
║   │    ↓                       │                                       ║
║   │ ConvPyramid (1,2,4,8)     │                                       ║
║   │    ↓                       │                                       ║
║   │ class_head → fg score      │                                       ║
║   │ coord_head → offsets       │                                       ║
║   │    ↓                       │                                       ║
║   │ Decode + NMS               │                                       ║
║   │    ↓                       │                                       ║
║   │ Top-100 proposals          │                                       ║
║   │ [(start, end, score)]      │                                       ║
║   └────────────────────────────┘                                       ║
║                                                                        ║
║   Training Loss:                                                       ║
║   total = LM_loss + (FocalLoss + L1Loss + NCELoss) / num_segments     ║
║                                                                        ║
║   Pipeline: Planner → Grounder → Verifier → Answerer                  ║
║   (Chain-of-LoRA: swap adapter per step, reuse backbone)               ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║   OUTPUT: Grounded timestamp + Answer                                  ║
╚════════════════════════════════════════════════════════════════════════╝
```

### Tại sao thiết kế như vậy?

| Thiết kế | Lý do |
|----------|-------|
| **Chain-of-LoRA** | 1 backbone, N adapters → tiết kiệm VRAM (swap ~100MB vs load ~14GB) |
| **`<|reg|>` trigger** | Cho phép LLM "suy nghĩ" trước khi predict timestamp (chain-of-thought) |
| **ConvPyramid multi-scale** | Phát hiện sự kiện ở nhiều temporal scale (ngắn lẫn dài) |
| **Verifier re-ranking** | Grounder sinh proposals noisy, Verifier lọc chất lượng |
| **Focal Loss** | Giải quyết extreme class imbalance (99% background) |
| **SEG tokens trong Verifier** | Đánh dấu trực tiếp vào visual sequence → model "nhìn thấy" boundary |
| **Temporal jittering (Answerer)** | Data augmentation → model robust hơn với grounding errors |
| **Multi-LR optimizer** | Detection head (from scratch) cần LR cao, LoRA (fine-tune) cần LR thấp |
| **GroupSampler** | Batch cùng data_type → tránh conflict giữa multimodal vs text-only samples |
