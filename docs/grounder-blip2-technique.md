# Grounder BLIP-2 Trong Dự Án NExT-GQA (Giải Thích Kỹ Thuật + Ví Dụ)

Tài liệu này giải thích chính xác cách `nextgqa/grounder.py` đang dùng BLIP-2/TFVTG để sinh temporal proposals.

File liên quan:
- `nextgqa/grounder.py`
- `vlm_localizer.py`
- `feature_extraction_nextgqa.py`
- `llm_prompting.py`

---

## 1. Grounder đang làm gì?

Grounder nhận:
1. `llm_outputs_{split}.json` (query rewrite từ Gemini)
2. `blip2_features/{video_id}.npy` (feature video đã extract)
3. metadata từ `gsub_{split}.json`, `{split}.csv`

Grounder trả ra cho mỗi câu hỏi:
- `top5_proposals`: danh sách 5 đoạn thời gian tốt nhất
- mỗi proposal có dạng `[start_sec, end_sec, confidence]`

Pipeline trong code:

```text
Query (text) -> BLIP-2 text branch -> frame-text similarity
-> multi-scale sliding window scoring
-> dynamic start refinement
-> NMS
-> merge/re-rank (main query + sub-query)
-> top-5 proposals
```

---

## 2. BLIP-2 features lấy như thế nào (Bước trước Grounder)

Trong `feature_extraction_nextgqa.py`:

1. Decode video bằng `decord`, sample frame theo `fps` (mặc định 3 fps).
2. Chạy BLIP-2 visual encoder + Q-Former.
3. Lấy `vision_proj(query_output.last_hidden_state)`.
4. Lưu mỗi video thành `.npy` shape:

```text
[T, 32, 256]
```

Ý nghĩa:
- `T`: số frame sampled
- `32`: số query tokens của Q-Former cho mỗi frame
- `256`: embedding dimension sau projection

Grounder không decode video trực tiếp, chỉ đọc `.npy` này.

---

## 3. Cách tính điểm text-video trong `vlm_localizer.calc_scores`

Input:
- `video_features`: `[T, 32, 256]`
- `sentences`: danh sách mô tả text

Các bước:
1. Tokenize text, chạy BLIP-2 Q-Former text branch.
2. Lấy `text_feat = text_proj([CLS])` -> shape `[M, 256]` (`M` là số câu).
3. Normalize cả text và video features.
4. Tính cosine-like similarity:

\[
S_{m,t,p} = \langle \hat{text}_{m}, \hat{video}_{t,p} \rangle
\]

- `m`: index câu text
- `t`: frame
- `p`: query token (0..31)

5. Lấy `max` theo token `p` -> mỗi câu còn vector theo frame.
6. Lấy `mean` theo các câu `m` -> còn 1 score theo frame:

\[
score_t = \frac{1}{M}\sum_m \max_p S_{m,t,p}
\]

Kết quả: `scores` shape `[1, T]`.

---

## 4. Tạo proposal bằng multi-scale sliding window

Trong `generate_proposal`:

### 4.1 Gate frame score
- Tạo mask: `masks = (scores > 0.2)`
- Chỉ giữ frame đủ liên quan: `scores = scores * masks`

### 4.2 Duyệt nhiều window size
Cho `kernel_size` chạy từ `stride` đến `max_stride`, bước `stride`.

Với mỗi cửa sổ:
- `inner_sum`: tổng score trong cửa sổ
- `outer_sum`: tổng score ngoài cửa sổ
- static score:

\[
static = \frac{inner\_sum}{kernel\_size} - \frac{outer\_sum}{outer\_num}
\]

Ý tưởng: đoạn đúng phải có score trong cao hơn nền ngoài.

### 4.3 Dynamic start (refine điểm bắt đầu)
`get_dynamic_scores` làm:
1. Gaussian smoothing chuỗi score.
2. Lấy đạo hàm rời rạc (diff).
3. Cộng dồn vùng gradient tăng theo heuristic `nchk`.
4. Sinh `dynamic_idxs` (start linh động) và `dynamic_scores`.

Khi xuất kết quả, code giữ cả:
- `static_start`
- `start` (dynamic start)
- `end`

### 4.4 NMS
- Áp dụng NMS theo IoU (`nms_thresh=0.3`) để bỏ proposal chồng lặp mạnh.
- Trả các proposal đã lọc + score.

---

## 5. Grounder kết hợp main query + sub-query ra sao

Trong `nextgqa/grounder.py`, hàm `ground_one_question`:

1. Nếu `llm response` có `query_json`:
- chạy localize cho từng sub-query (`sub_query_id >= 1`)
- mỗi sub-query lấy proposals tốt, sau đó `select_proposal` để re-rank

2. Chạy localize cho main query:
- luôn có câu hỏi raw (`sentence`)
- nếu có `sub_query_id == 0`, thêm mô tả rewrite chính

3. Trộn proposal:
- `proposals[:7]` từ main query
- cộng với kết quả `filter_and_integrate(sub_query_proposals, relation)`
  - `relation` có thể `single-query`, `sequentially`, `simultaneously`

4. Re-rank cuối bằng `select_proposal` (weighted IoU voting):

\[
score(j)=\sum_k IoU(p_j,p_k)^{\gamma} \cdot w_k,\; \gamma=0.6
\]

Proposal được nhiều proposal khác “đồng thuận” sẽ lên hạng cao.

---

## 6. Ví dụ thực tế từ dữ liệu hiện tại

Ví dụ lấy từ project của bạn:

- `video_id`: `2574374895`
- `qid`: `8`
- question:
  `what did the baby do after throwing the green cup away while on the floor near the end`
- `relationship`: `sequentially`

`query_json` (rút gọn):
1. `sub_query_id=0`: mô tả toàn bộ sự kiện
2. `sub_query_id=1`: "baby throws green cup away"
3. `sub_query_id=2`: "what happens immediately after"

Grounder output top proposals (giây):
1. `[2.33, 27.00, 0.939]`
2. `[0.33, 27.00, 0.689]`
3. `[2.33, 14.00, 1.000]`

Cách đọc ví dụ:
- Proposal #3 có confidence nội bộ cao nhất (1.0 sau normalize cục bộ), nhưng ngắn hơn.
- Proposal #1 phủ rộng hơn và được xếp top đầu sau bước tích hợp/re-rank tổng thể.
- Quan hệ `sequentially` giúp ưu tiên tổ hợp sub-events theo thứ tự thời gian.

---

## 7. Vì sao kỹ thuật này hợp với bài toán training-free?

1. Không fine-tune trên NExT-GQA.
2. Tận dụng BLIP-2 pretrained để có score frame-text.
3. Multi-scale window + contrast foreground/background giúp tìm đoạn theo thời gian.
4. Query decomposition từ LLM giảm độ khó câu hỏi phức hợp (before/after).
5. Voting IoU giúp kết quả ổn định hơn so với chọn score cục bộ thuần túy.

---

## 8. Gợi ý debug khi grounder sai

1. Kiểm tra quality query rewrite (`grounding_description`, `relationship`).
2. Vẽ frame-score theo thời gian để xem mask `scores > 0.2` có quá gắt không.
3. Thử `stride` nhỏ hơn để tăng temporal resolution.
4. Kiểm tra `max_stride_factor` (cửa sổ quá dài/ngắn đều gây miss).
5. So sánh top-1 với top-5 oracle để biết lỗi do ranking hay do proposal generation.

---

## 9. Lệnh chạy nhanh

Grounder test:

```bash
python -m nextgqa.grounder --split test
```

Dry run 10 câu:

```bash
python -m nextgqa.grounder --split test --dry_run
```

Eval grounder top-1 và oracle:

```bash
python -m nextgqa.eval_ground --split test --source grounder --topk 1
python -m nextgqa.eval_ground --split test --source grounder --topk 5
```

