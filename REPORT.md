# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Triệu Gia Khánh — 2A202600225
**Ngày nộp**: 2026-07-05
**Submission option**: B

## 1. Setup
- **Base model**: `<điền model bạn chọn — vd: unsloth/Qwen2.5-3B-bnb-4bit>`
- **Dataset**: `<tên dataset>, <số samples> (X train + Y eval)`
- **max_seq_length**: `<số> (p95 = <số>, rounded up)>`
- **GPU**: `<Tesla T4 / L4 / A100>, <X> GB VRAM`
- **Training cost**: `$<số> (~<phút> @ $<rate>/hr)`
- **HF Hub link** (nếu Option B): `https://huggingface.co/<username>/<adapter-name>`

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | ...             | ... min    | ... GB    | ...       | ...        |
| 16   | ...             | ... min    | ... GB    | ...       | ...        |
| 64   | ...             | ... min    | ... GB    | ...       | ...        |
| Base | -               | -          | -         | ...       | ...        |

## 3. Loss Curve Analysis
- Quan sát: `<có / không có overfitting? Lý do?>`
- Nếu `eval loss` tăng trong khi `train loss` giảm, ghi rõ dấu hiệu overfitting.
- Đính kèm hoặc tham chiếu `results/loss_curve.png`.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: `<improved / same / degraded?>`

### Example 2
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: `<improved / same / degraded?>`

### Example 3
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: `<improved / same / degraded?>`

### Example 4
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: `<improved / same / degraded?>`

### Example 5
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: `<improved / same / degraded?>`

## 5. Conclusion về Rank Trade-off

<Tối thiểu 100 từ. Trả lời 3 câu hỏi:>
- Rank nào cho ROI tốt nhất trên dataset này? Tại sao?
- Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?
- Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?

## 6. What I Learned
- <Bullet 1: insight cá nhân>
- <Bullet 2: insight cá nhân>
- <Bullet 3: optional>
