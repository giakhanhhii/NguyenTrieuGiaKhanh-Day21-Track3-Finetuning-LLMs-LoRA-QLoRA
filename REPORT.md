# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Triệu Gia Khánh — 2A202600225  
**Ngày nộp**: 2026-05-07  
**Submission option**: B

## 1. Setup
- **Base model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **Dataset**: `yahma/alpaca-cleaned`, **200 samples** (`180` train / `20` eval)
- **max_seq_length**: `512` (p95 = `400`, rounded up)
- **GPU**: `Tesla T4`, ~15.6 GB VRAM
- **Training cost**: `$0.06` (tổng thời gian ~ `10.2` phút, @ `$0.35/hr`)
- **HF Hub link** (Option B): `Chưa push trong run này`

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|------------------|------------|-----------|-----------|------------|
| 8    | 2,293,760        | 3.08 min   | 5.51 GB   | 1.4511    | 4.2678     |
| 16   | 4,587,520        | 4.03 min   | 4.75 GB   | 1.4359    | 4.2035     |
| 64   | 18,350,080       | 3.11 min   | 6.49 GB   | 1.4331    | 4.1915     |
| Base | -                | -          | -         | N/A       | N/A        |

> Nguồn số liệu: `results/rank_experiment_summary.csv`

## 3. Loss Curve Analysis
- File biểu đồ: `results/loss_curve.png`
- Quan sát chính: train loss giảm đều theo step ở cả 3 run, không có dấu hiệu diverge.
- Đánh giá overfitting: không kết luận chắc chắn overfitting vì cấu hình T4 tắt eval-during-training (`eval_strategy="no"`), chỉ đánh giá qua eval cuối run.
- Giải thích ngắn gọn theo xu hướng loss: rank cao hơn (r=64) cho perplexity thấp nhất nhưng chênh lệch rất nhỏ so với r=16, trong khi VRAM tăng rõ rệt.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Trả lời đúng ý chính nhưng trình bày còn ngắn và ít cấu trúc.  
**Fine-tuned (r=16)**: Diễn giải theo dạng nhập môn, mạch lạc hơn, dễ đọc hơn.  
**Nhận xét**: improved

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Trả về code trực tiếp (`def fibonacci(n): ...`) khá ngắn gọn.  
**Fine-tuned (r=16)**: Trả lời có thêm giải thích cách làm trước khi đưa code.  
**Nhận xét**: same (khác phong cách trình bày)

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Có liệt kê nhưng câu chữ hơi rời rạc.  
**Fine-tuned (r=16)**: Trình bày theo bullet rõ hơn, định hướng người dùng tốt hơn.  
**Nhận xét**: improved

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Có trả lời nhưng lẫn một số cụm từ chưa chuẩn thuật ngữ.  
**Fine-tuned (r=16)**: Câu trả lời có cấu trúc hơn, nhấn mạnh điểm khác biệt về quantization và VRAM.  
**Nhận xét**: improved

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Trả lời đúng trọng tâm nhưng hơi ngắn và chưa tách bạch rõ từng kỹ thuật.  
**Fine-tuned (r=16)**: Phân tách từng khái niệm rõ ràng hơn, dễ so sánh hơn.  
**Nhận xét**: improved

> Nguồn ví dụ: `results/qualitative_comparison.csv`

## 5. Conclusion về Rank Trade-off
Trên bài toán instruction tuning với dataset kích thước nhỏ-trung bình, `r=16` thường là điểm cân bằng tốt giữa chất lượng và chi phí huấn luyện. Rank thấp hơn như `r=8` có ưu thế về tốc độ và VRAM, phù hợp cho vòng lặp thử nghiệm nhanh, nhưng có thể giới hạn khả năng học các mẫu biểu đạt phức tạp. Ngược lại, `r=64` tăng đáng kể số tham số trainable và tài nguyên cần thiết; trong nhiều trường hợp mức cải thiện perplexity không tương xứng với phần chi phí tăng thêm, thể hiện hiện tượng diminishing returns. Với bối cảnh triển khai thực tế cần tối ưu cả latency, hạ tầng và chi phí bảo trì adapter, lựa chọn hợp lý nhất thường là bắt đầu từ `r=16`, sau đó chỉ nâng lên `r=64` nếu số liệu thực nghiệm cho thấy cải thiện rõ rệt, nhất quán ở cả chỉ số định lượng lẫn đánh giá qualitative.

## 6. What I Learned
- LoRA/QLoRA giúp fine-tune mô hình instruction hiệu quả trên GPU giới hạn như T4 mà vẫn theo dõi được trade-off rõ ràng giữa rank và tài nguyên.
- Chất lượng dữ liệu và định dạng Alpaca nhất quán có tác động rất lớn đến độ ổn định của loss và chất lượng đầu ra, đôi khi quan trọng hơn việc chỉ tăng rank.
- Việc đánh giá kết hợp cả perplexity và qualitative comparison là cần thiết để chọn rank triển khai, vì chỉ một metric định lượng có thể chưa phản ánh đầy đủ chất lượng thực tế.
