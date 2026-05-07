# Lab 21 Run Commands

## 1. Chạy training thật trên Colab

1. Upload notebook `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` lên Google Colab.
2. Chọn `Runtime > Change runtime type > GPU`.
3. Chạy `Run all`.

## 2. Sau khi Colab chạy xong

Tải các thư mục/file sau về máy và đặt vào repo:

- `adapters/r8`
- `adapters/r16`
- `adapters/r64`
- `results/rank_experiment_summary.csv`
- `results/qualitative_comparison.csv`
- `results/loss_curve.png`

Nếu bạn đã dùng `OUTPUT_DIR` trên Drive, chỉ cần copy chúng về đúng cấu trúc trên.

## 3. Tạo report local bằng lệnh

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_lab21_submission.ps1 `
  -ResultsDir results `
  -BaseModel "unsloth/Qwen2.5-3B-bnb-4bit" `
  -Dataset "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated, 200 samples (180 train + 20 eval)" `
  -MaxSeqLength "512 (p95 = 463, rounded up)" `
  -Gpu "Tesla T4, 16 GB VRAM" `
  -TrainingCost '$0.35 (~60 phut @ $0.35/hr)' `
  -HfLink "https://huggingface.co/<username>/<adapter-name>"
```

## 4. Validate output bundle

```powershell
python scripts/validate_lab21_outputs.py
```

## 5. Nếu cần check model OpenAI thật riêng

```powershell
python scripts/check_openai_gpt4o_mini.py
```
