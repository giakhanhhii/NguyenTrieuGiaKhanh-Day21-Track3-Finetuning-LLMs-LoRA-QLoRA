from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT
DEFAULT_STUDENT = "Nguyễn Triệu Gia Khánh"
DEFAULT_STUDENT_ID = "2A202600225"
DEFAULT_SUBMISSION_DATE = "2026-07-05"
DEFAULT_SUBMISSION_OPTION = "B"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return value or "N/A"


def build_rank_table(summary_rows: list[dict[str, str]]) -> str:
    indexed = {row.get("tag", ""): row for row in summary_rows}

    def cell(tag: str, key: str, digits: int | None = None, suffix: str = "") -> str:
        row = indexed.get(tag, {})
        value = row.get(key, "N/A")
        if digits is not None and value not in {"", "N/A", "-", "nan"}:
            try:
                value = f"{float(value):.{digits}f}"
            except Exception:
                pass
        if suffix and value not in {"", "N/A", "-"}:
            value = f"{value} {suffix}".strip()
        return value or "N/A"

    lines = [
        "| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |",
        "|------|-----------------|------------|-----------|-----------|------------|",
        f"| 8    | {cell('r8', 'trainable_params')} | {cell('r8', 'train_time_min', 2, 'min')} | {cell('r8', 'peak_vram_gb', 2, 'GB')} | {cell('r8', 'eval_loss', 4)} | {cell('r8', 'eval_perplexity', 2)} |",
        f"| 16   | {cell('r16', 'trainable_params')} | {cell('r16', 'train_time_min', 2, 'min')} | {cell('r16', 'peak_vram_gb', 2, 'GB')} | {cell('r16', 'eval_loss', 4)} | {cell('r16', 'eval_perplexity', 2)} |",
        f"| 64   | {cell('r64', 'trainable_params')} | {cell('r64', 'train_time_min', 2, 'min')} | {cell('r64', 'peak_vram_gb', 2, 'GB')} | {cell('r64', 'eval_loss', 4)} | {cell('r64', 'eval_perplexity', 2)} |",
        f"| Base | - | - | - | {cell('base', 'eval_loss', 4)} | {cell('base', 'eval_perplexity', 2)} |",
    ]
    return "\n".join(lines)


def build_examples(qual_rows: list[dict[str, str]]) -> str:
    sections = []
    for idx, row in enumerate(qual_rows[:5], start=1):
        sections.append(
            "\n".join(
                [
                    f"### Example {idx}",
                    f"**Prompt**: {row.get('prompt', '...')}",
                    f"**Base**: {row.get('base', '...')}",
                    f"**Fine-tuned (r=16)**: {row.get('finetuned', '...')}",
                    "**Nhận xét**: <improved / same / degraded?>",
                    "",
                ]
            )
        )
    while len(sections) < 5:
        idx = len(sections) + 1
        sections.append(
            "\n".join(
                [
                    f"### Example {idx}",
                    "**Prompt**: ...",
                    "**Base**: ...",
                    "**Fine-tuned (r=16)**: ...",
                    "**Nhận xét**: <improved / same / degraded?>",
                    "",
                ]
            )
        )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rubric-aligned REPORT.md from Lab 21 results CSVs.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "REPORT.md")
    parser.add_argument("--student-name", default=DEFAULT_STUDENT)
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--submission-date", default=DEFAULT_SUBMISSION_DATE)
    parser.add_argument("--submission-option", default=DEFAULT_SUBMISSION_OPTION)
    parser.add_argument("--base-model", default="<điền model bạn chọn — vd: unsloth/Qwen2.5-3B-bnb-4bit>")
    parser.add_argument("--dataset", default="<tên dataset>, <số samples> (X train + Y eval)")
    parser.add_argument("--max-seq-length", default="<số> (p95 = <số>, rounded up)>")
    parser.add_argument("--gpu", default="<Tesla T4 / L4 / A100>, <X> GB VRAM")
    parser.add_argument("--training-cost", default="$<số> (~<phút> @ $<rate>/hr)")
    parser.add_argument("--hf-link", default="https://huggingface.co/<username>/<adapter-name>")
    args = parser.parse_args()

    summary_csv = args.results_dir / "rank_experiment_summary.csv"
    qual_csv = args.results_dir / "qualitative_comparison.csv"

    summary_rows = read_csv_rows(summary_csv) if summary_csv.exists() else []
    qual_rows = read_csv_rows(qual_csv) if qual_csv.exists() else []

    rank_table = build_rank_table(summary_rows) if summary_rows else "\n".join(
        [
            "| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |",
            "|------|-----------------|------------|-----------|-----------|------------|",
            "| 8    | ...             | ... min    | ... GB    | ...       | ...        |",
            "| 16   | ...             | ... min    | ... GB    | ...       | ...        |",
            "| 64   | ...             | ... min    | ... GB    | ...       | ...        |",
            "| Base | -               | -          | -         | ...       | ...        |",
        ]
    )
    examples = build_examples(qual_rows)

    report = f"""# Lab 21 — Evaluation Report

**Học viên**: {args.student_name} — {args.student_id}
**Ngày nộp**: {args.submission_date}
**Submission option**: {args.submission_option}

## 1. Setup
- **Base model**: `{args.base_model}`
- **Dataset**: {args.dataset}
- **max_seq_length**: {args.max_seq_length}
- **GPU**: {args.gpu}
- **Training cost**: {args.training_cost}
- **HF Hub link** (nếu Option B): `{args.hf_link}`

## 2. Rank Experiment Results

{rank_table}

## 3. Loss Curve Analysis
- Quan sát: <có / không có overfitting? Lý do?>
- Nếu `eval loss` tăng trong khi `train loss` giảm, ghi rõ dấu hiệu overfitting.
- Đính kèm hoặc tham chiếu `results/loss_curve.png`.

## 4. Qualitative Comparison (5 examples)

{examples}

## 5. Conclusion về Rank Trade-off

<Tối thiểu 100 từ. Trả lời 3 câu hỏi:>
- Rank nào cho ROI tốt nhất trên dataset này? Tại sao?
- Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?
- Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?

## 6. What I Learned
- <Bullet 1: insight cá nhân>
- <Bullet 2: insight cá nhân>
- <Bullet 3: optional>
"""
    args.output.write_text(report, encoding="utf-8")
    print(f"Generated report: {args.output}")


if __name__ == "__main__":
    main()
