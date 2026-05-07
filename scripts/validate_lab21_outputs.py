from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Lab21_LoRA_Finetuning_T4.ipynb"
REQUIRED_ROOT_FILES = [
    ROOT / "README.md",
    ROOT / "REPORT.md",
    ROOT / "LINKS.md",
    ROOT / "requirements.txt",
    ROOT / ".env.example",
    ROOT / "scripts" / "check_openai_gpt4o_mini.py",
    ROOT / "scripts" / "generate_lab21_report.py",
]
REQUIRED_REPORT_SECTIONS = [
    "## 1. Setup",
    "## 2. Rank Experiment Results",
    "## 3. Loss Curve Analysis",
    "## 4. Qualitative Comparison",
    "## 5. Conclusion về Rank Trade-off",
    "## 6. What I Learned",
]
REQUIRED_NOTEBOOK_SNIPPETS = [
    "evaluate_base_model",
    "RUN_ALL_LAYERS_BONUS = True",
    "RUN_DORA_BONUS = True",
    "rank_experiment_summary.csv",
    "bonus_experiment_summary.csv",
    "qualitative_comparison.csv",
    "push_to_hub",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def validate_repo_files() -> None:
    missing = [path for path in REQUIRED_ROOT_FILES if not path.exists()]
    if missing:
        fail(f"Missing repo files: {', '.join(str(path.relative_to(ROOT)) for path in missing)}")

    report_text = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    missing_sections = [section for section in REQUIRED_REPORT_SECTIONS if section not in report_text]
    if missing_sections:
        fail(f"REPORT.md missing sections: {', '.join(missing_sections)}")


def validate_notebook() -> None:
    if not NOTEBOOK.exists():
        fail(f"Notebook not found: {NOTEBOOK}")

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if "cells" not in nb:
        fail("Notebook JSON missing 'cells'")

    combined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    missing_snippets = [snippet for snippet in REQUIRED_NOTEBOOK_SNIPPETS if snippet not in combined]
    if missing_snippets:
        fail(f"Notebook missing expected snippets: {', '.join(missing_snippets)}")

    dirty_cells = []
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            dirty_cells.append(idx)
        if cell.get("execution_count") is not None:
            dirty_cells.append(idx)
    if dirty_cells:
        fail(f"Notebook should be stripped, but code cells still contain outputs/execution counts: {dirty_cells}")


def validate_output_dir(output_dir: Path) -> None:
    report = output_dir / "REPORT.md"
    links = output_dir / "LINKS.md"
    summary_csv = output_dir / "results" / "rank_experiment_summary.csv"
    qual_csv = output_dir / "results" / "qualitative_comparison.csv"
    loss_curve = output_dir / "results" / "loss_curve.png"
    adapter_dirs = [
        output_dir / "adapters" / "r8",
        output_dir / "adapters" / "r16",
        output_dir / "adapters" / "r64",
    ]

    required_paths = [report, links, summary_csv, qual_csv, loss_curve, *adapter_dirs]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        fail(f"Output dir missing required artifacts: {', '.join(str(path) for path in missing)}")

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    tags = {row.get("tag", "").strip() for row in rows}
    for expected in {"base", "r8", "r16", "r64"}:
        if expected not in tags:
            fail(f"rank_experiment_summary.csv missing row for '{expected}'")

    with qual_csv.open("r", encoding="utf-8", newline="") as f:
        qual_rows = list(csv.DictReader(f))
    if len(qual_rows) < 5:
        fail("qualitative_comparison.csv must contain at least 5 rows")

    report_text = report.read_text(encoding="utf-8")
    missing_sections = [section for section in REQUIRED_REPORT_SECTIONS if section not in report_text]
    if missing_sections:
        fail(f"Generated REPORT.md missing sections: {', '.join(missing_sections)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Lab 21 repo scaffolding and generated artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional path to the generated OUTPUT_DIR from the notebook run.",
    )
    args = parser.parse_args()

    validate_repo_files()
    validate_notebook()
    if args.output_dir:
        validate_output_dir(args.output_dir)

    print("PASS: repository scaffolding looks good")
    if args.output_dir:
        print(f"PASS: output artifacts look good: {args.output_dir}")


if __name__ == "__main__":
    main()
