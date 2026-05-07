param(
    [string]$ResultsDir = "results",
    [string]$BaseModel = "unsloth/Qwen2.5-3B-bnb-4bit",
    [string]$Dataset = "<ten dataset>, <so samples> (X train + Y eval)",
    [string]$MaxSeqLength = "<so> (p95 = <so>, rounded up)",
    [string]$Gpu = "<Tesla T4 / L4 / A100>, <X> GB VRAM",
    [string]$TrainingCost = '$<so> (~<phut> @ $<rate>/hr)',
    [string]$HfLink = "https://huggingface.co/<username>/<adapter-name>"
)

$ErrorActionPreference = "Stop"

Write-Host "Generating REPORT.md from $ResultsDir ..." -ForegroundColor Cyan
python scripts/generate_lab21_report.py `
  --results-dir $ResultsDir `
  --student-name "Nguyen Trieu Gia Khanh" `
  --student-id "2A202600225" `
  --submission-date "2026-07-05" `
  --submission-option "B" `
  --base-model $BaseModel `
  --dataset $Dataset `
  --max-seq-length $MaxSeqLength `
  --gpu $Gpu `
  --training-cost $TrainingCost `
  --hf-link $HfLink

Write-Host "Validating repo scaffolding ..." -ForegroundColor Cyan
python scripts/validate_lab21_outputs.py

if (Test-Path $ResultsDir) {
  Write-Host "Results directory found: $ResultsDir" -ForegroundColor Green
} else {
  Write-Host "Results directory not found: $ResultsDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next manual checks before zipping:" -ForegroundColor Cyan
Write-Host "- Fill the conclusion and what-I-learned bullets in REPORT.md"
Write-Host "- Confirm results/loss_curve.png exists"
Write-Host "- Confirm adapters/r8, adapters/r16, adapters/r64 exist in your output bundle"
Write-Host "- Update LINKS.md with GitHub and Hugging Face URLs"
