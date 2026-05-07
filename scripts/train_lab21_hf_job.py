from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer


SEED = 42
MODEL_NAME = os.environ.get("LAB21_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_NAME = os.environ.get(
    "LAB21_DATASET_NAME",
    "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated",
)
DATASET_SPLIT = os.environ.get("LAB21_DATASET_SPLIT", "train")
MAX_SAMPLES = int(os.environ.get("LAB21_MAX_SAMPLES", "200"))
MAX_SEQ_CAP = int(os.environ.get("LAB21_MAX_SEQ_CAP", "1024"))
OUTPUT_DIR = Path(os.environ.get("LAB21_OUTPUT_DIR", "/tmp/lab21_hf_job"))
HF_USERNAME = os.environ.get("HF_USERNAME", "giakhanhjik")
HUB_MODEL_PREFIX = os.environ.get("HUB_MODEL_PREFIX", "lab21-qwen25-05b-vi")
RESULTS_REPO_ID = os.environ.get("RESULTS_REPO_ID", f"{HF_USERNAME}/lab21-results")
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "true").lower() == "true"
RUN_ALL_LAYERS_BONUS = os.environ.get("RUN_ALL_LAYERS_BONUS", "true").lower() == "true"
RUN_DORA_BONUS = os.environ.get("RUN_DORA_BONUS", "true").lower() == "true"
GPU_COST_USD_PER_HOUR = float(os.environ.get("GPU_COST_USD_PER_HOUR", "0.4"))
TEST_PROMPTS = [
    "Giải thích khái niệm machine learning cho người mới bắt đầu.",
    "Viết đoạn code Python tính số Fibonacci thứ n.",
    "Liệt kê 5 nguyên tắc thiết kế UI/UX.",
    "Tóm tắt sự khác biệt giữa LoRA và QLoRA.",
    "Phân biệt prompt engineering, RAG, và fine-tuning.",
]
BASE_TARGET_MODULES = ["q_proj", "v_proj"]
ALL_LAYER_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_hf_login() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        login(token=token)


def load_and_prepare_dataset(tokenizer):
    raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    raw = raw.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(raw))))

    cols = raw.column_names
    instruction_col = next((c for c in ["instruction", "instruction_vi", "prompt", "question"] if c in cols), None)
    input_col = next((c for c in ["input", "input_vi", "context"] if c in cols), None)
    output_col = next((c for c in ["output", "output_vi", "response", "answer"] if c in cols), None)
    if not instruction_col or not output_col:
        raise ValueError(f"Could not detect instruction/output columns in {cols}")

    def normalize(example):
        instruction = (example.get(instruction_col) or "").strip()
        input_text = (example.get(input_col) or "").strip() if input_col else ""
        output_text = (example.get(output_col) or "").strip()
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }

    ds = raw.map(normalize)

    def keep(example):
        if len(example["instruction"]) < 5 or len(example["output"]) < 10:
            return False
        output_tokens = len(tokenizer.encode(example["output"], add_special_tokens=False))
        return output_tokens >= 10

    ds = ds.filter(keep)

    def dedup_key(example):
        return (
            example["instruction"].strip().lower(),
            example["input"].strip().lower(),
            example["output"].strip().lower(),
        )

    seen = set()
    keep_indices = []
    for idx, row in enumerate(ds):
        key = dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(idx)
    ds = ds.select(keep_indices)

    def format_row(example):
        if example["input"]:
            text = (
                "### Instruction:\n"
                f"{example['instruction']}\n\n"
                "### Input:\n"
                f"{example['input']}\n\n"
                "### Response:\n"
                f"{example['output']}"
            )
        else:
            text = (
                "### Instruction:\n"
                f"{example['instruction']}\n\n"
                "### Response:\n"
                f"{example['output']}"
            )
        return {"text": text}

    ds = ds.map(format_row)
    lengths = [len(tokenizer.encode(row["text"], add_special_tokens=False)) for row in ds]
    p95 = int(np.percentile(lengths, 95))
    max_seq_length = min(MAX_SEQ_CAP, 1 << (max(p95, 256) - 1).bit_length())
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    meta = {
        "dataset_name": DATASET_NAME,
        "total_samples": len(ds),
        "train_samples": len(split["train"]),
        "eval_samples": len(split["test"]),
        "p95": p95,
        "max_seq_length": max_seq_length,
        "instruction_col": instruction_col,
        "input_col": input_col,
        "output_col": output_col,
    }
    return split["train"], split["test"], meta


def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_base_model(max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=get_quant_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def wrap_lora(model, rank: int, alpha: int, target_modules: list[str], use_dora: bool = False):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,
    )
    return get_peft_model(model, config)


def make_trainer(model, tokenizer, train_ds, eval_ds, output_dir: Path, max_seq_length: int):
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
    )
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=args,
    )


def manual_eval_loss(model, tokenizer, eval_ds):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "tmp_eval"),
            per_device_eval_batch_size=1,
            report_to="none",
            fp16=False,
        ),
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    try:
        metrics = trainer.evaluate()
        return float(metrics["eval_loss"])
    except Exception:
        dataloader = trainer.get_eval_dataloader()
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items() if hasattr(v, "to")}
                out = model(**batch)
                total += float(out.loss.item())
                count += 1
        return total / max(count, 1)


def plot_losses(log_history, save_path: Path):
    df = pd.DataFrame(log_history)
    train = df[df["loss"].notna()] if "loss" in df else pd.DataFrame()
    eval_df = df[df["eval_loss"].notna()] if "eval_loss" in df else pd.DataFrame()
    plt.figure(figsize=(8, 4))
    if not train.empty:
        plt.plot(train["step"], train["loss"], label="train", color="#0E2A52")
    if not eval_df.empty:
        plt.plot(eval_df["step"], eval_df["eval_loss"], label="eval", color="#C8102E", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve - r=16")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 160):
    text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_rank(rank: int, alpha: int, train_ds, eval_ds, max_seq_length: int, target_modules: list[str], tag: str, use_dora: bool = False):
    cleanup()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_base_model(max_seq_length)
    model = wrap_lora(model, rank=rank, alpha=alpha, target_modules=target_modules, use_dora=use_dora)
    trainer = make_trainer(model, tokenizer, train_ds, eval_ds, OUTPUT_DIR / tag, max_seq_length)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    adapter_dir = OUTPUT_DIR / "adapters" / tag
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    eval_loss = manual_eval_loss(model, tokenizer, eval_ds)
    result = {
        "tag": tag,
        "rank": rank,
        "alpha": alpha,
        "target_modules": ",".join(target_modules),
        "use_dora": use_dora,
        "trainable_params": int(trainable),
        "train_time_min": wall / 60,
        "peak_vram_gb": peak_vram,
        "eval_loss": eval_loss,
        "eval_perplexity": math.exp(eval_loss),
        "log_history": trainer.state.log_history,
    }
    del trainer
    cleanup()
    return result


def push_results(best_required_tag: str):
    api = HfApi()
    best_repo_id = f"{HF_USERNAME}/{HUB_MODEL_PREFIX}-{best_required_tag}"
    api.create_repo(repo_id=best_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=best_repo_id,
        repo_type="model",
        folder_path=str(OUTPUT_DIR / "adapters" / best_required_tag),
        commit_message=f"Upload best adapter {best_required_tag}",
    )

    if RUN_ALL_LAYERS_BONUS and (OUTPUT_DIR / "adapters" / "r16_all_layers").exists():
        repo_id = f"{HF_USERNAME}/{HUB_MODEL_PREFIX}-r16-all-layers"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(OUTPUT_DIR / "adapters" / "r16_all_layers"))

    if RUN_DORA_BONUS and (OUTPUT_DIR / "adapters" / "r16_dora").exists():
        repo_id = f"{HF_USERNAME}/{HUB_MODEL_PREFIX}-r16-dora"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(OUTPUT_DIR / "adapters" / "r16_dora"))

    api.create_repo(repo_id=RESULTS_REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=RESULTS_REPO_ID,
        repo_type="dataset",
        folder_path=str(OUTPUT_DIR),
        path_in_repo="lab21_outputs",
        commit_message="Upload Lab21 outputs",
    )


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "adapters").mkdir(exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(exist_ok=True)
    ensure_hf_login()

    base_model, base_tokenizer = load_base_model(MAX_SEQ_CAP)
    train_ds, eval_ds, meta = load_and_prepare_dataset(base_tokenizer)
    max_seq_length = meta["max_seq_length"]
    del base_model
    cleanup()

    base_model, base_tokenizer = load_base_model(max_seq_length)
    base_eval_loss = manual_eval_loss(base_model, base_tokenizer, eval_ds)
    base_metrics = {
        "tag": "base",
        "rank": "base",
        "alpha": 0,
        "target_modules": "-",
        "use_dora": False,
        "trainable_params": 0,
        "train_time_min": 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "eval_loss": base_eval_loss,
        "eval_perplexity": math.exp(base_eval_loss),
    }
    del base_model
    cleanup()

    r16 = train_rank(16, 32, train_ds, eval_ds, max_seq_length, BASE_TARGET_MODULES, "r16", use_dora=False)
    plot_losses(r16["log_history"], OUTPUT_DIR / "results" / "loss_curve.png")
    r8 = train_rank(8, 16, train_ds, eval_ds, max_seq_length, BASE_TARGET_MODULES, "r8", use_dora=False)
    r64 = train_rank(64, 128, train_ds, eval_ds, max_seq_length, BASE_TARGET_MODULES, "r64", use_dora=False)

    bonus_results = []
    if RUN_ALL_LAYERS_BONUS:
        bonus_results.append(train_rank(16, 32, train_ds, eval_ds, max_seq_length, ALL_LAYER_TARGET_MODULES, "r16_all_layers", use_dora=False))
    if RUN_DORA_BONUS:
        bonus_results.append(train_rank(16, 32, train_ds, eval_ds, max_seq_length, BASE_TARGET_MODULES, "r16_dora", use_dora=True))

    summary = pd.DataFrame(
        [
            base_metrics,
            {k: v for k, v in r8.items() if k != "log_history"},
            {k: v for k, v in r16.items() if k != "log_history"},
            {k: v for k, v in r64.items() if k != "log_history"},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "results" / "rank_experiment_summary.csv", index=False)

    if bonus_results:
        pd.DataFrame([{k: v for k, v in row.items() if k != "log_history"} for row in bonus_results]).to_csv(
            OUTPUT_DIR / "results" / "bonus_experiment_summary.csv",
            index=False,
        )

    base_eval_model, eval_tokenizer = load_base_model(max_seq_length)
    ft_model = PeftModel.from_pretrained(base_eval_model, OUTPUT_DIR / "adapters" / "r16")
    qualitative = []
    for prompt in TEST_PROMPTS:
        base_text = generate_text(base_eval_model, eval_tokenizer, prompt)
        ft_text = generate_text(ft_model, eval_tokenizer, prompt)
        qualitative.append({"prompt": prompt, "base": base_text[:500], "finetuned": ft_text[:500]})
    pd.DataFrame(qualitative).to_csv(OUTPUT_DIR / "results" / "qualitative_comparison.csv", index=False)

    total_minutes = summary.loc[summary["tag"] != "base", "train_time_min"].sum()
    if bonus_results:
        total_minutes += sum(item["train_time_min"] for item in bonus_results)
    report = {
        "student_name": "Nguyễn Triệu Gia Khánh",
        "student_id": "2A202600225",
        "submission_date": "2026-07-05",
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_meta": meta,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
        "training_cost_usd": (total_minutes / 60) * GPU_COST_USD_PER_HOUR,
        "best_required_tag": summary[summary["tag"].isin(["r8", "r16", "r64"])].sort_values("eval_perplexity").iloc[0]["tag"],
    }
    (OUTPUT_DIR / "results" / "run_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if PUSH_TO_HUB:
        push_results(report["best_required_tag"])

    print("Training complete.")
    print((OUTPUT_DIR / "results" / "rank_experiment_summary.csv").read_text(encoding="utf-8"))
    print((OUTPUT_DIR / "results" / "run_metadata.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
