from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI
import os


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "openai_checks"
DEFAULT_PROMPT = (
    "Say hello in Vietnamese and give one short sentence about why GPT-4o mini is a cost-efficient API model."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real API smoke test for gpt-4o-mini.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the OpenAI Responses API.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the response artifact.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    if not api_key or api_key == "paste_your_openai_api_key_here":
        print("OPENAI_API_KEY is missing in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=args.prompt,
    )

    output_text = getattr(response, "output_text", "").strip()
    if not output_text:
        output_text = json.dumps(response.model_dump(), ensure_ascii=False, indent=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = args.output_dir / f"gpt4o-mini-check-{timestamp}.json"

    artifact = {
        "timestamp_utc": timestamp,
        "requested_model": model,
        "prompt": args.prompt,
        "response_text": output_text,
        "response": response.model_dump(),
    }
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else {}
    created_at = getattr(response, "created_at", None)
    completed_at = getattr(response, "completed_at", None)
    latency = None
    if created_at is not None and completed_at is not None:
        latency = completed_at - created_at

    print(f"Requested model: {model}")
    print(f"Resolved model: {response.model}")
    if latency is not None:
        print(f"Latency: {latency:.2f}s")
    if usage:
        print(
            "Usage: "
            f"{usage.get('input_tokens', 0)} input, "
            f"{usage.get('output_tokens', 0)} output, "
            f"{usage.get('total_tokens', 0)} total tokens"
        )
    print("Response:")
    print(output_text)
    print(f"Saved artifact: {artifact_path}")


if __name__ == "__main__":
    main()
