#!/usr/bin/env python3
"""Generate GPT-2 text with the CUDA executable using greedy decoding."""

import argparse
from pathlib import Path
import subprocess

from transformers import AutoTokenizer

from gpt2_checkpoint import MODEL_ID, REVISION

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--engine", type=Path, default=ROOT / "build" / "mgie_generate")
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "model_weights")
    args = parser.parse_args()
    if not args.prompt.strip() or not 0 < args.max_new_tokens <= 1024:
        parser.error("provide a nonempty prompt and --max-new-tokens between 1 and 1024")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not 0 < len(prompt_ids) < 1024:
        parser.error("the prompt must contain between 1 and 1023 GPT-2 tokens")
    try:
        run = subprocess.run(
            [str(args.engine.resolve()), str(args.weights), str(args.max_new_tokens),
             *map(str, prompt_ids)],
            check=True, stdout=subprocess.PIPE, text=True,
        )
    except OSError as error:
        parser.exit(1, f"{error}\nBuild mgie_generate or select it with --engine.\n")
    except subprocess.CalledProcessError as error:
        return error.returncode if error.returncode > 0 else 1

    token_ids = [int(value) for value in run.stdout.split()]
    print(tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
