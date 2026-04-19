from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import requests


def run_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    dflash: bool = False,
    ignore_eos: bool = False,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        # Always pass an explicit dflash toggle so benchmark control runs cannot
        # inherit model-level dflash_enabled settings implicitly.
        "dflash": bool(dflash),
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if thinking_budget is not None:
        payload["thinking_budget"] = thinking_budget
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=1800)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tok_s = (completion_tokens / elapsed) if elapsed > 0 else 0.0
    message = data["choices"][0]["message"]
    preview_text = message.get("content")
    if preview_text is None:
        preview_text = message.get("reasoning_content") or ""

    dflash_usage = {
        key: value
        for key, value in usage.items()
        if key.startswith("dflash_") and isinstance(value, (int, float, bool, str))
    }

    return {
        "elapsed_s": elapsed,
        "tok_s": tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "text_preview": preview_text[:200],
        **dflash_usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple SiliconFlash baseline benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="omni9b-phase2prime")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument(
        "--prompt",
        default=(
            "You are helping with a performance engineering task. "
            "Explain, in detail, how speculative decoding can accelerate "
            "autoregressive generation on Apple Silicon."
        ),
    )
    args = parser.parse_args()

    results = [
        run_once(
            args.base_url,
            args.model,
            args.prompt,
            args.max_tokens,
            dflash=args.dflash,
            ignore_eos=args.ignore_eos,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )
        for _ in range(args.runs)
    ]
    summary = {
        "runs": args.runs,
        "model": args.model,
        "mean_tok_s": mean(r["tok_s"] for r in results),
        "mean_elapsed_s": mean(r["elapsed_s"] for r in results),
        "mean_completion_tokens": mean(r["completion_tokens"] for r in results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
