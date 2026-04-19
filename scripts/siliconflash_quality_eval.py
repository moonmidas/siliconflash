from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from statistics import mean
from typing import Any

import requests

_INT_RE = re.compile(r"-?\d+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _request_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    dflash: bool,
    ignore_eos: bool,
    enable_thinking: bool,
    thinking_budget: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "dflash": bool(dflash),
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if thinking_budget is not None:
        payload["thinking_budget"] = thinking_budget

    start = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=1800
    )
    elapsed_s = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()

    usage = data.get("usage", {})
    choice = data["choices"][0]
    message = choice.get("message", {})
    text = message.get("content") or message.get("reasoning_content") or ""

    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    tok_s = (completion_tokens / elapsed_s) if elapsed_s > 0 else 0.0

    dflash_usage = {
        key: value
        for key, value in usage.items()
        if key.startswith("dflash_") and isinstance(value, (int, float, bool, str))
    }

    return {
        "text": text,
        "elapsed_s": elapsed_s,
        "tok_s": tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": choice.get("finish_reason"),
        "dflash_usage": dflash_usage,
    }


def _analyze_repetition(text: str) -> dict[str, Any]:
    tokens = re.findall(r"\S+", text.lower())
    if not tokens:
        return {
            "token_count": 0,
            "repeat_ratio": 0.0,
            "max_run": 0,
            "short_token_max_ratio": 0.0,
            "punct_token_ratio": 0.0,
            "degenerate": True,
        }

    repeats = 0
    max_run = 1
    run = 1
    for prev_tok, tok in zip(tokens, tokens[1:]):
        if tok == prev_tok:
            repeats += 1
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1

    repeat_ratio = repeats / max(1, len(tokens) - 1)

    counts = Counter(tokens)
    short_token_ratios = [
        count / len(tokens) for tok, count in counts.items() if len(tok) <= 3
    ]
    short_token_max_ratio = max(short_token_ratios) if short_token_ratios else 0.0

    punct_tokens = [tok for tok in tokens if re.fullmatch(r"[^\w\s]+", tok)]
    punct_token_ratio = len(punct_tokens) / len(tokens)

    token_count = len(tokens)
    degenerate = (
        max_run >= 8
        or (token_count >= 32 and repeat_ratio >= 0.22)
        or (token_count >= 24 and short_token_max_ratio >= 0.45)
        or (token_count >= 24 and punct_token_ratio >= 0.45)
    )

    return {
        "token_count": len(tokens),
        "repeat_ratio": repeat_ratio,
        "max_run": max_run,
        "short_token_max_ratio": short_token_max_ratio,
        "punct_token_ratio": punct_token_ratio,
        "degenerate": degenerate,
    }


def _validate_addition(text: str) -> tuple[bool, str]:
    match = _INT_RE.search(text)
    if not match:
        return False, "missing_integer"
    value = int(match.group())
    return (value == 1015, f"value={value}")


def _validate_prime(text: str) -> tuple[bool, str]:
    cleaned = re.sub(r"[^a-zA-Z]", "", text.strip().lower())
    if not cleaned:
        return False, "missing_answer"
    return (cleaned.startswith("no"), f"answer={cleaned[:12]}")


def _validate_json(text: str) -> tuple[bool, str]:
    match = _JSON_RE.search(text)
    if not match:
        return False, "missing_json"
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return False, f"json_decode_error={exc.msg}"
    ok = (
        isinstance(obj, dict)
        and obj.get("alpha") == 1
        and obj.get("beta") == 2
        and obj.get("gamma") == 3
    )
    return ok, f"keys={sorted(obj.keys()) if isinstance(obj, dict) else 'non_dict'}"


def _validate_probe(text: str) -> tuple[bool, str]:
    analysis = _analyze_repetition(text)
    coherent = not analysis["degenerate"] and analysis["token_count"] >= 24
    return coherent, (
        f"degenerate={analysis['degenerate']}"
        f",max_run={analysis['max_run']}"
        f",repeat_ratio={analysis['repeat_ratio']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small fixed accuracy-drift eval suite")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument(
        "--probe-prompt",
        default=(
            "Write a detailed explanation of how speculative decoding works, including draft "
            "proposal, verification, acceptance criteria, and cache management tradeoffs on "
            "Apple Silicon."
        ),
    )
    args = parser.parse_args()

    cases = [
        {
            "id": "arith_addition",
            "prompt": "Answer with only the final integer: 947 + 68 = ?",
            "max_tokens": 16,
            "validator": _validate_addition,
        },
        {
            "id": "prime_check",
            "prompt": "Answer with exactly one word, yes or no: Is 221 a prime number?",
            "max_tokens": 16,
            "validator": _validate_prime,
        },
        {
            "id": "json_exact",
            "prompt": (
                "Output strict JSON only with exactly these integer key/value pairs: "
                '{"alpha": 1, "beta": 2, "gamma": 3}'
            ),
            "max_tokens": 48,
            "validator": _validate_json,
        },
        {
            "id": "coherence_probe",
            "prompt": args.probe_prompt,
            "max_tokens": 256,
            "validator": _validate_probe,
        },
    ]

    results: list[dict[str, Any]] = []
    pass_count = 0
    degenerate_cases = 0

    for case in cases:
        run = _request_once(
            base_url=args.base_url,
            model=args.model,
            prompt=case["prompt"],
            max_tokens=case["max_tokens"],
            dflash=args.dflash,
            ignore_eos=args.ignore_eos,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )
        rep = _analyze_repetition(run["text"])
        passed, check_info = case["validator"](run["text"])
        pass_count += int(passed)
        degenerate_cases += int(rep["degenerate"])

        result = {
            "id": case["id"],
            "pass": bool(passed),
            "check_info": check_info,
            "finish_reason": run["finish_reason"],
            "elapsed_s": run["elapsed_s"],
            "tok_s": run["tok_s"],
            "prompt_tokens": run["prompt_tokens"],
            "completion_tokens": run["completion_tokens"],
            "token_count": rep["token_count"],
            "repeat_ratio": rep["repeat_ratio"],
            "max_run": rep["max_run"],
            "degenerate": rep["degenerate"],
            "text_preview": run["text"][:240],
        }
        if run["dflash_usage"]:
            result["dflash_usage"] = run["dflash_usage"]
        results.append(result)

    case_count = len(results)
    summary = {
        "model": args.model,
        "dflash": bool(args.dflash),
        "case_count": case_count,
        "pass_count": pass_count,
        "failed_cases": case_count - pass_count,
        "pass_rate": (pass_count / case_count) if case_count else 0.0,
        "degenerate_cases": degenerate_cases,
        "repeat_ratio_mean": mean(r["repeat_ratio"] for r in results) if results else 0.0,
        "repeat_ratio_max": max((r["repeat_ratio"] for r in results), default=0.0),
        "max_run_max": max((r["max_run"] for r in results), default=0),
        "mean_tok_s": mean(r["tok_s"] for r in results) if results else 0.0,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
