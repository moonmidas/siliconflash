from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from omlx.dflash.mlx_native_drafter import load_mlx_dflash_draft_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the MLX-native DFlash drafter port")
    parser.add_argument("--model", default="z-lab/Qwen3.5-9B-DFlash")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    model = load_mlx_dflash_draft_model(args.model)
    noise = mx.zeros((1, args.seq_len, model.args.hidden_size), dtype=mx.float16)
    target_hidden = mx.zeros((1, args.ctx_len, model.args.conditioning_input_dim), dtype=mx.float16)

    # warmup
    for _ in range(2):
        out = model(noise_embedding=noise, target_hidden=target_hidden)
        mx.eval(out)

    start = time.perf_counter()
    for _ in range(args.iters):
        out = model(noise_embedding=noise, target_hidden=target_hidden)
        mx.eval(out)
    elapsed = time.perf_counter() - start

    print(json.dumps({
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "iters": args.iters,
        "seconds_per_iter": elapsed / args.iters,
    }, indent=2))


if __name__ == "__main__":
    main()
