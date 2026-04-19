from __future__ import annotations

import argparse
import json
from pathlib import Path

from omlx.model_settings import ModelSettingsManager
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig


async def main_async(args) -> None:
    base_path = Path(args.base_path)
    settings = ModelSettingsManager(base_path)
    current = settings.get_settings(args.model_id)
    current.dflash_enabled = args.enable_dflash
    current.dflash_draft_model = args.draft_model
    current.dflash_use_mlx_native_drafter = args.use_mlx_native_drafter
    settings.set_settings(args.model_id, current)

    pool = EnginePool(
        max_model_memory=48 * 1024**3,
        scheduler_config=SchedulerConfig(max_num_seqs=1),
    )
    pool._settings_manager = settings
    try:
        pool.discover_models(args.model_dir)
        engine = await pool.get_engine(args.model_id)
        stats = engine.get_stats()
        print(json.dumps({
            "model": args.model_id,
            "dflash": stats.get("dflash"),
        }, indent=2, default=str))
    finally:
        await pool.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DFlash runtime status for OmniCoder/oMLX")
    parser.add_argument("--base-path", default=".omlx-dflash-smoke")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model-id", default="omni9b-phase2prime")
    parser.add_argument("--draft-model", default="z-lab/Qwen3.5-9B-DFlash")
    parser.add_argument("--enable-dflash", action="store_true")
    parser.add_argument("--use-mlx-native-drafter", action="store_true")
    args = parser.parse_args()

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
