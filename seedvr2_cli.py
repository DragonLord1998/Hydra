#!/usr/bin/env python3
"""Thin CLI wrapper bridging Hydra's upscale endpoint to SeedVR2 7B."""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Single-process distributed setup (SeedVR2 expects torchrun-like env)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="Path to source image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--output_format", default="png")
    parser.add_argument("--resolution", type=int, default=1080,
                        help="Target resolution (shortest side)")
    parser.add_argument("--dit_offload_device", default="cpu")
    parser.add_argument("--vae_offload_device", default="cpu")
    args = parser.parse_args()

    from PIL import Image

    img = Image.open(args.input_image)
    w, h = img.size
    img.close()

    # Scale so shortest side == resolution, dimensions divisible by 16
    scale = args.resolution / min(w, h)
    res_w = round(w * scale / 16) * 16
    res_h = round(h * scale / 16) * 16

    # SeedVR2 reads all files from an input directory
    tmp_in = tempfile.mkdtemp(prefix="seedvr2_in_")
    shutil.copy2(args.input_image,
                 os.path.join(tmp_in, Path(args.input_image).name))

    try:
        from projects.inference_seedvr2_7b import configure_runner, generation_loop

        runner = configure_runner(sp_size=1)
        generation_loop(
            runner,
            video_path=tmp_in,
            output_dir=args.output,
            res_h=res_h,
            res_w=res_w,
            sp_size=1,
        )
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)


if __name__ == "__main__":
    main()
