"""Cold/warm end-to-end measurements, including transfers and full rendering."""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from utils.seamless_loop import LoopOptions, optimize_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quality", choices=("fast", "balanced", "high"), default="balanced")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--max-fade", type=int, default=12)
    parser.add_argument("--manual-overlap", type=int, default=-1, help="Optional forced fade to benchmark real full-resolution blending.")
    parser.add_argument("--input-tensor", type=Path, help="Trusted .pt tensor in temporal NHWC order; no video decoding.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.frames, args.height, args.width, args.runs) < 1:
        parser.error("Dimensions, frames and runs must be positive.")
    if args.input_tensor:
        images = torch.load(args.input_tensor, map_location="cpu", weights_only=True)
    else:
        # Deterministic moving texture. Fixture creation is outside measured work.
        generator = torch.Generator().manual_seed(7)
        base = torch.rand(args.height, args.width, 3, generator=generator)
        images = torch.stack([base.roll(i, dims=1) for i in range(args.frames)])
    opts = LoopOptions(device=args.device, quality=args.quality, max_trim_start=args.max_trim,
                       max_trim_end=args.max_trim, max_fade=args.max_fade, manual_overlap=args.manual_overlap)
    records = []
    for run in range(args.runs + 1):
        result, report = optimize_loop(images, opts)
        records.append({"run": "cold" if run == 0 else f"warm_{run}", "timings": report["timings"],
                        "selected": report["selected"], "candidate_count": report["candidate_count"],
                        "output_frames": len(result), "backend": report["backend"]})
        del result
    payload = {"fixture": "provided tensor" if args.input_tensor else "seed 7 translating random RGB texture",
               "shape": list(images.shape), "options": vars(args) | {"input_tensor": str(args.input_tensor) if args.input_tensor else None,
                                                                      "output": str(args.output) if args.output else None},
               "cpu_threads": torch.get_num_threads(), "records": records,
               "warm_median_seconds": {key: statistics.median(r["timings"][key] for r in records[1:])
                                       for key in records[0]["timings"]}}
    content = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
