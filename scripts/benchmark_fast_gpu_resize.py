"""Development benchmark: python scripts/benchmark_fast_gpu_resize.py --help."""

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.resize_kernels import resize_image_batch


CASES = ((1,1920,1080,1280,720), (81,1920,1080,1280,720),
         (121,1920,1080,1024,576), (121,1024,576,1920,1080))
METHODS = ("bilinear","bicubic","area","lanczos2","lanczos3","lanczos4","mitchell","catmull_rom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("fp32","fp16","bf16"), default="fp32")
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Use tiny test sizes instead of video-sized batches")
    parser.add_argument("--direct", action="store_true", help="Also measure direct interpolate for native methods")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.chunk_size < 0:
        parser.error("repeats >= 1, warmup >= 0 and chunk-size >= 0 are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; pass --device cpu (prefer --smoke for CPU)")
    device = torch.device(args.device)
    dtype = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[args.dtype]
    cases = ((1,192,108,128,72),(81,192,108,128,72),(121,192,108,104,56),(121,104,56,192,108)) if args.smoke else CASES
    print(f"torch={torch.__version__}; device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}; "
          f"dtype={args.dtype}; antialias=True; chunk_size={args.chunk_size}; repeats={args.repeats}",flush=True)
    print("case,method,median_ms,peak_extra_MiB",flush=True)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for batch,sw,sh,tw,th in cases:
            image = torch.rand((batch,sh,sw,3),device=device,dtype=dtype)
            for method in METHODS:
                runs = [(method,lambda:resize_image_batch(image,tw,th,method=method,chunk_size=args.chunk_size)[0])]
                if args.direct and method in ("bilinear","bicubic","area"):
                    kwargs = {"align_corners":False,"antialias":True} if method != "area" else {}
                    def direct():
                        x = image.permute(0,3,1,2)
                        if dtype != torch.float32 and method in ("bilinear","bicubic"):
                            x = x.float()
                        return F.interpolate(x,size=(th,tw),mode=method,**kwargs)
                    runs.append(("direct_"+method,direct))
                for label,run in runs:
                    for _ in range(args.warmup):
                        output = run()
                        del output
                    synchronize()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        baseline = torch.cuda.memory_allocated(device)
                    timings = []
                    for _ in range(args.repeats):
                        synchronize()
                        start = time.perf_counter()
                        output = run()
                        synchronize()
                        timings.append((time.perf_counter()-start)*1000)
                        expected = (batch,3,th,tw) if label.startswith("direct_") else (batch,th,tw,3)
                        assert output.shape == expected
                        del output
                    peak = (torch.cuda.max_memory_allocated(device)-baseline)/2**20 if device.type == "cuda" else float("nan")
                    print(f"{batch}x{sw}x{sh}->{tw}x{th},{label},{statistics.median(timings):.3f},{peak:.1f}",flush=True)
            del image
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
