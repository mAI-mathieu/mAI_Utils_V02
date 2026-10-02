# Fast GPU Resize implementation and measurements

`MAIFastGPUResize` is a small standard ComfyUI node wrapper in
`nodes/fast_gpu_resize.py`. It resolves explicit CUDA selection using ComfyUI's
device helper at execution time. `utils/resize_kernels.py` contains geometry,
method selection, dtype handling, sampling and batching with no ComfyUI dependency.
Registration adds only `MAIFastGPUResize` to the existing root mappings.

BHWC input is permuted to BCHW as a view; no unconditional contiguous copy is
made. Native methods call `torch.nn.functional.interpolate`. Custom methods
use two one-dimensional passes. The pass with the smaller intermediate runs
first, which reduces memory for mixed-axis scaling. Unchanged axes are skipped.

## Custom reconstruction

For each output pixel, the source coordinate is
`(output_index + 0.5) * source_size / target_size - 0.5`.
Lanczos uses `sinc(x) * sinc(x/a)` inside radius `a` (2/3/4), zero outside.
Mitchell and Catmull–Rom use the BC cubic with `(B,C)=(1/3,1/3)` and `(0,1/2)`.
When antialiasing a shrinking axis, distances are divided by source/target and
support is widened by the same factor. Every output's weights are normalized.
Indices clamp to the source boundary; distances are evaluated before clamping,
so the edge is replicated rather than wrapped.

Tables are computed on the processing device once per changed axis per call,
then shared across every frame and batch chunk. No persistent GPU cache is kept.
FP16/BF16 custom calculations use FP32 positions, weights and accumulation;
FP64 output uses FP64 work. Only final reconstruction is clamped to [0,1].
The output is cast to the requested dtype and permuted back to BHWC.

A small Python loop applies kernel taps to whole tensor batches using
`index_select` and `addcmul_`. It holds only one sampled plane at a time rather
than allocating a tensor with a taps dimension across all image pixels. Chunking
uses slices along the batch axis and writes into a preallocated output, preserving
order and avoiding a full-output `cat` copy. There are no per-frame or per-pixel
Python loops and no PIL, OpenCV, NumPy, SciPy or CPU image-conversion path.

Auto chunking uses a conservative 512 MiB estimate for custom working data,
including source casts, intermediate planes and geometry/output casts. This is
an estimate, not a strict allocator limit. The full input/output are additional;
one very large frame can exceed the estimate. Native methods process the whole
batch at chunk size 0, so explicit chunking can also reduce their working memory.

GPU execution now uses `utils/gpu_memory.py` to release unused CUDA cache before
work and after success/failure/cancellation. It estimates full output plus chunk
scratch and requests ComfyUI model offloading on the processing GPU only when
driver-visible free memory is below that budget. Source CUDA cache is also
released when explicitly moving results to CPU. Models may reload later, and
live input/output tensors remain allocated. CPU operations and matching
device/dtype identity bypasses skip CUDA cleanup. No node sockets changed.
Cleanup cannot free other processes' memory or guarantee that a huge output fits.

## CUDA benchmark, 2026-10-02

NVIDIA GeForce RTX 5090, PyTorch 2.7.1+cu128, FP32 RGB, Windows.
Antialias true, chunk size 0 (automatic custom chunking), two warmups and median
of five measured runs. Each timed run includes table construction, allocation,
filtering, final clipping and output processing. CUDA synchronization occurs
before/after timings in the script. These historical measurements predate the
memory cleanup, which now synchronizes inside GPU execution too. Data is already on CUDA; transfers
from a CPU IMAGE are excluded. No other test process ran during this measurement.

Command:

```text
python scripts/benchmark_fast_gpu_resize.py --direct --warmup 2 --repeats 5
```

Elapsed time in milliseconds for the complete batch:

| Method | 1 frame 1080p→720p | 81 frames 1080p→720p | 121 frames 1080p→1024×576 | 121 frames 1024×576→1080p |
|---|---:|---:|---:|---:|
| bilinear | 0.103 | 8.509 | 9.940 | 14.202 |
| bicubic | 0.147 | 11.394 | 14.527 | 24.120 |
| area | 0.071 | 3.133 | 3.063 | 5.417 |
| lanczos2 | 1.671 | 93.612 | 133.295 | 114.498 |
| lanczos3 | 2.021 | 132.858 | 191.493 | 155.848 |
| lanczos4 | 2.144 | 166.553 | 234.967 | 195.646 |
| mitchell | 1.958 | 93.173 | 145.734 | 114.532 |
| catmull_rom | 2.011 | 94.744 | 129.608 | 113.759 |

Direct `interpolate` comparison (same CUDA input/layout/antialias settings):

| Direct method | 1 frame down | 81 frames down | 121 frames down | 121 frames up |
|---|---:|---:|---:|---:|
| bilinear | 0.085 | 7.889 | 9.876 | 14.515 |
| bicubic | 0.156 | 9.949 | 13.515 | 18.904 |
| area | 0.052 | 2.616 | 2.933 | 5.464 |

Direct bicubic excludes the node's final range clamp. Sub-millisecond timings
are sensitive to Python/launch overhead and GPU clock changes; these are local
development measurements, not a universal speed guarantee.

Peak additional allocated GPU memory (includes output; excludes input):

| Path | 1 frame down | 81 frames down | 121 frames down | 121 frames up |
|---|---:|---:|---:|---:|
| Native bilinear/bicubic | 45 MiB | 3631 MiB | 4506 MiB | 6561 MiB |
| Area | 11 MiB | 854 MiB | 817 MiB | 2872 MiB |
| Custom separable, approximately | 37 MiB | 1039 MiB | 974 MiB | 3113 MiB |

The custom implementation trades speed for bounded temporary memory. Native
PyTorch remains much faster. Chunk size and FP32 promotion can change both time
and memory. The script accepts `--dtype fp16/bf16`, `--chunk-size 32` and `--smoke`.
CPU development smoke command:
`python scripts/benchmark_fast_gpu_resize.py --device cpu --smoke`.

## Verification and limits

`python -m pytest -q`: **811 passed**. The CPU benchmark smoke run also passed.

Tests cover all algorithms, exact geometry, rounded dimensions, fit/fill and odd
padding, identity object reuse, unchanged input pixels, 121-frame ordering,
chunk equivalence, constant colors, replicated edges, high-frequency antialias
suppression, independent scalar kernel references, transposed axis equivalence,
tiny/extreme-aspect images, and native `interpolate` equivalence. CPU/CUDA tests
cover FP32/FP16/BF16 with antialias on/off. Node tests verify sockets, defaults,
device transfers, ComfyUI helper selection and no CUDA calls at import time.

The pack imports in the available Python environment, with all existing root
mapping entries preserved. Its existing optional CropAndStitch node cannot load
in that environment because `comfy_aimdo` is missing. The local ComfyUI venv
launcher references an unavailable base Python installation, so execution inside
that venv and live ComfyUI UI validation could not be performed. No runtime or
ComfyUI core files were changed. Linux portability follows standard torch APIs
and pathlib in the benchmark; a Linux host was not available for validation.

Alpha is resized independently without premultiplication. CPU `auto` stays on
CPU. Full output allocation, extreme fill intermediates and substantial
downscale tap counts remain limits even with chunking. No models are unloaded
or moved by this node.
