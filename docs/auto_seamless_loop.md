# Auto Seamless Loop

`MAIAutoSeamlessLoop` / **mAI auto seamless loop**, under `mAI / Image`.
The independent module `utils/seamless_loop.py` needs only ComfyUI's existing
PyTorch. It makes no network calls and needs no file, codec, FFmpeg, flow model,
or GPU video hardware. Loader/encoder nodes in the example are separate IO steps.

## Contract and controls

Input is a finite floating IMAGE tensor `[N,H,W,C]` in `[0,1]`; temporal order is
batch order. N is read from shape. RGB, RGBA and grayscale channels are preserved;
the first up to three channels provide color information. Premultiplied alpha
is not inferred. Resolution and dtype are preserved; all scoring uses FP32 with
autocast disabled. Input is never mutated. FPS must be finite and positive.

Basic controls: `fps=24`, `device=auto`, `quality=balanced`, `max_trim_start=8`,
`max_trim_end=8`, `max_fade=12`, `min_retained_percent=70`. Limits are inclusive
frame counts, not seconds. Retention applies to the **final** cycle, including
overlap loss: `Nout >= ceil(N * percent / 100)`. Invalid manual constraints raise
a useful error. Search is capped at 50,000 valid candidates; reduce limits if hit.

Advanced controls expose blend space, seven objective weights, satisfactory
threshold (0.025), tie tolerance (0.002), and three manual overrides. Each override
defaults to -1 (auto); nonnegative values fix that dimension **within the basic
limits**. Set all three for a fully manual construction. A manual override may
exclude the unchanged candidate. Missing optional controls use Python defaults.
The frontend only hides widgets; their values persist, and all seven output
sockets remain available even if the frontend extension fails.

Stable output order: `images:IMAGE`, `fps:FLOAT`, `frame_count:INT`,
`trim_start:INT`, `trim_end:INT`, `overlap:INT`, `report:STRING`.
The JSON includes exact durations, objective terms, selected backend, GPU smoke
test, fallback attempts, timing phases, alternative candidates and uncertainty.
FPS is never changed. Original audio is not edited and is intentionally unconnected
in the example, since shortening or rotating the cycle also changes audio alignment.

## Exact construction

Let `T = images[a:N-b]`, `L = len(T)`, and `K` be overlap. For `K=0`, output is a
fresh copy of T with no rotation, removal of duplicates, or endpoint adjustment.
For `K>0`, require `L >= 2K+1` (at least one unblended middle frame). Define

```text
w[j] = j / (K-1),                    K >= 2, 0 <= j < K
w[0] = 1/2,                         K = 1
B[j] = (1-w[j]) T[L-K+j] + w[j] T[j]
O = concat(T[K:L-K], B)              len(O) = L-K
```

The inclusive weights keep exact tail/head endpoints for K>=2. The first bridge
frame follows `T[L-K-1]` with the original next frame `T[L-K]`. The final bridge
frame is `T[K-1]`, followed at the last-to-first boundary by `O[0]=T[K]`.
These connections are checked with neighboring temporal differences, as well as
internal bridge frames. K=1 uses a midpoint and both connections are still scored.
K=2 contains only unmixed endpoints and may act as a cut with trimming; it is not
assumed to be smooth. No extra black, repeated or freeze frame is inserted.
Existing natural duplicates are retained unless the objective favors a repair.
The phase rotates by K frames for positive overlap; there is no reversal/ping-pong.

Default `srgb` means interpolation directly in the encoded pixel values. `linear`
decodes the sRGB transfer function, blends in linear light, then encodes back.
For RGB channel x, decode `x/12.92` below 0.04045, otherwise
`((x+0.055)/1.055)^2.4`; encode `12.92*x` below 0.0031308, otherwise
`1.055*x^(1/2.4)-0.055`. Additional channels (including alpha) blend numerically.
This assumes sRGB color, not scene-linear, PQ, HLG or HDR input. Linear-light proxy
scoring approximates blending after full-resolution conversion; nonlinear color
conversion and downsampling do not commute exactly.

## Transparent heuristic objective

Every valid integer triple `(a,b,K)` is searched jointly, including unchanged and
all valid zero-fade choices. This is an exhaustive bounded **proxy** grid followed
by a shortlist refinement, not a guarantee of the global full-resolution optimum.

Stage 1 uses aspect-preserving area proxies with long edge 64 / 96 / 128 for
fast / balanced / high, without upscaling. Candidates are grouped by K and scored
in tensor batches. Stage 2 uses original boundary frames at long edge 128 / 256 /
384 and RGB information; refine the top 4 / 8 / 16, unchanged if permitted, plus
the cheapest satisfactory repairs (at most another 4 / 8 / 16). Reference quartiles
are recomputed from consecutive pairs/triples around 16 uniformly sampled starts.
Only these needed boundary/reference frames are resized for refinement.

Each candidate supplies the **actual constructed** periodic window: three frames
before the bridge, its K frames, and three after. For zero fade use three frames
on either side of the seam. Short cycles use cyclic indices, so even windows with
repeated context reflect actual playback. Analyze all junctions and the bridge;
matching endpoints alone never establish a good loop.

For an error map E and detail map D, use
`W = 1 + 3*clamp(D/(mean(D)+1e-6), max=20)` and
`R(E,D) = .7*sum(E*W)/sum(W) + .3*max(mean(E) in a 4x4 region grid)`.
Detail is luminance finite-difference edges plus local temporal activity. This
gives small textured/moving regions more influence than blank backgrounds.
It is a salience approximation, **not face/hand detection**; tiny details below
proxy resolution, textureless subjects, occlusions and arbitrary alpha can fool it.

Let d be signed adjacent-frame differences, u the corresponding R(abs(d)),
v the R(abs(d[t+1]-d[t])), e the absolute per-channel mean of d, and
`P(z)=.75*max(z)+.25*mean(z)` over the window. Let r=fps/24, and q_u, q_e, q_v
be robust upper quartiles over original consecutive frame differences/accelerations.
Upper quartiles tolerate ordinary motion when a clip contains long pauses, while
a single corrupt exposure frame cannot establish a large normal-change baseline.
The reported metrics are:

| Term | Definition | Default weight |
|---|---|---:|
| appearance | `max(0,P(u)-1.5*q_u)*r` | 1.0 |
| exposure | `max(0,P(e)-1.5*q_e)*r` | 0.5 |
| motion | `max(0,P(v)-1.5*q_v)*r^2` | 0.6 |
| smoothness | `max(abs(u[t+1]-u[t]))*r` | 0.4 |
| ghosting | P of tail/head regional mismatch plus 0.25 edge mismatch, times `4*w*(1-w)` | 1.0 |
| duration | `(N-Nout)/N` | 0.25 |
| fade | `K/N` | 0.12 |
| activity_loss | excess proportion of original step activity removed by trims, times `q_u*r` | duration weight |

Ghost edge differences are scaled by proxy long edge / 96 to use normalized image
coordinates. Spatial reductions are means/weighted means, not resolution-dependent
sums. Temporal peak and mean have fixed contributions, so adding bridge frames
cannot dilute a bad connection. First/second temporal derivatives scale with frame
rate / squared frame rate. Pixel errors are in the input's `[0,1]` units; every
metric is nonnegative and the final score is their weighted sum. Activity loss uses
stage-1 activity prefix sums and refinement q_u; a one-frame exposure outlier is
not treated as essential action when normal motion is zero. It is approximate
activity retention, not semantic usefulness. Timing and score changes across
quality levels are expected; this is not a perceptually calibrated metric.

Selection is explicitly satisficing: unchanged wins when its refined score meets
the threshold. Otherwise, among satisfactory refined choices minimize total lost
frames `a+b+K`, then K, then score. If none is satisfactory select the lowest
refined score and report **low confidence**. Report near ties within tie tolerance,
including alternatives with lower scores when a cheaper satisfactory repair wins.
Duration/activity/fade penalties discourage excessive trimming, static subsequence
selection, and long fades. For demanding cases reduce satisfactory_score, adjust
weights, or override the dimensions. High mode increases proxy/refine resolution
and shortlist size; no optical-flow backend is installed or used.

## Device and memory behavior

Auto respects `comfy.model_management.get_torch_device()`: a configured CPU stays
CPU; CUDA is checked for availability and tested by an actual FP32 reduction and
matrix multiplication before search. Standalone auto detects CUDA. A configured
non-CPU/non-CUDA backend uses CPU with a reason. There are no CUDA queries at module
import, global precision/thread/allocator changes, or custom architecture kernels.

Reuse installed PyTorch. Selected CUDA must have working kernels and a compatible
driver for the device. Diagnostics record torch version, CUDA runtime, device
name/capability and actual smoke result; a capability number alone proves nothing.
RTX PRO 6000 Blackwell workstation/server variants, RTX 5090, B200 and B300 use the
same ordinary operations. NVIDIA lists capabilities 12.0 / 12.0 / 10.0 / 10.3,
respectively ([official GPU table](https://developer.nvidia.com/cuda/gpus)).
CPU-only installations need no CUDA build. Follow
[PyTorch's compatibility selector](https://pytorch.org/get-started/locally/) when
maintaining ComfyUI; this node never installs or upgrades torch.

Proxy frames resize on the input device before upload. Cached proxies and edge
features serve candidate batches; full-resolution frames are blended **only for
the winner**. Candidate/render chunks use at most a conservative share of currently
free VRAM (10%, capped at 128 MiB for estimated working temporaries), accounting for
loaded ComfyUI models. Refinement further reduces the candidate chunk. The budget
is an estimate, not a hard allocation bound. Input/output storage and the small
proxy cache still need space, and at least one frame/candidate must fit.

CUDA OOM unwinds temporary tensors, halves candidate/render chunks, and retries
twice; auto then retries on CPU. Forced CUDA retries twice and gives an actionable
failure. Compatibility fallback only catches specifically identified CUDA kernel
or driver failures; bugs, invalid input and cancellation propagate. Tensors become
reusable by the allocator when references are dropped; no global cache flush or
unloading other nodes' models is performed. CPU fallback needs a CPU output policy;
with `--gpu-only`, fail clearly if GPU output cannot be safely returned.

The ComfyUI adapter returns on `intermediate_device()` (normally CPU); standalone
defaults to CPU and allows explicit output_device. The independent optimizer uses
inference mode. Cancellation is checked between validation/proxy/search/refinement/
render chunks. Progress tracks search and rendering; reduced-chunk retries may
restart progress. There is no hidden retiming or preserve-duration mode.

## Installation and workflow

1. Place this repository under `ComfyUI/custom_nodes/mAI_Utils_V02` as usual.
2. Use the same Python environment as ComfyUI. The new node adds **no dependencies**;
   the pack's existing IO nodes use its existing `requirements.txt`.
3. Restart ComfyUI and refresh the page; add **mAI auto seamless loop**.
4. Connect an ordered IMAGE batch and its real FPS. Connect output images/FPS to
   your usual video-combine node, or core CreateVideo followed by SaveVideo.
5. Load `examples/auto_seamless_loop.json` in ComfyUI, put `loop_input.mp4` in its
   input folder and select it in the loader. The example uses the pack's software
   `libx264` output and encodes three repetitions so you can inspect boundaries.
   Decoding/encoding are separate from core tensor operation. Software encoding
   works on data-center GPUs without NVENC; even output dimensions are recommended
   because the existing combine node pads odd sizes to even.

For a headless API server, `examples/auto_seamless_loop_api.json` is the prompt
graph. POST `{"prompt": <contents of that file>}` to `/prompt`, then poll
`/history/<prompt_id>` for the output. Select a different input filename if needed.
The report output can connect to any STRING consumer; it is not automatically
written to disk or shown by a new output node. Original audio is intentionally
not connected. No models are needed for this example.

## Tests and benchmarking

```bash
python -m pytest
python -m pytest tests/test_seamless_loop.py -q
python scripts/benchmark_seamless_loop.py --device cpu --runs 3
python scripts/benchmark_seamless_loop.py --device cuda --runs 3
```

Benchmarks create deterministic input outside the measured work. First call is
marked cold; warm runs follow in the same process. Timings separately cover device
initialization/smoke test, proxy creation/transfers, search, refinement/transfers,
and winner rendering/transfers. End-to-end includes validation, enumeration,
selection, retries and those phases; it excludes Python startup, fixture creation,
codecs and ComfyUI graph scheduling. CUDA phases synchronize correctly. A CUDA
context may already be warm in ComfyUI; benchmark cold includes context startup.
`--input-tensor clip.pt` supports a trusted saved tensor without video dependencies.
Resolution, frames, quality and grid limits are configurable. No family clip or
identified image batch was supplied/found inside this repository, so the suggested
1/0/4 trim/overlap is not encoded as a fixture or preference.

### Local verification, 2026-10-06

Full suite: **1,140 passed, 14 skipped**, including **61 focused loop tests**.
The new node and existing registrations import together; frontend and deployment
sources passed syntax checks, and example socket/widget wiring is tested. Full
ComfyUI GUI playback was not performed. The host's standalone Python environment
provided PyTorch 2.7.1+cu128, CUDA 12.8 and an RTX 5090 (12.0); the ComfyUI venv
launcher on this machine points to an unavailable interpreter, so these results
are not a claim that its venv ran the test suite. Installed ComfyUI source APIs
were inspected directly. No core files or existing node contracts were changed.

Sequential isolated-process benchmarks (two warm runs; host default 24 CPU threads):

| Case | Candidates | Warm search | Warm refinement + transfers | Warm render + transfers | Warm end-to-end |
|---|---:|---:|---:|---:|---:|
| CPU, 48 x 180 x 320 RGB, balanced | 564 | 0.615 s | 0.062 s | 0.001 s | 0.682 s |
| RTX 5090, same automatic search | 564 | 0.188 s | 0.027 s | 0.002 s | 0.222 s |
| RTX 5090, 16 x 720 x 1280 RGB, forced K=4 | 1 | 0.004 s | 0.024 s | 0.026 s | 0.064 s |

The automatic cases selected unchanged frames for the deterministic translating
texture. The manual case specifically measures full-resolution blending/transfers
(12 output frames, 0.5 seconds at 24 FPS), **not** a joint-search speed comparison.
CUDA automatic cold device initialization was 10.767 s, cold search 4.698 s, and
cold end-to-end 15.500 s. Startup/kernel warm-up is material and these timings are
not video decode/encode or complete ComfyUI workflow timings. None is a performance
guarantee or a comparison with the earlier 2,197-candidate prototype.
Raw phase records: [CPU](auto_seamless_loop_benchmark_cpu.json),
[CUDA](auto_seamless_loop_benchmark_cuda.json), and
[full-resolution rendering](auto_seamless_loop_benchmark_render.json).
Reproduce the rendering case with:

```bash
python scripts/benchmark_seamless_loop.py --device cuda --frames 16 --height 720 --width 1280 --max-trim 0 --max-fade 4 --manual-overlap 4 --runs 2
```

Synthetic tests independently check exact sample indices, both bridge connections,
sRGB endpoints and known linear midpoint, static and periodic translation behavior,
exposure outlier removal, localized offsets, motion reversal, moving-object ghosts,
invalid/noncontiguous data, joint constraints, CPU/CUDA ranking and rendering,
narrow fallback, bounded OOM retry, cancellation, node hooks and workflow wiring.
FP32 CPU/CUDA score/output tolerance: absolute 2e-5, relative 2e-4; rankings are
equivalent within 2e-5 for close candidates. CUDA tests skip without hardware.
Real subjective seam quality still needs repeated playback and cannot be certified
by these synthetic tests. See deployment examples and verification notes in
`deployment/auto_seamless_loop/README.md`.
