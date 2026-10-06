# mAI Utils V02

ComfyUI custom node pack for small, reusable mAI utility nodes.

Install this folder under ComfyUI's `custom_nodes` directory, then restart ComfyUI.
The currently registered nodes are listed below.

## mAI Image Logic Check

Location: `mAI / Logic`. Registered as `MAIImageLogicCheck`.

Compares an IMAGE batch property with a number for conditional workflow branching.
Required inputs: `image` (`IMAGE`), `property` (`Megapixels`, `Width`, `Height`,
`Aspect Ratio`, or `Batch Size`), `operator` (`>`, `>=`, `<`, `<=`, `==`, or `!=`),
and `compare_value` (`FLOAT`, default `2.2`, range `0`–`1e15`, step `0.01`).
Outputs, in order: `result` (`BOOLEAN`) and `actual_value` (`FLOAT`).

Defaults to `Megapixels > 2.2`. For a `[B, H, W, C]` batch, megapixels are
`W * H / 1_000_000.0` per image, width and height are pixel counts, aspect ratio
is `W / H`, and batch size is `B`. A 1024 × 1024 image returns
`False` and `1.048576` with the defaults. Equality uses `math.isclose` with
relative and absolute tolerances of `1e-6`; `!=` inverts that check.

Reads only tensor shape: input pixels are never modified, copied, or moved
between CPU and GPU. Requires a non-empty four-dimensional IMAGE batch; reports
one result for the entire batch using its shared dimensions. The boolean can
connect to conditional/lazy nodes, whose execution behavior determines branching.
This node must evaluate its input image before it can measure its shape.

Test in ComfyUI: restart, search for **mAI Image Logic Check**, connect a
1024 × 1024 IMAGE, and display `actual_value` or connect `result` to a BOOLEAN
input on a conditional node. Check the default result above, then change the
operator to `<` and expect `True`. Try `Batch Size` with a multi-image batch.
Automated tests: `python -m pytest tests/test_image_logic_check.py`.

## mAI GPU Video Combine

Location: `mAI / IO`. Registered as `MAIGPUVideoCombine`.

Combines an IMAGE frame batch with optional ComfyUI AUDIO into an MP4 using
NVIDIA NVENC or a selected software encoder. The basic frame/audio, saving and playback controls are inspired
by [VideoHelperSuite's Video Combine](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite#video-combine);
this node is implemented independently and does not require VHS.

Inputs: `frames` (IMAGE), optional `audio` (AUDIO), `frame_rate`,
`filename_prefix`, `codec`, `quality`, `preset`, `pingpong`, `loop_count`,
`trim_to_audio`, `save_output`, `save_metadata`, `gpu_device`, and `chunk_size`.
Defaults: 24 FPS, prefix `video/mAI`, `h264_nvenc`, quality 23, preset p4,
no ping-pong/repeats, full video duration, save to output with workflow metadata,
GPU -1 (first available), automatic chunks. Connect the loader's `fps` output
to `frame_rate` to preserve playback speed.

Outputs, in order:

* `filenames` (VHS_FILENAMES): `(save_output, [absolute_video_path])`, compatible
  with VHS filename consumers when VHS is installed.
* `file_path` (STRING): the completed MP4's absolute path.
* `frame_count` (INT): encoded frames, including repeats/ping-pong or audio trim.
* `duration` (FLOAT): encoded frame count divided by FPS, in seconds.

The node saves the video and returns ComfyUI's native animated/video preview.
`save_output=false` writes a preview file in ComfyUI's temp directory instead.
Filename prefixes support subfolders and ComfyUI's standard `%width%`, `%height%`,
`%year%`, `%month%`, `%day%`, `%hour%`, `%minute%`, `%second%` placeholders.
Unique suffixes prevent concurrent runs from overwriting videos. MP4 workflow
and prompt tags are embedded when supplied; global `--disable-metadata` is honored.

`codec` offers H.264, HEVC and AV1 NVENC, plus `libx264` (H.264), `libx265`
(HEVC) and `libsvtav1` (AV1) software encoding. `quality` is NVENC VBR CQ or
software CRF (0–51, lower gives higher quality/larger files). Values are not
equivalent between encoders. Presets p1–p7 range from fastest to slowest;
p4 is the default balance. Software H.264/HEVC map to ultrafast, superfast,
veryfast, medium, slow, slower, veryslow; software AV1 maps to 12, 10, 8, 6, 5, 4, 3.
H.264 MP4 is the most broadly usable preview choice. HEVC/AV1 playback depends
on the browser/player. `gpu_device` is FFmpeg's NVENC GPU index, not a torch
device selector; -1 chooses an available encoder device. Software codecs ignore it.

Frames are clipped to [0,1] and converted to 8-bit, limited-range BT.709 NV12
in tensor chunks on their existing device (CUDA for GPU frames). NVENC performs
video compression on the GPU; software codecs compress on CPU and deinterleave
NV12 to planar YUV420P. FFmpeg stdin requires a CPU byte buffer, so CUDA frames transfer
to CPU and FFmpeg uploads them to its encoder. Audio PCM, AAC encoding and
muxing use CPU. This is GPU-accelerated encoding, with a CPU bridge rather than
a direct torch-to-NVENC CUDA surface connection when using NVENC. No PNG/PIL/OpenCV path or
per-frame Python conversion loop is used. NumPy is only a zero-copy CPU byte
view of torch data. `imageio-ffmpeg` provides a bundled FFmpeg fallback when installed.

`chunk_size=0` chooses at most 32 frames and approximately 64 MiB of packed
NV12 pixels per chunk (at least one frame). Positive values specify frames per chunk.
Input is not modified, and frame order is preserved. Odd width/height gain one
black column/row at the right/bottom for even 4:2:0 dimensions; no resizing occurs.
One-channel input expands to RGB; RGBA alpha is discarded. Outputs are SDR 8-bit
4:2:0; this is not an HDR, alpha-video or lossless export node.

Missing audio produces a silent video. Audio must have waveform shape
`[1, channels, samples]` and an integer sample rate. Long audio is trimmed to the
video; short audio is padded with silence. `trim_to_audio=true` shortens the
video to the frame containing the audio endpoint. `pingpong` appends reversed
interior frames without duplicating endpoints. `loop_count` means additional
encoded video repetitions. Audio plays once, then pads with silence; it does
not repeat or reverse with the frames.

Requires FFmpeg with the chosen encoder. Discovery order: `MAI_FFMPEG_EXE`,
`IMAGEIO_FFMPEG_EXE`, PATH, then the `imageio-ffmpeg` package installed by this
pack's requirements. Overrides must identify an executable, without command
arguments. NVENC additionally requires an NVENC-capable NVIDIA GPU/driver;
unavailable codecs/drivers produce a clear error without automatic software
fallback. Software codecs work with CPU or GPU input tensors. GPU dimension limits still apply,
especially to tiny images or very large resolutions. The full input batch remains
in memory; chunking bounds conversion buffers. Before encoding, unused CUDA
cache is released and ComfyUI is asked to offload models if free VRAM is below
the estimated conversion budget plus 2 GiB for NVENC's separate CUDA context.
After success, failure or cancellation, unused cache is released again.
An explicit FFmpeg `CUDA_ERROR_OUT_OF_MEMORY` triggers one retry after requesting
model offloading on the affected GPUs, with one-frame conversion chunks.
Models may reload later in the workflow. This cannot free live frame tensors or
memory owned by other processes and does not guarantee against OOM.
Software encoding reserves only tensor conversion memory, with no NVENC context
headroom or NVENC retry. With multiple GPUs, set `gpu_device` explicitly: automatic NVENC preparation
targets the first visible CUDA GPU, while FFmpeg can select another capable GPU.

### Modal and Runpod

**RTX PRO 6000 Blackwell:** use NVENC when FFmpeg and the container's NVIDIA
video driver libraries are available. **B200/B300:** select `libx264` (recommended
for previews), `libx265`, or `libsvtav1`; these GPUs have no NVENC engines.
They can run CUDA frame conversion, but video compression uses CPU.
See [NVIDIA's support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix).

For the reported `FFmpeg was not found` error, install this pack's updated
`requirements.txt` using the same Python environment that runs ComfyUI, then
restart. For Debian/Ubuntu Runpod containers you can also install system FFmpeg:

```bash
apt-get update && apt-get install -y ffmpeg
```

For Modal, extend your existing ComfyUI image with `.apt_install("ffmpeg")`
and rebuild/redeploy. See the [cloud setup and runtime verification](docs/gpu_video_combine.md#cloud-setup-modal-and-runpod)
for driver library requirements and a real encoding check. A bundled FFmpeg
does not supply NVIDIA host drivers or guarantee every encoder on every platform;
software AV1 requires a build with `libsvtav1` (absent in the tested Windows wheel).

Test in ComfyUI: restart, search **mAI GPU Video Combine**, connect `frames`,
`audio` and `fps` from **mAI video loader** (or frames from **mAI Fast GPU Resize**),
then queue. Check the video preview, sound, first/last frame, and saved MP4 in
`output/video`. Disconnect audio to check silent output; try ping-pong/repeats
and `trim_to_audio`. Save/reload the workflow. There are no GIF/WebP/custom VHS
format presets, latent decoding or VHS meta-batch inputs in this initial version.

Tests: `python -m pytest tests/test_video_encoding.py tests/test_gpu_video_combine.py`.
Real NVENC tests are opt-in: set `MAI_TEST_NVENC=1` and run those tests with
FFmpeg/ffprobe on PATH. Software integration tests use `MAI_TEST_FFMPEG=1`.
Detailed validation and implementation notes:
[GPU Video Combine](docs/gpu_video_combine.md).

## mAI Fast GPU Resize

Location: `mAI / Image`. Registered as `MAIFastGPUResize`.

Resizes a complete ComfyUI `IMAGE` batch using PyTorch tensors. Supports RGB,
RGBA and other channel counts. Inputs: `image`, `width`, `height`, `resize_mode`,
`method`, `antialias`, `multiple_of`, `device`, `precision`, and `chunk_size`.
Outputs in order: `image` (IMAGE), `width` (INT), `height` (INT),
`method_used` (STRING, including `identity` for a resize bypass).

Defaults: 1280×720, `exact`, `auto` method/device/precision, antialias enabled,
multiple 1, chunk size 0. Modes: `exact` stretches; `keep_aspect_fit` centers the
image on a black canvas; `keep_aspect_fill` centers and crops overflow.
`keep_aspect` fits the whole image within the requested width/height without
padding or cropping, returning its actual dimensions: 1920×1080 into 1024×1024
produces 1024×576. It supports both upscaling and downscaling. Integer pixel
rounding can slightly alter the ratio; use `multiple_of=1` for the closest match.
With larger multiples, the bounds are rounded first and the fitted output
dimensions are also rounded to that multiple, which can further alter the ratio.
Odd padding/crop differences put the extra pixel on the bottom/right. Dimensions
round to the nearest `multiple_of` number (INT widget, 1–16384, also connectable
to an INT output), ties upward, minimum one multiple. Common choices are
8/16/32/64; 1 disables rounding. For multiple 8: 1023→1024, 1020→1024,
1019→1016. Any integer in the widget range is supported, such as 3 or 128.

Methods: `nearest`, `nearest-exact`, `bilinear`, `bicubic`, `area`, true
`lanczos2`/`lanczos3`/`lanczos4`, `mitchell` (B=C=1/3), `catmull_rom`
(B=0, C=1/2), and `auto`. Auto chooses Lanczos3 if either resize scale ≤0.65,
Lanczos2 for other shrinking, bicubic up to 1.5×, and Catmull–Rom above that.
It uses the actual resized image dimensions before padding/cropping.
Custom kernels use shared separable weights, replicated edges and wider
low-pass kernels when downsampling with antialias enabled. Native antialias
applies only to bilinear/bicubic; area always averages and nearest ignores it,
following [PyTorch's interpolation API](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.interpolate.html).
Lanczos/custom cubic/bicubic results clamp to [0,1] after reconstruction.

`device=auto` preserves the input device, including CPU. Choose `gpu` to move
a CPU batch to ComfyUI's preferred CUDA device, or `cpu` to explicitly move it
back. Output stays on the processing device. `precision=auto` preserves input
dtype; explicit fp32/fp16/bf16 controls output dtype. Custom kernels accumulate
in FP32 (FP64 for FP64 input); native antialiased reduced precision and CPU
linear/cubic/area use FP32 internally when needed. Matching-size inputs bypass
filtering and copying, while honoring explicit device/dtype changes.

`chunk_size=0` processes native methods as a whole batch; custom methods select
batch chunks from a conservative 512 MiB working-memory estimate. Positive
values specify the batch chunk size. Tables are built once and reused by every
chunk. Only chunk and kernel-tap loops are used, never per-frame resizing.
The node uses no PIL/OpenCV/NumPy path and adds no dependencies. GPU execution
releases unused CUDA cache, requests ComfyUI model offloading when the estimated
output/scratch budget exceeds free VRAM, and releases unused cache after each
operation, including errors/cancellation. Live input/output tensors stay on
their selected device. CPU execution and unchanged device/dtype identity
bypasses do not run CUDA cleanup. Model offloading can cause later reloads.

Limitations: input must be one nonempty BHWC floating-point tensor; IMAGE lists
are not accepted. Chunking bounds temporary working memory, but the full output
and input still need memory; memory preparation does not guarantee against OOM.
Extreme fill aspect ratios or very large single images can need large intermediates.
Custom kernels are slower than native methods; FP32 accumulation costs memory.
Alpha is filtered like any other channel, without premultiplication. CUDA is
required for explicit `gpu`; CPU remains supported. Linux uses the same portable
torch code, but was not tested on a Linux host.

Test in ComfyUI: restart, search **mAI Fast GPU Resize**, connect an H3/LTX IMAGE
batch to `image`, set 1024×576 and `device=gpu`, then connect the IMAGE output
to Preview Image or a video encoder. Check frame order and the width/height/
method diagnostics. Try 1024×1024 with fit/fill and chunk size 32; save and
reload the workflow. Try `keep_aspect` on a 16:9 image with width/height 1024,
`multiple_of=1`: verify a 1024×576 output with no borders or cropping.
Automated tests:
`python -m pytest tests/test_resize_kernels.py tests/test_fast_gpu_resize.py tests/test_gpu_memory.py`.
Benchmark: `python scripts/benchmark_fast_gpu_resize.py --direct` (CUDA), or
`--device cpu --smoke`. Measurements and architecture details:
[Fast GPU Resize notes](docs/fast_gpu_resize.md).

## mAI JSON parser

Location: `mAI / Text`. Registered as `MAIJsonParser`.

Selects one value from pasted or connected JSON text. Inputs: `json_text`
(multiline STRING), `key_path` (STRING, default `prompt`), and `output_type`
(`auto` by default, or `string`). One stable wildcard output: `value`.
In auto mode, strings, integers, floats and booleans keep their Python types.
Objects and arrays return JSON text; JSON null returns the text `null`, never
Python `None`. String mode always returns text (booleans become `true`/`false`,
and selected strings are not surrounded by quotes). Numeric strings stay strings.

Paths: `prompt`, `settings.width`, `items[0].name`, or `$.items[0].name`.
Use `["key.with.dots"]` for literal keys containing punctuation, `["0"]` for
a numeric object key, and `[""]` for an empty key. Blank or `$` selects the root;
`[0]` selects the first item of a root array. Indices start at zero.
Invalid JSON, missing keys, out-of-range indices and incompatible traversal
produce clear errors. Duplicate object keys use the last value, as in Python's
standard JSON parser. No external dependencies or frontend extension required.

Test in ComfyUI: restart, add **mAI JSON parser**, and use
`{"prompt":"a forest","settings":{"width":1024},"enabled":true}`.
With `key_path=prompt`, connect `value` to a text input; with
`key_path=settings.width` in auto mode, connect it to an INT input.
Try string mode with a text display node, and save/reload the workflow.
Both text widgets can be connected to upstream STRING outputs.

Limitations: the wildcard socket permits connections, but the selected runtime
type must match the downstream input. This is a single-value selector, with no
wildcards, filters, negative indices or expression evaluation. Accepts valid JSON
only, without Markdown fences or comments. Objects/arrays are serialized text,
not ComfyUI batches. Automated tests: `python -m pytest tests/test_json_parser.py`.

## mAI image gate

Location: `mAI / Image`. Registered as `MAIImageGate`.

Inputs: `enabled` (BOOLEAN, default true; can be converted to an input for an
external boolean) and optional `image` (IMAGE). Output: `image` (IMAGE).
True passes the exact original image or image batch without copying or modifying
it. False outputs `None`. An unconnected image input also outputs `None`.
The image input is lazy: when disabled, this gate does not request its upstream
image branch. Other consumers may still cause that branch to execute.

Only connect to downstream inputs that explicitly handle `None` as a missing
image. This does not actually disconnect the socket, restore a widget default,
or skip the downstream node. Even an optional input is not guaranteed to accept
`None`; nodes that inspect graph connections can still see the connection.
Preview Image, Save Image, and other nodes requiring a real image cannot consume
the disabled output. No additional dependencies or frontend extension required.

Test in ComfyUI: restart, add **mAI image gate**, and connect Load Image to its
image input. Connect its output to a compatible optional image input. Queue with
enabled true, then false: the receiving node should use the image, then its
missing-image behavior. Toggle true again to restore the image. Also try leaving
the gate's image input unconnected. Automated tests:
`python -m pytest tests/test_image_gate.py`.

## mAI conditional LoRA

Location: `mAI / Utils`. Registered as `MAIConditionalLora`.

Applies a selected LoRA only when the prompt contains any configured trigger.
Inputs: `model` (MODEL), optional `clip` (CLIP), `lora_name` (installed LoRA dropdown),
`prompt` (STRING socket), `strength` (FLOAT, default 1.0, range -100 to 100),
and `trigger_words` (multiline STRING, blank by default; can be converted to an input).
Without CLIP, only MODEL is patched. When CLIP is connected, the same strength
applies to both MODEL and CLIP.
Outputs, in order: `model` (MODEL), `clip` (CLIP), `prompt` (STRING, unchanged).
If CLIP is omitted, its output is `None`; leave that output disconnected and
connect your original text encoder directly to the text encoding nodes.

Separate triggers with commas or newlines; each entry is a whole word or phrase.
Any match activates the LoRA once. Matching ignores case and normalizes whitespace;
`cat` matches `(cat:1.2)` but not `cathedral`. Special characters are literal,
not regular expressions. Blank triggers, an empty prompt, no match, or strength
zero return the original inputs without loading a LoRA file. Switching from a
match to no match returns the incoming MODEL and CLIP, without this node's LoRA.

Test in ComfyUI: restart, add **mAI conditional LoRA**, and connect a checkpoint's
MODEL. Select a compatible LoRA, set `trigger_words` to its trigger, and
connect a text source to `prompt`. Connect output MODEL to the sampler and output
`prompt` to the positive text encoder. For a model-only LoRA, leave CLIP input
and output disconnected. To also patch the text encoder, connect CLIP through
this node and use its CLIP output for text encoding.
Compare prompts with and without the trigger at a fixed seed; also try strength
zero. Multiple nodes can be chained to conditionally apply different LoRAs.

Limitations: requires MODEL; uses ComfyUI's native LoRA compatibility
and loading behavior. Matching is literal text, so negated wording such as
`no cat` still matches `cat`. Triggers cannot contain commas or newlines.
Does not insert trigger words or remove LoRAs already applied upstream.
No extra dependencies or frontend extension required.
Automated tests: `python -m pytest tests/test_lora_triggers.py tests/test_conditional_lora.py`.

## mAI Cinematic Post

Location: `mAI / Image`. Registered as `mAI_CinematicPost` (the explicitly
requested identifier). Input: `image` (`IMAGE`, float RGB `[B,H,W,3]` in 0–1).
Output: `image` (`IMAGE`), preserving batch size, resolution, device and dtype.
Optional `subject_mask` (`MASK`): white protects subject color from grading and
gently boosts sharpening/local contrast, scaled by `skin_protect`. Single masks
broadcast across the batch; differently sized masks are bilinearly resized.
Missing or zero-element masks use gentle orange-hue protection; an all-black mask
explicitly disables subject protection. A white mask protects the whole image.

Applies exposure, a filmic luminance curve, highlight shoulder, black lift,
midtone contrast, cool shadows/warm highlights, saturation and green/blue taming,
color density, warm highlight-edge halation, diffuse neutral bloom, lens softness,
selective sharpening, local contrast, vignette, subtle radial chromatic aberration,
optional lens distortion, and seeded luminance-dependent color grain.

Start with **Subtle Film**, `strength=1.0`, and the supplied defaults: contrast
0.15, highlight rolloff 0.25, saturation -0.08, color density 0.12, halation 0.06,
bloom 0.04, grain 0.06, vignette 0.08. The defaults are intentionally subtle.
Use strength 0.5 for a lighter finish. `enabled=false`, `strength=0`, and
**Off / Neutral** each return the original image exactly without processing.

Main controls:

* `strength` blends the completed look with the original (0–1).
* `preset`: **Subtle Film**, **Commercial Cinematic**,
  **Commercial Cinematic - Preserve Colors**, **Moody**, **Warm Premium**,
  **Cool Night**, or **Off / Neutral**. The Preserve Colors variant keeps the
  commercial contrast, black level and optical effects while removing split
  toning, global saturation reduction, and green/blue taming; density drops from
  0.18 to 0.04. It preserves the original palette more closely, but tone and glow
  still affect the final colors. Presets apply in Python, including API
  workflows. Sliders are trims relative to the Subtle Film defaults: for example,
  Moody sets contrast to 0.22; moving the contrast slider from 0.15 to 0.20 makes
  effective contrast 0.27. Effective values stay within the widget limits.
  Presets do not rewrite widget values; Off / Neutral bypasses all slider edits.
* `exposure` multiplies input brightness by powers of two before tone mapping;
  `contrast`, `black_lift`, `highlight_rolloff`, and `midtone_contrast` shape tone.
* `saturation` adjusts chroma; `color_density` darkens chromatic midtones for richer
  color without a saturation boost. `shadow_cool`/`highlight_warm` split tone;
  `green_tame`/`blue_tame` reduce dominant greens/blues; `skin_protect` weights
  color protection and subject detail enhancement.
* Halation/bloom strengths set glow amount; thresholds isolate bright sources;
  blur values are Gaussian sigma in pixels at a 1080-pixel short edge, scaled
  with resolution (minimum scale 0.5). Bloom has an additional diffuse radius.
* `lens_softness` mixes in slight blur; `sharpen_amount` restores luminance detail
  with noise suppression and bounded halos; `local_contrast` shapes broader detail.
* `grain_strength`, `grain_size`, and `grain_chroma` adjust amplitude, particle
  size and color variation. Size scales with resolution, with a one-pixel minimum.
  `grain_seed` is reproducible without changing global Torch random state.
  `grain_animation_safe=false` repeats the same pattern at the same seed and size;
  true uses seed + batch index for repeatable variation. For separately queued
  frames, vary the seed explicitly if desired. Neither mode tracks subject motion
  or guarantees flicker-free video; zero grain gives the most stable finish.
* `vignette_feather` sets edge softness; `chromatic_aberration` is approximately
  per-channel pixels at a 1080-pixel frame edge; `lens_distortion` is a small
  signed radial coefficient, default 0.
* `advanced_mode` defaults false: show only enabled, preset, advanced_mode,
  strength, contrast, highlight_rolloff, saturation, color_density,
  halation_strength, bloom_strength, grain_strength and vignette_strength.
  Set true to show all controls. Hidden controls keep their values, still affect
  processing, and survive workflow save/load. Connected controls remain visible.

Usage/test in ComfyUI: restart, add **mAI Cinematic Post**, and connect
**Load Image / VAE Decode → mAI Cinematic Post → Preview Image / Save Image**.
Compare Subtle Film with Off / Neutral on a portrait and an image containing bright
lights. Optionally connect a subject mask, try presets, and requeue with a fixed
grain seed. Automated checks: `python -m pytest tests/test_cinematic_post.py`.
After this update, restart ComfyUI and hard-refresh the browser (Ctrl+F5) to load
the new frontend extension and preset. Toggle advanced_mode false/true and reload
a saved workflow to check compact layout and retained values. Existing workflows
with advanced_mode=true remain expanded. Existing slider edits also remain preset
trims; use the default sliders on a fresh node to evaluate the new base look.

Limitations/tradeoffs: this is an artistic display-referred RGB finish, not a
scene-linear/HDR color-management transform, stock emulation or skin detector.
Use RGB images; alpha and empty image batches are rejected. Processing uses
float32 internally and clamps the final result to 0–1; extreme settings can lose
detail. Grain is repeatable on the same device/runtime, not bit-identical across
CPU and CUDA. Frames are processed one at a time to limit intermediate memory;
large blurs use reduced resolution, trading some accuracy for speed. Processing
stays on the input device (typical ComfyUI images arrive on CPU); it does not move
images to a GPU automatically. No added dependencies. Widget hiding uses the
frontend's [native hidden flag](https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.51.10/src/lib/litegraph/src/LGraphNode.ts#L3772)
(checked against frontend 1.51.10 source). If the
extension fails to load, processing still works with all controls visible.
The optional before/after output is omitted; compare with a separate preview.

## mAI image aspect ratio

Location: `mAI / Image`. Registered as `MAIImageAspectRatio`.

Input: `image` (`IMAGE`). Output: `aspect_ratio` (`STRING`), exactly `1:1`,
`16:9`, or `9/16` (portrait).

Selects the preset with the smallest absolute difference from the image's
width divided by height. Exact ties prefer `1:1`. Reads dimensions only and
returns one string for the entire batch, whose images share dimensions.
Only these three presets are supported; empty images are rejected.

Test in ComfyUI: restart, add **mAI image aspect ratio**, connect Load Image,
and connect the output to a text display node. Images sized 1024×1024,
1920×1080, and 1080×1920 should return `1:1`, `16:9`, and `9/16` respectively.
Automated tests: `python -m pytest tests/test_aspect_ratios.py tests/test_image_aspect_ratio.py`.

## mAI Krea2 image conditioning

Location: `mAI / Conditioning`. Registered as `MAIKrea2ImageConditioning`.

Generates a reconstruction caption and Krea 2 conditioning using the connected
native Krea 2 Qwen3-VL-4B encoder. It reuses the loaded CLIP and ComfyUI's image
preprocessing, tokenizer, generation, and 12-layer conditioning extraction. No
separate caption model, downloads, extra dependencies, or core modifications.

Inputs:

* `image`: non-empty floating-point ComfyUI IMAGE batch in `[0, 1]`, RGB or RGBA
  (alpha is ignored).
* `clip`: full Qwen3-VL-4B encoder loaded with CLIPLoader type `krea2`.
  Other encoder types are rejected, including a generic Qwen3-VL CLIP.
* `instruction`: editable reconstruction instructions; blank restores the default.
* `conditioning_mode`: `text_plus_vl` by default; modes below.
* `detail_level`: `low`, `medium`, `high` (default), or `extreme`; adjusts caption
  instructions, without increasing the token budget automatically.
* `caption_max_new_tokens`: 32–1024, default 192, per image.

Modes:

* `text_only`: generate a caption from the image, then encode only that caption
  using the native Krea 2 text template.
* `vl_only`: encode the actual image with an empty user text using Krea's native
  system/user template with image placeholders inside the user turn. Caption
  generation is skipped entirely and `caption` returns an empty string (`""`).
  `instruction`, `detail_level`, and `caption_max_new_tokens` are unused in this mode.
* `text_plus_vl`: encode the generated caption and actual image together through
  the native multimodal path. Both participate in the same transformer pass;
  separate tensor concatenation or averaging is unnecessary. Independent text/VL
  strength sliders are omitted because the native joint path does not expose them.

Both image-conditioning modes preserve Krea's system/user prefix. The generic
Qwen image-generation chat template is used only for caption generation: its
user/assistant layout is incompatible with Krea's conditioning prefix stripping,
especially after image placeholders expand into many hidden-state positions.
After updating from the original implementation, restart ComfyUI and re-run the
node to regenerate conditioning. Compare at the same seed and sampler settings.

Outputs, in order: `conditioning` (CONDITIONING), `caption` (STRING).
The readable caption is available in `text_only` and `text_plus_vl` for prompt
inspection or saving. The output socket remains present but empty in `vl_only`.

Batch behavior: in the text modes, captions are generated sequentially, one per image, and joined
with `Image 1`, `Image 2`, etc. labels. All images become ordered references in
one shared multimodal conditioning (text-only uses the joined captions). This
does not pair separate conditionings with individual latent batch items. Process
images individually when each latent needs its own reference. Larger batches and
longer captions increase context length, generation time and memory use. Greedy
caption decoding uses no random sampling; numerical results can vary by hardware.

Usage/test in ComfyUI: restart, add this node, connect Load Image and the Krea 2
CLIP. VAE Encode the source image separately and feed that latent into your Krea 2
sampler/refine workflow. Connect `conditioning` to its positive input, preserving
the workflow's model, negative conditioning, VAE and sampling settings. Inspect
`caption` with a text display/save node; compare all three modes at the same seed
and denoise. The node does not create latents or change denoise.
For `vl_only`, verify that no caption token-generation progress appears, the
caption output is empty, and image conditioning still reaches the sampler.

Limitations: requires ComfyUI's native Krea 2 generation and multimodal APIs and
encoder weights with vision support. Native conditioning format compatibility
does not guarantee pixel-accurate reconstruction or prevent drift at high denoise.
Caption length can be cut off by the token budget. An empty or reasoning-only
response triggers one automatic retry with an explicit caption request, using
the same image, encoder and token budget. A warning is logged only when retrying.
If both attempts are empty, the node stops with the image number and returned
token counts; it never substitutes a generic or previous image's caption.
Increasing the budget may help exhausted generations, but not immediate stops.
Native execution errors and cancellation are not retried. To check recovery,
rerun the image that failed and inspect the caption and any retry warning.
Image-aware conditioning is experimental: the
[Krea reference encoder](https://github.com/krea-ai/krea-2/blob/main/encoder.py)
defines a text-only, 512-position layout. Full-resolution image embeddings can
produce much longer sequences; matching the tensor format does not establish
equal generation quality. This node does not silently truncate image embeddings
or caption text to fit that layout. If textures spread between objects, compare
`text_only` and `text_plus_vl` with the same seed, caption and sampling settings.
No fallback to an unrelated encoder is attempted if native execution fails.
Stop uses ComfyUI's native interruption mechanism. The adapter restores encoder
options on exit and temporarily disables the native decoder's
`graph_dynamic_vbar_blocks` optimization when exposed. This avoids new decoder
graph captures in this node, with a possible caption speed cost. It preserves
native interruption exceptions and discards cancelled results without forcing
GPU frees or garbage collection. Existing graphs from other nodes and external
compile/graph wrappers are outside this guard. Stop may wait for an active GPU
operation to finish. To check cancellation, restart ComfyUI, stop during caption
generation, then queue again with the same encoder; repeat during conditioning.

If Stop aborts the whole server with `CUDAMallocAsyncAllocator`, "uncaptured free
of a captured allocation", and `CUDA error: invalid argument`, Python cannot catch
that native allocator abort. As a diagnostic workaround, restart ComfyUI with
`--disable-cuda-graphs --disable-cuda-malloc` added to the existing launch command.
These flags affect the whole server and may change speed/memory use; this node
does not change launch settings automatically. The local cancellation tests do
not reproduce or prove resolution of a remote CUDA/driver crash.
Adapter tests: `python -m pytest tests/test_krea2_conditioning.py`; these use small
test doubles and do not measure model caption quality or GPU compatibility.

## mAI background lighting match

Location:

```text
mAI / Image
```

Purpose:
Correct a brightness and contrast shift after inpainting. The node measures the
untouched background in both the original and edited images, then applies the
resulting lighting correction to the complete edited image.

Inputs:

* `original_image` — the image before inpainting
* `edited_image` — the complete image after inpainting or object removal
* optional `edit_mask` — white marks the edited area; black marks the untouched
  background

Output:

* `image` — the corrected edited image

Default behavior:

* Uses only the inverse of `edit_mask` for analysis.
* When `edit_mask` is not connected, uses the whole image for analysis.
* Treats soft mask values as proportional background weights.
* Matches the mean (brightness) and standard deviation (contrast) independently
  for the red, green, and blue channels.
* Applies one affine correction per RGB channel to every edited pixel, including
  pixels inside the mask.
* Supports image batches and broadcasts a batch of one where possible.

Known limitations:

* When connected, the mask dimensions must match the image dimensions.
* The mask must leave some untouched background visible.
* Contrast cannot be reconstructed if an edited background channel is completely
  flat while the corresponding original channel has contrast; the node reports a
  clear error in that case.
* Output values are limited to `[0, 1]`. Clipping at the range boundaries can make
  the final measured statistics differ slightly from the mathematical target.

## mAI video loader

Location:

```text
mAI / IO
```

Purpose:
Upload or select a video from ComfyUI's input directory and decode it into a
ComfyUI image batch, along with its source metadata and audio stream.

Input:

* `video`

Outputs, in order:

* `frames` (`IMAGE`) — all decoded video frames as one image batch
* `fps` (`FLOAT`) — the video's average source frame rate
* `audio` (`AUDIO`) — the decoded audio stream, or no value when the video has no audio
* `frame_count` (`INT`) — the number of decoded frames
* `width` (`INT`) — decoded frame width
* `height` (`INT`) — decoded frame height

Default behavior:

* Lists video files in ComfyUI's input directory and provides a video upload button.
* Uses ComfyUI's native video decoder and returns frames in
  `[frame, height, width, channel]` format.
* Applies video rotation metadata through ComfyUI's decoder before reporting width
  and height.

Known limitations:

* The complete decoded frame sequence is held in memory; long or high-resolution
  videos can require substantial RAM.
* Variable-frame-rate sources report their average frame rate.
* A video without an audio stream returns no `AUDIO` value. Nodes that require
  audio should only be connected when the source contains audio.

## mAI trim frame sequence

Location: `mAI / Image`. Registered as `MAITrimFrameSequence`.

Trims frames from an image batch or an ordered list of image frames/batches.
Inputs: `frames` (`IMAGE`), `trim_start`, and `trim_end` (non-negative integers).
Outputs, in order: `frames` (`IMAGE`, one batch), `frame_count` (`INT`, remaining frames).

* `trim_start` removes that many frames from the beginning.
* `trim_end` removes that many frames from the end.
* Set either count to `0` to keep that end; the counts can differ.

Defaults: both counts `0` (keeps all frames). List inputs are treated as one
continuous sequence in their original order, including lists of multi-frame batches.
For 10 frames with `trim_start=2` and `trim_end=3`, five frames remain
(original frames 3 through 7).
Pixel values, tensor dtype, and device are preserved.

Limitations: at least one frame must remain; empty inputs or trims removing every
frame raise a clear error. All frames must share dimensions, channels, dtype, and
device. Output is always a batch; combining lists allocates memory for retained
frames. This node trims images only; audio and FPS are not adjusted. Connected
trim settings must each supply one value for the entire sequence.

The node name, registration, IMAGE input, and outputs are unchanged. The old
`trim_mode`/`trim_amount` controls are replaced by the two counts. The frontend
automatically converts saved dropdown/count widget settings when loading old
workflows: `start` becomes `amount/0`, `end` becomes `0/amount`, and `both ends`
becomes `amount/amount`. Old connections to converted trim widgets must be
reconnected to the new count inputs; API prompts must use `trim_start` and
`trim_end`. Without the frontend extension, old widget settings need to be set
manually; execution of the new inputs requires no JavaScript.

Test in ComfyUI: restart, add **mAI trim frame sequence**, connect an image batch
(for example, `frames` from **mAI video loader**), and connect the result to
Preview Image or a video encoder. Try counts `0/0`, `2/0`, `0/3`, and `2/3` and check
the retained first/last frames and `frame_count`. Also try an IMAGE list source.
Automated tests: `python -m pytest tests/test_frame_sequence.py tests/test_trim_frame_sequence.py`.

## mAI frame loop fade

Location: `mAI / Image`. Registered as `MAIFrameLoopFade`.

Extends an image sequence with a linear crossfade from its last frame back to its
first frame for loop playback. Inputs: `images` (`IMAGE` batch or ordered list of
frames/batches) and `fade_frames` (`INT`, default `24`, widget range `0`–`10000`).
Outputs, in order: `images` (`IMAGE`, one batch) and `frame_count` (`INT`).

All original frames remain unchanged, followed by exactly `fade_frames` new
images. The first image's opacity increases from `1 / fade_frames` to `1` over
the appended frames; the final frame matches the original first frame exactly.
For example, 100 input frames with `fade_frames=24` produce 124 frames.
`0` keeps the sequence unchanged; `1` appends the first image directly.
A single input image is repeated. Tensor dtype and device are preserved.

Limitations: frames must be non-empty floating-point tensors with matching
dimensions, channels, dtype, and device. This is a pixel blend between two still
frames, so moving subjects can ghost during the fade. The final first-frame copy
is repeated when playback loops. Additional images require memory. Audio and FPS
are not adjusted; at a fixed FPS, the video becomes `fade_frames / FPS` seconds
longer. Connected fade settings must supply one value for the whole sequence.

Test in ComfyUI: restart, add **mAI frame loop fade**, connect an image sequence
(such as `frames` from **mAI video loader**) to `images`, and send its `images`
output to Preview Image or a video encoder. Try `fade_frames=4`: the frame count
should increase by four, and the added images should show 25%, 50%, 75%, and 100%
of the first frame over the last. Try `0` to confirm the original sequence.
Automated tests: `python -m pytest tests/test_frame_loop_fade.py`.

## mAI H3 to LTX Frame Adapter

Location: `mAI / Image`. Registered as `MAIH3ToLTXFrameAdapter`.

Prepares an H3 frame sequence for LTX 2.5 refinement by selecting the largest
frame count `1 + 8 * ((N - 1) // 8)` no greater than the source count. Input:
`images` (`IMAGE`). Outputs, in order: `images` (`IMAGE` batch), `source_frames`,
`target_frames`, and `removed_frames` (all `INT`). No widgets or optional inputs.

For example, 124 frames become 121, 107 become 105, and 90 become 89. The first
and last source frames are preserved exactly; intermediate removals are spread
evenly using rounded indices across the full sequence. Compatible tensor batches
(such as 73 or 209 frames) return unchanged, without copying. No interpolation,
resizing, color processing, dtype conversion, or device transfer occurs. One
concise console message reports conversion or an already-compatible count.

Ordered lists of HWC or one-frame BHWC tensors are also supported and returned
as one ComfyUI IMAGE batch. The selection helper preserves the original list
elements; assembling the output batch allocates memory. List frames must share
dimensions, channels, dtype, and device. Multi-frame batches within a list are
not supported. Empty inputs and sequences shorter than 9 frames raise clear
errors. The node accepts any count of at least 9, without enforcing H3's count
rule. It handles images only: audio, FPS, and timestamps are not adjusted.
At an unchanged playback FPS, fewer frames mean a slightly shorter duration,
although the selected frames cover the complete source temporal span.

Test in ComfyUI: restart, search for **mAI H3 to LTX Frame Adapter**, connect the
H3 IMAGE batch to `images`, and send the output to the LTX refinement image input.
With 124 source frames, check diagnostics `124 / 121 / 3` and preview the first
and last frames against the source. A compatible 121-frame batch should report
`121 / 121 / 0`. Automated tests:
`python -m pytest tests/test_h3_to_ltx_frame_adapter.py`.

## mAI Inpaint Crop - Separate Stitch Mask

Location:

```text
mAI / Image
```

Purpose:
Crop an image using the current `ComfyUI-Inpaint-CropAndStitch` workflow while
generating a separate full-size stitch/blend mask inside the node.

```text
render mask = what the model edits
stitch mask = where the generated result is blended back
```

The generated stitch mask has two modes:

* `rectangle (full crop)` blends the complete cropped rectangle back into the
  source image.
* `extended mask` follows the render-mask shape, expands it with
  `stitch_mask_expand_pixels`, and ensures the crop is large enough to contain
  that expanded region.

Only `cropped_mask` is sent to the inpainting model.
`stitch_mask_blend_pixels` feathers the generated stitch boundary in either
mode. Feathering happens inward so the outer crop boundary reaches exact black
(`0.0`) on only its final perimeter row while preserving a fully white center
whenever the crop size allows it.

Inputs:

* All current inputs from the installed upstream `Inpaint Crop` node.
* `stitch_mask_mode` (`rectangle (full crop)` or `extended mask`)
* `stitch_mask_expand_pixels` (used by `extended mask`, default `32`)
* `stitch_mask_blend_pixels` (default `32`; supports ComfyUI's full resolution range)

Outputs:

* `stitcher`
* `cropped_image`
* `cropped_mask` (the render/inpainting mask)
* `cropped_stitch_mask` (the exact mask stored in the stitcher for compositing)

Example workflow:

```text
IMAGE ─────────────────────────────┐
                                  │
RENDER MASK ──────────────────────┤
                                  ▼
                     mAI Inpaint Crop
                                  │
                    ┌─────────────┼──────────────────────┐
                    │             │                      │
              cropped image   render mask            STITCHER
                    │             │            (internal stitch mask)
                    └──────► IMAGE MODEL                 │
                              │                          │
                              ▼                          │
                       rendered crop                     │
                              │                          │
                              └────────► Inpaint Stitch ◄┘
                                               │
                                               ▼
                                          FINAL IMAGE
```

Connect `stitcher` and the model's rendered crop to the standard upstream
`Inpaint Stitch` node. `cropped_stitch_mask` previews the exact internally
generated mask used by that stitch operation.

Known limitations:

* This integration follows the locally installed upstream v3 stitcher format and
  requires the `cropped_mask_for_blend` field exposed by that format.
* The optional dependency must be available when ComfyUI loads this node pack.

### CropAndStitch dependency installation

`ComfyUI-Inpaint-CropAndStitch` must also be installed. It is intentionally not
listed in `requirements.txt` because it is a separate ComfyUI custom-node pack.

Recommended installation:

```text
ComfyUI Manager
→ search for ComfyUI-Inpaint-CropAndStitch
→ install
→ restart ComfyUI
```

Manual installation, following the current upstream README:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git
```

Restart ComfyUI after installation. If the dependency is absent, this node is
skipped and the rest of mAI Utils continues to load.

Acknowledgment: this node integrates with
[`ComfyUI-Inpaint-CropAndStitch`](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch),
which is licensed under GNU GPL v3. No upstream source or license text is
vendored here.

## mAI prepare image for Minimax H3

Location:

```text
mAI / Image
```

Purpose:
Resize an image for Minimax H3 while keeping its proportions as close as possible
and ensuring that both output dimensions are multiples of 32.

Inputs:

* `image`
* `target_megapixels` (`0.2 MP` through `2.0 MP` in `0.1 MP` steps)
* `resize_mode` (`Target megapixels` or `Short side 768 px`)

Output:

* `image`

Default behavior:

* Targets approximately `1.0 MP` by default (`1024 × 1024` total pixels in ComfyUI's convention).
* Calculates a proportional target size, then rounds width and height to the nearest multiples of 32.
* Uses Lanczos resampling without cropping.
* Supports image batches.

With `Short side 768 px` selected, the megapixel value is ignored. The shorter
dimension is set to exactly 768 pixels and the longer dimension is rounded to
the nearest multiple of 32.

Known limitations:

* Rounding both dimensions to a 32-pixel grid can introduce a small aspect-ratio difference.
* The selected megapixel value is approximate because valid output dimensions are discrete.

## mAI Composite Layer

Location:

```text
mAI / Image
```

Purpose:
Composite one image layer over a base image.
Use multiple copies of the node chained together to build multi-layer compositions.

Inputs:

* `base_image`
* optional `base_mask`
* `layer_image`
* optional `layer_mask`

Widgets:

* `x`
* `y`
* `scale`
* `opacity`
* `anchor`
* `fit_mode`

Outputs:

* `image`
* `mask`

Default behavior:

* Outputs RGB image data.
* Uses `layer_mask` when connected.
* Otherwise uses embedded alpha when `layer_image` has 4 channels.
* Otherwise treats the layer as opaque.
* Supports negative `x` and `y` values.
* Allows layers to be partially outside the base image.

Mask behavior:

* The `mask` output is the placed layer alpha.
* If `base_mask` is connected, the output mask combines `base_mask` and the current layer alpha.
* This makes it possible to chain Composite Layer nodes while preserving mask coverage.

Known limitations:

* The old `mAI Image Layer Stack` implementation remains in the codebase but is not registered by default.

## mAI mask bounding box

Location:

```text
mAI / Mask
```

Purpose:
Create a pure white, filled rectangular mask covering the bounding box of the
original mask's nonzero pixels.

Input:

* `mask`

Output:

* `mask`

Default behavior:

* Treats every input value greater than zero as part of the mask.
* Fills the smallest axis-aligned rectangle containing those pixels with white (`1.0`).
* Keeps pixels outside the rectangle black (`0.0`).
* Processes each mask in a batch independently.
* Returns an empty mask when the input mask is empty.

Known limitations:

* Very small positive mask values count toward the bounding box.

## mAI mask outline

Location:

```text
mAI / Mask
```

Purpose:
Draw a colored outline that follows the mask's actual contour on the original
image.

Inputs:

* `image` — the original image to annotate
* `mask` — the shape to outline
* `color` — outline color in `#RRGGBB` format (default `#00FF00`)
* `thickness` — outline width in pixels (default `10`)
* `padding` — pixels used to expand the mask shape before outlining (default `0`)
* `threshold` — minimum included mask value on a `0` to `255` scale (default `0`)

Output:

* `image` — a copy of the input image with the outline drawn over it

Default behavior:

* Treats every mask value greater than the selected threshold as part of the shape.
* Expands the mask by `padding` pixels before tracing its contour.
* Centers even stroke widths across the contour; odd widths place the extra pixel inside.
* Processes each batch item independently and broadcasts a batch of one where possible.
* Leaves the image unchanged when the mask contains no included pixels.
* Preserves additional image channels, including alpha; the outline changes RGB only.

Known limitations:

* The mask height and width must match the image.
* `threshold` uses `0` to `255` while ComfyUI stores mask values internally as `0.0` to `1.0`.
* The morphology uses square pixel neighborhoods, so heavily padded diagonal corners can look slightly squared.
* Outlines and padding are clipped at image edges.

## mAI Save Text File

Location:

```text
mAI / IO
```

Purpose:
Save text to a UTF-8 file.

Inputs:

* `text`
* `file_name`
* `folder`
* `overwrite`

Outputs:

* `file_path`
* `saved`

Default behavior:

* Writes text using UTF-8 encoding.
* Saves into the current working directory when `folder` is empty.
* Creates `folder` if it does not exist.
* Uses the exact provided `file_name`.
* Does not add timestamps or change the file name.
* Does not auto-add `.txt`; include `.txt` in `file_name` when wanted.
* Raises `FileExistsError` when `overwrite` is false and the target file already exists.

Known limitations:

* `file_name` must be a simple file name, not a nested or absolute path.
* Put folder paths in `folder`, not in `file_name`.

## mAI Type Converter

Location:

```text
mAI / Utils
```

Purpose:
Convert one selected input type into boolean, string, int, and float outputs.

Inputs:

* `source_type`
* `boolean`
* `string`
* `int`
* `float`
* `strict`

Outputs:

* `boolean`
* `string`
* `int`
* `float`

Default behavior:

* `source_type` decides which input value is used for conversion.
* `strict = false` falls back to safe defaults for invalid conversions.
* `strict = true` raises errors for invalid conversions.

Safe defaults:

* Invalid boolean conversions return `false`.
* Invalid int conversions return `0`.
* Invalid float conversions return `0.0`.

## mAI Random Line

Location:

```text
mAI / Utils
```

Purpose:
Randomly select one non-empty line from a multiline text input.

Input:

* `text`

Output:

* `line`

Default behavior:

* Uses Windows and Unix line endings.
* Ignores empty and whitespace-only lines.
* Preserves the selected line text except for the removed line break.
* Returns an empty string when there are no non-empty lines.
* Produces a fresh random choice on each queued execution.

Example:

Input:

```text
Line 1
Line 2
```

Possible output:

```text
Line 2
```

Known limitations:

* There is no visible seed input, so results are intentionally not reproducible.

## mAI text sequence randomizer

Location:

```text
mAI / Text
```

Purpose:
Randomize the order of items in a text sequence with a reproducible seed.

Inputs:

* `text`
* `separator` (`comma` or `line break`)
* `seed`

Output:

* `text`

Default behavior:

* Uses comma-separated items by default.
* Trims whitespace around each item and ignores empty items.
* Joins comma-separated output with a comma and one space.
* A fixed seed always produces the same order for the same text and separator.
* The seed widget supports ComfyUI's Fixed, Increment, Decrement, and Randomize modes.
* Returns an empty string when there are no non-empty items.

Example:

Input:

```text
text 1, text 2, text 3
```

Possible output:

```text
text 2, text 1, text 3
```

Known limitations:

* Commas inside an item are treated as separators in `comma` mode.
* Random shuffling can occasionally leave the sequence in its original order.

## mAI Example Text Node

Location:

```text
mAI / Template
```

Purpose:
Small registered example node that returns a text string, optionally stripping leading and trailing whitespace.

Inputs:

* `text`
* `strip_whitespace`

Output:

* `text`

Default behavior:

* `strip_whitespace = true` removes leading and trailing whitespace.
* `strip_whitespace = false` returns the input text unchanged.

## mAI auto seamless loop

Location: `mAI / Image`. Registered as `MAIAutoSeamlessLoop`.
Automatically search beginning/end trims and overlap jointly for an ordered IMAGE
batch. Tensor-only PyTorch operation; no encoder, downloaded model or extra dependency.

Inputs: `images`, positive `fps` (24 by default), `device` (auto/cuda/cpu), `quality`
(fast/balanced/high), `max_trim_start` (8), `max_trim_end` (8), `max_fade` (12), and
`min_retained_percent` (70% of input, applied after overlap). Advanced controls
expose score weights, sRGB/linear-light blending and manual trims/overlap (-1=auto).
Missing optional controls use defaults. CPU/CUDA execution respects ComfyUI's
device policy, supports progress/cancellation and reports bounded OOM fallback.

Outputs, in order: `images`, `fps`, `frame_count`, `trim_start`, `trim_end`,
`overlap`, `report` (JSON). The cycle is the unblended middle followed by the
blended bridge; length is `N - trim_start - trim_end - overlap`, with exact duration
in the report. FPS stays unchanged. Satisfactory unchanged clips retain all frames.

Install this pack under `ComfyUI/custom_nodes`, restart ComfyUI, and load
[the example workflow](examples/auto_seamless_loop.json); select a video from the
input folder. Inspect the encoded repeats and connect `report` to a STRING viewer.
The example uses existing IO nodes and software libx264; only those IO steps need
video codecs. An existing generated IMAGE batch can connect directly to the node.

The objective is a documented two-stage heuristic, not a guarantee of perceptual
perfection. Tiny subjects, occlusions and motion may remain difficult; uncertain
or poor results are reported. Default shortening/phase rotation does not preserve
audio alignment. High mode uses larger proxies rather than optical flow.
For clips already intended to loop, objective version 4 searches every permitted
start/end trim pair and each valid overlap from zero through `max_fade`. Both
scoring stages inspect only candidate boundary neighborhoods and the actual
fade; the middle does not supply motion tolerances or an activity penalty.
The score checks microjumps, nearby motion, brightness/color, contrast changes,
and structural ghosting using an 8x8 detail grid. `exposure_weight` controls both
brightness/color and contrast. Ghost scoring compares structure at matched tone,
so a lighting change alone is not treated as a doubled image; rendered colors
are never automatically normalized. Retention and duration/fade penalties keep
repairs short. Unchanged wins only if satisfactory and within `tie_tolerance`
of the best refined score. For equally short satisfactory repairs, prefer fewer
fully discarded source frames; a cut no longer wins just because K is smaller.
Ghosting measures the energy of the weaker mixed contour, so faint short blends
are not charged almost the whole endpoint mismatch. Local region checks prevent
overall endpoint motion from hiding a quiet subject's jump. Refinement includes
representatives of all fade lengths for small ranges (including 0..12 in balanced),
and evenly samples large ranges within a bounded budget.
The JSON reports `contrast`, `objective_version=4`, `fade_comparison`, and the scoring
scope; the existing `activity_loss` report key remains, fixed at zero.
Fast mode can still miss subtle
localized jumps; use balanced/high and inspect repeated playback. If automatic
trimming leaves a visible cut, advanced manual overlap can force a short fade
(1 is a midpoint blend; 2 contains only unmixed endpoints). This can introduce
ghosting and does not align objects or repair generated geometry.
See [construction, scoring and controls](docs/auto_seamless_loop.md),
[RunPod/Modal examples](deployment/auto_seamless_loop/README.md), and
[benchmark utility](scripts/benchmark_seamless_loop.py). Run
`python -m pytest tests/test_seamless_loop.py -q` for focused tests; CUDA cases skip
on CPU hosts. RTX 5090 and CPU tested locally; RTX PRO 6000, B200/B300 and cloud
deployments remain unverified on hardware.

## Testing

Run the Python tests from this folder:

```powershell
python -m pytest
```

The tests cover pure node and utility behavior where possible.
