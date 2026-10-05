# GPU Video Combine

`nodes/gpu_video_combine.py` provides the `MAIGPUVideoCombine` ComfyUI output
node. `utils/video_encoding.py` handles pure settings/sequence logic, input
validation, chunk conversion and the restricted FFmpeg encoding bridge. No
VHS code is copied or imported, and no ComfyUI core files are modified.

The scope follows the basic functionality of
[VHS Video Combine](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite#video-combine):
IMAGE frames, optional AUDIO, frame rate, filename prefix, output/temp saving,
quality, ping-pong, loops, workflow metadata and a video preview. This node
exports MP4 using NVENC or explicit software codecs; it does not reproduce GIF/WebP, arbitrary format JSON,
latent/VAE decoding or VHS's incremental meta-batch protocol.

## Encoding path

1. Validate BHWC frames, options, and optional `[1, channels, samples]` audio.
2. Determine output count/FPS timing, including ping-pong and repeats. Sequence
   indices are generated per chunk, without constructing a second full batch.
3. Convert/clamp/quantize a batch chunk to 8-bit NV12 on its current tensor
   device. Small odd dimensions are padded to the next even size. FP32 work
   handles RGB-to-limited-range BT.709 conversion, with box-filtered 2×2 chroma
   sampling. Luma/chroma use SDR ranges 16–235 / 16–240, respectively.
4. Transfer that packed chunk to CPU and stream a zero-copy NumPy byte view
   to FFmpeg. There are no per-frame Python conversion loops or image files.
5. FFmpeg accepts `nv12` directly into the selected NVENC encoder; compression
   runs on the GPU. Input and output are tagged limited-range BT.709 with
   centered chroma, matching the tensor conversion. NV12 requires 1.5 bytes per
   pixel instead of RGBX's 4, reducing bridge data by 62.5% without an additional
   CPU color-conversion filter.
   With `libx264`, `libx265` or `libsvtav1`, FFmpeg instead deinterleaves NV12 to
   YUV420P and compresses on CPU. The tensor conversion still uses the frame device.
6. Audio is written as temporary, interleaved float PCM in sample chunks;
   FFmpeg performs AAC encoding and muxing. Excess samples are discarded and
   short audio is padded. Metadata uses a temporary FFmetadata input, avoiding
   Windows command-line length limits for large saved workflows.
7. After successful completion, the encoded file replaces an exclusively
   reserved empty destination owned by this run. Existing files are preserved;
   temporary files and an unfinished encoder are cleaned up on exceptions.

NVENC use through FFmpeg follows
[NVIDIA's FFmpeg documentation](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html).
There is no `-hwaccel cuda` flag because the input is raw frames, with no
compressed video to decode. This implementation does not claim a zero-copy
CUDA torch-to-encoder path: FFmpeg's process/stdin boundary requires CPU
staging. NVIDIA's encoder is independent of CUDA tensor compute, and FFmpeg
is allowed to select a different available GPU unless a device index is set.

All subprocess calls use a known FFmpeg executable, fixed argument lists and
`shell=False`. No shell scripts, custom command fields or network inputs are
exposed. Importing the pack or querying the node schema does not launch
FFmpeg, test a GPU, or initialize CUDA. Capability checks occur at execution.
FFmpeg error output is captured in a temporary file to avoid stderr pipe
deadlocks; failed encoding reports the actual diagnostics. Software encoding
requires selecting a software codec; no automatic fallback hides NVENC failures.

Automatic chunks target 64 MiB of packed NV12 (up to 32 frames), not a strict
total-memory limit. Float conversion, source slices, selected indices and
encoder buffers consume additional memory. User-selected chunk size can use
more memory. Full frames stay allocated by ComfyUI. Interrupted execution is
checked between chunks and while draining the encoder; a blocking pipe write
must finish or fail before another cancellation check can run.

## CUDA memory lifecycle

Both GPU nodes share `utils/gpu_memory.py`. Before work, garbage collection,
device synchronization and `torch.cuda.empty_cache()` return unused allocator
memory to the driver. For encoding, the node requests an estimated 48 bytes per
pixel per conversion frame and 2 GiB of headroom for FFmpeg's separate CUDA
context. Only when driver-visible free memory is below the budget does it call
ComfyUI's `free_memory` hook to offload models on that device. The hook's
reclaimable-cache accounting is adjusted so that external CUDA context memory
is not confused with cached PyTorch memory. No ComfyUI core code is changed.

An explicit FFmpeg `CUDA_ERROR_OUT_OF_MEMORY` retries once after the failed
encoder and temporary files have been cleaned up. The retry requests all
ComfyUI-managed models on the encoding/conversion devices to be offloaded and
converts one frame per chunk. Codec/device settings stay the same. Other errors
and cancellation propagate immediately; a second OOM propagates its diagnostics.
Cache cleanup runs after success/failure/cancellation. Traceback locations are
preserved, but inactive worker locals are cleared on errors to release scratch.

Live input tensors, output tensors and allocations belonging to other processes
cannot be freed this way. Models can reload at later nodes, and synchronization
and cache clearing add overhead. Budgets are estimates, not guarantees. On
multiple GPUs, an explicit `gpu_device` is recommended: preparation for `-1`
targets CUDA device 0 plus the frame device; FFmpeg may choose another capable
encoder. GPU indices follow the process's CUDA-visible device ordering.
Software codecs require conversion memory only and do not reserve the NVENC
context budget, check hardware support or retry hardware OOM errors.

## Cloud setup: Modal and Runpod

According to [NVIDIA's encoding support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix),
RTX PRO 6000 Blackwell (including Server Edition) has four NVENC engines and
supports H.264, HEVC and AV1. HGX B200/B300 and GB200/GB300 have zero NVENC
engines. NVDEC decoding support does not imply encoding support. Installing
FFmpeg or changing CUDA versions cannot add an encoder to these GPUs.

| GPU | Node codec | Compression |
| --- | --- | --- |
| RTX PRO 6000 Blackwell | `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` | NVENC, provided the container exposes the driver libraries |
| B200 or B300 | `libx264`, `libx265`, `libsvtav1` | CPU; CUDA tensor conversion remains available |

Existing workflows keep `h264_nvenc` as their default. On B200/B300, switch
the existing codec widget to `libx264` for broadly playable H.264 MP4 output.
All sockets, timing, audio, metadata and output paths are unchanged. Quality is
software CRF instead of NVENC CQ; equal numbers do not imply equal quality.
Presets p1–p7 map to ultrafast/superfast/veryfast/medium/slow/slower/veryslow for
x264/x265 and 12/10/8/6/5/4/3 for SVT-AV1. CPU encoding may take much longer;
use p1–p4 initially. `gpu_device` is ignored for software compression.

### FFmpeg installation

Install this repository's `requirements.txt` in ComfyUI's active Python
environment, not another venv. It now includes `imageio-ffmpeg`, whose bundled
executable is a fallback when system FFmpeg is absent. There is no download,
package installation or network call during node execution.

Discovery order is `MAI_FFMPEG_EXE`, `IMAGEIO_FFMPEG_EXE`, PATH, then
`imageio_ffmpeg.get_ffmpeg_exe()`. Overrides accept one executable path/name,
including spaces, without embedded quotes or arguments. An invalid override
fails clearly instead of silently using a different binary. A selected binary
must provide the requested encoder; installing a Python wrapper named `ffmpeg`
alone does not install the executable. Wheels/encoder availability can differ
on ARM64; install system FFmpeg there when needed.

For a Debian/Ubuntu-based Runpod image, include this in the image build:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

For an already running Debian/Ubuntu pod, `apt-get update && apt-get install -y ffmpeg`
fixes the missing executable if run with package-install permissions. Put the
installation into the template/image too so replacement pods retain it.

For Modal, extend the existing ComfyUI image definition:

```python
image = image.apt_install("ffmpeg").env({
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
})
```

Use that image in your existing function/sandbox and rebuild/redeploy it.
These methods follow [Modal's image documentation](https://modal.com/docs/guide/images).
NVENC on RTX PRO 6000 additionally requires `libnvidia-encode.so.1` and a
compatible host driver. In NVIDIA Container Toolkit deployments, the
`video` capability mounts the video driver libraries; defaults usually expose
only compute/utility. Configure it before container creation, as explained by
[NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
Changing the environment inside a running container cannot mount missing
libraries, and the capability variable alone does not guarantee Modal's
runtime exposes them. Verify with the actual encoding command below; if they
are unavailable, use an explicit software codec or an environment with video
driver support. Do not install host NVIDIA drivers into the ComfyUI venv.

Use ComfyUI/PyTorch/CUDA versions compatible with your GPU separately from
FFmpeg. [Modal's GPU guide](https://modal.com/docs/guide/gpu) currently specifies
CUDA 13.1+ for B300. This pack uses existing PyTorch operations and does not
install or replace the platform's CUDA/PyTorch packages.

### Verify inside the running cloud container

First confirm the actual executable and encoder (from this repository):

```bash
python -c "from utils.video_encoding import find_ffmpeg; print(find_ffmpeg())"
ffmpeg -hide_banner -h encoder=h264_nvenc
```

An encoder help/list entry proves build support only. On RTX PRO 6000, this
one-frame encode also checks driver visibility and hardware initialization:

```bash
ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=128x128:r=24 \
    -frames:v 1 -c:v h264_nvenc -gpu 0 -f null -
```

On B200/B300 replace `-c:v h264_nvenc -gpu 0` with `-c:v libx264`. If using
only the bundled executable, substitute the path printed by `find_ffmpeg()`
for `ffmpeg`; ffprobe is provided by the system package for integration tests.
Then restart ComfyUI, select the matching codec, connect a short frame batch,
queue the node and check the preview, audio and saved video. Test HEVC/AV1
separately if needed; browser playback support varies. Cloud machines were
not available for live testing in this workspace.

## Validation

On 2026-10-05, the full suite passed **1,008 tests** on the local Windows /
RTX 5090 / PyTorch 2.7.1+cu128 environment with both `MAI_TEST_NVENC=1` and
`MAI_TEST_FFMPEG=1`. System FFmpeg encoded all three NVENC and all three
software codecs; software integration ran with both CPU and CUDA frame tensors.
The pack imported successfully with 22 registered nodes, including the unchanged
`MAIGPUVideoCombine` mapping and socket contract.

The separate imageio-ffmpeg 0.6.0 Windows-wheel check confirmed executable
discovery when system FFmpeg is hidden, and seven successful encode/decode cases:
H.264/HEVC software with CPU/CUDA frames, plus all three NVENC codecs. This
wheel lacks `libsvtav1`; its two software AV1 cases correctly failed with the
missing-encoder diagnostic. Use a system FFmpeg with SVT-AV1 for that selection.
The Linux wheel and live Modal/Runpod RTX PRO 6000/B200/B300 were not tested;
the runtime check above is needed on each deployed cloud image.

On 2026-10-02, the complete suite with `MAI_TEST_NVENC=1` passed **913 tests**
on Windows, RTX 5090 and PyTorch 2.7.1+cu128. All three NVENC codecs encoded
successfully. The node pack imported with its new registration, and all existing
class/display mappings were checked against the previous Git version.

Pure/tensor tests cover option ranges, path validation, NV12 packing and
independent BT.709 reference colors,
FP32/FP16/BF16, RGB/RGBA/grayscale, odd padding, clipping, input preservation,
frame sequence/count, audio timing and interleaving, metadata escaping,
partial pipe writes, fixed NVENC command construction, optional audio, output
sockets, native preview descriptors and output/temp selection.
Memory regression tests cover targeted model offloading, driver-visible cache
accounting, cleanup on success/failure/cancellation, failed traceback locals,
CPU/identity bypasses, preserved live CUDA tensors, and the single NVENC OOM retry.

Opt-in integration tests actually encode and decode files using H.264, HEVC
and AV1 NVENC. They inspect streams with ffprobe, decode pixels with FFmpeg,
verify ordered endpoints, a 121-frame batch, ping-pong/repeats, mono/stereo
AAC, trim/pad timing, odd-sized dimensions, large Unicode workflow metadata,
unavailable-device diagnostics, and cleanup after interruption/failure.

Run on an NVIDIA host with an NVENC-enabled FFmpeg and ffprobe:

```powershell
$env:MAI_TEST_NVENC = '1'
python -m pytest tests/test_video_encoding.py tests/test_gpu_video_combine.py
```

On Linux:

```bash
MAI_TEST_NVENC=1 python -m pytest tests/test_video_encoding.py tests/test_gpu_video_combine.py
```

For software-only integration (also suitable for B200/B300):

```bash
MAI_TEST_FFMPEG=1 python -m pytest tests/test_video_encoding.py tests/test_gpu_video_combine.py
```

Software integration covers all three codecs with actual FFmpeg/ffprobe,
audio, Unicode metadata, odd dimensions, ping-pong/repeats and decoded frames.
Discovery regression tests simulate missing PATH binaries, bundled executables,
environment overrides and package errors without requiring FFmpeg or a GPU.

Tests without the environment flags skip encoding integration and do not require
an FFmpeg installation. Standalone pack imports were verified; preview UI
behavior inside a live cloud ComfyUI instance still requires deployment testing.
The native preview response follows ComfyUI's `PreviewVideo` format. Linux uses
the same subprocess/torch APIs but was not tested on a Linux host.
