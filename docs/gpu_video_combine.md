# GPU Video Combine

`nodes/gpu_video_combine.py` provides the `MAIGPUVideoCombine` ComfyUI output
node. `utils/video_encoding.py` handles pure settings/sequence logic, input
validation, chunk conversion and the restricted FFmpeg encoding bridge. No
VHS code is copied or imported, and no ComfyUI core files are modified.

The scope follows the basic functionality of
[VHS Video Combine](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite#video-combine):
IMAGE frames, optional AUDIO, frame rate, filename prefix, output/temp saving,
quality, ping-pong, loops, workflow metadata and a video preview. This node
exports MP4 using NVENC; it does not reproduce GIF/WebP, arbitrary format JSON,
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
deadlocks; failed encoding reports the actual diagnostics. No software codec
fallback hides an unavailable hardware encoder.

Automatic chunks target 64 MiB of packed NV12 (up to 32 frames), not a strict
total-memory limit. Float conversion, source slices, selected indices and
encoder buffers consume additional memory. User-selected chunk size can use
more memory. Full frames stay allocated by ComfyUI. Interrupted execution is
checked between chunks and while draining the encoder; a blocking pipe write
must finish or fail before another cancellation check can run.

## Validation

On 2026-10-02, the complete suite with `MAI_TEST_NVENC=1` passed **895 tests**
on Windows, RTX 5090 and PyTorch 2.7.1+cu128. All three NVENC codecs encoded
successfully. The node pack imported with its new registration, and all existing
class/display mappings were checked against the previous Git version.

Pure/tensor tests cover option ranges, path validation, NV12 packing and
independent BT.709 reference colors,
FP32/FP16/BF16, RGB/RGBA/grayscale, odd padding, clipping, input preservation,
frame sequence/count, audio timing and interleaving, metadata escaping,
partial pipe writes, fixed NVENC command construction, optional audio, output
sockets, native preview descriptors and output/temp selection.

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

Tests without the environment flag skip hardware integration and do not require
an FFmpeg installation. Runtime imports and preview UI behavior inside the
user's live ComfyUI remain subject to the previously identified broken local
venv launcher; the native preview response follows the installed ComfyUI core's
`PreviewVideo` format. Linux uses the same subprocess/torch APIs but was not
tested on a Linux host.
