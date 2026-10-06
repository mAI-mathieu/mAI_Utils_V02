# RunPod and Modal examples

Definitions only: no GPU is rented and no service is published during development.
Core node use is offline after installation. Hosting/IO dependencies are separate.
Official documentation inspected on 2026-10-06:

- [ComfyUI IMAGE interface](https://github.com/Comfy-Org/docs/blob/main/custom-nodes/walkthrough.mdx)
  and [ComfyUI implementation](https://github.com/Comfy-Org/ComfyUI).
- [Modal GPU requirements](https://modal.com/docs/guide/gpu): supports
  `RTX-PRO-6000`, `B200`, `B300`; **B300 requires CUDA 13.1+**.
- [NGC PyTorch 26.02 release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-02.html):
  `nvcr.io/nvidia/pytorch:26.02-py3` bundles CUDA 13.1.1 and
  PyTorch 2.11.0a0+eb65b36914 on Python 3.12/Ubuntu 24.04. This is an NVIDIA
  tested framework build, not upstream stable PyTorch. Check provider drivers
  against the linked CUDA compatibility requirements before deployment.
- [RunPod templates](https://docs.runpod.io/pods/templates/overview) accept custom
  images, exposed ports and persistent volume mount paths.
- [Modal image API](https://modal.com/docs/sdk/py/latest/Image),
  [web_server API](https://modal.com/docs/sdk/py/latest/web_server), and
  [Volumes](https://modal.com/docs/guide/volumes) for image inclusion,
  authenticated HTTP proxy and persistence.

Both examples pin the framework image and ComfyUI commit
`fb2315f11db0ebfaafa9099a5df5227dc6bb42bc`, the inspected installed upstream
revision. The build helper filters direct torch installs and constrains torch and
torchvision to image versions; it never upgrades them. NGC's own pip constraints
also apply. A pip conflict fails the build rather than replacing the framework.
Builds save `image-requirements.lock.txt` under `/opt/ComfyUI`. ComfyUI's other
transitive dependency ranges remain upstream ranges: retain the generated lock,
image digest and built image for byte-for-byte deployment reproduction. These are
documented compatible choices, not a claim that the Linux image was built locally.

## RunPod Pod

From the custom node repository root (Docker with Linux containers):

```bash
docker build --platform linux/amd64 -f deployment/auto_seamless_loop/Dockerfile -t mai-comfy-loop:26.02 .
docker run --gpus all --rm -p 8188:8188 -v "$PWD/workspace:/workspace" mai-comfy-loop:26.02
```

To host it, explicitly push the built image to your own registry and select it in
a RunPod Pod template. Expose `8188/http`; choose the requested GPU and a persistent
volume mounted at `/workspace`. Place video input under `/workspace/input`, output
under `/workspace/output`, user settings under `/workspace/user`. The source and
ComfyUI are baked into `/opt/ComfyUI`, so the volume does not hide the installation.
For local development optionally bind-mount this repository read-only over
`/opt/ComfyUI/custom_nodes/mAI_Utils_V02`, then restart the container after edits.
Use a provider-supported authenticated access path for the ComfyUI HTTP API.

Open the Pod's HTTP endpoint and load `examples/auto_seamless_loop.json`, or POST
the API example to `/prompt` with a file named `loop_input.mp4` in the input folder.
For CPU testing omit `--gpus all` and override command to
`python custom_nodes/mAI_Utils_V02/deployment/auto_seamless_loop/start_comfyui.py --cpu`.
The core optimizer does not require GPU encoders; the example explicitly uses
libx264 on RTX PRO, B200/B300 and CPU.

## Modal

In a separate hosting environment install the Modal SDK and configure your own
account. No SDK or credentials are required by the ComfyUI node. `MAI_MODAL_GPU`
selects `RTX-PRO-6000` (default), `B200`, or `B300`; all use the pinned CUDA 13.1.1
image to avoid assuming the RTX 5090's CUDA 12.8 runtime covers B300.

Explicit commands (Linux/macOS; PowerShell uses `$env:MAI_MODAL_GPU = 'B300'`):

```bash
MAI_MODAL_GPU=B300 modal run deployment/auto_seamless_loop/modal_app.py
MAI_MODAL_GPU=B200 modal serve deployment/auto_seamless_loop/modal_app.py
MAI_MODAL_GPU=RTX-PRO-6000 modal deploy deployment/auto_seamless_loop/modal_app.py
```

`run` executes the benchmark and incurs GPU use only when explicitly invoked.
`serve` creates a temporary development server; `deploy` publishes it. The HTTP
server requires Modal proxy authentication (`Modal-Key` / `Modal-Secret` headers;
use your own proxy credentials). Keep credentials out of workflows/source. Maximum
containers per function is one to keep ComfyUI queue/history on a single worker;
this is an example, not distributed queue storage.

The node pack is copied into the image. `mai-comfy-workspace` is mounted at
`/workspace`. Upload `loop_input.mp4` to `input/` with Modal's Volume tools, then
start/restart the ComfyUI container to see it. Submit the same API prompt graph to
the authenticated URL. Volumes commit in the background/on shutdown; changes from
another container require reload or a new container. Use the Volume tools to
retrieve output. Model assets are unnecessary for this sample; other workflows
must mount/install their own models separately.

## Verification status

CPU and RTX 5090 were exercised locally with synthetic tensor tests and phase
benchmarks, using existing PyTorch 2.7.1+cu128 / CUDA 12.8, compute capability 12.0.
RTX PRO 6000 workstation/server, B200, B300 and the Linux Docker/Modal deployments
remain **supported by design/documented compatibility, not hardware verified**.
No container was built and no cloud service was started here. These hosts require
their own smoke test and repeated-playback inspection. No architecture extensions,
flow hardware, NVENC or downloaded models are required for optimization.
