# Build from this repository root, for linux/amd64.
# NGC 26.02 pins PyTorch 2.11.0a0+eb65b36914 and CUDA 13.1.1, including B300.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
FROM ${BASE_IMAGE}
ARG COMFYUI_REF=fb2315f11db0ebfaafa9099a5df5227dc6bb42bc
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout ${COMFYUI_REF}
COPY . /opt/ComfyUI/custom_nodes/mAI_Utils_V02
RUN python /opt/ComfyUI/custom_nodes/mAI_Utils_V02/deployment/auto_seamless_loop/install_dependencies.py
WORKDIR /opt/ComfyUI
EXPOSE 8188
# A mount at /workspace does not obscure the installed ComfyUI or node pack.
CMD ["python", "custom_nodes/mAI_Utils_V02/deployment/auto_seamless_loop/start_comfyui.py"]
