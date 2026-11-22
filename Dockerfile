FROM python:3.8-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/prakashdeepika/SadTalker"
LABEL org.opencontainers.image.description="CPU runtime and model weights for the SadTalker video workflow"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/sadtalker
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade "pip<25" \
    && python -m pip install \
        torch==1.12.1+cpu \
        torchvision==0.13.1+cpu \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

RUN mkdir -p checkpoints gfpgan/weights \
    && curl --fail --location --retry 3 \
        https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar \
        --output checkpoints/mapping_00109-model.pth.tar \
    && curl --fail --location --retry 3 \
        https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors \
        --output checkpoints/SadTalker_V0.0.2_512.safetensors \
    && curl --fail --location --retry 3 \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth \
        --output gfpgan/weights/alignment_WFLW_4HG.pth \
    && curl --fail --location --retry 3 \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        --output gfpgan/weights/detection_Resnet50_Final.pth \
    && curl --fail --location --retry 3 \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
        --output gfpgan/weights/GFPGANv1.4.pth \
    && curl --fail --location --retry 3 \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth \
        --output gfpgan/weights/parsing_parsenet.pth
