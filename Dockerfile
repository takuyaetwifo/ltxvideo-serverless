# flash_attn非依存のクリーンベース
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorchを先にインストール(flash_attnなし)
RUN pip3 install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 残りの依存
RUN pip3 install --no-cache-dir \
    "diffusers>=0.32.0" \
    "transformers>=4.45.0" \
    accelerate \
    sentencepiece \
    "imageio[ffmpeg]" \
    Pillow \
    runpod \
    huggingface_hub

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
