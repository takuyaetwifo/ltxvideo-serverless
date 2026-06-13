FROM runpod/base:0.6.2-cuda12.2.0

ENV PYTHONUNBUFFERED=1 HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "diffusers>=0.32.0" \
    "transformers>=4.45.0" \
    accelerate sentencepiece \
    "imageio[ffmpeg]" Pillow \
    runpod huggingface_hub

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
