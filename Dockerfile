# RunPod Serverless 用 LTX-Video ワーカー
# TripoSRと違いコンパイル不要なのでruntimeベースで十分
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存インストール
RUN pip install --no-cache-dir \
    "diffusers>=0.32.0" \
    transformers>=4.45.0 \
    accelerate \
    sentencepiece \
    "imageio[ffmpeg]" \
    runpod \
    huggingface_hub

# モデルをイメージに焼き込み(コールドスタート短縮・起動10-20秒に)
RUN python -c "\
from diffusers import LTXImageToVideoPipeline; \
import torch; \
LTXImageToVideoPipeline.from_pretrained('Lightricks/LTX-Video', torch_dtype=torch.bfloat16); \
print('LTX-Video モデルキャッシュ完了') \
"

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
