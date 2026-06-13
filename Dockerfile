# torch入り公式イメージ(pipでtorchを入れ直さない=依存事故ゼロ)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 全ピン: LTX-Video対応直後(2024年末〜2025年1月)の整合セット
# 未ピンだと最新transformersがFA3カーネルをtorch.libraryに登録しようとして
# infer_schemaで即死する(これが今までのexit code 1の真因)
RUN pip install --no-cache-dir \
    diffusers==0.32.2 \
    transformers==4.47.1 \
    accelerate==1.2.1 \
    huggingface_hub==0.27.1 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    protobuf==5.29.3 \
    numpy==1.26.4 \
    Pillow==10.4.0 \
    imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    runpod==1.7.7

# ガード1: FA3/kernels系が紛れ込んでいないことを保証
RUN python -c "import importlib.util, sys; \
    bad=[b for b in ('kernels','flash_attn','flash_attn_interface') if importlib.util.find_spec(b)]; \
    sys.exit('NG: '+str(bad)) if bad else print('OK: FA3/kernels なし')"

# ガード2: import時クラッシュをビルド段階で検出(GPU不要)
RUN python -c "import torch, transformers, diffusers; \
    from diffusers import LTXImageToVideoPipeline; \
    from diffusers.utils import export_to_video; \
    import runpod; \
    print('import smoke OK:', torch.__version__, diffusers.__version__, transformers.__version__)"

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
