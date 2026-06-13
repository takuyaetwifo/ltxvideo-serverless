# -*- coding: utf-8 -*-
"""RunPod Serverless ワーカー: 画像(base64) + プロンプト -> LTX-Video -> MP4(base64)を返す。
   入力 job["input"]:
     image_b64      : 画像のbase64 (必須)
     prompt         : 動きの指示 (任意)
     negative_prompt: ネガティブ (任意)
     width          : 幅 px (任意, デフォルト704, 32の倍数に丸める)
     height         : 高さ px (任意, デフォルト480, 32の倍数に丸める)
     num_frames     : フレーム数 (任意, デフォルト97=約5秒, 8n+1に丸める)
     steps          : 推論ステップ数 (任意, デフォルト30)
   出力: {"video_b64": "...", "bytes": N, "frames": N}
"""
import os, sys

# importより前にキャッシュ先を確定。Network Volumeがあればそちら(コールドスタートDL回避)
if os.path.isdir("/runpod-volume"):
    os.environ["HF_HOME"] = "/runpod-volume/hf_cache"

import io, base64, tempfile, traceback

print(f"Python: {sys.version}", flush=True)

from PIL import Image
import torch
import runpod
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
import diffusers, transformers
print(f"torch={torch.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__} CUDA={torch.cuda.is_available()}", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("LTX-Video モデルをロード中...(初回は約15GBダウンロード)", flush=True)
_pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16,
)
# .to(cuda) と enable_model_cpu_offload は排他。VRAMで切替
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
if vram_gb >= 20:
    _pipe.to(DEVICE)          # 24GB級: 全載せが最速
    print(f"VRAM {vram_gb:.0f}GB -> 全モデルGPU載せ", flush=True)
else:
    _pipe.enable_model_cpu_offload()   # 16GB級: T5-XXLが重いのでオフロード
    print(f"VRAM {vram_gb:.0f}GB -> CPUオフロード", flush=True)
_pipe.vae.enable_tiling()     # 高解像度VAEデコードのOOM保険
print("モデルロード完了", flush=True)


def handler(job):
    inp = job.get("input", {}) or {}
    b64 = inp.get("image_b64")
    if not b64:
        return {"error": "image_b64 がありません"}

    prompt   = inp.get("prompt", "cinematic motion, natural movement, hair flowing, cloak billowing")
    negative = inp.get("negative_prompt", "worst quality, blurry, jittery, distorted, artifacts")
    width    = int(inp.get("width",  704))
    height   = int(inp.get("height", 480))
    frames   = int(inp.get("num_frames", 97))
    steps    = int(inp.get("steps", 30))
    fps      = int(inp.get("fps", 25))
    seed     = int(inp.get("seed", 42))

    try:
        raw   = base64.b64decode(b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # LTX-Videoは32の倍数サイズ・フレーム数は8n+1が必要
        width  = max(32, width  // 32 * 32)
        height = max(32, height // 32 * 32)
        frames = max(9, (frames - 1) // 8 * 8 + 1)

        with torch.inference_mode():
            result = _pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=3.0,
                generator=torch.Generator(DEVICE).manual_seed(seed),
            )

        video_frames = result.frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        export_to_video(video_frames, path, fps=fps)

        video_bytes = open(path, "rb").read()
        os.unlink(path)
        return {
            "video_b64": base64.b64encode(video_bytes).decode("ascii"),
            "bytes":     len(video_bytes),
            "frames":    frames,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}


runpod.serverless.start({"handler": handler})
