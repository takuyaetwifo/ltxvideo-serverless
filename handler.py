# -*- coding: utf-8 -*-
"""RunPod Serverless ワーカー: 画像(base64) + プロンプト -> LTX-Video -> MP4(base64)を返す。
   入力 job["input"]:
     image_b64      : 画像のbase64 (必須)
     prompt         : 動きの指示 (任意)
     negative_prompt: ネガティブ (任意)
     width          : 幅 px (任意, デフォルト704)
     height         : 高さ px (任意, デフォルト480)
     num_frames     : フレーム数 (任意, デフォルト97=約5秒)
     steps          : 推論ステップ数 (任意, デフォルト30)
   出力: {"video_b64": "...", "bytes": N, "frames": N}
"""
import os, io, base64, tempfile, sys, traceback

print(f"Python: {sys.version}", flush=True)

try:
    from PIL import Image
    print("PIL OK", flush=True)
except Exception as e:
    print(f"PIL ERROR: {e}", flush=True); sys.exit(1)

try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"torch ERROR: {e}", flush=True); sys.exit(1)

try:
    import runpod
    print("runpod OK", flush=True)
except Exception as e:
    print(f"runpod ERROR: {e}", flush=True); sys.exit(1)

try:
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
    import diffusers
    print(f"diffusers: {diffusers.__version__} OK", flush=True)
except Exception as e:
    print(f"diffusers ERROR: {e}", flush=True); traceback.print_exc(); sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

print("LTX-Video モデルをロード中...", flush=True)
try:
    _pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(DEVICE)
    _pipe.enable_model_cpu_offload()
    print("モデルロード完了", flush=True)
except Exception as e:
    print(f"モデルロードERROR: {e}", flush=True); traceback.print_exc(); sys.exit(1)


def handler(job):
    inp = job.get("input", {}) or {}
    b64 = inp.get("image_b64")
    if not b64:
        return {"error": "image_b64 がありません"}

    prompt   = inp.get("prompt", "cinematic motion, natural movement, hair flowing, cloak billowing")
    negative = inp.get("negative_prompt", "worst quality, blurry, jittery, distorted, artifacts")
    width    = int(inp.get("width",  704))
    height   = int(inp.get("height", 480))
    frames   = int(inp.get("num_frames", 97))   # 97 ≈ 5秒 @25fps
    steps    = int(inp.get("steps", 30))
    fps      = int(inp.get("fps", 25))
    seed     = int(inp.get("seed", 42))

    try:
        raw   = base64.b64decode(b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # LTX-Videoは8の倍数サイズが必要
        width  = (width  // 8) * 8
        height = (height // 8) * 8

        with torch.no_grad():
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
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}


runpod.serverless.start({"handler": handler})
