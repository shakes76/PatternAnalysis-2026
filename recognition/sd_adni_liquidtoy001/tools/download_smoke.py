"""Fetch the minimum pinned SD components for real LoRA acceptance."""

import hashlib
import json
from pathlib import Path

from diffusers import DDPMScheduler, UNet2DConditionModel
from download_vae import MODEL, REVISION
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer


def main():
    destination = Path("models/sd15-smoke")
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=[
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/*",
            "scheduler/scheduler_config.json",
        ],
    )
    CLIPTokenizer.from_pretrained(
        destination, subfolder="tokenizer", local_files_only=True
    )
    CLIPTextModel.from_pretrained(
        destination,
        subfolder="text_encoder",
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    )
    DDPMScheduler.from_pretrained(
        destination, subfolder="scheduler", local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        destination,
        subfolder="unet",
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    )
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"]
        )
    )
    assert any(p.requires_grad for p in unet.parameters())
    files = {}
    for path in sorted(destination.rglob("*")):
        if (
            not path.is_file()
            or ".cache" in path.parts
            or path.name == "provenance.json"
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[str(path.relative_to(destination))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    result = {"model": MODEL, "revision": REVISION, "files": files}
    (destination / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
