"""Fetch only the pinned SD 1.5 VAE; never download a full pipeline."""

import hashlib
import json
import shutil
from pathlib import Path

from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def main():
    destination = Path("models/sd15-vae")
    if shutil.disk_usage(".").free < 2 * 1024**3:
        raise RuntimeError("Require at least 2 GiB free before VAE preparation")
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
    )
    AutoencoderKL.from_pretrained(
        destination, subfolder="vae", local_files_only=True, use_safetensors=True
    )
    files = {}
    for path in sorted((destination / "vae").iterdir()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    record = {"model": MODEL, "revision": REVISION, "files": files}
    (destination / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
