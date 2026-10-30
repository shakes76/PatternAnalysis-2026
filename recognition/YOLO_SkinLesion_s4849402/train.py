"""Train YOLO on train/validation only."""

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from modules import CLASS_NAMES, DetectorConfig, build_detector, model_summary
from utils import read_config, save_json, sha256


@dataclass(frozen=True)
class TrainConfig:
    data_yaml: Path
    output_dir: Path
    image_size: int = 640
    epochs: int = 50
    batch_size: int = 8
    device: str = "0"
    learning_rate: float = 0.001
    seed: int = 42
    workers: int = 2


def validate_training_data(config: TrainConfig) -> dict:
    path = config.data_yaml.resolve()
    data = read_config(path)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if tuple(names or []) != CLASS_NAMES:
        raise ValueError("Dataset class order does not match the model")
    root = Path(data.get("path", path.parent))
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    resolved = {}
    for split in ("train", "val"):
        value = data.get(split)
        if not isinstance(value, str):
            raise ValueError("This entry point expects exported image directories")
        folder = (root / value).resolve()
        if not folder.is_dir() or not any(folder.iterdir()):
            raise ValueError(f"Missing/empty {split}: {folder}")
        resolved[split] = folder
    if resolved["train"] == resolved["val"]:
        raise ValueError("Training and validation paths overlap")
    image_ids = {split: {p.stem for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
                 for split, folder in resolved.items()}
    if image_ids['train'] & image_ids['val']:
        raise ValueError('Training and validation image IDs overlap')
    audit_path = root / "audit.json"
    audit = read_config(audit_path)
    if audit.get("split_policy") not in ("patient-id", "official-isic"):
        raise ValueError("Missing split provenance; export with dataset.py first")
    if audit["split_policy"] == "official-isic" and not audit.get("split_evidence"):
        raise ValueError("Official split requires documented confirmation")
    for split in ('train', 'val'):
        if set(audit.get('image_ids', {}).get(split, [])) != image_ids[split]:
            raise ValueError(f'{split} image inventory changed since export')
    return {"dataset_config_sha256": sha256(path), "audit_sha256": sha256(audit_path),
            "audit": audit, "paths": resolved}


def run_training(model_config: DetectorConfig, train_config: TrainConfig) -> Path:
    import torch

    if train_config.epochs < 1 or train_config.batch_size < 1 or train_config.workers < 0:
        raise ValueError("Invalid training budget")
    if train_config.image_size < 32 or train_config.image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    if train_config.learning_rate <= 0 or model_config.seed != train_config.seed:
        raise ValueError("Invalid learning rate or inconsistent seeds")
    provenance = validate_training_data(train_config)
    output = train_config.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new experiment directory: {output}")
    detector = build_detector(model_config)
    output.mkdir(parents=True)
    save_json(output / "experiment.json", {
        "model": asdict(model_config), "training": asdict(train_config),
        "provenance": provenance, "initial_model": model_summary(detector),
    })

    def check_head(trainer):
        if tuple(trainer.data["names"][i] for i in range(3)) != CLASS_NAMES:
            raise ValueError("Trainer changed class order")
        if trainer.model.model[-1].nc != 3:
            raise ValueError('Trainer did not build a three-class detection head')
        save_json(output / "training_model.json", {
            "head_classes": trainer.model.model[-1].nc,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
        })
        if train_config.device != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(trainer.device)

    detector.add_callback("on_train_start", check_head)
    start = time.perf_counter()
    detector.train(
        data=str(train_config.data_yaml.resolve()), project=str(output.parent), name=output.name,
        exist_ok=True, imgsz=train_config.image_size, epochs=train_config.epochs,
        batch=train_config.batch_size, device=train_config.device, workers=train_config.workers,
        optimizer="AdamW", lr0=train_config.learning_rate, seed=train_config.seed,
        deterministic=True, patience=0, val=True, save=True, plots=True,
        pretrained=True, amp=False, resume=False, cache=False, fraction=1.0,
        multi_scale=False, cos_lr=False, freeze=0,
        box=7.5, cls=0.5, dfl=1.5,
    )
    best = Path(detector.trainer.best)
    if not best.is_file():
        raise FileNotFoundError("Training did not produce best.pt")
    peak = None
    if train_config.device != "cpu" and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(detector.trainer.device)
    save_json(output / "training_summary.json", {
        "seconds": time.perf_counter() - start, "peak_gpu_allocated_bytes": peak,
        "best_checkpoint": str(best), "best_sha256": sha256(best),
        "final_model": model_summary(detector), "test_evaluated": False,
        "best_selection": "Ultralytics validator fitness; inspect installed version and results.csv",
    })
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    config = TrainConfig(args.data, args.output, args.imgsz, args.epochs, args.batch,
                         args.device, args.lr, args.seed, args.workers)
    print(run_training(DetectorConfig(weights=args.weights, seed=args.seed), config))


if __name__ == "__main__":
    main()
