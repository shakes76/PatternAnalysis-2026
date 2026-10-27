"""Display one checkpoint sample; test examples require explicit separate opt-in.

Argument parsing and the test gate use only the standard library. Importing this
module, --help, or requesting test without --allow-test-example cannot import a
model framework, scan a data split, load a checkpoint or query CUDA.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re


SPLIT_FOLDERS = {
    "training": ("keras_slices_train", "keras_slices_seg_train"),
    "validation": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[4]
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=workspace / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--split", choices=tuple(SPLIT_FOLDERS), default="training")
    parser.add_argument("--filename", default="case_004_week_0_slice_12.nii.gz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_reference"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-test-example", action="store_true",
                        help="Explicitly authorize a new test-sample forward; absent by default. Do not use for submission checks.")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test_example:
        parser.error("Test example inference is disabled by default. A separate authorized run requires --allow-test-example; existing test scores do not authorize it.")
    if args.allow_test_example and args.split != "test":
        parser.error("--allow-test-example is only meaningful with --split test.")
    if (Path(args.filename).name != args.filename or "\\" in args.filename
            or not re.fullmatch(r"case_\d+_week_\d+_slice_\d+\.nii\.gz", args.filename)):
        parser.error("filename must be an exact MRI basename case_<id>_week_<n>_slice_<n>.nii.gz.")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve previous inference evidence.")
    return args


def checkpoint_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_preprocessed_sample(model, sample, device, num_classes):
    """Apply the existing single-sample collate/valid/loss/crop rules in eval mode.

    The caller owns data selection and checkpoint provenance. No optimizer is
    created. B1 inspection is not equivalent to the formal B4 split evaluation.
    """
    import torch
    from dataset import collate_padded, crop_to_original
    from losses import segmentation_cross_entropy
    from metrics import SegmentationMetrics

    model.eval()
    batch = collate_padded([sample])
    image, target, valid = [batch[key].to(device) for key in ("image", "target", "valid_mask")]
    with torch.inference_mode():
        logits = model(image)
        if logits.shape != (1, num_classes, *target.shape[-2:]):
            raise ValueError("Model output must match [1,num_classes,padded_H,padded_W].")
        loss = segmentation_cross_entropy(logits, target, valid)
        prediction = logits.argmax(dim=1)
    metrics = SegmentationMetrics(num_classes)
    metrics.update(prediction, target, valid)
    size = sample["original_size"]
    return {
        "image": crop_to_original(sample["image"], size)[0].cpu().numpy(),
        "target": crop_to_original(sample["target"], size).cpu().numpy(),
        "prediction": crop_to_original(prediction[0].cpu(), size).numpy(),
        "cross_entropy": float(loss.item()), "metrics": metrics.compute(),
        "logit_shape": list(logits.shape), "original_size": list(size),
    }


def save_legacy_prediction(output_path, result, checkpoint, split, filename):
    """Keep the existing training/validation three-panel layout and palette."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    num_classes = checkpoint["model_config"]["num_classes"]
    palette = ["black", "gold", "deepskyblue", "magenta", "limegreen", "orange"]
    colors = ListedColormap(palette[:num_classes]) if num_classes <= 6 else plt.get_cmap("tab20", num_classes)
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), layout="constrained")
    axes[0].imshow(result["image"], cmap="gray", interpolation="nearest")
    axes[0].set_title({"training":"Training MRI", "validation":"Validation MRI", "test":"Test MRI"}[split])
    for axis, values, title in ((axes[1], result["target"], "Reference mask"),
                                (axes[2], result["prediction"], f"{checkpoint['model']} prediction")):
        shown = axis.imshow(values, cmap=colors, vmin=-0.5, vmax=num_classes-0.5, interpolation="nearest")
        axis.set_title(title)
    fig.colorbar(shown, ax=list(axes[1:]), ticks=list(range(num_classes)), label="Class ID")
    fig.suptitle(f"{split}: {filename} | checkpoint epoch {checkpoint['epoch']}")
    try:
        with Path(output_path).open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150)
    finally:
        plt.close(fig)


def execute_prediction(args):
    """Run a caller-authorized example, with the test gate repeated before imports."""
    if args.split not in SPLIT_FOLDERS:
        raise ValueError("Unsupported split.")
    if args.split == "test" and not getattr(args,"allow_test_example",False):
        raise PermissionError("Test example inference requires explicit separate opt-in.")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve previous inference evidence.")
    if (Path(args.filename).name != args.filename or "\\" in args.filename
            or not re.fullmatch(r"case_\d+_week_\d+_slice_\d+\.nii\.gz",args.filename)):
        raise ValueError("filename must be an exact MRI basename.")
    import torch
    from dataset import collect_pairs, load_pair
    from modules import build_model
    from prediction_panels import save_prediction_panels

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    image_folder, mask_folder = SPLIT_FOLDERS[args.split]
    pairs = collect_pairs(args.data_root / image_folder, args.data_root / mask_folder)
    pair = next((pair for pair in pairs if pair[0].name == args.filename), None)
    if pair is None:
        raise ValueError("Requested filename not found in the explicitly selected split.")
    checkpoint_digest = checkpoint_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_name = checkpoint["model"]
    model = build_model(model_name, **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sample = load_pair(*pair)
    result = infer_preprocessed_sample(model,sample,device,checkpoint["model_config"]["num_classes"])
    if checkpoint_sha256(args.checkpoint) != checkpoint_digest:
        raise RuntimeError("Checkpoint file changed during sample inspection.")
    args.output_dir.mkdir(parents=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".matplotlib"))
    save_legacy_prediction(args.output_dir/"prediction.png",result,checkpoint,args.split,args.filename)
    panel = save_prediction_panels(
        args.output_dir/"prediction_panels.png",result["image"],result["target"],
        {f"{model_name}: epoch {checkpoint['epoch']}":result["prediction"]},
        {"split":args.split,"filename":args.filename,
         "claim_boundary":"Single-sample B1 inspection; not the frozen B4 full-split score or a newly selected checkpoint."},
        num_classes=checkpoint["model_config"]["num_classes"],
        target_id=min(5,checkpoint["model_config"]["num_classes"]-1))
    summary = {"model":model_name,"performed_by":"Codex automation","split":args.split,
               "is_improved_run":model_name=="improved_unet","filename":args.filename,
               "checkpoint_epoch":checkpoint["epoch"],"checkpoint_sha256":checkpoint_digest,
               "limited_training_smoke":checkpoint.get("limited_training_smoke",False),
               "logit_shape":result["logit_shape"],"restored_shape":list(result["prediction"].shape),
               "cross_entropy":result["cross_entropy"],"metrics":result["metrics"],
               "test_split_read":args.split=="test","test_example_explicit_opt_in":getattr(args,"allow_test_example",False),
               "sample_count":1,"model_forward_calls":1,"optimizer_steps":0,
               "prediction_panels":panel,
               "claim_boundary":"Single-sample inspection; not independent test evaluation or final model quality."}
    with (args.output_dir/"prediction_summary.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({key:summary[key] for key in ("split","filename","checkpoint_epoch","logit_shape","restored_shape","cross_entropy")},indent=2),flush=True)
    return summary


def main(argv=None):
    return execute_prediction(parse_args(argv))


if __name__ == "__main__":
    main()
