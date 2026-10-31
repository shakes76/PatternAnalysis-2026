from pathlib import Path

import torch
from PIL import Image
from torch import nn

from modules import ConvNeXt, BaselineCNN, ConvNeXtBlock
from predict import load_weights, save_gradcam_examples
from dataset import CLASS_TO_IDX, make_transforms
def save_grad_cam(
    image_path: Path,
    weights: Path,
    output: Path = Path("gradcam_output"),
    target_class: str | None = None,
    actual_class: str | None = None,
    model_type: str = "convnext"
    ):
    """
    save a gradcam of a given image
    Args:
        image (Path): Path to image
        model (str | None): convnext or baseline, defaults to convnext
        weights (Path): the location of the model weights
        output (Path | None): the directory to save the image, if none, creates a gradcam_output folder
        target_class (str| None): AD or NC
        actual_class (str): The true class, AD or NC
    """
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model_class = ConvNeXt if model_type == "convnext" else BaselineCNN
    model = load_weights(
        model_class(num_classes=2),
        weights,
        device,
    ).to(device).eval()

    # Preprocess image using transforms with no augmentation
    transform = make_transforms(False)
    image = transform(Image.open(image_path)).unsqueeze(0).to(device) # type: ignore

    # Predict class
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)
        predicted = probabilities.argmax(dim=1).item()

    # Select class to explain (predicted class by default)
    target = CLASS_TO_IDX[target_class] if target_class else predicted

    # Find a suitable convolutional layer
    target_layer = next(
        layer for layer in reversed(list(model.modules()))
        if isinstance(layer, ConvNeXtBlock)
    )

    class_names = {CLASS_TO_IDX["AD"]: "AD", CLASS_TO_IDX["NC"]: "NC"}

    stem = image_path.stem
    patient_id, separator, slice_id = stem.partition("_")
    fname = f"{patient_id}_{slice_id}"


    # Format expected by save_gradcam_examples()
    examples = [{
        "image": image[0].cpu(), # type: ignore
        "predicted": target,
        "filename": fname,
        "actual_name": actual_class or "unknown",
        "predicted_name": class_names[target], # type: ignore
    }]


    save_gradcam_examples(
        model, examples, device, output / fname, target_layer
    )

save_grad_cam(
    image_path=Path("datasets/test/AD/388206_78.jpeg"),
    weights=Path("models/ConvNeXt_best.pt")
    )

