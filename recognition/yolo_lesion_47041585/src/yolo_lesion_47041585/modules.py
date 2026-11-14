"""
Detector construction for the lesion experiments.

Wraps the Ultralytics YOLO11n detector (COCO-pretrained `yolo11n.pt`) and the trainer
subclass that plugs the project's sampler into it. The detector architecture, loss and
NMS are supplied by the framework and are attributed as such in the README. This module
does not split data, choose thresholds or draw figures.
"""

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, nms, ops

from yolo_lesion_47041585.dataset import EpochSampler, build_sampling_weights

# Ultralytics' metric key for mAP averaged over IoU 0.5 to 0.95 for boxes
MAP_50_95 = "metrics/mAP50-95(B)"


def _task(model: YOLO) -> str:
    """The model's task, which pyright can't see through Ultralytics' dynamic lookup."""
    return cast(str, model.task)  # pyright: ignore[reportUnknownMemberType]


def build_detector(checkpoint: str) -> YOLO:
    """Load a detection checkpoint (not a classification or segmentation variant).

    Official weights such as `weights/yolo11n.pt` are downloaded to that path if they
    are missing.

    Args:
        checkpoint (str): Path to the weights to load.

    Returns:
        YOLO: The loaded detector.

    Raises:
        ValueError: If the checkpoint is not for object detection.
    """
    model = YOLO(checkpoint)
    if (task := _task(model)) != "detect":
        raise ValueError(f"{checkpoint} is a {task!r} model, expected 'detect'.")
    return model


def validate_model_contract(model: YOLO, class_names: Sequence[str]) -> None:
    """Check the model is a detector whose class order matches the project mapping.

    Args:
        model (YOLO): A model fine-tuned on the project's classes.
        class_names (Sequence[str]): The class names in class ID order.

    Raises:
        ValueError: If the task or class names differ from `class_names`.
    """
    if (task := _task(model)) != "detect":
        raise ValueError(f"Expected a 'detect' model, got {task!r}.")
    names = [str(name) for _, name in sorted(model.names.items())]
    if names != list(class_names):
        raise ValueError(f"Model classes {names} do not match {list(class_names)}.")


class LesionDetectionTrainer(DetectionTrainer):
    """DetectionTrainer whose training loader draws images with the project's sampler.

    The stock `build_dataloader` only honours a sampler when training across processes,
    so for the "replacement" scheme this builds the training `InfiniteDataLoader` itself
    and hands it an `EpochSampler`, drawing with replacement (uniformly if there is no
    target fraction). The "stock" scheme, and validation under either, use the stock
    loader unchanged, which shuffles without replacement and is never weighted.

    Args:
        cfg: Ultralytics default configuration.
        overrides (dict[str, Any] | None): Training arguments for Ultralytics.
        _callbacks (dict[str, Any] | None): Ultralytics callbacks.
        train_classes (Mapping[str, int]): The class of each training image by ISIC ID,
            which must match the images in the training folder exactly.
        scheme (Literal["stock", "replacement"]): How training images are drawn.
        target_fraction (float | None): Expected malignant share of each epoch, or None
            to keep the natural class balance. Only valid with "replacement".

    Raises:
        ValueError: If a target fraction is given with the "stock" scheme.

    Attributes:
        train_sampler (EpochSampler | None): The sampler last given to the training
            loader, kept so a run can log what it drew.
        train_class_ids (NDArray[np.int_] | None): The class of each training image in
            the order the sampler indexes them.
        validation_seconds (list[float]): How long each epoch's validation took.
    """

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, Any] | None = None,
        *,
        train_classes: Mapping[str, int],
        scheme: Literal["stock", "replacement"],
        target_fraction: float | None,
    ) -> None:
        if scheme == "stock" and target_fraction is not None:
            raise ValueError("The stock loader cannot sample to a malignant fraction.")

        # pyright doesnt know DetectionTrainer.__init__ takes overrides and _callbacks
        super().__init__(cfg, overrides, _callbacks)  # pyright: ignore[reportUnknownMemberType]
        self.train_sampler: EpochSampler | None = None
        self.train_class_ids: NDArray[np.int_] | None = None
        self.validation_seconds: list[float] = []
        self._scheme = scheme
        self._target_fraction = target_fraction
        self._train_classes = dict(train_classes)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> InfiniteDataLoader:
        # only use weighted loader for training
        if mode != "train" or self._scheme == "stock":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        if rank != -1:
            raise ValueError("Weighted sampling supports single-process training only.")

        # build the dataset and line up the classes with the order it loaded
        dataset = cast(YOLODataset, self.build_dataset(dataset_path, mode, batch_size))

        # pyright doesnt know YOLODataset has an im_files attribute
        files = cast(list[str], dataset.im_files)  # pyright: ignore[reportUnknownMemberType]
        ids = [Path(file).stem for file in files]
        if set(ids) != self._train_classes.keys() or len(ids) != len(set(ids)):
            raise ValueError(
                f"Training images do not match the training records: found {len(ids)} "
                f"images for {len(self._train_classes)} records."
            )

        self.train_class_ids = np.array([self._train_classes[i] for i in ids])
        weights = build_sampling_weights(
            self.train_class_ids.tolist(), self._target_fraction
        )

        # pyright doesnt know self.args has seed and workers attributes
        args = cast(Any, self.args)  # pyright: ignore[reportUnknownMemberType]
        seed, workers = int(args.seed), min(os.cpu_count() or 1, int(args.workers))
        self.train_sampler = EpochSampler(weights, len(dataset), seed)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            sampler=self.train_sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=cast(Any, dataset).collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
            prefetch_factor=4 if workers > 0 else None,
        )

    def validate(self) -> tuple[dict[str, float] | None, float | None]:
        """Validate, scoring the epoch by mAP50:95 alone so best.pt follows that metric.

        Ultralytics' default fitness weights mAP50 at 10% and mAP50:95 at 90%. When an
        epoch exactly ties the best score, its fitness is nudged down by one step so
        the earliest epoch keeps best.pt, since Ultralytics overwrites it on equality.
        """
        # Ultralytics' trainer attributes are untyped, so reach them through Any
        trainer = cast(Any, self)
        start = time.perf_counter()
        metrics: dict[str, float] | None = trainer.validator(trainer)
        self.validation_seconds.append(time.perf_counter() - start)
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)

        fitness = float(metrics[MAP_50_95])
        best: float | None = trainer.best_fitness
        if best is None or fitness > best:
            trainer.best_fitness = fitness
        elif fitness == best:
            fitness = float(np.nextafter(fitness, -np.inf))

        return metrics, fitness


class LesionPredictor(DetectionPredictor):
    """DetectionPredictor that matches the validator's non-maximum suppression.

    Ultralytics' predictor keeps only the highest-scoring class for each candidate box,
    while its validator (`multi_label=True`) keeps a box for every class scoring above
    the confidence floor. This model scores a malignant lesion about 0.8 benign and 0.1
    to 0.2 malignant on the same box, so the predictor dropped the malignant boxes that
    the triage policy needs and validation counted. This keeps them, so predictions and
    validation agree. NMS is still applied per class.
    """

    def postprocess(self, preds: Any, img: Any, orig_imgs: Any, **kwargs: Any) -> Any:
        args = cast(Any, self.args)  # pyright: ignore[reportUnknownMemberType]
        kept: Any = nms.non_max_suppression(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            preds,
            args.conf,
            kwargs.pop("iou", args.iou),
            args.classes,
            args.agnostic_nms,
            multi_label=True,
            max_det=args.max_det,
            nc=0,
            end2end=getattr(self.model, "end2end", False),
        )
        if not isinstance(orig_imgs, list):  # a batch tensor, not a list of images
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]  # pyright: ignore[reportUnknownMemberType]
        return self.construct_results(kept, img, orig_imgs, **kwargs)  # pyright: ignore[reportUnknownMemberType]


def trainer_overrides(
    config: Mapping[str, Any], model_path: Path, run_dir: Path
) -> dict[str, Any]:
    """Flatten a resolved run config into the keyword overrides Ultralytics expects.

    Args:
        config (Mapping[str, Any]): The resolved config.
        model_path (Path): The local starting checkpoint.
        run_dir (Path): Where this run's outputs go. It must not exist yet.

    Returns:
        dict[str, Any]: Training, augmentation, data and output settings.
    """
    return {
        **config["train"],
        **config["augment"],
        "model": str(model_path),
        "data": str(config["data"]),
        "seed": config["seed"],
        "project": str(run_dir.parent.resolve()),
        "name": run_dir.name,
        "exist_ok": False,
    }
