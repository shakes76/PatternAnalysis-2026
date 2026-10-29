"""Training-only positive-term weighting; ordinary two-class YOLO checkpoints.

Uses upstream v8DetectionLoss unchanged except its elementwise BCE module.
YOLO targets are soft assignment scores, not hard diagnostic probabilities.
https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/utils/loss.py
https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html
"""
import torch
from ultralytics.utils.loss import v8DetectionLoss


def install_training_loss(model, weights):
    if list(weights) not in ([1.0, 1.0], [1.0, 2.0]):
        raise ValueError('Only the predeclared positive weights are permitted')
    criterion = v8DetectionLoss(model)
    criterion.bce = torch.nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor(weights, device=next(model.parameters()).device))
    model.criterion = criterion
    return criterion
