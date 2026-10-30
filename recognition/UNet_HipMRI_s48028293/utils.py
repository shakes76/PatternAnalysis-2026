"""
utils.py computes per-class Dice and IoU on the HipMRI predictions, per volume and per slice, and formats the results.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

class SegMeter:
    # Accumulates per-class overlap counts per volume and per slice.

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.volumes = {}
        self.slices = []
        self.n_items = 0

    def update(self, preds, targets, cases, weeks, slice_ids):
        # Update the meter with a batch of predictions and targets.
        p = F.one_hot(preds, self.num_classes)
        t = F.one_hot(targets, self.num_classes)
        if preds.dim() == 3:
            p, t = p.flatten(1, 2)[:, None], t.flatten(1, 2)[:, None]
        else:                 
            p, t = p.movedim(3, 1).flatten(2, 3), t.movedim(3, 1).flatten(2, 3)
        counts = torch.stack([(p * t).sum(2), p.sum(2), t.sum(2)], dim=3)
        counts = counts.cpu().numpy().astype(np.int64)
        for i in range(counts.shape[0]):
            key = (int(cases[i]), int(weeks[i]))
            if key not in self.volumes:
                self.volumes[key] = np.zeros((self.num_classes, 3), dtype=np.int64)
            self.volumes[key] += counts[i].sum(0)
            if preds.dim() == 3:
                slice_numbers = [int(slice_ids[i])]
            else:
                slice_numbers = range(counts.shape[1])
            for s, z in enumerate(slice_numbers):
                self.slices.append((self.n_items, key[0], key[1], z, counts[i, s]))
            self.n_items += 1
    
    @staticmethod
    def _dice_iou(counts):
        # Return (dice, iou) for the given counts array of shape (..., 3).
        inter, pred, target = counts[..., 0], counts[..., 1], counts[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = 2.0 * inter / (pred + target)
            iou = inter / (pred + target - inter)
        return dice, iou

    def per_volume(self):
        #Return (keys, dice, iou); dice and iou have shape (n_volumes, C).
        keys = sorted(self.volumes)
        counts = np.stack([self.volumes[k] for k in keys])
        dice, iou = self._dice_iou(counts)
        return keys, dice, iou

    def per_slice(self, class_idx):
        #Return a list of dicts with the Dice of one class on every slice.
        #'index' is the position in the (unshuffled) dataset
        rows = []
        for index, case, week, slice_id, counts in self.slices:
            dice, _ = self._dice_iou(counts[class_idx])
            rows.append({"index": index, "case": case, "week": week, "slice": slice_id,
                        "dice": float(dice), "target_px": int(counts[class_idx, 2]),
                        "pred_px": int(counts[class_idx, 1])})
        return rows

    def mean_dice(self):
        #Mean Dice per class over volumes, shape (C,)
        return np.nanmean(self.per_volume()[1], axis=0)

@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None):
    #Run the model over a loader.
    model.eval()
    meter = SegMeter(num_classes)
    use_cuda = device.type == "cuda"
    total_loss, n_batches, forward_time = 0.0, 0, 0.0
    for images, masks, cases, weeks, slice_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            logits = model(images)
        if use_cuda:
            torch.cuda.synchronize()
        forward_time += time.perf_counter() - start
        if criterion is not None:
            total_loss += criterion(logits, masks).item()
        meter.update(logits.argmax(dim=1), masks, cases, weeks, slice_ids)
        n_batches += 1
    return meter, total_loss / max(n_batches, 1), forward_time

def format_results(meter, class_names):
    #Per-class Dice and IoU (mean +/- std over volumes) as printable text.
    keys, dice, iou = meter.per_volume()
    
    lines = [f"Volumes evaluated: {len(keys)}",
            f"{'class':12s} {'DSC mean':>9s} {'DSC std':>8s} {'DSC min':>8s} "
            f"{'IoU mean':>9s} {'IoU std':>8s}"]
    for c, name in enumerate(class_names):
        lines.append(f"{name:12s} {np.nanmean(dice[:, c]):9.4f} {np.nanstd(dice[:, c]):8.4f} "
                    f"{np.nanmin(dice[:, c]):8.4f} {np.nanmean(iou[:, c]):9.4f} "
                    f"{np.nanstd(iou[:, c]):8.4f}")
    lines.append(f"{'mean':12s} {np.nanmean(dice):9.4f} {'':8s} {'':8s} {np.nanmean(iou):9.4f}")
    return "\n".join(lines)
