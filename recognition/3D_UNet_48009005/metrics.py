import torch

def calculate_dice_iou(pred, target, num_classes=6):
    """
    Calculate Dice and IoU for each anatomical class
    Background class 0 is excluded from the reported metrics as it is just the background
    
    pred: integer class predictions
    target: integer ground truth labels
    """
    
    results = {}
    #start at 1 to exclude the background label 0
    for class_id in range(1, num_classes):
        
        pred_class = pred == class_id
        target_class = target == class_id
        
        intersection = (pred_class & target_class).sum().float()
        
        pred_count = pred_class.sum().float()
        target_count = target_class.sum().float()
        
        #skip class if it is absent from both prediction and target
        if pred_count == 0 and target_count == 0:
            continue
        
        dice = (2.0 * intersection / (pred_count + target_count))
        union = pred_count + target_count - intersection
        IoU = intersection / union
        
        results[class_id] = {"dice": dice.item(), "iou": IoU.item()}
    return results
    