import torch
from metrics import calculate_dice_iou

#Using fake lables 0,1 and 2
target = torch.tensor([[[0,1],[2,2]]])
pred = torch.tensor([[[0,1],[2,0]]])

results = calculate_dice_iou(pred, target, num_classes=6)

print(results)

#Class 1
assert abs(results[1]["dice"] - 1.0) < 1e-6
assert abs(results[1]["iou"] - 1.0) < 1e-6 

#Class 2
#Dice = 2 * 1 / (1 + 2) = 2/3
#IoU = 1 / (1 + 2 - 1) = 1/2

assert abs(results[2]["dice"] - (2/3)) < 1e-6
assert abs(results[2]["iou"] - 0.5) < 1e-6

assert 0 not in results

print("Passed")