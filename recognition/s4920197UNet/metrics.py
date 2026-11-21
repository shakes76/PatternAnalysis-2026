import numpy as np 
import torch
import torch.nn as nn

'''simple class that calculates the dice loss of any output image returned by our models'''
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, output):
        loss = 0.0
        epsilon = 1e-6
        dc = []
        for k in range(6):
            p = output[:, k, :, :]
            y = mask[:, k, :, :]

            p_flat = p.contiguous().view(-1)
            y_flat = y.contiguous().view(-1)
            intersection = torch.sum(p_flat * y_flat)
            total = torch.sum(p_flat) + torch.sum(y_flat)

            dice_coef = (2.0 * intersection + epsilon) / (total + epsilon)
            dc.append(dice_coef)
            loss += (1.0 - dice_coef)

        return loss/6, dc

'''used to generate a heatmap which shows which regions of an output image where most significant in the models decisionmaking process'''
class HeatMapGenerator:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.segment
        self.feature_maps = None
        self.gradients = None
        # Register hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        """generates the heatmap for a given model, takes in an input vector and class index, returns the segmentation map and heatmap"""
        self.model.eval()
        output = self.model(x)
        
        # if class_idx is None:
        #     class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx].sum()
        loss.backward()
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the feature maps by the gradients
        feature_maps = self.feature_maps[0]
        for i in range(feature_maps.shape[0]): 
            feature_maps[i, :, :] *= pooled_gradients[i]
            
        # Average the weighted feature maps across channels
        heatmap = torch.mean(feature_maps, dim=0).cpu().numpy()
        
        # Apply ReLU to keep only positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap between 0 and 1
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return output, heatmap

            
    def get_map(self, heatmap, mask, out, c):
        # Normalize the heatmap between 0 and 1
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        mask_img = mask[0][c].T
        out_img = out[0][c].T
        # aplt[0].imshow(img)
        # aplt[1].imshow(mask)
        # aplt[2].imshow(out_img[0][cidx])
        gt = (mask_img > 0.5).astype(int)
        pred = (out_img > 0.5).astype(int)
        # Create an RGB image to map errors visually
        h, w = gt.shape
        error_map = np.zeros((h, w, 3), dtype=np.uint8)
        # Perfect match (White)
        error_map[np.logical_and(gt == 1, pred == 1)] = [255, 255, 255]
        # False Positive (Red) -> Model predicted mask where there shouldn't be one
        error_map[np.logical_and(gt == 0, pred == 1)] = [255, 0, 0]
        # False Negative (Blue) -> Model missed the actual target
        error_map[np.logical_and(gt == 1, pred == 0)] = [0, 0, 255]
        error_map[np.logical_and(gt == 0, pred == 0)] = [211, 211, 211]
        return error_map
