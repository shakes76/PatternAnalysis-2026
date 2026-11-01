import sys 
from train import BasicUnet, ImprovedUnet
from torch.utils.dataset import DataLoader
import matplotlib.pyplot as plt
import torch
import random

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

"""runs testing on both models with the same dataset, and generates error maps of the outputs on 3 images"""
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = sys.argv[1]
    unet = BasicUnet(dev, data_dir)
    unet.load_model(sys.argv[2])
    unet.test_model()
    gen = HeatMapGenerator(unet.model)
    im_unet = ImprovedUnet(dev, data_dir)
    im_unet.load_model(sys.argv[3])
    im_unet.test_model()
    gen2 = HeatMapGenerator(im_unet.model)
    train = DataLoader(
        dataset = TrainData(data_dir),
        shuffle = False,
        batch_size = 1
    )
    print(f"unet parameters = {sum(p.numel() for p in unet.model.parameters() if p.requires_grad)}")
    print(f"improved unet parameters = {sum(p.numel() for p in im_unet.model.parameters() if p.requires_grad)}")

    img_set = random.choices(range(600), k = 3)
    for i, (mask, image) in enumerate(train):
        if i in img_set:
            cidx = 0
            output, map = gen.generate(image, 3)
            output2, map2 = gen2.generate(image, 2)
            cmap = 'plasma'
            #shows the heatmaps of both models for a class
            plt.imshow(map, alpha=0.5, cmap=cmap)
            plt.figure()
            plt.imshow(img)
            plt.imshow(map2, alpha=0.5,cmap=cmap)
            plt.figure()
            plt.imshow(img)
            plt.imshow(map_3, alpha=0.5, cmap=cmap)
            plt.figure()
            plt.imshow(img)
            plt.imshow(map2_3, alpha=0.5, cmap=cmap)
            out = output.detach().numpy()
            out2 = output2.detach().numpy()
            img = image[0][0]
            plt.figure()
            plt.imshow(img)
            plt.imshow(map)
            plt.imshow(map2)
            mask = mask.detach().numpy()
            for c in range(6):
                #error maps for all classes
                error_map = gen.get_map(map, mask, out, c)
                error_map2 = gen2.get_map(map2, mask, out2, c)
                ax, plot = plt.subplots(2, figsize=(20,20))
                plot[0].imshow(error_map)
                plot[1].imshow(error_map2)

    plt.show()