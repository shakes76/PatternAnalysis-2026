import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from params import OUT_DIM, HIDDEN_DIMS, TRIPLET_MARGIN, RESNET_OUT

class SiameseNet(nn.Module):
    """
    Siamese Network with Resnet50 sub-network.
    """
    
    def __init__(self, embed_dim=OUT_DIM):
        super().__init__()

        subnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.feature_extractor = nn.Sequential(*list(subnet.children())[:-1])

        self.fc_layers = nn.Sequential(
            nn.Linear(RESNET_OUT, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, embed_dim)
        )

        self.classifier = nn.Linear(embed_dim, 2)
    
    def get_embedding(self, x):
        """
        Compute an embedding for a batch of images
        Args: x (torch.Tensor): input tensor of shape (B, C, H, W)
        Returns: torch.Tensor: embedding tensor of shape (B, embedding_dim)
        """
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

    def classify(self, x):
        """
        Predict class logits from raw images 
        Args: x (torch.Tensor): input tensor of shape (B, C, H, W)
        Returns: torch.Tensor: logits of shape (B, num_classes)
        """
        embedding = self.get_embedding(x)
        return self.classifier(embedding)

    def forward(self, x1, x2, x3):
        """
        Forward pass for a triplet: anchor, positive, negative
        Args:
            x1 (torch.Tensor): anchor batch tensor of shape (B, C, H, W)
            x2 (torch.Tensor): positive batch tensor of shape (B, C, H, W)
            x3 (torch.Tensor): negative batch tensor of shape (B, C, H, W)
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            embeddings for (anchor, positive, negative), each of shape (B, embedding_dim)
        """
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        out3 = self.get_embedding(x3)
        return out1, out2, out3

class TripletLoss(nn.Module):

    def __init__(self, margin=TRIPLET_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        """
        Compute squared Euclidean distance per sample
        Args:
            x1 (torch.Tensor): tensor of shape (B, D)
            x2 (torch.Tensor): tensor of shape (B, D)
        Returns:
            torch.Tensor: distances of shape (B,) with squared L2 values
        """
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calculate triplet margin loss for three input embeddings
        Args: anchor, positive, negative embeddings (B, D)
        Returns: mean loss over batch as a scalar tensor
        """
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class ResNetBaseline(nn.Module):
    """
    Standard ResNet-50 baseline matching the feature extractor backbone 
    and classifier head dimensions used in the Siamese network.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Reuse the exact same backbone and classification head structure
        siamese_model = SiameseNet()
        self.feature_extractor = siamese_model.feature_extractor
        self.fc_layers = siamese_model.fc_layers
        self.classifier = siamese_model.classifier

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return self.classifier(out)