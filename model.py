"""
CNN Model for Cat vs Dog Classification
"""
import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    """
    Convolutional Neural Network for binary classification of cat and dog images.
    
    Architecture:
        - 3 convolutional blocks (Conv2d -> ReLU -> MaxPool2d)
        - Fully connected layers with dropout for classification
    
    Input: (batch_size, 3, 64, 64) RGB images
    Output: (batch_size, 2) class logits
    """
    
    def __init__(self):
        super(CatDogCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Binary classification: Cat or Dog
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
