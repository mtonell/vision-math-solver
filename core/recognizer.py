import torch
import torch.nn as nn
import torch.nn.functional as F

class MathCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for recognizing handwritten math characters.
    Output Classes (14):
    0-9: Digits 0 to 9
    10: '+'
    11: '-'
    12: 'x' (Multiplication)
    13: '/'
    """
    def __init__(self, num_classes=14):
        super(MathCNN, self).__init__()
        # 1 input image channel (grayscale 28x28), 32 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After two 2x2 poolings, the 28x28 image becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Input size: (Batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # Size becomes: (Batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # Size becomes: (Batch, 64, 7, 7)
        
        # Flatten the tensor
        x = x.view(-1, 64 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
