import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image

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

class MathRecognizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MathCNN(num_classes=14).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Map class indices to characters
        self.class_map = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: '+', 11: '-', 12: '*', 13: '/' # Mapped class 12 to '*' for Sympy evaluation
        }
        
    def predict(self, crop):
        """
        Takes a grayscale numpy image (2D array) of the extracted bounding box,
        resizes it, and returns the predicted character label.
        """
        # Resize to 28x28 as expected by the CNN
        resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image for the transform
        pil_img = Image.fromarray(resized)
        
        # Apply transform and move to device
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output.data, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return self.class_map[predicted.item()], confidence.item()
