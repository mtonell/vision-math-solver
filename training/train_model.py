import os
import sys
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from model_architecture import MathCNN

# Setup device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Operator mapping
# MNIST covers 0-9. We add 5 more classes for math operators.
operator_mapping = {
    '+': 10,
    '-': 11,
    'x': 12,
    '/': 13
}

class SyntheticOperatorDataset(Dataset):
    """
    Generates synthetic 28x28 grayscale images of math operators 
    to combine with the MNIST dataset.
    """
    def __init__(self, num_samples_per_op=6000, transform=None):
        self.samples = []
        self.transform = transform
        
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX
        ]
        
        print("Generating synthetic images for operators...")
        for op, label in operator_mapping.items():
            for _ in range(num_samples_per_op):
                # Blank black image 28x28 (same as MNIST format)
                img = np.zeros((28, 28), dtype=np.uint8)
                
                # Randomize thickness and scale to make the model robust
                scale = random.uniform(0.6, 1.1)
                thickness = random.randint(1, 3)
                font = random.choice(fonts)
                
                # Get the size of the text to center it
                text_size = cv2.getTextSize(op, font, scale, thickness)[0]
                text_x = (28 - text_size[0]) // 2 + random.randint(-3, 3)
                text_y = (28 + text_size[1]) // 2 + random.randint(-3, 3)
                
                # Draw the operator in white (255)
                cv2.putText(img, op, (text_x, text_y), font, scale, 255, thickness)
                
                # Convert to PIL Image for torchvision transforms
                pil_img = Image.fromarray(img)
                self.samples.append((pil_img, label))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    print(f"Using device: {device}")
    
    # Standard MNIST transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 1. Load MNIST
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    print("Loading MNIST dataset...")
    mnist_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    
    # 2. Generate Synthetic Operators
    synth_dataset = SyntheticOperatorDataset(num_samples_per_op=6000, transform=transform)
    
    # 3. Combine them
    combined_dataset = ConcatDataset([mnist_dataset, synth_dataset])
    print(f"Total training samples: {len(combined_dataset)}")
    
    # 4. Create DataLoader
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
    
    # 5. Initialize Model, Loss, Optimizer
    model = MathCNN(num_classes=14).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training Loop
    epochs = 3
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 300 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
                
    print("Training Complete!")
    
    # 7. Save Model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'cnn_model.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to: {save_path}")

if __name__ == "__main__":
    main()
