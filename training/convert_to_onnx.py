import torch
import os
import sys

# Get the directory where this script is located (training/)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Import directly since we are now in the same folder
from model_architecture import MathCNN

def convert():
    model = MathCNN(num_classes=14)
    model_path = os.path.join(base_dir, "models", "cnn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = os.path.join(base_dir, "models", "cnn_model.onnx")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported to {onnx_path}")

if __name__ == "__main__":
    convert()
