import cv2
import numpy as np
import onnxruntime as ort

class MathRecognizer:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        
        # Map class indices to characters
        self.class_map = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: '+', 11: '-', 12: '*', 13: '/' # Mapped class 12 to '*' for Sympy evaluation
        }
        
    def predict(self, crop):
        """
        Takes a grayscale numpy image (2D array) of the extracted bounding box,
        resizes it, and returns the predicted character label using ONNX.
        """
        # Resize to 28x28 as expected by the CNN
        resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Manual PyTorch transforms: ToTensor() and Normalize((0.1307,), (0.3081,))
        img_array = resized.astype(np.float32) / 255.0
        img_array = (img_array - 0.1307) / 0.3081
        
        # Add channel and batch dimensions: (1, 1, 28, 28)
        tensor = np.expand_dims(np.expand_dims(img_array, axis=0), axis=0)
        
        # Run ONNX inference
        ort_inputs = {self.ort_session.get_inputs()[0].name: tensor}
        output = self.ort_session.run(None, ort_inputs)[0]
        
        # Softmax
        exp_vals = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_vals / np.sum(exp_vals)
        predicted = np.argmax(probabilities)
        confidence = probabilities[predicted]
        
        return self.class_map[predicted], float(confidence)
