# src/inference.py
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from train import CustomResNet

"""
Refactored inference.py so we can test each individual function
"""



def load_model(model_path, device):
    """Loads a trained model from a given path."""
    model = CustomResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_image(image_path):
    """Loads and preprocesses an image for model inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load the image. Check the file path.")
    return image


def preprocess_image(image, device):
    image = image / 255.0
    image = cv2.resize(image, (275, 275))
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().to(device)
    image = image.unsqueeze(0)
    return image

def predict(model, image_tensor):
    """Runs inference and returns the predicted class index."""
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction.argmax(dim=1).item()

def get_class_name(predicted_class, csv_path='archive/names.csv'):
    """Fetches the class name from the CSV file based on predicted class index."""
    df = pd.read_csv(csv_path)
    return df['Model'].iloc[predicted_class - 1]

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    classes = [i for i in range(1, 197)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('model_weights/best_model.pth', device)
    image = load_image('inference_images/00076.jpg')
    image_tensor = preprocess_image(image, device)
    predicted_class = classes[predict(model, image_tensor)]

    print(f"Predicted Model: {get_class_name(predicted_class)}")

if __name__ == "__main__":
    main()
