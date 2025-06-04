import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import sys
from model import CustomResNet
import argparse

def load_model(model_path, device):
    model = CustomResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

def preprocess_image(image, device):
    image = image / 255.0
    image = cv2.resize(image, (275, 275))
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().to(device)
    image = image.unsqueeze(0)
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction.argmax(dim=1).item()

def get_class_name(predicted_class, csv_path='archive/names.csv'):
    df = pd.read_csv(csv_path)
    return df['Model'].iloc[predicted_class - 1]

def main():
    parser = argparse.ArgumentParser(description="Run inference on images with a given model.")
    parser.add_argument('--weights', type=str, default='model_weights/best_model.pth',
                        help='Path to the model weights file')
    args = parser.parse_args()

    folder_path = 'inference_images'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(f'model_weights/{args.weights}', device)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                image = load_image(image_path)
                image_tensor = preprocess_image(image, device)
                start_time = time.time()
                predicted_class = predict(model, image_tensor)
                elapsed_time = time.time() - start_time
                class_name = get_class_name(predicted_class)
                print(f"{filename} -> Predicted Model: {class_name} (Inference time: {elapsed_time:.4f} seconds)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
