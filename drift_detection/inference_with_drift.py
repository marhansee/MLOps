import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import sys
from model import CustomResNet
import argparse
import torchdrift
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class SimpleImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image  

def load_model(model_path, device):
    model = CustomResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def predict(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction.argmax(dim=1).item()

def get_class_name(predicted_class, csv_path='archive/names.csv'):
    df = pd.read_csv(csv_path)
    return df['Model'].iloc[predicted_class - 1]

def detect_drift(input_data, feature_extractor, drift_detector):
    # Extract features
    features = feature_extractor(input_data)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(f"Drift score: {score}")
    print(f"P-value: {p_val}")
    return p_val


def synthesize_data_drift(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

def main():
    parser = argparse.ArgumentParser(description="Run inference on images with a given model.")
    parser.add_argument('--weights', type=str, default='model_weights/new_best_model.pth',
                        help='Path to the model weights file')
    parser.add_argument('--add_drift', action='store_true', 
                        help="If set, drift is added to the data")
    args = parser.parse_args()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(f'{args.weights}', device)

    inference_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_kwargs = {'batch_size': 16}
    folder = 'data/car_data/car_data/train/Volvo XC90 SUV 2007'
    train_data = SimpleImageFolderDataset(folder, transform=inference_transform)
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)

    # Load drift detector
    drift_detector = torch.load('drift_detector.pth', map_location=device)
    print("Loaded drift detector:", drift_detector)

    # Load feature extractor
    feature_extractor = model.resnet.to(device)
    feature_extractor.eval()
    drift_detector.eval()

    for batch_idx, x in enumerate(train_loader):
        x = x.to(device)
        with torch.no_grad():
            if args.add_drift:
                x = synthesize_data_drift(x)

            features = feature_extractor(x)
            score = drift_detector(features)
            p_val = drift_detector.compute_p_value(features)

        if p_val.item() < 0.01:
            print("DATA DRIFT DETECTED!")
            print(f"[Batch {batch_idx}] Drift score: {score.item():.4f} | p-value: {p_val.item():.4f}")
            sys.exit(1)
        else:
            print("No drift detected!")

if __name__ == "__main__":
    main()

