import os
import torch
from model import CustomResNet
import argparse
from torchvision import datasets, transforms
import yaml
import torchdrift
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

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

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run inference on images with a given model.")
    parser.add_argument('--weights', type=str, default='model_weights/new_best_model.pth',
                        help='Path to the model weights file')
    args = parser.parse_args()

    config_path = 'train_config.yaml'
    config = load_config(config_path)

    # Load train data
    train_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_kwargs = {'batch_size': config['batch_size']}


    folder = 'data/car_data/car_data/train/Volvo XC90 SUV 2007'

    # Datasets and loaders
    train_data = SimpleImageFolderDataset(folder, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load feature extractor (the model)
    model = load_model(f'{args.weights}', device)
    feature_extractor = model.resnet.to(device)
    feature_extractor.eval()

    # Drift detector
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    with torch.no_grad():
        torchdrift.utils.fit(train_loader, feature_extractor, drift_detector, device=device)


    # Save the drift detector
    torch.save(drift_detector, 'drift_detector.pth')
    print("Saved drift detector fitted on train data")


if __name__ == "__main__":
    main()
