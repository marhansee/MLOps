import sys
import torch
import cv2
import numpy as np
from train import CustomResNet
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    classes = [i for i in range(1, 197)]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = CustomResNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load the image. Check the file path.")

    image = image / 255.0
    image = cv2.resize(image, (275, 275))
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)

    predicted_class = classes[prediction.argmax(dim=1).item()]
    df = pd.read_csv('archive/names.csv')
    print(f"Predicted Model: {df['Model'].iloc[predicted_class-1]}")

if __name__ == "__main__":
    main()
