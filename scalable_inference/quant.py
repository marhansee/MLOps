import torch
import torch.quantization
from model import CustomResNet  # Your model class
import os

def dynamic_quantization(model, weights_path, output_path):
    # Load the weights
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Apply dynamic quantization (int8 on Linear layers)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(quantized_model, f"{output_path}/dynamic_quantized_model_int8.pth")
    print("Dynamic quantized model (int8) saved successfully!")


def main():
    quant_folder = "model_weights/quantized_weights"
    os.makedirs(quant_folder, exist_ok=True)

    model = CustomResNet()
    weights_path = "model_weights/best_model.pth"

    # Only INT8 quantization
    dynamic_quantization(model, weights_path, quant_folder)

if __name__ == '__main__':
    main()
