import torch
import torch.quantization
from model import CustomResNet # Import the model
import os

# Load the model
def dynamic_quantization(model, weights_path, output_path):
    checkpoint = torch.load(weights_path, map_location="cpu") # Load Weights
    model.load_state_dict(checkpoint)
    model.eval() # Set the model to evaluation mode

    # Apply quantization on linear layers
    data_types = {
        'int8': torch.qint8,
        'fp16': torch.float16
    }

    for dtype_name, dtype in data_types.items():
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=dtype
        )

        # Save the quantized model
        torch.save(quantized_model, 
                   f"{output_path}/dynamic_quantized_model_{dtype_name}.pth")
        print(f"Dynamic quantized model with {data_types} saved successfully!")

def static_quantization(model, weights_path, output_path, input_size=(1, 3, 275, 275)):
    checkpoint = torch.load(weights_path, map_location="cpu") # Load Weights

    # Define the quantization configurations
    qconfigs = {
        "int8": torch.quantization.get_default_qconfig("fbgemm"),
        "fp16": torch.quantization.float16_qconfig

    }

    for dtype_name, qconfig in qconfigs.items():
        model.load_state_dict(checkpoint)
        model.eval() # Set the model to evaluation mode

        model.qconfig = qconfig

        # Prepare the model for static quantization
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # Calibrate the model with representative data
        for _ in range(10):
            dummy_input = torch.randn(input_size)
            model_prepared(dummy_input)

        model_quantized = torch.quantization.convert(model_prepared, inplace=False)

        torch.save(model_quantized.state_dict(), 
                   f"{output_path}/static_quantized_model_{dtype_name}.pth")
        print(f"Static quantized model {dtype_name} saved successfully!")

def main():
    quant_folder = "model_weights/quantized_weights"
    os.makedirs(quant_folder, exist_ok=True)

    # Load model
    model = CustomResNet()

    # Define weights path
    weights_path = "model_weights/best_model.pth"

    # Apply dynamic quantization
    dynamic_quantization(
        model=model,
        weights_path=weights_path,
        output_path=quant_folder
    )

    # Apply static quantization
    static_quantization(
        model=model,
        weights_path=weights_path,
        output_path=quant_folder
    )


if __name__ == '__main__':
    main()