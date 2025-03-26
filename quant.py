import torch
import torch.quantization
from model import CustomResNet # Import the model

# Load the model
model = CustomResNet()
checkpoint = torch.load("best_model.pth", map_location="cpu") # Load Weights
model.load_state_dict(checkpoint)
model.eval() # Set the model to evaluation mode

# Apply quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model, "quantized_model.pth")
print("Quantized model saved successfully!")
