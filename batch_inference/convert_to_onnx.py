import torch
from model import CustomResNet
from onnxruntime.quantization import quantize_dynamic, QuantType

device = torch.device('cpu')  # quantized model only supports CPU

model = CustomResNet()
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, 275, 275)

# Export model to ONNX (since triton expects onnx models)
torch.onnx.export(
    model,                        # your PyTorch model
    dummy_input,                  # dummy input tensor
    "models/model.onnx",                 # output file path
    export_params=True,           # store the trained parameter weights inside the model file
    opset_version=11,             # ONNX version, 11 is usually safe and compatible
    do_constant_folding=True,     # optimize constants
    input_names=['input'],        # the model's input names
    output_names=['output'],      # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},   # variable batch size
        'output': {0: 'batch_size'}
    }
)
print("Model successfully exported to ONNX!")

# Quantize to Dynamic INT8 model
quantize_dynamic("models/model.onnx", "models/model_quant.onnx", weight_type=QuantType.QUInt8)