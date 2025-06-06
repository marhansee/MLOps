import os
import time
import threading
import webbrowser
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from PIL import Image
from torchvision import transforms
import numpy as np
import onnxruntime as ort
import time

app = FastAPI()
host = "127.0.0.1"
port = 8000
TEST_IMAGES_ROOT = "data/car_data/car_data/test_2"
BATCH_SIZE = 4  


# Preprocessing transform
test_transform = transforms.Compose([
    transforms.Resize((275, 275)),
    transforms.ToTensor()
])
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8
# Load ONNX model session once at startup
ort_session = ort.InferenceSession("batch_inference/model_repository/model_quant.onnx", sess_options=sess_options)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = test_transform(img)
    return img.numpy()

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

@app.get("/", response_class=RedirectResponse)
def index():
    return RedirectResponse(url="/docs")


@app.get("/predict_batch_from_folder")
def predict_batch_from_folder():
    images = []
    image_paths = []

    for root, dirs, files in os.walk(TEST_IMAGES_ROOT):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                img_np = preprocess_image(full_path)
                images.append(img_np)

    if not images:
        return {"error": "No images found in folder."}

    batch_times = []
    per_sample_times = []
    all_preds = []

    for batch_images in batchify(images, BATCH_SIZE):
        batch_np = np.stack(batch_images).astype(np.float32)
        input_name = ort_session.get_inputs()[0].name

        start_time = time.time()
        outputs = ort_session.run(None, {input_name: batch_np})
        end_time = time.time()

        batch_inference_time = end_time - start_time
        batch_times.append(batch_inference_time)

        per_sample_inference_time = batch_inference_time / len(batch_images)
        per_sample_times.append(per_sample_inference_time)

        batch_preds = outputs[0].tolist()
        all_preds.extend(batch_preds)

    avg_batch_time_ms = (sum(batch_times) / len(batch_times)) * 1000
    avg_per_sample_time_ms = (sum(per_sample_times) / len(per_sample_times)) * 1000

    total_images = len(images)
    total_time = sum(batch_times)
    throughput = total_images / total_time  # images per second

    return {
        "average_batch_inference_time_ms": avg_batch_time_ms,
        "NO_average_per_sample_inference_time_ms": avg_per_sample_time_ms,
        "throughput_images_per_second": throughput
    }

def open_browser():
    time.sleep(1)
    webbrowser.open(f"http://{host}:{port}/docs")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host=host, port=port)
