import pytest
import torch
import pandas as pd
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
from src.new_inference import load_model, preprocess_image, predict, get_class_name, load_image
#from model import CustomResNet


@pytest.fixture
def dummy_model():
    """
    This function creates a dummy model for testing.
    """
    model = MagicMock()
    model.eval()
    
    # Define dummy output
    model.return_value = torch.tensor([[0.1, 0.9]])
    return model

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_image():
    """Creates a dummy image (300x300 with 3 color channels)."""
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)




def test_load_model():
    """Test model loading with an invalid path."""
    weight_path = "model_weights/best_model.pth"
    assert os.path.exists(weight_path), f"{weight_path} does not exist!"


def test_load_image_success(dummy_image):
    """Test successful image loading with a mock."""
    dummy_path = "fake_image.jpg"

    with patch("cv2.imread", return_value=dummy_image):
        image = load_image(dummy_path)

    assert image.shape == (300, 300, 3)  # Check expected shape
    assert image.dtype == np.uint8  # Ensure correct data type
    

def test_load_image_failure():
    """Test failure case when the image file is invalid."""
    dummy_path = "nonexistent.jpg"

    with patch("cv2.imread", return_value=None):  # Simulate failure
        with pytest.raises(ValueError, match="Failed to load the image"):
            load_image(dummy_path)


def test_preprocess_image(dummy_image, device):
    """Test preprocessing function with a dummy image."""
    processed_image = preprocess_image(dummy_image, device)

    # Expected output shape (1, 3, 275, 275)
    assert processed_image.shape == (1, 3, 275, 275)
    assert processed_image.dtype == torch.float32  # Ensure float type
    assert processed_image.device == device  # Ensure correct device