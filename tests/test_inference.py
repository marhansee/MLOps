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
    model.return_value = torch.tensor([[0.1, 0.9, 0.0]])
    return model

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_image():
    """Creates a dummy image (300x300 with 3 color channels)."""
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

@pytest.fixture
def dummy_image_tensor():
    """Creates a dummy tensor shaped like a real image batch."""
    return torch.randn(1, 3, 275, 275)


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


def test_predict(dummy_model, dummy_image_tensor):
    """Test predict() function to ensure it returns correct class index."""
    predicted_class = predict(dummy_model, dummy_image_tensor)

    # The mocked model should return class index 1 (highest in logits)
    assert predicted_class == 1

@patch("pandas.read_csv")
def test_get_class_name(mock_read_csv):
    """Test get_class_name() function using a mocked CSV file."""

    # Mock DataFrame with fake class names
    mock_df = pd.DataFrame({"Model": ["Car A", "Car B", "Car C"]})
    mock_read_csv.return_value = mock_df

    predicted_class = 2
    class_name = get_class_name(predicted_class)

    # Expecting index 2 → "Car B"
    assert class_name == "Car B"
