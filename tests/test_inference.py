import pytest
import torch
import pandas as pd
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
from src.new_inference import load_model, preprocess_image, predict, get_class_name
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
    tensor = torch.randn(1, 3, 300, 300)
    return tensor

def test_load_model(device):
    """Test model loading with an invalid path."""
    with pytest.raises(FileNotFoundError):
        load_model("best_model.pth", device)

# Create dummy image
def test_preprocess_image(dummy_image, device):
    """Test that image is resized to 275x275"""
    image = preprocess_image(dummy_image, device)
    assert image.shape == (1, 3, 275, 275), "Image should be resized!"

 


