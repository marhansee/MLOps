import pytest
import torch
import pandas as pd
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
from src.new_inference import load_model, preprocess_image, predict, get_class_name
from model import CustomResNet




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

def test_modelweights_exist():
    model_path = "best_model.pth"
    assert os.path.exists(model_path) == True
