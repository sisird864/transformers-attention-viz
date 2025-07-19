"""
Pytest configuration and fixtures
"""

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_image():
    """Create a mock image for testing"""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def mock_attention_data():
    """Create mock attention data"""
    return {
        "attention_maps": [
            np.random.rand(8, 10, 10),  # 8 heads, 10x10 attention
            np.random.rand(8, 10, 10),
            np.random.rand(8, 10, 10),
        ],
        "token_info": {
            "input_ids": np.array([[101, 102, 103, 104, 105, 106, 107, 108, 109, 110]]),
            "attention_mask": np.ones((1, 10)),
            "modality_boundaries": {
                "text_start": 0,
                "text_end": 5,
                "image_start": 5,
                "image_end": 10,
                "total_length": 10,
            },
        },
        "model_outputs": None,
        "layer_names": ["layer_0", "layer_1", "layer_2"],
        "num_layers": 3,
        "num_heads": 8,
    }


@pytest.fixture
def mock_inputs():
    """Create mock model inputs"""
    return {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10),
        "pixel_values": torch.randn(1, 3, 224, 224),
    }
