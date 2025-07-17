"""
Basic tests for attention visualizer
"""

import pytest
import torch
import numpy as np
from PIL import Image

from attention_viz import AttentionVisualizer, ensure_numpy_available
from attention_viz.extractors import AttentionExtractor
from attention_viz.visualizers import AttentionHeatmap

# Check numpy availability at test start
NUMPY_AVAILABLE = ensure_numpy_available()


class MockModel(torch.nn.Module):
    """Mock model for testing"""
    
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(256, 8)
    
    def forward(self, **kwargs):
        # Return mock outputs
        batch_size = 1
        seq_len = 10
        hidden_size = 256
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attentions = torch.softmax(torch.randn(batch_size, 8, seq_len, seq_len), dim=-1)
        
        return type('MockOutput', (), {
            'last_hidden_state': hidden_states,
            'attentions': (attentions,)
        })()


class MockProcessor:
    """Mock processor for testing"""
    
    def __call__(self, text=None, images=None, **kwargs):
        return {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
    
    def decode(self, token_ids, **kwargs):
        return " ".join([f"token_{i}" for i in range(len(token_ids))])


class TestAttentionVisualizer:
    """Test attention visualizer functionality"""
    
    def test_initialization(self):
        """Test visualizer initialization"""
        model = MockModel()
        processor = MockProcessor()
        visualizer = AttentionVisualizer(model, processor)
        
        assert visualizer.model is model
        assert visualizer.processor is processor
        assert visualizer.extractor is not None
    
    def test_visualization_creation(self):
        """Test creating visualizations"""
        model = MockModel()
        processor = MockProcessor()
        visualizer = AttentionVisualizer(model, processor)
        
        # Create dummy image
        image = Image.new('RGB', (224, 224), color='red')
        text = "test text"
        
        # Test heatmap visualization
        viz = visualizer.visualize(
            image=image,
            text=text,
            visualization_type="heatmap"
        )
        
        assert viz is not None
        assert hasattr(viz, 'show')
        assert hasattr(viz, 'save')
    
    def test_attention_extraction(self):
        """Test attention extraction"""
        model = MockModel()
        extractor = AttentionExtractor(model)
        
        inputs = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }
        
        attention_data = extractor.extract(inputs)
        
        assert "attention_maps" in attention_data
        assert "token_info" in attention_data
        assert len(attention_data["attention_maps"]) > 0
    
    def test_statistics_calculation(self):
        """Test attention statistics calculation"""
        model = MockModel()
        processor = MockProcessor()
        visualizer = AttentionVisualizer(model, processor)
        
        image = Image.new('RGB', (224, 224), color='blue')
        text = "test text"
        
        stats = visualizer.get_attention_stats(image, text)
        
        assert "entropy" in stats
        assert "top_tokens" in stats
        assert "concentration" in stats
        assert isinstance(stats["entropy"], np.ndarray)


class TestVisualizers:
    """Test individual visualizer components"""
    
    def test_heatmap_creation(self):
        """Test heatmap visualizer"""
        heatmap_viz = AttentionHeatmap()
        
        # Create mock attention data
        attention_data = {
            "attention_maps": [np.random.rand(8, 10, 10)],
            "token_info": {
                "modality_boundaries": {
                    "text_start": 0,
                    "text_end": 5,
                    "image_start": 5,
                    "image_end": 10,
                    "total_length": 10
                }
            }
        }
        
        inputs = {"input_ids": torch.randint(0, 1000, (1, 10))}
        
        viz = heatmap_viz.create(attention_data, inputs)
        assert viz is not None


def test_imports():
    """Test that all modules can be imported"""
    from attention_viz import AttentionVisualizer, launch_dashboard
    from attention_viz.extractors import AttentionExtractor
    from attention_viz.visualizers import AttentionHeatmap, AttentionFlow
    
    assert AttentionVisualizer is not None
    assert launch_dashboard is not None
    assert AttentionExtractor is not None
    assert AttentionHeatmap is not None