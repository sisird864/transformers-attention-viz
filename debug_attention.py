#!/usr/bin/env python3
"""
Debug script to test attention extraction
"""

import torch
import numpy as np
from PIL import Image

# Import the modules
from attention_viz import AttentionVisualizer, ensure_numpy_available
from attention_viz.extractors import AttentionExtractor

# Simple mock model for testing
class SimpleMockModel(torch.nn.Module):
    """Simplified mock model that directly returns attention"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, output_attentions=True, **kwargs):
        batch_size = 1
        seq_len = 10
        num_heads = 8
        num_layers = 3
        
        # Generate mock attention weights for each layer
        attentions = []
        for _ in range(num_layers):
            attn = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            attentions.append(attn)
        
        # Return output with attentions
        return type('Output', (), {
            'last_hidden_state': torch.randn(batch_size, seq_len, 256),
            'attentions': tuple(attentions) if output_attentions else None
        })()


def test_simple_extraction():
    """Test basic attention extraction"""
    print("Testing simple attention extraction...")
    
    model = SimpleMockModel()
    extractor = AttentionExtractor(model)
    
    # Create dummy inputs
    inputs = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10)
    }
    
    # Extract attention
    attention_data = extractor.extract(inputs)
    
    print(f"Number of attention maps: {len(attention_data['attention_maps'])}")
    print(f"Layer names: {attention_data['layer_names']}")
    print(f"Number of layers: {attention_data['num_layers']}")
    print(f"Number of heads: {attention_data['num_heads']}")
    
    if attention_data['attention_maps']:
        print(f"First attention map shape: {attention_data['attention_maps'][0].shape}")
    else:
        print("No attention maps extracted!")
        
        # Check if model outputs have attentions
        outputs = model(output_attentions=True)
        if hasattr(outputs, 'attentions') and outputs.attentions:
            print(f"Model outputs contain {len(outputs.attentions)} attention tensors")
            print(f"First attention tensor shape: {outputs.attentions[0].shape}")
    
    return attention_data


if __name__ == "__main__":
    # Test numpy availability
    print(f"NumPy available: {ensure_numpy_available()}")
    
    # Run extraction test
    attention_data = test_simple_extraction()
    
    # Test with MockProcessor
    print("\nTesting with MockProcessor...")
    
    class MockProcessor:
        def __call__(self, text=None, images=None, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "pixel_values": torch.randn(1, 3, 224, 224)
            }
        
        def decode(self, token_ids, **kwargs):
            return " ".join([f"token_{i}" for i in range(len(token_ids))])
    
    # Test visualizer
    model = SimpleMockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    # Try to visualize
    try:
        viz = visualizer.visualize(
            image=Image.new('RGB', (224, 224), color='red'),
            text="test text",
            visualization_type="heatmap"
        )
        print("✅ Visualization created successfully!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        
        # Debug: manually check attention extraction
        inputs = processor(text="test", images=Image.new('RGB', (224, 224)))
        attention_data = visualizer.extractor.extract(inputs)
        print(f"Attention maps in failed viz: {len(attention_data['attention_maps'])}")