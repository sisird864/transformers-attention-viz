#!/bin/bash

echo "Running debug test to identify the issue..."
echo "========================================="

# Save the debug script
cat > debug_attention_test.py << 'EOF'
#!/usr/bin/env python3
"""
Debug script to test attention extraction
"""

import torch
import numpy as np
from PIL import Image
import sys

# Import the modules
try:
    from attention_viz import AttentionVisualizer, ensure_numpy_available
    from attention_viz.extractors import AttentionExtractor
    from attention_viz.extractors.base import BaseModelAdapter
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

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


def test_adapter():
    """Test the base model adapter"""
    print("\n1. Testing BaseModelAdapter...")
    model = SimpleMockModel()
    adapter = BaseModelAdapter()
    
    # Add some fake attention modules to the model
    model.layer_0_attention = torch.nn.MultiheadAttention(256, 8)
    model.layer_1_self_attn = torch.nn.MultiheadAttention(256, 8)
    
    modules = adapter.get_attention_modules(model)
    print(f"Found {len(modules)} attention modules")
    for name, module in modules:
        print(f"  - {name}: {module.__class__.__name__}")
    
    return len(modules) > 0


def test_extraction():
    """Test attention extraction"""
    print("\n2. Testing AttentionExtractor...")
    model = SimpleMockModel()
    
    # Forward pass to check outputs
    outputs = model(output_attentions=True)
    print(f"Model outputs attentions: {hasattr(outputs, 'attentions') and outputs.attentions is not None}")
    if hasattr(outputs, 'attentions') and outputs.attentions:
        print(f"Number of attention layers: {len(outputs.attentions)}")
        print(f"First attention shape: {outputs.attentions[0].shape}")
    
    extractor = AttentionExtractor(model)
    
    # Create dummy inputs
    inputs = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10)
    }
    
    # Extract attention
    attention_data = extractor.extract(inputs)
    
    print(f"\nExtraction results:")
    print(f"  - attention_maps: {len(attention_data['attention_maps'])}")
    print(f"  - layer_names: {attention_data['layer_names']}")
    print(f"  - num_layers: {attention_data['num_layers']}")
    
    if attention_data['attention_maps']:
        print(f"  - First map shape: {attention_data['attention_maps'][0].shape}")
        print(f"  - num_heads: {attention_data['num_heads']}")
    
    return len(attention_data['attention_maps']) > 0


def test_visualizer():
    """Test the full visualizer"""
    print("\n3. Testing AttentionVisualizer...")
    
    class MockProcessor:
        def __call__(self, text=None, images=None, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "pixel_values": torch.randn(1, 3, 224, 224)
            }
        
        def decode(self, token_ids, **kwargs):
            return " ".join([f"token_{i}" for i in range(len(token_ids))])
    
    model = SimpleMockModel()
    processor = MockProcessor()
    
    try:
        visualizer = AttentionVisualizer(model, processor)
        viz = visualizer.visualize(
            image=Image.new('RGB', (224, 224), color='red'),
            text="test text",
            visualization_type="heatmap"
        )
        print("✅ Visualization created successfully!")
        return True
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Debug Attention Extraction")
    print("=" * 50)
    
    # Test numpy
    print(f"NumPy available: {ensure_numpy_available()}")
    
    # Run tests
    adapter_ok = test_adapter()
    extraction_ok = test_extraction()
    viz_ok = test_visualizer()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  - Adapter test: {'✅ PASS' if adapter_ok else '❌ FAIL'}")
    print(f"  - Extraction test: {'✅ PASS' if extraction_ok else '❌ FAIL'}")
    print(f"  - Visualizer test: {'✅ PASS' if viz_ok else '❌ FAIL'}")
    
    sys.exit(0 if all([adapter_ok, extraction_ok, viz_ok]) else 1)
EOF

# Run the debug script
python debug_attention_test.py

# Clean up
rm debug_attention_test.py

echo -e "\n\nNow running the actual tests..."
echo "================================"
pytest tests/test_basic.py::TestAttentionVisualizer::test_attention_extraction -v -s