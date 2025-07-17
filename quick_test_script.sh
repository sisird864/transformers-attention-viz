#!/bin/bash

# Quick test to verify the fixes work

echo "Running quick test of the fixes..."
echo "=================================="

# First, make sure numpy compatibility is OK
echo -e "\n1. Testing NumPy compatibility:"
python -c "
import torch
import numpy as np
from attention_viz.utils import tensor_to_numpy, ensure_numpy_available

print(f'NumPy available: {ensure_numpy_available()}')
t = torch.tensor([1.0, 2.0, 3.0])
arr = tensor_to_numpy(t)
print(f'Tensor to numpy conversion: {arr}')
"

# Run the specific failing tests
echo -e "\n2. Running the previously failing tests:"
pytest tests/test_basic.py::TestAttentionVisualizer::test_visualization_creation -v
pytest tests/test_basic.py::TestAttentionVisualizer::test_attention_extraction -v
pytest tests/test_basic.py::TestAttentionVisualizer::test_statistics_calculation -v

# Run all tests
echo -e "\n3. Running all tests:"
pytest tests/test_basic.py -v --tb=short

echo -e "\nTest run complete!"