# Fixing NumPy Initialization Issues

If you encounter "RuntimeError: Numpy is not available" when running tests or using the library, follow these steps:

## Quick Fix (Already Applied)

The code has been updated to handle this issue automatically by using a fallback conversion method when the standard PyTorch to NumPy conversion fails.

## Environment Fix (Recommended)

### Option 1: Reinstall NumPy and PyTorch

```bash
# Deactivate and reactivate your virtual environment
deactivate
source venv/bin/activate

# Uninstall and reinstall numpy and torch
pip uninstall numpy torch torchvision -y
pip install numpy
pip install torch torchvision
```

### Option 2: Use Conda (Recommended for Mac)

```bash
# Create a new conda environment
conda create -n attention-viz python=3.11
conda activate attention-viz

# Install PyTorch through conda (better Mac support)
conda install pytorch torchvision -c pytorch

# Install the rest of the dependencies
pip install -e ".[dev]"
```

### Option 3: Set Environment Variable

Add this to your shell profile (~/.zshrc or ~/.bash_profile):

```bash
export OMP_NUM_THREADS=1
```

Then restart your terminal.

## Verify the Fix

Run this Python script to verify NumPy is working:

```python
import torch
import numpy as np
from attention_viz.utils import ensure_numpy_available

# Check if numpy is available
if ensure_numpy_available():
    print("✅ NumPy is properly initialized!")
else:
    print("⚠️  Using fallback conversion (slightly slower but fully functional)")

# Test conversion
tensor = torch.tensor([1, 2, 3])
try:
    array = tensor.numpy()
    print("✅ Direct conversion works!")
except:
    from attention_viz.utils import tensor_to_numpy
    array = tensor_to_numpy(tensor)
    print("✅ Fallback conversion works!")
    
print(f"Result: {array}")
```

## Platform-Specific Notes

### Apple Silicon Macs (M1/M2/M3)

This issue is most common on Apple Silicon. The conda installation method usually works best:

```bash
# Install Miniforge (optimized for Apple Silicon)
brew install miniforge

# Create environment
conda create -n attention-viz python=3.11
conda activate attention-viz
conda install pytorch::pytorch torchvision -c pytorch
```

### Intel Macs

The pip installation should work, but if not, try:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Testing After Fix

Run the tests again:

```bash
pytest tests/ -v
```

All tests should now pass! 🎉