"""
Utility functions for handling numpy compatibility issues
"""

import warnings

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert PyTorch tensor to numpy array, handling initialization issues.

    Args:
        tensor: PyTorch tensor to convert

    Returns:
        numpy array
    """
    try:
        # Try standard conversion
        return tensor.cpu().numpy()
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            # Fallback method for numpy initialization issues
            # Convert to list first, then to numpy
            if tensor.dim() == 0:
                # Scalar tensor
                return np.array(tensor.item())
            else:
                # Multi-dimensional tensor
                return np.array(tensor.cpu().tolist())
        else:
            raise


def ensure_numpy_available():
    """
    Check if numpy is properly initialized with PyTorch.

    Returns:
        bool: True if numpy is available, False otherwise
    """
    try:
        test_tensor = torch.tensor([1.0])
        test_tensor.numpy()
        return True
    except RuntimeError:
        warnings.warn(
            "NumPy initialization issue detected. Using fallback conversion method. "
            "This may impact performance slightly but functionality remains intact."
        )
        return False
