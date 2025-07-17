# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- CUDA (optional, for GPU acceleration)

## Quick Install

### From PyPI (Recommended)

```bash
pip install transformers-attention-viz
```

### From Source

```bash
git clone https://github.com/YOUR_USERNAME/transformers-attention-viz.git
cd transformers-attention-viz
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/YOUR_USERNAME/transformers-attention-viz.git
cd transformers-attention-viz
pip install -e ".[dev]"
```

## Verify Installation

```python
import attention_viz
print(attention_viz.__version__)
```

## GPU Support

The library automatically detects and uses GPU if available. To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Troubleshooting

### ImportError with Transformers

Make sure you have the correct version:
```bash
pip install transformers>=4.30.0
```

### Gradio Issues

If you encounter issues with the dashboard:
```bash
pip install gradio>=3.0.0 --upgrade
```

### Memory Issues

For large models, you may need to:
- Use a smaller batch size
- Enable gradient checkpointing
- Use CPU instead of GPU for visualization

## Platform-Specific Notes

### Windows

- Use Anaconda for easier installation
- May need Visual C++ Build Tools

### macOS

- Works with both Intel and Apple Silicon
- For M1/M2, use PyTorch nightly builds

### Linux

- Most straightforward installation
- Ensure CUDA drivers match PyTorch version

## Docker Installation

```dockerfile
FROM python:3.9

RUN pip install transformers-attention-viz

CMD ["attention-viz-dashboard"]
```

Build and run:
```bash
docker build -t attention-viz .
docker run -p 7860:7860 attention-viz
```