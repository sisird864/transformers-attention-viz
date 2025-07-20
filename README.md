# 🔍 Transformers Attention Viz

Interactive attention visualization for multi-modal transformer models (CLIP, BLIP, Flamingo, etc.)
![Transformers Attention Viz](https://raw.githubusercontent.com/sisird864/transformers-attention-viz/main/docs/images/hero_image.png)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Transformers](https://img.shields.io/badge/transformers-4.30+-orange.svg)](https://github.com/huggingface/transformers)

## 🎯 Features

- 📊 **Interactive Visualizations**: Explore attention patterns between text and images
- 🔄 **Multi-Layer Support**: Visualize attention across all transformer layers
- 🎨 **Customizable**: Multiple color schemes and visualization styles
- 📸 **Export Ready**: Generate publication-quality figures
- 🚀 **Easy Integration**: Works seamlessly with HuggingFace models

## 🚀 Quick Start

### Installation

```bash
pip install transformers-attention-viz
```

### Basic Usage

```python
from transformers import CLIPModel, CLIPProcessor
from attention_viz import AttentionVisualizer

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create visualizer
visualizer = AttentionVisualizer(model)

# Visualize attention
image = Image.open("cat.jpg")
text = "a photo of a cat"

viz = visualizer.visualize(
    image=image,
    text=text,
    layer_index=-1  # Last layer
)
viz.show()
```

## 📸 Examples

### Cross-Modal Attention Heatmap

![Cross-modal attention example](https://raw.githubusercontent.com/sisird864/transformers-attention-viz/main/docs/images/cross_attention_example.png)

### Layer-wise Attention Evolution

![Cross-modal attention example](https://raw.githubusercontent.com/sisird864/transformers-attention-viz/main/docs/images/layer_evolution_example.png)

### Interactive Dashboard

![Cross-modal attention example](https://raw.githubusercontent.com/sisird864/transformers-attention-viz/main/docs/images/dashboard_example.png)

## 🛠️ Advanced Usage

### Visualizing Specific Layers and Heads

```python
# Visualize specific layers
viz = visualizer.visualize(
    image=image,
    text=text,
    layer_indices=[0, 6, 11],  # First, middle, and last layers
    head_indices=[0, 4, 8]     # Specific attention heads
)

# Get attention statistics
stats = visualizer.get_attention_stats(image, text)
print(f"Attention entropy: {stats['entropy']}")
print(f"Top attended tokens: {stats['top_tokens']}")
```

### Comparing Multiple Inputs

```python
# Compare attention patterns
comparison = visualizer.compare_attention(
    images=[image1, image2],
    texts=["a cat", "a dog"],
    layer_index=-1
)
comparison.save("attention_comparison.png")
```

### Interactive Dashboard

```python
from attention_viz import launch_dashboard

# Launch interactive exploration tool
launch_dashboard(model, processor)
# Opens at http://localhost:7860
```

## 🔧 Supported Models

- ✅ CLIP (all variants)
- ✅ BLIP
- ✅ BLIP-2
- ✅ Flamingo (coming soon)
- ✅ CoCa (coming soon)
- ✅ Custom vision-language models

## 📚 Documentation

Full documentation available at <https://transformers-attention-viz.readthedocs.io>

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Tutorials](docs/tutorials.md)
- [Contributing](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/transformers-attention-viz.git
cd transformers-attention-viz

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📖 Citation

If you use this tool in your research, please cite:

```bibtex
@software{transformers-attention-viz,
  author = {Your Name},
  title = {Transformers Attention Viz: Interactive Attention Visualization for Multi-Modal Transformers},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/transformers-attention-viz}
}
```

## 📄 License

MIT License - see <LICENSE> for details.

## 🙏 Acknowledgments

- HuggingFace team for the amazing Transformers library
- OpenAI for CLIP
- Salesforce Research for BLIP

## 🛤️ Roadmap

- [ ] Support for Flamingo models
- [ ] 3D attention visualization
- [ ] Attention pattern export to TensorBoard
- [ ] Real-time video attention tracking
- [ ] Attention-based model debugging tools

## 🚧 Known Issues (v0.1.3)

- Evolution and Flow visualizations may fail with certain text lengths
- Working on fixes for v0.1.4
- Heatmap visualization works reliably for all inputs