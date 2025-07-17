# API Reference

## Core Module

### AttentionVisualizer

Main class for creating attention visualizations.

```python
class AttentionVisualizer(model, processor=None)
```

#### Parameters:
- `model`: HuggingFace transformer model
- `processor`: Optional processor for the model

#### Methods:

##### visualize()
```python
visualize(image, text, layer_indices=None, head_indices=None, 
         visualization_type="heatmap", **kwargs)
```

Create attention visualization.

**Parameters:**
- `image`: Input image (PIL Image, tensor, or path)
- `text`: Input text string
- `layer_indices`: Which layers to visualize (default: last layer)
- `head_indices`: Which attention heads to visualize (default: all)
- `visualization_type`: Type of visualization ("heatmap", "flow", "evolution")

**Returns:**
- `VisualizationResult`: Object with `show()` and `save()` methods

##### get_attention_stats()
```python
get_attention_stats(image, text, layer_index=-1)
```

Get statistical analysis of attention patterns.

**Returns:**
- Dictionary with entropy, top tokens, concentration metrics

##### compare_attention()
```python
compare_attention(images, texts, layer_index=-1, comparison_type="side_by_side")
```

Compare attention patterns across multiple inputs.

## Extractors Module

### AttentionExtractor

Extract attention weights from transformer models.

```python
class AttentionExtractor(model)
```

#### Methods:

##### extract()
```python
extract(inputs, layer_indices=None, head_indices=None)
```

Extract attention maps for given inputs.

**Returns:**
- Dictionary containing attention maps and metadata

## Visualizers Module

### AttentionHeatmap

Create heatmap visualizations.

```python
class AttentionHeatmap()
```

#### Methods:

##### create()
```python
create(attention_data, inputs, cmap=None, show_values=False, 
       mask_padding=True, aggregate_heads=True)
```

### AttentionFlow

Create flow diagram visualizations.

```python
class AttentionFlow()
```

#### Methods:

##### create()
```python
create(attention_data, inputs, threshold=0.1, top_k=None, bidirectional=False)
```

### LayerEvolution

Visualize attention evolution across layers.

```python
class LayerEvolution()
```

#### Methods:

##### create()
```python
create(attention_data, inputs, metric="entropy", animate=False, 
       selected_tokens=None)
```

## Dashboard Module

### launch_dashboard()

Launch interactive Gradio dashboard.

```python
launch_dashboard(model=None, processor=None, **kwargs)
```

**Parameters:**
- `model`: Pre-loaded model (optional)
- `processor`: Pre-loaded processor (optional)
- `**kwargs`: Additional Gradio launch arguments

## Utilities

### VisualizationResult

Container for visualization outputs.

#### Methods:

##### show()
Display the visualization.

##### save()
```python
save(path, dpi=300, **kwargs)
```

Save visualization to file.

##### to_image()
Convert to PIL Image.

## Model Support

Currently supported models:
- CLIP (all variants)
- BLIP
- BLIP-2
- More coming soon!

## Examples

### Basic Usage
```python
from transformers import CLIPModel, CLIPProcessor
from attention_viz import AttentionVisualizer

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

visualizer = AttentionVisualizer(model, processor)

viz = visualizer.visualize(
    image="cat.jpg",
    text="a photo of a cat"
)
viz.show()
```

### Advanced Options
```python
# Visualize specific layers and heads
viz = visualizer.visualize(
    image=image,
    text=text,
    layer_indices=[0, 6, 11],
    head_indices=[0, 4, 8],
    visualization_type="flow",
    threshold=0.15,
    top_k=10
)

# Save high-resolution figure
viz.save("attention_analysis.png", dpi=600)
```