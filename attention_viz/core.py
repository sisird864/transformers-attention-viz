"""
Core attention visualization class that orchestrates the entire pipeline
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .extractors import AttentionExtractor
from .utils import tensor_to_numpy
from .visualizers import AttentionFlow, AttentionHeatmap, LayerEvolution


class AttentionVisualizer:
    """Main class for visualizing attention in multi-modal transformers"""

    def __init__(
        self, model: Union[CLIPModel, torch.nn.Module], processor: Optional[CLIPProcessor] = None
    ):
        """
        Initialize the attention visualizer

        Args:
            model: HuggingFace multi-modal model
            processor: Optional processor for the model
        """
        self.model = model
        self.processor = processor
        self.extractor = AttentionExtractor(model)

        # Initialize visualizers
        self.heatmap_viz = AttentionHeatmap()
        self.flow_viz = AttentionFlow()
        self.evolution_viz = LayerEvolution()

    def visualize(
        self,
        image: Union[Image.Image, torch.Tensor, str],
        text: Union[str, List[str]],
        layer_indices: Optional[Union[int, List[int]]] = None,
        head_indices: Optional[Union[int, List[int]]] = None,
        visualization_type: str = "heatmap",
        **kwargs,
    ):
        """
        Main visualization method

        Args:
            image: Input image (PIL Image, tensor, or path)
            text: Input text
            layer_indices: Which layers to visualize (default: last layer)
            head_indices: Which attention heads to visualize (default: all)
            visualization_type: Type of visualization ("heatmap", "flow", "evolution")
            **kwargs: Additional arguments for specific visualizers

        Returns:
            Visualization object with show() and save() methods
        """
        # Preprocess inputs
        inputs = self._preprocess_inputs(image, text)

        # Extract attention
        attention_data = self.extractor.extract(
            inputs, layer_indices=layer_indices, head_indices=head_indices
        )

        # Create visualization
        if visualization_type == "heatmap":
            return self.heatmap_viz.create(attention_data, inputs, **kwargs)
        elif visualization_type == "flow":
            return self.flow_viz.create(attention_data, inputs, **kwargs)
        elif visualization_type == "evolution":
            return self.evolution_viz.create(attention_data, inputs, **kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

    def get_attention_stats(
        self,
        image: Union[Image.Image, torch.Tensor, str],
        text: Union[str, List[str]],
        layer_index: int = -1,
    ) -> Dict[str, np.ndarray]:
        """
        Get statistical analysis of attention patterns

        Returns:
            Dictionary containing entropy, top tokens, attention distribution stats
        """
        inputs = self._preprocess_inputs(image, text)
        attention_data = self.extractor.extract(inputs, layer_indices=[layer_index])

        # Calculate statistics
        attention_matrix = attention_data["attention_maps"][0]  # Shape: [heads, seq_len, seq_len]

        # Calculate entropy for each head
        entropy = self._calculate_entropy(attention_matrix)

        # Find top attended tokens
        top_tokens = self._get_top_attended_tokens(attention_matrix, inputs["input_ids"])

        # Calculate attention concentration
        concentration = self._calculate_concentration(attention_matrix)

        return {
            "entropy": entropy,
            "top_tokens": top_tokens,
            "concentration": concentration,
            "mean_attention": attention_matrix.mean(axis=(0, 1)),
            "std_attention": attention_matrix.std(axis=(0, 1)),
        }

    def compare_attention(
        self,
        images: List[Union[Image.Image, torch.Tensor, str]],
        texts: List[str],
        layer_index: int = -1,
        comparison_type: str = "side_by_side",
    ):
        """
        Compare attention patterns across multiple input pairs
        """
        all_attention_data = []
        all_inputs = []

        for image, text in zip(images, texts):
            inputs = self._preprocess_inputs(image, text)
            attention_data = self.extractor.extract(inputs, layer_indices=[layer_index])
            all_attention_data.append(attention_data)
            all_inputs.append(inputs)

        # Create comparison visualization
        return self.heatmap_viz.create_comparison(
            all_attention_data, all_inputs, comparison_type=comparison_type
        )

    def _preprocess_inputs(
        self, image: Union[Image.Image, torch.Tensor, str], text: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Preprocess inputs using the model's processor"""
        if self.processor is None:
            raise ValueError("Processor not provided. Please initialize with a processor.")

        # Handle different image input types
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:
                image = image.squeeze(0)
            image_array = tensor_to_numpy(image.permute(1, 2, 0))
            image = Image.fromarray((image_array * 255).astype(np.uint8))

        # Process inputs
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        return inputs  # type: ignore[no-any-return]

    def _calculate_entropy(self, attention_matrix: np.ndarray) -> np.ndarray:
        """Calculate attention entropy for each head"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_matrix = attention_matrix + eps

        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(attention_matrix * np.log(attention_matrix), axis=-1)
        return entropy.mean(axis=-1)  # Average over sequence length

    def _get_top_attended_tokens(
        self, attention_matrix: np.ndarray, input_ids: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get the top-k most attended tokens"""
        # Average attention across heads and source positions
        avg_attention = attention_matrix.mean(axis=(0, 1))

        # Get top-k indices
        top_indices = np.argsort(avg_attention)[-top_k:][::-1].copy()

        # Decode tokens
        if self.processor is not None:
            tokens = self.processor.decode(input_ids[0][top_indices], skip_special_tokens=False)
            tokens = tokens.split()
        else:
            tokens = [f"token_{i}" for i in top_indices]

        # Create (token, attention_score) pairs
        top_tokens = [
            (tokens[i] if i < len(tokens) else f"token_{idx}", avg_attention[idx])
            for i, idx in enumerate(top_indices)
        ]

        return top_tokens

    def _calculate_concentration(self, attention_matrix: np.ndarray) -> float:
        """Calculate how concentrated the attention is (Gini coefficient)"""
        # Flatten and sort attention values
        attention_flat = attention_matrix.flatten()
        attention_sorted = np.sort(attention_flat)

        # Calculate Gini coefficient
        n = len(attention_sorted)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * attention_sorted)) / (n * np.sum(attention_sorted)) - (n + 1) / n

        return float(gini)
