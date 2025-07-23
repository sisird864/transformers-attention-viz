"""
Core attention visualization class that correctly handles different model types
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
        
        # Identify model capabilities
        self.model_type = self._identify_model_type()
        self.has_cross_attention = self._check_cross_attention_support()

        # Initialize visualizers
        self.heatmap_viz = AttentionHeatmap()
        self.flow_viz = AttentionFlow()
        self.evolution_viz = LayerEvolution()
        
        # Warn about model limitations
        if not self.has_cross_attention:
            warnings.warn(
                f"Model type '{self.model_type}' does not support cross-modal attention. "
                "Only self-attention (text-to-text or image-to-image) can be visualized. "
                "Consider using BLIP or Flamingo for cross-modal attention visualization."
            )

    def _identify_model_type(self) -> str:
        """Identify the model type"""
        model_class_name = self.model.__class__.__name__
        
        # Check for BLIP models (handles BlipForConditionalGeneration, etc.)
        if any(x in model_class_name for x in ["BLIP", "Blip", "blip"]):
            return "BLIP"
        elif "CLIP" in model_class_name:
            return "CLIP"
        elif "Flamingo" in model_class_name:
            return "Flamingo"
        else:
            # Also check the module path (e.g., transformers.models.blip.modeling_blip)
            module_path = self.model.__class__.__module__
            if "blip" in module_path.lower():
                return "BLIP"
            elif "clip" in module_path.lower():
                return "CLIP"
            else:
                return "Unknown"

    
    def _check_cross_attention_support(self) -> bool:
        """Check if model supports cross-modal attention"""
        # CLIP does NOT have cross-modal attention
        if self.model_type == "CLIP":
            return False
        # BLIP DOES have cross-modal attention
        elif self.model_type == "BLIP":
            return True
        elif self.model_type in ["Flamingo"]:
            return True
        else:
            # Check for cross-attention layers in unknown models
            for name, module in self.model.named_modules():
                if "cross" in name.lower() and "attention" in name.lower():
                    return True
            return False

    def visualize(
        self,
        image: Union[Image.Image, torch.Tensor, str],
        text: Union[str, List[str]],
        layer_indices: Optional[Union[int, List[int]]] = None,
        head_indices: Optional[Union[int, List[int]]] = None,
        visualization_type: str = "heatmap",
        attention_type: str = "auto",  # New parameter
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
            attention_type: Type of attention to visualize ("auto", "text_self", "vision_self", "cross")
            **kwargs: Additional arguments for specific visualizers

        Returns:
            Visualization object with show() and save() methods
        """
        # Auto-select attention type based on model
        if attention_type == "auto":
            if self.has_cross_attention:
                attention_type = "cross"
                print(f"Model supports cross-modal attention. Visualizing cross-attention.")
            else:
                attention_type = "vision_self"
                print(f"Model does NOT support cross-modal attention. Visualizing vision self-attention.")
                print("To see text self-attention, use attention_type='text_self'")
        
        # Validate attention type for model
        if attention_type == "cross" and not self.has_cross_attention:
            raise ValueError(
                f"Model '{self.model_type}' does not support cross-modal attention. "
                f"Use attention_type='text_self' or 'vision_self' instead."
            )
        
        # Update the visualization title based on attention type
        if attention_type == "text_self":
            kwargs['title'] = f"Text Self-Attention ({self.model_type})"
        elif attention_type == "vision_self":
            kwargs['title'] = f"Vision Self-Attention ({self.model_type})"
        elif attention_type == "cross":
            kwargs['title'] = f"Cross-Modal Attention ({self.model_type})"
        
        # Store attention type for use by visualizers
        kwargs['attention_type'] = attention_type
        
        # Rest of the original method...
        inputs = self._preprocess_inputs(image, text)
        attention_data = self.extractor.extract(
            inputs, 
            layer_indices=layer_indices, 
            head_indices=head_indices,
            attention_type=attention_type
        )
        
        # Pass model info to visualizers
        attention_data['model_type'] = self.model_type
        attention_data['attention_type'] = attention_type

        if visualization_type == "heatmap":
            return self.heatmap_viz.create(attention_data, inputs, **kwargs)
        elif visualization_type == "flow":
            return self.flow_viz.create(attention_data, inputs, **kwargs)
        elif visualization_type == "evolution":
            return self.evolution_viz.create(attention_data, inputs, **kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

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

        return inputs

    def get_attention_stats(
        self,
        image: Union[Image.Image, torch.Tensor, str],
        text: Union[str, List[str]],
        layer_index: int = -1,
        attention_type: str = "auto",
    ) -> Dict[str, np.ndarray]:
        """
        Get statistical analysis of attention patterns

        Returns:
            Dictionary containing entropy, top tokens, attention distribution stats
        """
        # Auto-select attention type
        if attention_type == "auto":
            attention_type = "vision_self" if not self.has_cross_attention else "cross"
        
        inputs = self._preprocess_inputs(image, text)
        attention_data = self.extractor.extract(
            inputs, 
            layer_indices=[layer_index],
            attention_type=attention_type
        )

        # Calculate statistics
        attention_matrix = attention_data["attention_maps"][0]  # Shape: [heads, seq_len, seq_len]

        # Calculate entropy for each head
        entropy = self._calculate_entropy(attention_matrix)

        # Find top attended tokens/patches
        # Note: _get_top_attended_tokens now automatically detects if it's vision or text
        top_tokens = self._get_top_attended_tokens(attention_matrix, inputs["input_ids"])

        # Calculate attention concentration
        concentration = self._calculate_concentration(attention_matrix)

        return {
            "entropy": entropy,
            "top_tokens": top_tokens,
            "concentration": concentration,
            "mean_attention": attention_matrix.mean(axis=(0, 1)),
            "std_attention": attention_matrix.std(axis=(0, 1)),
            "attention_type": attention_type,
            "model_type": self.model_type,
            "sequence_length": attention_matrix.shape[-1],  # Add this to help identify what type
        }


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
        """Get the top-k most attended tokens/patches"""
        # Average attention across heads and source positions
        avg_attention = attention_matrix.mean(axis=(0, 1))

        # Get top-k indices
        top_indices = np.argsort(avg_attention)[-top_k:][::-1].copy()

        # Check if this is vision attention based on matrix size
        # Text attention is typically small (< 20), vision is larger (50 for CLIP)
        is_vision_attention = attention_matrix.shape[-1] > 20

        if is_vision_attention:
            # For vision attention, return patch information
            top_items = []
            for idx in top_indices:
                if idx == 0:
                    # First patch is typically CLS token
                    label = "CLS_token"
                else:
                    # Calculate which patch this is (assuming 7x7 grid for CLIP)
                    # Subtract 1 because patch 0 is CLS token
                    patch_idx = idx - 1
                    row = patch_idx // 7
                    col = patch_idx % 7
                    label = f"Patch_({row},{col})"
                
                top_items.append((label, avg_attention[idx]))
            
            return top_items
        else:
            # For text attention, decode tokens
            tokens = []
            if self.processor is not None:
                for idx in top_indices:
                    # Decode each token separately
                    if idx < len(input_ids[0]):
                        token = self.processor.decode([input_ids[0][idx]], skip_special_tokens=False)
                        tokens.append(token.strip())
                    else:
                        tokens.append(f"token_{idx}")
            else:
                tokens = [f"token_{i}" for i in top_indices]

            # Create (token, attention_score) pairs
            top_tokens = [
                (tokens[i], avg_attention[idx])
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