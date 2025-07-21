# attention_viz/visualizers/heatmap.py - FIXED VERSION
"""
Attention heatmap visualizations - Fixed for CLIP
"""

import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


class AttentionHeatmap:
    """Creates heatmap visualizations of attention patterns"""

    def __init__(self):
        self.default_cmap = "Blues"
        self.figure_size = (12, 8)

    def create(
        self,
        attention_data: Dict[str, Any],
        inputs: Dict[str, Any],
        cmap: Optional[str] = None,
        show_values: bool = False,
        mask_padding: bool = True,
        aggregate_heads: bool = True,
        **kwargs,
    ) -> "VisualizationResult":
        """
        Create attention heatmap visualization

        Args:
            attention_data: Extracted attention data
            inputs: Original inputs
            cmap: Colormap to use
            show_values: Whether to show attention values in cells
            mask_padding: Whether to mask padding tokens
            aggregate_heads: Whether to average across attention heads
        """
        attention_maps = attention_data["attention_maps"]
        if not attention_maps:
            raise ValueError("No attention maps found in data")

        # Use the last layer by default
        attention_matrix = attention_maps[-1]

        # Aggregate heads if requested
        if aggregate_heads and attention_matrix.ndim > 2:
            attention_matrix = attention_matrix.mean(axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # FIXED: Handle masking properly
        if mask_padding and "attention_mask" in attention_data["token_info"]:
            mask = attention_data["token_info"]["attention_mask"][0]
            # Only apply mask if dimensions match
            if mask.shape[0] == attention_matrix.shape[0]:
                attention_matrix = self._apply_mask(attention_matrix, mask)
            else:
                print(f"Warning: Mask shape {mask.shape} doesn't match attention shape {attention_matrix.shape}. Skipping masking.")

        title = kwargs.pop('title', None)
        attention_type = kwargs.pop('attention_type', None)
        
        # Create heatmap - DON'T pass **kwargs to avoid errors
        sns.heatmap(
            attention_matrix,
            cmap=cmap or self.default_cmap,
            square=True,
            cbar_kws={"label": "Attention Weight"},
            annot=show_values,
            fmt=".2f" if show_values else None,
            ax=ax,
            vmin=0,
            vmax=attention_matrix.max() if attention_matrix.max() > 0 else 1,
        )
        
        # Add labels
        self._add_labels(ax, attention_data, inputs)
        
        # Use appropriate title
        if title:
            plt.title(title, fontsize=14, pad=20)
        else:
            # Default title based on what we're actually showing
            model_type = attention_data.get('model_type', 'Transformer')
            attn_type = attention_data.get('attention_type', 'self')
            
            if attn_type == 'text_self':
                default_title = f"{model_type} Text Self-Attention"
            elif attn_type == 'vision_self':
                default_title = f"{model_type} Vision Self-Attention"
            elif attn_type == 'cross':
                default_title = f"{model_type} Cross-Modal Attention"
            else:
                default_title = "Attention Heatmap"
                
            plt.title(default_title, fontsize=14, pad=20)
        
        plt.tight_layout()

        return VisualizationResult(fig)

    def _apply_mask(self, attention_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply attention mask to hide padding tokens"""
        # FIXED: Ensure mask dimensions match attention matrix
        if len(mask.shape) == 1:
            # Create 2D mask from 1D mask
            mask_2d = mask[:, None] * mask[None, :]
            
            # Make sure dimensions match
            if mask_2d.shape != attention_matrix.shape:
                print(f"Warning: Mask shape {mask_2d.shape} doesn't match attention shape {attention_matrix.shape}")
                return attention_matrix
        else:
            mask_2d = mask

        # Set masked positions to NaN (will appear white in heatmap)
        masked_attention = attention_matrix.copy()
        masked_attention[mask_2d == 0] = np.nan

        return masked_attention

    def _add_labels(self, ax, attention_data: Dict[str, Any], inputs: Dict[str, Any]):
        """Add token labels to axes"""
        # Get sequence length from attention matrix
        seq_len = ax.get_xlim()[1]
        
        # Create simple numeric labels
        labels = [str(i) for i in range(int(seq_len))]
        
        # Set labels (truncate if too many)
        max_labels = 30
        if len(labels) > max_labels:
            step = len(labels) // max_labels
            labels_subset = labels[::step]
            ax.set_xticks(range(0, len(labels), step))
            ax.set_yticks(range(0, len(labels), step))
            ax.set_xticklabels(labels_subset, rotation=45, ha="right")
            ax.set_yticklabels(labels_subset)
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Token Position", fontsize=12)


class VisualizationResult:
    """Container for visualization results with save/show methods"""

    def __init__(self, figure: plt.Figure):
        self.figure = figure

    def show(self):
        """Display the visualization"""
        plt.show()

    def save(self, path: str, dpi: int = 300, **kwargs):
        """Save visualization to file"""
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)

    def to_image(self) -> Image.Image:
        """Convert to PIL Image"""
        buf = io.BytesIO()
        self.figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return Image.open(buf)

    def __repr__(self):
        return f"<VisualizationResult with figure size {self.figure.get_size_inches()}>"