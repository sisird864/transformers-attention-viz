"""
Attention heatmap visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import io


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
        **kwargs
    ) -> 'VisualizationResult':
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
        
        # Apply padding mask if available
        if mask_padding and "attention_mask" in attention_data["token_info"]:
            mask = attention_data["token_info"]["attention_mask"][0]
            attention_matrix = self._apply_mask(attention_matrix, mask)
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            cmap=cmap or self.default_cmap,
            square=True,
            cbar_kws={"label": "Attention Weight"},
            annot=show_values,
            fmt=".2f" if show_values else None,
            ax=ax,
            **kwargs
        )
        
        # Add labels
        self._add_labels(ax, attention_data, inputs)
        
        plt.title("Cross-Modal Attention Heatmap", fontsize=14, pad=20)
        plt.tight_layout()
        
        return VisualizationResult(fig)
    
    def create_comparison(
        self,
        attention_data_list: List[Dict[str, Any]],
        inputs_list: List[Dict[str, Any]],
        comparison_type: str = "side_by_side",
        **kwargs
    ) -> 'VisualizationResult':
        """Create comparison visualization of multiple attention patterns"""
        n_comparisons = len(attention_data_list)
        
        if comparison_type == "side_by_side":
            fig, axes = plt.subplots(1, n_comparisons, figsize=(6 * n_comparisons, 6))
            if n_comparisons == 1:
                axes = [axes]
        else:  # grid layout
            n_rows = int(np.ceil(np.sqrt(n_comparisons)))
            n_cols = int(np.ceil(n_comparisons / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
            axes = axes.flatten()
        
        for idx, (attention_data, inputs) in enumerate(zip(attention_data_list, inputs_list)):
            ax = axes[idx]
            attention_matrix = attention_data["attention_maps"][-1]
            
            if attention_matrix.ndim > 2:
                attention_matrix = attention_matrix.mean(axis=0)
            
            sns.heatmap(
                attention_matrix,
                cmap=self.default_cmap,
                square=True,
                cbar=True,
                ax=ax,
                **kwargs
            )
            
            ax.set_title(f"Input {idx + 1}", fontsize=12)
        
        # Hide unused subplots
        for idx in range(n_comparisons, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Attention Pattern Comparison", fontsize=16)
        plt.tight_layout()
        
        return VisualizationResult(fig)
    
    def _apply_mask(self, attention_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply attention mask to hide padding tokens"""
        # Expand mask to match attention matrix dimensions
        mask_2d = mask[:, None] * mask[None, :]
        
        # Set masked positions to NaN (will appear white in heatmap)
        masked_attention = attention_matrix.copy()
        masked_attention[mask_2d == 0] = np.nan
        
        return masked_attention
    
    def _add_labels(self, ax, attention_data: Dict[str, Any], inputs: Dict[str, Any]):
        """Add token labels to axes"""
        # Get modality boundaries
        boundaries = attention_data["token_info"]["modality_boundaries"]
        text_end = boundaries["text_end"]
        image_end = boundaries["image_end"]
        
        # Create labels
        labels = []
        for i in range(boundaries["total_length"]):
            if i < text_end:
                labels.append(f"T{i}")  # Text token
            else:
                labels.append(f"I{i-text_end}")  # Image patch
        
        # Set labels (truncate if too many)
        max_labels = 50
        if len(labels) > max_labels:
            step = len(labels) // max_labels
            labels = labels[::step]
            ax.set_xticks(range(0, len(labels) * step, step))
            ax.set_yticks(range(0, len(labels) * step, step))
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
        
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel("Target Tokens", fontsize=12)
        ax.set_ylabel("Source Tokens", fontsize=12)


class VisualizationResult:
    """Container for visualization results with save/show methods"""
    
    def __init__(self, figure: plt.Figure):
        self.figure = figure
    
    def show(self):
        """Display the visualization"""
        plt.show()
    
    def save(self, path: str, dpi: int = 300, **kwargs):
        """Save visualization to file"""
        self.figure.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    
    def to_image(self) -> Image.Image:
        """Convert to PIL Image"""
        buf = io.BytesIO()
        self.figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf)
    
    def __repr__(self):
        return f"<VisualizationResult with figure size {self.figure.get_size_inches()}>"