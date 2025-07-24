# attention_viz/visualizers/heatmap.py - FIXED VERSION for BLIP
"""
Attention heatmap visualizations - Fixed for CLIP and BLIP
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

        # FIXED: Handle different attention types differently
        attention_type = attention_data.get("attention_type", "self")
        model_type = attention_data.get("model_type", "Unknown")

        # For BLIP cross-attention, we need special handling
        if model_type == "BLIP" and attention_type == "cross":
            # BLIP cross-attention shape: [heads, text_tokens, image_tokens]
            # We want to visualize how text tokens attend to image patches

            if aggregate_heads and attention_matrix.ndim > 2:
                attention_matrix = attention_matrix.mean(axis=0)

            # Create figure with subplots for each text token
            n_text_tokens = attention_matrix.shape[0]
            n_cols = min(n_text_tokens, 4)
            n_rows = (n_text_tokens + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_cols > 1 else [[axes]]
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            # Get token names if available
            token_names = self._get_token_names(inputs, attention_data)

            for token_idx in range(n_text_tokens):
                row = token_idx // n_cols
                col = token_idx % n_cols
                ax = axes[row, col] if n_rows > 1 or n_cols > 1 else axes

                # Get attention for this text token to all image patches
                token_attention = attention_matrix[token_idx]  # Shape: (577,) for BLIP

                # Skip CLS token and reshape to grid
                if len(token_attention) == 577:  # BLIP's typical size
                    image_attention = token_attention[1:].reshape(24, 24)
                elif len(token_attention) == 50:  # CLIP's typical size
                    image_attention = token_attention[1:].reshape(7, 7)
                else:
                    # Try to find reasonable square shape
                    grid_size = int(np.sqrt(len(token_attention) - 1))
                    if grid_size * grid_size == len(token_attention) - 1:
                        image_attention = token_attention[1:].reshape(grid_size, grid_size)
                    else:
                        # Fallback: just show as 1D
                        image_attention = token_attention.reshape(-1, 1)

                # Create heatmap
                sns.heatmap(
                    image_attention,
                    cmap=cmap or self.default_cmap,
                    square=True,
                    cbar=True,
                    ax=ax,
                    vmin=0,
                    vmax=image_attention.max() if image_attention.max() > 0 else 1,
                    cbar_kws={"shrink": 0.8},
                )

                # Set title
                token_name = (
                    token_names[token_idx] if token_idx < len(token_names) else f"Token {token_idx}"
                )
                ax.set_title(f"{token_name}", fontsize=12)
                ax.set_xlabel("")
                ax.set_ylabel("")

                # Remove ticks for clarity
                ax.set_xticks([])
                ax.set_yticks([])

            # Hide unused subplots
            for idx in range(n_text_tokens, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)

            # Overall title
            title = kwargs.get("title", f"{model_type} Cross-Modal Attention")
            plt.suptitle(title, fontsize=16)

        else:
            # Original behavior for self-attention
            if aggregate_heads and attention_matrix.ndim > 2:
                attention_matrix = attention_matrix.mean(axis=0)

            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Apply masking if requested
            if mask_padding and "attention_mask" in attention_data["token_info"]:
                mask = attention_data["token_info"]["attention_mask"][0]
                if mask.shape[0] == attention_matrix.shape[0]:
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
                vmin=0,
                vmax=attention_matrix.max() if attention_matrix.max() > 0 else 1,
            )

            # Add labels
            self._add_labels(ax, attention_data, inputs)

            # Title
            title = kwargs.get("title", "Attention Heatmap")
            plt.title(title, fontsize=14, pad=20)

        plt.tight_layout()

        return VisualizationResult(fig)

    def _get_token_names(self, inputs: Dict[str, Any], attention_data: Dict[str, Any]) -> List[str]:
        """Get token names from inputs"""
        token_names = []

        if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
            input_ids = (
                inputs["input_ids"][0] if inputs["input_ids"].dim() > 1 else inputs["input_ids"]
            )

            # Try to decode tokens if we have a processor/tokenizer reference
            # For now, return generic names
            for i in range(len(input_ids)):
                if i == 0:
                    token_names.append("[CLS/BOS]")
                elif i == len(input_ids) - 1:
                    token_names.append("[SEP/EOS]")
                else:
                    token_names.append(f"Token {i}")

        return token_names

    def _apply_mask(self, attention_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply attention mask to hide padding tokens"""
        # FIXED: Ensure mask dimensions match attention matrix
        if len(mask.shape) == 1:
            # Create 2D mask from 1D mask
            mask_2d = mask[:, None] * mask[None, :]

            # Make sure dimensions match
            if mask_2d.shape != attention_matrix.shape:
                print(
                    f"Warning: Mask shape {mask_2d.shape} doesn't match attention shape {attention_matrix.shape}"
                )
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
