"""
Head-wise attention comparison visualization
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .heatmap import VisualizationResult


class HeadComparison:
    """Compares attention patterns across different attention heads"""

    def __init__(self):
        self.figure_size = (16, 12)

    def create(
        self,
        attention_data: Dict[str, Any],
        inputs: Dict[str, Any],
        layer_index: int = -1,
        max_heads: int = 12,
        **kwargs,
    ) -> "VisualizationResult":
        """
        Create head comparison visualization

        Args:
            attention_data: Extracted attention data
            inputs: Original inputs
            layer_index: Which layer to analyze
            max_heads: Maximum number of heads to display
        """
        from .heatmap import VisualizationResult

        attention_maps = attention_data["attention_maps"]
        if not attention_maps:
            raise ValueError("No attention maps found")

        # Get attention for specified layer
        layer_attention = attention_maps[layer_index]

        if layer_attention.ndim == 2:
            # Single head, nothing to compare
            raise ValueError("Model has only one attention head, nothing to compare")

        n_heads = min(layer_attention.shape[0], max_heads)

        # Create subplots grid
        n_cols = 4
        n_rows = int(np.ceil(n_heads / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each head
        for head_idx in range(n_heads):
            row = head_idx // n_cols
            col = head_idx % n_cols
            ax = axes[row, col]

            head_attention = layer_attention[head_idx]

            # Create heatmap
            sns.heatmap(
                head_attention,
                cmap="Blues",
                square=True,
                cbar=True,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )

            ax.set_title(f"Head {head_idx}", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Reduce tick labels for clarity
            if head_attention.shape[0] > 20:
                ax.set_xticks([])
                ax.set_yticks([])

        # Hide unused subplots
        for idx in range(n_heads, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        # Add head statistics subplot
        self._add_head_statistics(fig, layer_attention, n_heads)

        plt.suptitle(f"Attention Head Comparison - Layer {layer_index}", fontsize=16)
        plt.tight_layout()

        return VisualizationResult(fig)

    def _add_head_statistics(self, fig, layer_attention: np.ndarray, n_heads: int):
        """Add statistical analysis of head behaviors"""
        # Create new axes for statistics
        gs = fig.add_gridspec(3, 4, hspace=0.3)
        ax_stats = fig.add_subplot(gs[2, :])

        # Calculate statistics for each head
        head_stats: Dict[str, List[float]] = {"entropy": [], "sparsity": [], "diagonal_focus": []}

        for head_idx in range(n_heads):
            head_attention = layer_attention[head_idx]

            # Entropy
            eps = 1e-8
            safe_attention = head_attention + eps
            entropy = -np.sum(safe_attention * np.log(safe_attention), axis=-1).mean()
            head_stats["entropy"].append(entropy)

            # Sparsity (percentage of near-zero values)
            sparsity = (head_attention < 0.01).sum() / head_attention.size
            head_stats["sparsity"].append(sparsity)

            # Diagonal focus (attention to same position)
            diagonal_mean = np.diag(head_attention).mean()
            head_stats["diagonal_focus"].append(diagonal_mean)

        # Plot statistics
        x = np.arange(n_heads)
        width = 0.25

        ax_stats.bar(x - width, head_stats["entropy"], width, label="Entropy", alpha=0.8)
        ax_stats.bar(x, head_stats["sparsity"], width, label="Sparsity", alpha=0.8)
        ax_stats.bar(
            x + width, head_stats["diagonal_focus"], width, label="Diagonal Focus", alpha=0.8
        )

        ax_stats.set_xlabel("Head Index")
        ax_stats.set_ylabel("Value")
        ax_stats.set_title("Head Behavior Statistics")
        ax_stats.legend()
        ax_stats.grid(True, alpha=0.3)
