"""
Visualization of attention evolution across layers
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

from .heatmap import VisualizationResult


class LayerEvolution:
    """Visualizes how attention patterns evolve across transformer layers"""

    def __init__(self):
        self.figure_size = (15, 10)

    def create(
        self,
        attention_data: Dict[str, Any],
        inputs: Dict[str, Any],
        metric: str = "entropy",
        animate: bool = False,
        selected_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> "VisualizationResult":
        """
        Create layer evolution visualization

        Args:
            attention_data: Extracted attention data
            inputs: Original inputs
            metric: What to track ("entropy", "concentration", "diversity")
            animate: Create animated visualization
            selected_tokens: Specific tokens to track
        """
        from .heatmap import VisualizationResult

        attention_maps = attention_data["attention_maps"]
        if not attention_maps:
            raise ValueError("No attention maps found")

        # Calculate metrics for each layer
        layer_metrics = self._calculate_layer_metrics(attention_maps, metric)

        if animate:
            fig = self._create_animation(attention_maps, layer_metrics, metric)
        else:
            fig = self._create_static_plot(attention_maps, layer_metrics, metric, selected_tokens)

        return VisualizationResult(fig)

    def _calculate_layer_metrics(
        self, attention_maps: List[np.ndarray], metric: str
    ) -> Dict[str, np.ndarray]:
        """Calculate specified metric for each layer"""
        metrics: Dict[str, List[float]] = {
            "entropy": [],
            "concentration": [],
            "diversity": [],
            "mean_attention": [],
        }

        for layer_attention in attention_maps:
            # Average across heads if needed
            if layer_attention.ndim > 2:
                layer_attention = layer_attention.mean(axis=0)

            # Calculate entropy
            eps = 1e-8
            safe_attention = layer_attention + eps
            entropy = -np.sum(safe_attention * np.log(safe_attention), axis=-1).mean()
            metrics["entropy"].append(entropy)

            # Calculate concentration (Gini coefficient)
            flat_attention = layer_attention.flatten()
            sorted_attention = np.sort(flat_attention)
            n = len(sorted_attention)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_attention)) / (n * np.sum(sorted_attention)) - (
                n + 1
            ) / n
            metrics["concentration"].append(gini)

            # Calculate diversity (1 - max attention)
            diversity = 1 - layer_attention.max(axis=-1).mean()
            metrics["diversity"].append(diversity)

            # Mean attention
            metrics["mean_attention"].append(layer_attention.mean())

        return {k: np.array(v) for k, v in metrics.items()}

    def _create_static_plot(
        self,
        attention_maps: List[np.ndarray],
        layer_metrics: Dict[str, np.ndarray],
        metric: str,
        selected_tokens: Optional[List[int]],
    ) -> plt.Figure:
        """Create static layer evolution plot"""
        n_layers = len(attention_maps)

        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        axes = axes.ravel()

        # Plot 1: Metric evolution
        ax1 = axes[0]
        ax1.plot(range(n_layers), layer_metrics[metric], "b-o", linewidth=2, markersize=8)
        ax1.set_xlabel("Layer", fontsize=12)
        ax1.set_ylabel(metric.capitalize(), fontsize=12)
        ax1.set_title(f"{metric.capitalize()} Evolution Across Layers", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: All metrics comparison
        ax2 = axes[1]
        for metric_name, values in layer_metrics.items():
            # Normalize to [0, 1] for comparison
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
            ax2.plot(range(n_layers), normalized, "-o", label=metric_name, alpha=0.7)
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("Normalized Value", fontsize=12)
        ax2.set_title("All Metrics Comparison", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Attention pattern snapshots
        ax3 = axes[2]
        # Show attention patterns at different layers
        layer_indices = [0, n_layers // 2, n_layers - 1]
        patterns = []
        for idx in layer_indices:
            pattern = attention_maps[idx]
            if pattern.ndim > 2:
                pattern = pattern.mean(axis=0)
            patterns.append(pattern[:10, :10])  # Show first 10x10 for clarity

        combined = np.hstack(patterns)
        im = ax3.imshow(combined, cmap="Blues", aspect="auto")
        ax3.set_title("Attention Patterns: First, Middle, Last Layer", fontsize=14)
        ax3.set_xticks([5, 15, 25])
        ax3.set_xticklabels(["Layer 0", f"Layer {n_layers//2}", f"Layer {n_layers-1}"])
        plt.colorbar(im, ax=ax3)

        # Plot 4: Token-specific tracking
        ax4 = axes[3]
        if selected_tokens is None:
            selected_tokens = [
                0,
                5,
                min(9, attention_maps[0].shape[-1] - 1) if attention_maps else 0,
            ]  # Default selection

        for token_idx in selected_tokens[:5]:  # Limit to 5 tokens
            token_attention = []
            for layer_attention in attention_maps:
                if layer_attention.ndim > 2:
                    layer_attention = layer_attention.mean(axis=0)
                # Get average attention this token pays to others
                if token_idx < layer_attention.shape[0]:
                    token_attention.append(layer_attention[token_idx].mean())

            if token_attention:
                ax4.plot(
                    range(len(token_attention)),
                    token_attention,
                    "-o",
                    label=f"Token {token_idx}",
                    alpha=0.7,
                )

        ax4.set_xlabel("Layer", fontsize=12)
        ax4.set_ylabel("Average Attention", fontsize=12)
        ax4.set_title("Token-Specific Attention Evolution", fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle("Attention Evolution Analysis", fontsize=16)
        plt.tight_layout()

        return fig

    def _create_animation(
        self, attention_maps: List[np.ndarray], layer_metrics: Dict[str, np.ndarray], metric: str
    ) -> plt.Figure:
        """Create animated visualization of attention evolution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Initialize plots
        (line,) = ax1.plot([], [], "b-o")
        ax1.set_xlim(0, len(attention_maps) - 1)
        ax1.set_ylim(layer_metrics[metric].min() * 0.9, layer_metrics[metric].max() * 1.1)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel(metric.capitalize())
        ax1.set_title(f"{metric.capitalize()} Evolution")
        ax1.grid(True, alpha=0.3)

        # Heatmap
        attention = attention_maps[0]
        if attention.ndim > 2:
            attention = attention.mean(axis=0)
        im = ax2.imshow(attention, cmap="Blues", animated=True)
        ax2.set_title("Attention Pattern - Layer 0")
        cbar = plt.colorbar(im, ax=ax2)

        def animate(frame):
            # Update line plot
            line.set_data(range(frame + 1), layer_metrics[metric][: frame + 1])

            # Update heatmap
            attention = attention_maps[frame]
            if attention.ndim > 2:
                attention = attention.mean(axis=0)
            im.set_array(attention)
            im.set_clim(vmin=attention.min(), vmax=attention.max())
            ax2.set_title(f"Attention Pattern - Layer {frame}")

            return line, im

        anim = FuncAnimation(
            fig, animate, frames=len(attention_maps), interval=500, blit=True, repeat=True
        )

        plt.suptitle("Animated Attention Evolution", fontsize=16)
        plt.tight_layout()

        # Note: To save animation, use: anim.save('evolution.gif', writer='pillow')

        return fig
