"""
Attention flow visualization showing connections between tokens
"""

from typing import Any, Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

from .heatmap import VisualizationResult


class AttentionFlow:
    """Creates flow diagrams showing attention connections"""

    def __init__(self):
        self.figure_size = (14, 10)
        self.min_attention_threshold = 0.1

    def create(
        self,
        attention_data: Dict[str, Any],
        inputs: Dict[str, Any],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        bidirectional: bool = False,
        **kwargs,
    ) -> "VisualizationResult":
        """
        Create attention flow visualization

        Args:
            attention_data: Extracted attention data
            inputs: Original inputs
            threshold: Minimum attention weight to show
            top_k: Show only top-k connections per token
            bidirectional: Show bidirectional connections
        """
        from .heatmap import VisualizationResult

        attention_maps = attention_data["attention_maps"]
        if not attention_maps:
            raise ValueError("No attention maps found")

        # Use last layer, average across heads
        attention_matrix = attention_maps[-1]
        if attention_matrix.ndim > 2:
            attention_matrix = attention_matrix.mean(axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Get token positions
        boundaries = attention_data["token_info"]["modality_boundaries"]
        positions = self._calculate_token_positions(boundaries)

        # Draw tokens
        self._draw_tokens(ax, positions, boundaries)

        # Draw attention flows
        self._draw_attention_flows(
            ax,
            attention_matrix,
            positions,
            threshold or self.min_attention_threshold,
            top_k,
            bidirectional,
        )

        # Style the plot
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, max(len(positions["text"]), len(positions["image"])) + 1)
        ax.axis("off")
        ax.set_title("Attention Flow Diagram", fontsize=16, pad=20)

        # Add legend
        self._add_legend(ax)

        plt.tight_layout()
        return VisualizationResult(fig)

    def _calculate_token_positions(self, boundaries: Dict[str, int]) -> Dict[str, List[tuple]]:
        """Calculate x,y positions for tokens"""
        text_count = boundaries["text_end"]
        image_count = boundaries["image_end"] - boundaries["text_end"]

        # Text tokens on the left
        text_positions = [(2, i) for i in range(text_count)]

        # Image patches on the right
        image_positions = [(8, i) for i in range(image_count)]

        return {
            "text": text_positions,
            "image": image_positions,
            "all": text_positions + image_positions,
        }

    def _draw_tokens(self, ax, positions: Dict[str, List[tuple]], boundaries: Dict[str, int]):
        """Draw token representations"""
        # Draw text tokens
        for i, (x, y) in enumerate(positions["text"]):
            circle = plt.Circle((x, y), 0.3, color="skyblue", ec="black", lw=2)
            ax.add_patch(circle)
            ax.text(x, y, f"T{i}", ha="center", va="center", fontsize=8)

        # Draw image patches
        for i, (x, y) in enumerate(positions["image"]):
            rect = plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, color="lightcoral", ec="black", lw=2)
            ax.add_patch(rect)
            ax.text(x, y, f"I{i}", ha="center", va="center", fontsize=8)

        # Add labels
        ax.text(2, -0.5, "Text Tokens", ha="center", fontsize=12, weight="bold")
        ax.text(8, -0.5, "Image Patches", ha="center", fontsize=12, weight="bold")

    def _draw_attention_flows(
        self,
        ax,
        attention_matrix: np.ndarray,
        positions: Dict[str, List[tuple]],
        threshold: float,
        top_k: Optional[int],
        bidirectional: bool,
    ):
        """Draw attention flow lines"""
        all_positions = positions["all"]
        n_tokens = len(all_positions)

        # Create line segments
        lines = []
        colors = []
        linewidths = []

        for i in range(n_tokens):
            for j in range(n_tokens):
                if i == j:  # Skip self-attention for clarity
                    continue

                attention_weight = attention_matrix[i, j]

                # Apply threshold
                if attention_weight < threshold:
                    continue

                # Apply top-k if specified
                if top_k is not None:
                    row_top_k = np.argsort(attention_matrix[i])[-top_k:]
                    if j not in row_top_k:
                        continue

                # Create line
                start = all_positions[i]
                end = all_positions[j]

                # Add slight curve to avoid overlapping lines
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2 + 0.5 * np.sign(end[0] - start[0])

                # Use quadratic bezier curve
                t = np.linspace(0, 1, 20)
                x_curve = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_x + t**2 * end[0]
                y_curve = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_y + t**2 * end[1]

                for k in range(len(t) - 1):
                    lines.append([(x_curve[k], y_curve[k]), (x_curve[k + 1], y_curve[k + 1])])
                    colors.append(attention_weight)
                    linewidths.append(1 + 3 * attention_weight)

        # Create line collection
        if lines:
            lc = LineCollection(lines, cmap="viridis", alpha=0.6)
            lc.set_array(np.array(colors))
            lc.set_linewidths(linewidths)
            ax.add_collection(lc)

            # Add colorbar
            cbar = plt.colorbar(lc, ax=ax, pad=0.02)
            cbar.set_label("Attention Weight", fontsize=10)

    def _add_legend(self, ax):
        """Add legend explaining the visualization"""
        legend_elements = [
            plt.Circle((0, 0), 0.3, color="skyblue", ec="black", label="Text Token"),
            plt.Rectangle((0, 0), 0.6, 0.6, color="lightcoral", ec="black", label="Image Patch"),
            plt.Line2D([0], [0], color="green", lw=3, alpha=0.6, label="High Attention"),
            plt.Line2D([0], [0], color="blue", lw=1, alpha=0.6, label="Low Attention"),
        ]

        ax.legend(
            handles=legend_elements, loc="upper right", frameon=True, fancybox=True, shadow=True
        )
