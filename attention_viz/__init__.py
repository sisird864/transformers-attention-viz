"""
Transformers Attention Viz - Interactive attention visualization for multi-modal transformers
"""

from .version import __version__, __author__, __email__, __description__

from .core import AttentionVisualizer
from .dashboard.app import launch_dashboard
from .extractors import AttentionExtractor
from .visualizers import (
    AttentionHeatmap,
    AttentionFlow,
    LayerEvolution,
    HeadComparison
)
from .utils import ensure_numpy_available, tensor_to_numpy

__all__ = [
    "AttentionVisualizer",
    "launch_dashboard",
    "AttentionExtractor",
    "AttentionHeatmap",
    "AttentionFlow",
    "LayerEvolution",
    "HeadComparison",
    "ensure_numpy_available",
    "tensor_to_numpy",
]