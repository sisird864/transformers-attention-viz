"""
Transformers Attention Viz - Interactive attention visualization for multi-modal transformers
"""

from .core import AttentionVisualizer
from .dashboard.app import launch_dashboard
from .extractors import AttentionExtractor
from .utils import ensure_numpy_available, tensor_to_numpy
from .version import __author__, __description__, __email__, __version__
from .visualizers import AttentionFlow, AttentionHeatmap, HeadComparison, LayerEvolution

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
