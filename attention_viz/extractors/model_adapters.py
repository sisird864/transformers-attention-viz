"""
Model-specific adapters for different transformer architectures
"""

from typing import Dict, List, Tuple

import torch

from .base import BaseModelAdapter


class CLIPAdapter(BaseModelAdapter):
    """Adapter for CLIP models"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
        """Get CLIP-specific attention modules"""
        attention_modules = []

        # Text encoder attention layers
        if hasattr(model, "text_model"):
            for i, layer in enumerate(model.text_model.encoder.layers):
                if hasattr(layer, "self_attn"):
                    attention_modules.append((f"text_encoder.layer_{i}", layer.self_attn))

        # Vision encoder attention layers
        if hasattr(model, "vision_model"):
            for i, layer in enumerate(model.vision_model.encoder.layers):
                if hasattr(layer, "self_attn"):
                    attention_modules.append((f"vision_encoder.layer_{i}", layer.self_attn))

        # Cross-attention layers if present
        if hasattr(model, "cross_attention"):
            attention_modules.append(("cross_attention", model.cross_attention))

        return attention_modules

    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get boundaries for CLIP inputs"""
        text_length = inputs["input_ids"].shape[1]

        # CLIP typically processes text and images separately
        # For joint attention visualization, we need to identify patch tokens
        image_length = 0
        if "pixel_values" in inputs:
            # Calculate number of image patches
            # Standard CLIP uses 14x14 patches for 224x224 images
            patch_size = 16  # Default CLIP patch size
            image_size = inputs["pixel_values"].shape[-1]
            patches_per_side = image_size // patch_size
            image_length = patches_per_side * patches_per_side + 1  # +1 for CLS token

        return {
            "text_start": 0,
            "text_end": text_length,
            "image_start": text_length,
            "image_end": text_length + image_length,
            "total_length": text_length + image_length,
        }


class BLIPAdapter(BaseModelAdapter):
    """Adapter for BLIP models"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
        """Get BLIP-specific attention modules"""
        attention_modules = []

        # BLIP has different architecture than CLIP
        # Text encoder
        if hasattr(model, "text_encoder"):
            for i, layer in enumerate(model.text_encoder.encoder.layer):
                if hasattr(layer, "attention"):
                    attention_modules.append((f"text_encoder.layer_{i}", layer.attention))

        # Vision encoder
        if hasattr(model, "vision_model"):
            for i, layer in enumerate(model.vision_model.encoder.layers):
                if hasattr(layer, "self_attn"):
                    attention_modules.append((f"vision_encoder.layer_{i}", layer.self_attn))

        # Cross-modal encoder (if present in BLIP variant)
        if hasattr(model, "text_decoder"):
            for i, layer in enumerate(model.text_decoder.bert.encoder.layer):
                if hasattr(layer, "crossattention"):
                    attention_modules.append((f"cross_attention.layer_{i}", layer.crossattention))

        return attention_modules

    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get boundaries for BLIP inputs"""
        # BLIP processes inputs differently depending on the task
        text_length = 0
        image_length = 0

        if "input_ids" in inputs:
            text_length = inputs["input_ids"].shape[1]

        if "pixel_values" in inputs:
            # BLIP uses similar patch encoding as CLIP
            patch_size = 16
            image_size = inputs["pixel_values"].shape[-1]
            patches_per_side = image_size // patch_size
            image_length = patches_per_side * patches_per_side + 1

        return {
            "text_start": 0,
            "text_end": text_length,
            "image_start": text_length,
            "image_end": text_length + image_length,
            "total_length": text_length + image_length,
        }


class FlamingAdapter(BaseModelAdapter):
    """Adapter for Flamingo models (placeholder for future implementation)"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
        """Get Flamingo-specific attention modules"""
        # Flamingo has a more complex architecture with perceiver resampler
        # This is a placeholder for future implementation
        raise NotImplementedError("Flamingo support coming soon!")

    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get boundaries for Flamingo inputs"""
        raise NotImplementedError("Flamingo support coming soon!")
