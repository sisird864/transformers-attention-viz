# attention_viz/extractors/model_adapters.py - FIXED VERSION
"""
Model-specific adapters for different transformer architectures
"""

from typing import Dict, List, Tuple

import torch

from .base import BaseModelAdapter


class CLIPAdapter(BaseModelAdapter):
    """Adapter for CLIP models - Fixed to properly extract attention"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
        """Get CLIP-specific attention modules"""
        attention_modules = []

        # Text encoder attention layers
        if hasattr(model, "text_model") and hasattr(model.text_model, "encoder"):
            # Handle different CLIP model structures
            if hasattr(model.text_model.encoder, "layers"):
                for i, layer in enumerate(model.text_model.encoder.layers):
                    if hasattr(layer, "self_attn"):
                        attention_modules.append((f"text_encoder.layer_{i}", layer.self_attn))
            elif hasattr(model.text_model, "transformer"):
                # Some CLIP models have transformer attribute
                for i, layer in enumerate(model.text_model.transformer.resblocks):
                    if hasattr(layer, "attn"):
                        attention_modules.append((f"text_encoder.layer_{i}", layer.attn))

        # Vision encoder attention layers
        if hasattr(model, "vision_model"):
            if hasattr(model.vision_model, "encoder") and hasattr(model.vision_model.encoder, "layers"):
                for i, layer in enumerate(model.vision_model.encoder.layers):
                    if hasattr(layer, "self_attn"):
                        attention_modules.append((f"vision_encoder.layer_{i}", layer.self_attn))
            elif hasattr(model.vision_model, "transformer"):
                # Some CLIP models have transformer attribute
                for i, layer in enumerate(model.vision_model.transformer.resblocks):
                    if hasattr(layer, "attn"):
                        attention_modules.append((f"vision_encoder.layer_{i}", layer.attn))

        # If no attention modules found, try generic search
        if not attention_modules:
            print("Warning: No CLIP-specific attention modules found. Using generic search.")
            return super().get_attention_modules(model)

        print(f"Found {len(attention_modules)} attention modules in CLIP model")
        return attention_modules

    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get boundaries for CLIP inputs"""
        text_length = 0
        image_length = 0
        
        if "input_ids" in inputs:
            text_length = inputs["input_ids"].shape[1]

        if "pixel_values" in inputs:
            # Calculate number of image patches
            # Standard CLIP uses different patch sizes
            image_size = inputs["pixel_values"].shape[-1]
            
            # Common CLIP configurations
            if image_size == 224:
                patch_size = 32  # For clip-vit-base-patch32
                patches_per_side = image_size // patch_size  # 7
            elif image_size == 336:
                patch_size = 14  # For larger models
                patches_per_side = image_size // patch_size
            else:
                # Fallback
                patch_size = 16
                patches_per_side = image_size // patch_size
                
            image_length = patches_per_side * patches_per_side + 1  # +1 for CLS token

        return {
            "text_start": 0,
            "text_end": text_length,
            "image_start": 0,  # Note: CLIP processes text and image separately
            "image_end": image_length,
            "total_length": max(text_length, image_length),  # They're processed separately
        }


class BLIPAdapter(BaseModelAdapter):
    """Adapter for BLIP models"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
        """Get BLIP-specific attention modules"""
        attention_modules = []

        # BLIP has different architecture than CLIP
        # Text encoder
        if hasattr(model, "text_encoder"):
            if hasattr(model.text_encoder, "encoder"):
                for i, layer in enumerate(model.text_encoder.encoder.layer):
                    if hasattr(layer, "attention"):
                        attention_modules.append((f"text_encoder.layer_{i}", layer.attention))

        # Vision encoder
        if hasattr(model, "vision_model"):
            if hasattr(model.vision_model, "encoder") and hasattr(model.vision_model.encoder, "layers"):
                for i, layer in enumerate(model.vision_model.encoder.layers):
                    if hasattr(layer, "self_attn"):
                        attention_modules.append((f"vision_encoder.layer_{i}", layer.self_attn))

        # Cross-modal encoder (if present in BLIP variant)
        if hasattr(model, "text_decoder"):
            if hasattr(model.text_decoder, "bert") and hasattr(model.text_decoder.bert, "encoder"):
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