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
    """Adapter for BLIP models - FIXED for cross-attention"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
        """Get BLIP-specific attention modules including cross-attention"""
        attention_modules = []

        # BLIP has a text decoder with cross-attention layers
        if hasattr(model, "text_decoder"):
            # Get all decoder layers
            if hasattr(model.text_decoder, "bert") and hasattr(model.text_decoder.bert, "encoder"):
                encoder = model.text_decoder.bert.encoder
                
                # Iterate through decoder layers
                for i, layer in enumerate(encoder.layer):
                    # Self-attention
                    if hasattr(layer, "attention"):
                        attention_modules.append((f"text_decoder.layer_{i}.self_attention", layer.attention))
                    
                    # Cross-attention (this is what we want for cross-modal visualization)
                    if hasattr(layer, "crossattention"):
                        attention_modules.append((f"text_decoder.layer_{i}.cross_attention", layer.crossattention))
            
            # Alternative structure for some BLIP variants
            elif hasattr(model.text_decoder, "layers"):
                for i, layer in enumerate(model.text_decoder.layers):
                    if hasattr(layer, "self_attn"):
                        attention_modules.append((f"text_decoder.layer_{i}.self_attention", layer.self_attn))
                    if hasattr(layer, "encoder_attn"):  # Cross-attention
                        attention_modules.append((f"text_decoder.layer_{i}.cross_attention", layer.encoder_attn))

        # Vision encoder self-attention
        if hasattr(model, "vision_model"):
            if hasattr(model.vision_model, "encoder") and hasattr(model.vision_model.encoder, "layers"):
                for i, layer in enumerate(model.vision_model.encoder.layers):
                    if hasattr(layer, "self_attn"):
                        attention_modules.append((f"vision_encoder.layer_{i}.self_attention", layer.self_attn))

        # If no modules found, try generic search
        if not attention_modules:
            print("Warning: No BLIP-specific attention modules found. Using generic search.")
            # Look for cross-attention specifically
            for name, module in model.named_modules():
                if "crossattention" in name.lower() or "cross_attention" in name.lower():
                    attention_modules.append((name, module))
            
            # Then add other attention modules
            attention_modules.extend(super().get_attention_modules(model))

        print(f"Found {len(attention_modules)} attention modules in BLIP model")
        # Debug: print module names
        for name, _ in attention_modules[:5]:  # Show first 5
            print(f"  - {name}")
        if len(attention_modules) > 5:
            print(f"  ... and {len(attention_modules) - 5} more")

        return attention_modules

    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get boundaries for BLIP inputs"""
        text_length = 0
        image_length = 0

        if "input_ids" in inputs:
            text_length = inputs["input_ids"].shape[1]

        if "pixel_values" in inputs:
            # BLIP typically uses 577 patches (24x24 grid + 1 CLS token)
            # for 384x384 images with 16x16 patches
            image_length = 577  # Standard for BLIP-base

        return {
            "text_start": 0,
            "text_end": text_length,
            "image_start": 0,
            "image_end": image_length,
            "total_length": max(text_length, image_length),
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