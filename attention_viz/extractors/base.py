"""
Base attention extraction functionality - FIXED FOR BOTH BLIP AND CLIP
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..utils import tensor_to_numpy


class AttentionExtractor:
    """Extracts attention maps from multi-modal transformer models"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.attention_maps: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._identify_model_type()
        self._register_hooks()

    def _identify_model_type(self):
        """Identify the model type to use appropriate extraction strategy"""
        model_class_name = self.model.__class__.__name__

        # Better BLIP detection
        if any(x in model_class_name for x in ["BLIP", "Blip", "blip"]):
            from .model_adapters import BLIPAdapter

            self.adapter = BLIPAdapter()
            self.model_type = "BLIP"
        elif "CLIP" in model_class_name:
            from .model_adapters import CLIPAdapter

            self.adapter = CLIPAdapter()
            self.model_type = "CLIP"
        elif "Flamingo" in model_class_name:
            from .model_adapters import BLIPAdapter  # Placeholder

            self.adapter = BLIPAdapter()
            self.model_type = "Flamingo"
        else:
            # Check module name as fallback
            module_name = self.model.__class__.__module__
            if "blip" in module_name.lower():
                from .model_adapters import BLIPAdapter

                self.adapter = BLIPAdapter()
                self.model_type = "BLIP"
            elif "clip" in module_name.lower():
                from .model_adapters import CLIPAdapter

                self.adapter = CLIPAdapter()
                self.model_type = "CLIP"
            else:
                # Default adapter
                self.adapter = BaseModelAdapter()
                self.model_type = "Unknown"

    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""

        def create_hook(name):
            def hook(module, input, output):
                # BLIP cross-attention handling
                if "cross" in name.lower() or "encoder_attn" in name.lower():
                    if isinstance(output, tuple):
                        attention_weights = None

                        # Check each element in the tuple for cross-attention
                        for i, elem in enumerate(output):
                            if isinstance(elem, torch.Tensor) and elem.dim() == 4:
                                # For BLIP cross-attention: check if last dim is ~577 (image tokens)
                                if elem.shape[-1] > 500 and elem.shape[-1] < 600:
                                    attention_weights = elem
                                    break

                        if attention_weights is not None:
                            # Apply softmax if needed
                            if (attention_weights < 0).any():
                                attention_weights = torch.softmax(attention_weights, dim=-1)

                            if name not in self.attention_maps:
                                self.attention_maps[name] = []
                            self.attention_maps[name].append(attention_weights.detach().cpu())

                # Self-attention handling (for both CLIP and BLIP)
                elif "self" in name.lower() or (
                    "attention" in name.lower() and "cross" not in name.lower()
                ):
                    if isinstance(output, tuple) and len(output) > 1:
                        # Standard transformer output format
                        attention_weights = output[1]

                        if attention_weights is not None and isinstance(
                            attention_weights, torch.Tensor
                        ):
                            # Verify it's attention (4D tensor)
                            if attention_weights.dim() == 4:
                                # Apply softmax if needed
                                if (attention_weights < 0).any():
                                    attention_weights = torch.softmax(attention_weights, dim=-1)

                                if name not in self.attention_maps:
                                    self.attention_maps[name] = []
                                self.attention_maps[name].append(attention_weights.detach().cpu())

                    # Some models return attention directly as tensor
                    elif isinstance(output, torch.Tensor) and output.dim() == 4:
                        attention_weights = output
                        if (attention_weights < 0).any():
                            attention_weights = torch.softmax(attention_weights, dim=-1)

                        if name not in self.attention_maps:
                            self.attention_maps[name] = []
                        self.attention_maps[name].append(attention_weights.detach().cpu())

            return hook

        # Register hooks for each attention layer
        attention_modules = self.adapter.get_attention_modules(self.model)

        for name, module in attention_modules:
            hook = module.register_forward_hook(create_hook(name))
            self.hooks.append(hook)

    def extract(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_indices: Optional[Union[int, List[int]]] = None,
        head_indices: Optional[Union[int, List[int]]] = None,
        attention_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract attention maps for given inputs

        Args:
            inputs: Preprocessed inputs from the model processor
            layer_indices: Which layers to extract (default: all)
            head_indices: Which heads to extract (default: all)
            attention_type: Which attention to extract ('text_self', 'vision_self', 'cross')
        """
        # Clear previous attention maps
        self.attention_maps.clear()

        # Forward pass with hooks
        with torch.no_grad():
            # BLIP-specific handling for cross-attention
            if (
                self.model_type == "BLIP"
                and hasattr(self.model, "text_decoder")
                and attention_type == "cross"
            ):
                # Run vision model first
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs.get("pixel_values"),
                    output_attentions=True,
                    return_dict=True,
                )

                # Then run text decoder with cross-attention
                outputs = self.model.text_decoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    encoder_hidden_states=vision_outputs.last_hidden_state,
                    encoder_attention_mask=torch.ones(
                        vision_outputs.last_hidden_state.size()[:-1],
                        dtype=torch.long,
                        device=vision_outputs.last_hidden_state.device,
                    ),
                    output_attentions=True,
                    return_dict=True,
                )

                # Manually add cross-attentions if hooks didn't capture them
                if hasattr(outputs, "cross_attentions") and outputs.cross_attentions:
                    for i, cross_attn in enumerate(outputs.cross_attentions):
                        key = f"text_decoder.layer_{i}.cross_attention"
                        if key not in self.attention_maps or not self.attention_maps[key]:
                            self.attention_maps[key] = [cross_attn.detach().cpu()]

            else:
                # Standard forward pass for other cases (CLIP, BLIP self-attention)
                outputs = self.model(**inputs, output_attentions=True)

                # If hooks didn't capture attention, try to extract from outputs
                if len(self.attention_maps) == 0:
                    # CLIP-specific extraction
                    if self.model_type == "CLIP":
                        # Text encoder attentions
                        if hasattr(outputs, "text_model_output") and hasattr(
                            outputs.text_model_output, "attentions"
                        ):
                            if outputs.text_model_output.attentions is not None:
                                for i, attn in enumerate(outputs.text_model_output.attentions):
                                    if attn is not None:
                                        self.attention_maps[f"text_encoder.layer_{i}"] = [
                                            attn.detach().cpu()
                                        ]

                        # Vision encoder attentions
                        if hasattr(outputs, "vision_model_output") and hasattr(
                            outputs.vision_model_output, "attentions"
                        ):
                            if outputs.vision_model_output.attentions is not None:
                                for i, attn in enumerate(outputs.vision_model_output.attentions):
                                    if attn is not None:
                                        self.attention_maps[f"vision_encoder.layer_{i}"] = [
                                            attn.detach().cpu()
                                        ]

                    # Generic fallback for other models
                    elif hasattr(outputs, "attentions") and outputs.attentions:
                        for i, attn in enumerate(outputs.attentions):
                            if attn is not None:
                                self.attention_maps[f"layer_{i}"] = [attn.detach().cpu()]

        # Filter by attention type if specified
        if attention_type:
            filtered_maps = {}

            if attention_type == "cross":
                # Only keep cross-attention maps
                for k, v in self.attention_maps.items():
                    if "cross" in k.lower():
                        filtered_maps[k] = v
            elif attention_type == "text_self":
                # Only keep text self-attention
                for k, v in self.attention_maps.items():
                    if ("text" in k.lower() or "decoder" in k.lower()) and "cross" not in k.lower():
                        filtered_maps[k] = v
            elif attention_type == "vision_self":
                # Only keep vision self-attention
                for k, v in self.attention_maps.items():
                    if "vision" in k.lower() and "cross" not in k.lower():
                        filtered_maps[k] = v

            self.attention_maps = filtered_maps

        # Process collected attention maps
        processed_attention = self._process_attention_maps(layer_indices, head_indices)

        # Get token information
        token_info = self._get_token_info(inputs)

        return {
            "attention_maps": processed_attention,
            "token_info": token_info,
            "model_outputs": outputs if "outputs" in locals() else None,
            "layer_names": list(self.attention_maps.keys()),
            "num_layers": len(self.attention_maps),
            "num_heads": processed_attention[0].shape[0] if processed_attention else 0,
            "attention_type": attention_type,
            "model_type": self.model_type,
        }

    def _process_attention_maps(
        self,
        layer_indices: Optional[Union[int, List[int]]],
        head_indices: Optional[Union[int, List[int]]],
    ) -> List[np.ndarray]:
        """Process raw attention maps based on indices"""
        all_layers = list(self.attention_maps.values())

        # If no attention maps collected, return empty list
        if not all_layers:
            return []

        # Handle layer selection
        if layer_indices is not None:
            if isinstance(layer_indices, int):
                layer_indices = [layer_indices]
            # Support negative indexing
            layer_indices = [idx if idx >= 0 else len(all_layers) + idx for idx in layer_indices]
            selected_layers = [
                all_layers[idx] for idx in layer_indices if 0 <= idx < len(all_layers)
            ]
        else:
            selected_layers = all_layers

        # Process each layer
        processed = []
        for layer_attention in selected_layers:
            if isinstance(layer_attention, list):
                layer_attention = layer_attention[0]  # Take first element if list

            attention_array = tensor_to_numpy(layer_attention)

            # Handle head selection
            if head_indices is not None:
                if isinstance(head_indices, int):
                    head_indices = [head_indices]
                attention_array = attention_array[:, head_indices, :, :]

            # For most uses, we want to remove batch dimension
            if attention_array.shape[0] == 1:
                attention_array = attention_array.squeeze(0)

            processed.append(attention_array)

        return processed

    def _get_token_info(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract token information from inputs"""
        token_info = {}

        # Get input IDs
        if "input_ids" in inputs:
            token_info["input_ids"] = tensor_to_numpy(inputs["input_ids"])
            token_info["num_tokens"] = inputs["input_ids"].shape[1]

        # Get attention mask
        if "attention_mask" in inputs:
            token_info["attention_mask"] = tensor_to_numpy(inputs["attention_mask"])

        # Get token type IDs if available (for models that use them)
        if "token_type_ids" in inputs:
            token_info["token_type_ids"] = tensor_to_numpy(inputs["token_type_ids"])

        # Get image-text boundary
        token_info["modality_boundaries"] = self.adapter.get_modality_boundaries(inputs)

        return token_info

    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()

    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self.cleanup()


class BaseModelAdapter:
    """Base adapter for model-specific functionality"""

    def get_attention_modules(self, model: torch.nn.Module) -> List[tuple]:
        """Get list of (name, module) pairs for attention layers"""
        attention_modules = []

        for name, module in model.named_modules():
            # Skip submodules of attention layers (like out_proj)
            if "." in name and any(
                parent in name for parent in ["attention.", "self_attn.", "attn."]
            ):
                continue

            # Common attention module names
            if any(
                attn_name in name.lower()
                for attn_name in [
                    "attention",
                    "attn",
                    "mha",
                    "multihead",
                    "self_attn",
                    "crossattention",
                ]
            ):
                # Check if it's actually an attention module (not norm, dropout, etc.)
                if hasattr(module, "forward") and not any(
                    skip in name.lower() for skip in ["norm", "ln", "dropout", "proj"]
                ):
                    # Also check module class name
                    module_class = module.__class__.__name__.lower()
                    if any(attn in module_class for attn in ["attention", "attn"]):
                        attention_modules.append((name, module))

        return attention_modules

    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get the boundaries between different modalities in the input"""
        # Default implementation - override in specific adapters
        text_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        return {
            "text_start": 0,
            "text_end": text_length,
            "image_start": 0,
            "image_end": 0,
            "total_length": text_length,
        }
