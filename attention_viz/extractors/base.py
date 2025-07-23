"""
Base attention extraction functionality - FIXED FOR BLIP
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

        # FIXED: Better BLIP detection
        if any(x in model_class_name for x in ["BLIP", "Blip", "blip"]):
            from .model_adapters import BLIPAdapter
            self.adapter = BLIPAdapter()
        elif "CLIP" in model_class_name:
            from .model_adapters import CLIPAdapter
            self.adapter = CLIPAdapter()
        elif "Flamingo" in model_class_name:
            from .model_adapters import BLIPAdapter  # Placeholder
            self.adapter = BLIPAdapter()
        else:
            # Check module name as fallback
            module_name = self.model.__class__.__module__
            if "blip" in module_name.lower():
                from .model_adapters import BLIPAdapter
                self.adapter = BLIPAdapter()
            else:
                # Default adapter
                self.adapter = BaseModelAdapter()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""

        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights = output[1]
                    
                    if attention_weights is not None:
                        if isinstance(attention_weights, torch.Tensor):
                            # Check if this needs softmax (for BLIP cross-attention)
                            # If values have negatives, it's pre-softmax
                            if (attention_weights < 0).any():
                                # Apply softmax to convert logits to attention weights
                                attention_weights = torch.softmax(attention_weights, dim=-1)
                            
                            self.attention_maps[name].append(attention_weights.detach().cpu())
                            
                        elif isinstance(attention_weights, tuple):
                            # BLIP case - tuple of (self_attention, cross_attention)
                            for idx, attn_component in enumerate(attention_weights):
                                if attn_component is not None and isinstance(attn_component, torch.Tensor):
                                    # Check if needs softmax
                                    if (attn_component < 0).any():
                                        attn_component = torch.softmax(attn_component, dim=-1)
                                    
                                    if idx == 0:
                                        key = f"{name}_self"
                                    else:
                                        key = f"{name}_cross"
                                    self.attention_maps[key].append(attn_component.detach().cpu())
                
                elif hasattr(output, "attentions"):
                    if output.attentions is not None:
                        for attn in output.attentions:
                            if attn is not None:
                                # Check if needs softmax
                                if (attn < 0).any():
                                    attn = torch.softmax(attn, dim=-1)
                                self.attention_maps[name].append(attn.detach().cpu())

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
            outputs = self.model(**inputs, output_attentions=True)

        # If hooks didn't capture attention but model outputs have them
        if len(self.attention_maps) == 0 and hasattr(outputs, "attentions") and outputs.attentions:
            # Use attention from model outputs directly
            for i, attn in enumerate(outputs.attentions):
                self.attention_maps[f"layer_{i}"] = [attn.detach().cpu()]

        # For BLIP, also check for cross_attentions in outputs
        if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
            for i, cross_attn in enumerate(outputs.cross_attentions):
                if cross_attn is not None:
                    self.attention_maps[f"cross_layer_{i}"] = [cross_attn.detach().cpu()]

        # Filter by attention type if specified
        if attention_type:
            filtered_maps = {}
            
            if attention_type == "cross":
                # Only keep cross-attention maps
                filtered_maps = {k: v for k, v in self.attention_maps.items() if 'cross' in k}
            elif attention_type == "text_self":
                # Only keep text self-attention
                filtered_maps = {k: v for k, v in self.attention_maps.items() 
                               if ('text' in k or 'decoder' in k) and 'cross' not in k}
            elif attention_type == "vision_self":
                # Only keep vision self-attention
                filtered_maps = {k: v for k, v in self.attention_maps.items() 
                               if 'vision' in k and 'cross' not in k}
            
            if filtered_maps:
                self.attention_maps = filtered_maps

        # Process collected attention maps
        processed_attention = self._process_attention_maps(layer_indices, head_indices)

        # Get token information
        token_info = self._get_token_info(inputs)

        return {
            "attention_maps": processed_attention,
            "token_info": token_info,
            "model_outputs": outputs,
            "layer_names": list(self.attention_maps.keys()),
            "num_layers": len(self.attention_maps),
            "num_heads": processed_attention[0].shape[0] if processed_attention else 0,
            "attention_type": attention_type,
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

            # Average over batch dimension if needed
            if attention_array.shape[0] > 1:
                attention_array = attention_array.mean(axis=0)
            else:
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
                for attn_name in ["attention", "attn", "mha", "multihead", "self_attn", "crossattention"]
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