"""
Base attention extraction functionality
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

# Import from parent package
try:
    from ..utils import tensor_to_numpy
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import tensor_to_numpy


class AttentionExtractor:
    """Extracts attention maps from multi-modal transformer models"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.hooks = []
        self.attention_maps = defaultdict(list)
        self._identify_model_type()
        self._register_hooks()
    
    def _identify_model_type(self):
        """Identify the model type to use appropriate extraction strategy"""
        model_class_name = self.model.__class__.__name__
        
        if "CLIP" in model_class_name:
            from .model_adapters import CLIPAdapter
            self.adapter = CLIPAdapter()
        elif "BLIP" in model_class_name:
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
                    # Most transformers return (hidden_states, attention_weights)
                    attention_weights = output[1]
                    if attention_weights is not None:
                        self.attention_maps[name].append(attention_weights.detach().cpu())
                elif hasattr(output, 'attentions'):
                    # Some models have attention in output object
                    if output.attentions is not None:
                        self.attention_maps[name].extend([attn.detach().cpu() for attn in output.attentions])
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
        head_indices: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Extract attention maps for given inputs
        
        Args:
            inputs: Preprocessed inputs from the model processor
            layer_indices: Which layers to extract (default: all)
            head_indices: Which heads to extract (default: all)
            
        Returns:
            Dictionary containing attention maps and metadata
        """
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Forward pass with hooks
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Process collected attention maps
        processed_attention = self._process_attention_maps(
            layer_indices,
            head_indices
        )
        
        # Get token information
        token_info = self._get_token_info(inputs)
        
        return {
            "attention_maps": processed_attention,
            "token_info": token_info,
            "model_outputs": outputs,
            "layer_names": list(self.attention_maps.keys()),
            "num_layers": len(self.attention_maps),
            "num_heads": processed_attention[0].shape[0] if processed_attention else 0
        }
    
    def _process_attention_maps(
        self,
        layer_indices: Optional[Union[int, List[int]]],
        head_indices: Optional[Union[int, List[int]]]
    ) -> List[np.ndarray]:
        """Process raw attention maps based on indices"""
        all_layers = list(self.attention_maps.values())
        
        # Handle layer selection
        if layer_indices is not None:
            if isinstance(layer_indices, int):
                layer_indices = [layer_indices]
            # Support negative indexing
            layer_indices = [idx if idx >= 0 else len(all_layers) + idx 
                           for idx in layer_indices]
            selected_layers = [all_layers[idx] for idx in layer_indices 
                             if 0 <= idx < len(all_layers)]
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
            # Common attention module names
            if any(attn_name in name.lower() for attn_name in ["attention", "attn", "mha", "multihead"]):
                if hasattr(module, "forward") and not any(skip in name for skip in ["norm", "ln", "dropout"]):
                    attention_modules.append((name, module))
        
        return attention_modules
    
    def get_modality_boundaries(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get the boundaries between different modalities in the input"""
        # Default implementation - override in specific adapters
        return {
            "text_start": 0,
            "text_end": inputs["input_ids"].shape[1],
            "image_start": 0,
            "image_end": 0
        }