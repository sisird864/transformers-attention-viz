"""
Fix the KeyError and check if attention is actually meaningful
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from attention_viz import AttentionVisualizer

def simple_semantic_test():
    """Simple test to check if attention is semantically meaningful"""
    
    print("SEMANTIC ATTENTION TEST")
    print("="*60)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    visualizer = AttentionVisualizer(model, processor)
    
    # Create very simple test case
    img = Image.new('RGB', (224, 224), 'white')
    pixels = img.load()
    
    # Just a red square in the top-left
    for i in range(56):
        for j in range(56):
            pixels[i, j] = (255, 0, 0)
    
    # Test with "red"
    text = "red"
    
    try:
        inputs = processor(images=img, text=text, return_tensors="pt")
        attention_data = visualizer.extractor.extract(inputs, attention_type="cross")
        
        if attention_data["attention_maps"]:
            attn = attention_data["attention_maps"][-1]  # Last layer
            if attn.ndim > 2:
                attn = attn.mean(axis=0)
            
            # Get attention for "red" token (should be index 1, after CLS)
            red_token_attn = attn[1].reshape(8, 8)
            
            # Calculate attention distribution
            top_left_quadrant = red_token_attn[:4, :4].mean()
            top_right_quadrant = red_token_attn[:4, 4:].mean()
            bottom_left_quadrant = red_token_attn[4:, :4].mean()
            bottom_right_quadrant = red_token_attn[4:, 4:].mean()
            
            print(f"\nAttention distribution for 'red' token:")
            print(f"Top-left (where red square is): {top_left_quadrant:.4f}")
            print(f"Top-right: {top_right_quadrant:.4f}")
            print(f"Bottom-left: {bottom_left_quadrant:.4f}")
            print(f"Bottom-right: {bottom_right_quadrant:.4f}")
            
            # Visualize
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            ax1.imshow(img)
            ax1.set_title("Input: Red square top-left")
            ax1.axis('off')
            
            im = ax2.imshow(red_token_attn, cmap='hot', vmin=0, vmax=red_token_attn.max())
            ax2.set_title("'red' token attention\n(8×8 grid)")
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046)
            
            # Show with grid lines
            ax3.imshow(red_token_attn, cmap='hot', vmin=0, vmax=red_token_attn.max())
            ax3.set_title("With grid lines")
            for i in range(8):
                ax3.axhline(i+0.5, color='blue', linewidth=0.5)
                ax3.axvline(i+0.5, color='blue', linewidth=0.5)
            ax3.set_xticks(range(8))
            ax3.set_yticks(range(8))
            
            plt.tight_layout()
            plt.show()
            
            # Check if attention is meaningful
            ratio = top_left_quadrant / (bottom_right_quadrant + 1e-8)
            print(f"\nRatio (top-left / bottom-right): {ratio:.2f}")
            
            if ratio > 1.5:
                print("✅ GOOD - Attention focuses on red region!")
            elif ratio > 1.1:
                print("⚠️  OKAY - Slight preference for red region")
            else:
                print("❌ CONCERNING - Attention is nearly uniform!")
                print("\nPossible issues:")
                print("1. The model might need more context than single words")
                print("2. Early layers might be more uniform (try different layers)")
                print("3. The softmax temperature might be very high (uniform distribution)")
                
    except KeyError as e:
        print(f"\n❌ KeyError: {e}")
        print("\nTo fix this, ensure your base.py has:")
        print("from collections import defaultdict")
        print("self.attention_maps = defaultdict(list)")


def check_which_layer_works_best():
    """Check different layers to find most semantic attention"""
    
    print("\n\nCHECKING DIFFERENT LAYERS")
    print("="*60)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Simple test image
    img = Image.new('RGB', (224, 224), 'white')
    pixels = img.load()
    for i in range(112):
        for j in range(112):
            pixels[i, j] = (255, 0, 0)  # Red top-left
    
    text = "red"
    inputs = processor(images=img, text=text, return_tensors="pt")
    
    # Get all layers manually
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
        image_embeds = vision_outputs[0]
        
        outputs = model.text_decoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long),
            output_attentions=True,
            return_dict=True
        )
    
    if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
        print(f"Found {len(outputs.cross_attentions)} cross-attention layers")
        
        # Check each layer
        for i, cross_attn in enumerate(outputs.cross_attentions):
            # Get attention for "red" token
            attn = cross_attn[0].mean(0)  # Average over heads
            if hasattr(attn, 'detach'):
                attn = attn.detach().cpu().numpy()
            
            # Apply softmax if needed
            if (attn < 0).any():
                attn = torch.softmax(torch.from_numpy(attn), dim=-1).numpy()
            
            red_attn = attn[1].reshape(8, 8)  # "red" token
            
            # Calculate focus
            top_half = red_attn[:4, :].mean()
            bottom_half = red_attn[4:, :].mean()
            ratio = top_half / (bottom_half + 1e-8)
            
            print(f"\nLayer {i}: Top/Bottom ratio = {ratio:.2f}")
            
        print("\nTIP: Later layers often show more semantic attention")


def test_more_complex_prompt():
    """Test with more descriptive prompts"""
    
    print("\n\nTESTING WITH MORE CONTEXT")
    print("="*60)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    visualizer = AttentionVisualizer(model, processor)
    
    # Same image
    img = Image.new('RGB', (224, 224), 'white')
    pixels = img.load()
    for i in range(80):
        for j in range(80):
            pixels[i, j] = (255, 0, 0)
    
    # Test different prompts
    prompts = [
        "red",
        "red square",
        "a red square in the corner",
        "there is a red square"
    ]
    
    for prompt in prompts:
        try:
            inputs = processor(images=img, text=prompt, return_tensors="pt")
            attention_data = visualizer.extractor.extract(inputs, attention_type="cross")
            
            if attention_data["attention_maps"]:
                attn = attention_data["attention_maps"][-1]
                if attn.ndim > 2:
                    attn = attn.mean(axis=0)
                
                # Average attention across all tokens
                avg_attn = attn.mean(axis=0).reshape(8, 8)
                
                top_left = avg_attn[:4, :4].mean()
                rest = avg_attn[4:, 4:].mean()
                ratio = top_left / (rest + 1e-8)
                
                print(f"\nPrompt: '{prompt}'")
                print(f"Focus ratio: {ratio:.2f}")
                
        except Exception as e:
            print(f"\nPrompt: '{prompt}' - Error: {e}")


if __name__ == "__main__":
    # First, try to fix the KeyError by checking the setup
    print("First, let's check your setup...")
    
    # Run tests
    simple_semantic_test()
    check_which_layer_works_best()
    test_more_complex_prompt()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("\nIf ratios are all close to 1.0, the attention might be:")
    print("1. Too uniform (not semantically meaningful)")
    print("2. Looking at wrong layer (try last layer)")
    print("3. Needing more context than single words")
    print("\nBLIP often needs fuller sentences to show meaningful attention.")