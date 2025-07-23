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
    img = Image.new('RGB', (384, 384), 'white')  # BLIP uses 384x384
    pixels = img.load()
    
    # Red square in the top-left (about 1/4 of the image)
    for i in range(96):
        for j in range(96):
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
            # Skip first token (CLS) of image patches
            red_token_attn = attn[1, 1:]  # Shape: (576,) - excluding CLS token
            red_token_attn = red_token_attn.reshape(24, 24)  # BLIP uses 24x24 grid
            
            # Calculate attention distribution
            # Top-left quadrant (where red square is) is roughly 6x6 in the 24x24 grid
            top_left_quadrant = red_token_attn[:6, :6].mean()
            top_right_quadrant = red_token_attn[:6, 18:].mean()
            bottom_left_quadrant = red_token_attn[18:, :6].mean()
            bottom_right_quadrant = red_token_attn[18:, 18:].mean()
            
            print(f"\nAttention distribution for 'red' token:")
            print(f"Top-left (where red square is): {top_left_quadrant:.4f}")
            print(f"Top-right: {top_right_quadrant:.4f}")
            print(f"Bottom-left: {bottom_left_quadrant:.4f}")
            print(f"Bottom-right: {bottom_right_quadrant:.4f}")
            
            # Visualize
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Show the input image
            ax1.imshow(img)
            ax1.set_title("Input: Red square top-left")
            ax1.axis('off')
            
            # Show attention heatmap
            im = ax2.imshow(red_token_attn, cmap='hot', vmin=0, vmax=red_token_attn.max())
            ax2.set_title("'red' token attention\n(24×24 grid)")
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046)
            
            # Show with grid lines
            ax3.imshow(red_token_attn, cmap='hot', vmin=0, vmax=red_token_attn.max())
            ax3.set_title("With grid overlay (6x6 regions)")
            # Draw 6x6 region boundaries
            for i in range(0, 25, 6):
                ax3.axhline(i-0.5, color='blue', linewidth=1)
                ax3.axvline(i-0.5, color='blue', linewidth=1)
            # Highlight the red square region
            rect = plt.Rectangle((-0.5, -0.5), 6, 6, fill=False, edgecolor='red', linewidth=2)
            ax3.add_patch(rect)
            ax3.set_xlim(-0.5, 23.5)
            ax3.set_ylim(23.5, -0.5)
            
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
                print("2. Try looking at different layers (earlier layers might be better)")
                print("3. The attention might be more diffuse in BLIP")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def check_which_layer_works_best():
    """Check different layers to find most semantic attention"""
    
    print("\n\nCHECKING DIFFERENT LAYERS")
    print("="*60)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Simple test image
    img = Image.new('RGB', (384, 384), 'white')
    pixels = img.load()
    # Red in top-left quarter
    for i in range(192):
        for j in range(192):
            pixels[i, j] = (255, 0, 0)
    
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
        
        best_ratio = 0
        best_layer = -1
        
        # Check each layer
        for i, cross_attn in enumerate(outputs.cross_attentions):
            # Get attention for "red" token
            attn = cross_attn[0].mean(0)  # Average over heads
            if hasattr(attn, 'detach'):
                attn = attn.detach().cpu().numpy()
            
            # Apply softmax if needed
            if (attn < 0).any():
                attn = torch.softmax(torch.from_numpy(attn), dim=-1).numpy()
            
            # Get attention for "red" token, skip CLS token of image
            red_attn = attn[1, 1:].reshape(24, 24)  # "red" token attention to image patches
            
            # Calculate focus (red is in top-left half)
            top_half = red_attn[:12, :].mean()
            bottom_half = red_attn[12:, :].mean()
            ratio = top_half / (bottom_half + 1e-8)
            
            print(f"\nLayer {i}: Top/Bottom ratio = {ratio:.2f}")
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_layer = i
        
        print(f"\n🏆 Best layer: {best_layer} with ratio {best_ratio:.2f}")
        print("\nTIP: If all ratios are close to 1.0, try more descriptive prompts")


def test_more_complex_prompt():
    """Test with more descriptive prompts"""
    
    print("\n\nTESTING WITH MORE CONTEXT")
    print("="*60)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    visualizer = AttentionVisualizer(model, processor)
    
    # Create image with red square in top-left
    img = Image.new('RGB', (384, 384), 'white')
    pixels = img.load()
    for i in range(128):
        for j in range(128):
            pixels[i, j] = (255, 0, 0)
    
    # Test different prompts
    prompts = [
        "red",
        "red square",
        "a red square in the corner",
        "there is a red square in the top left"
    ]
    
    results = []
    
    for prompt in prompts:
        try:
            inputs = processor(images=img, text=prompt, return_tensors="pt")
            attention_data = visualizer.extractor.extract(inputs, attention_type="cross", layer_indices=[-1])
            
            if attention_data["attention_maps"]:
                attn = attention_data["attention_maps"][-1]
                if attn.ndim > 2:
                    attn = attn.mean(axis=0)
                
                # Average attention across all text tokens to image patches
                avg_attn = attn[:, 1:].mean(axis=0).reshape(24, 24)  # Skip CLS token
                
                # Calculate focus on red region (top-left ~8x8 area)
                top_left = avg_attn[:8, :8].mean()
                rest = avg_attn[8:, 8:].mean()
                ratio = top_left / (rest + 1e-8)
                
                results.append((prompt, ratio))
                print(f"\nPrompt: '{prompt}'")
                print(f"Focus ratio: {ratio:.2f}")
                
                # Show best one
                if ratio > 1.2:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    ax1.imshow(img)
                    ax1.set_title(f"Input")
                    ax1.axis('off')
                    
                    im = ax2.imshow(avg_attn, cmap='hot')
                    ax2.set_title(f"Attention for: '{prompt}'\nRatio: {ratio:.2f}")
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046)
                    
                    # Add red square outline
                    rect = plt.Rectangle((-0.5, -0.5), 8, 8, fill=False, edgecolor='red', linewidth=2)
                    ax2.add_patch(rect)
                    
                    plt.tight_layout()
                    plt.show()
                
        except Exception as e:
            print(f"\nPrompt: '{prompt}' - Error: {e}")
    
    # Summary
    if results:
        best_prompt, best_ratio = max(results, key=lambda x: x[1])
        print(f"\n🏆 Best prompt: '{best_prompt}' with ratio {best_ratio:.2f}")


def test_different_visual_features():
    """Test attention on different visual features"""
    
    print("\n\nTESTING DIFFERENT VISUAL FEATURES")
    print("="*60)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    visualizer = AttentionVisualizer(model, processor)
    
    # Create image with multiple colored regions
    img = Image.new('RGB', (384, 384), 'white')
    pixels = img.load()
    
    # Red top-left
    for i in range(192):
        for j in range(192):
            pixels[i, j] = (255, 0, 0)
    
    # Blue top-right
    for i in range(192):
        for j in range(192, 384):
            pixels[i, j] = (0, 0, 255)
    
    # Green bottom-left
    for i in range(192, 384):
        for j in range(192):
            pixels[i, j] = (0, 255, 0)
    
    # Test each color
    colors = ["red", "blue", "green", "yellow"]
    expected_regions = ["top-left", "top-right", "bottom-left", "bottom-right"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show input
    axes[0].imshow(img)
    axes[0].set_title("Input: 4 colored quadrants")
    axes[0].axis('off')
    
    for idx, (color, region) in enumerate(zip(colors[:3], expected_regions[:3])):
        try:
            inputs = processor(images=img, text=color, return_tensors="pt")
            attention_data = visualizer.extractor.extract(inputs, attention_type="cross", layer_indices=[-1])
            
            if attention_data["attention_maps"]:
                attn = attention_data["attention_maps"][-1]
                if attn.ndim > 2:
                    attn = attn.mean(axis=0)
                
                # Get attention for color token
                color_attn = attn[1, 1:].reshape(24, 24)
                
                axes[idx+1].imshow(color_attn, cmap='hot')
                axes[idx+1].set_title(f"'{color}' attention\n(should focus on {region})")
                axes[idx+1].axis('off')
                
        except Exception as e:
            print(f"Error with {color}: {e}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # First, try to fix the KeyError by checking the setup
    print("Testing BLIP cross-modal attention visualization...")
    print("\nNote: BLIP uses 24x24 patch grid (384x384 image with 16x16 patches)")
    print("=" * 60)
    
    # Run tests
    simple_semantic_test()
    check_which_layer_works_best()
    test_more_complex_prompt()
    test_different_visual_features()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("\nIf attention is uniform across all tests:")
    print("1. BLIP may need full sentences, not single words")
    print("2. Try earlier layers (0-6) which might show more localized attention")
    print("3. The model might be using global context rather than local features")
    print("\nNext steps:")
    print("- Use more descriptive prompts")
    print("- Try with real images where BLIP can leverage its training")
    print("- Visualize attention at multiple layers simultaneously")