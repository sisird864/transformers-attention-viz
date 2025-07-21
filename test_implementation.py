# Test script to verify your implementation works correctly

from transformers import CLIPModel, CLIPProcessor
from attention_viz import AttentionVisualizer
from PIL import Image
import requests
from io import BytesIO

# Load model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create visualizer
visualizer = AttentionVisualizer(model, processor)

# Load test image
url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

text = "a fluffy orange cat"

# Test 1: Auto mode (should show vision self-attention and warn about no cross-modal)
print("\n" + "="*50)
print("Test 1: Auto mode")
print("="*50)
viz1 = visualizer.visualize(
    image=image,
    text=text,
    visualization_type="heatmap",
    attention_type="auto"  # Should default to vision_self for CLIP
)
viz1.show()

# Test 2: Explicitly request text self-attention
print("\n" + "="*50)
print("Test 2: Text self-attention")
print("="*50)
viz2 = visualizer.visualize(
    image=image,
    text=text,
    visualization_type="heatmap",
    attention_type="text_self"
)
viz2.show()

# Test 3: Try to request cross-modal (should fail with clear error)
print("\n" + "="*50)
print("Test 3: Cross-modal attention (should fail)")
print("="*50)
try:
    viz3 = visualizer.visualize(
        image=image,
        text=text,
        visualization_type="heatmap",
        attention_type="cross"
    )
except ValueError as e:
    print(f"Expected error: {e}")

# Test 4: Get attention statistics
print("\n" + "="*50)
print("Test 4: Attention statistics")
print("="*50)
stats = visualizer.get_attention_stats(image, text)
print(f"Model type: {stats['model_type']}")
print(f"Attention type: {stats['attention_type']}")
print(f"Average entropy: {stats['entropy'].mean():.3f}")
print(f"Top tokens: {stats['top_tokens'][:3]}")