"""
Download example images for demos
"""

import os
import requests
from PIL import Image
from io import BytesIO

# Create images directory
os.makedirs("examples/images", exist_ok=True)

# Example images to download (using Unsplash for free images)
examples = {
    "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
    "dog.jpg": "https://images.unsplash.com/photo-1547407139-3c921a66005c?w=400", 
    "landscape.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
    "person.jpg": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
    "city.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400"
}

print("Downloading example images...")

for filename, url in examples.items():
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Open and save image
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")  # Ensure RGB format
        img = img.resize((224, 224))  # Resize to standard size
        
        filepath = os.path.join("examples/images", filename)
        img.save(filepath)
        print(f"✓ Downloaded {filename}")
        
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

print("\nExample images ready!")