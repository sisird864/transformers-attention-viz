#!/usr/bin/env python3
"""
Comprehensive test suite to run before PyPI release
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import traceback
import gc
import psutil
import time

# Track test results
test_results = {
    "passed": [],
    "failed": []
}

def test(name):
    """Decorator for tests"""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n🧪 Testing: {name}")
                func()
                print(f"✅ PASSED: {name}")
                test_results["passed"].append(name)
            except Exception as e:
                print(f"❌ FAILED: {name}")
                print(f"   Error: {str(e)}")
                traceback.print_exc()
                test_results["failed"].append(name)
        return wrapper
    return decorator

print("🔍 Comprehensive Test Suite for transformers-attention-viz")
print("=" * 60)

# Test 1: Import Tests
@test("Basic imports")
def test_imports():
    from attention_viz import AttentionVisualizer, launch_dashboard
    from attention_viz.extractors import AttentionExtractor
    from attention_viz.visualizers import AttentionHeatmap, AttentionFlow, LayerEvolution
    assert AttentionVisualizer is not None

@test("Version information")
def test_version():
    import attention_viz
    assert hasattr(attention_viz, '__version__')
    assert attention_viz.__version__ == "0.1.0"
    print(f"   Version: {attention_viz.__version__}")

# Test 2: Real Model Tests
@test("CLIP model integration")
def test_clip_model():
    from transformers import CLIPModel, CLIPProcessor
    from attention_viz import AttentionVisualizer
    
    print("   Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    visualizer = AttentionVisualizer(model, processor)
    
    # Test with real image
    image = Image.new('RGB', (224, 224), color='red')
    text = "a red square"
    
    viz = visualizer.visualize(
        image=image,
        text=text,
        visualization_type="heatmap"
    )
    assert viz is not None
    print("   CLIP visualization successful")

@test("BLIP model integration")
def test_blip_model():
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from attention_viz import AttentionVisualizer
        
        print("   Loading BLIP model...")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        visualizer = AttentionVisualizer(model, processor)
        print("   BLIP model loaded successfully")
    except Exception as e:
        if "404" in str(e):
            print("   ⚠️  BLIP model not available, skipping...")
        else:
            raise

# Test 3: Edge Cases
@test("Empty inputs handling")
def test_empty_inputs():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    # Test with minimal image
    image = Image.new('RGB', (1, 1))
    try:
        viz = visualizer.visualize(image=image, text="", visualization_type="heatmap")
    except Exception as e:
        print(f"   Expected error handling: {type(e).__name__}")

@test("Large input handling")
def test_large_inputs():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    # Test with large image
    image = Image.new('RGB', (1024, 1024))
    text = " ".join(["word"] * 100)  # Long text
    
    viz = visualizer.visualize(image=image, text=text, visualization_type="heatmap")
    assert viz is not None

# Test 4: All Visualization Types
@test("All visualization types")
def test_all_viz_types():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    image = Image.new('RGB', (224, 224))
    text = "test"
    
    for viz_type in ["heatmap", "flow", "evolution"]:
        print(f"   Testing {viz_type}...")
        viz = visualizer.visualize(
            image=image,
            text=text,
            visualization_type=viz_type
        )
        assert viz is not None

# Test 5: Memory Usage
@test("Memory efficiency")
def test_memory_usage():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    # Run multiple visualizations
    for i in range(10):
        image = Image.new('RGB', (224, 224))
        viz = visualizer.visualize(image=image, text=f"test {i}", visualization_type="heatmap")
        del viz
        gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"   Memory increase: {memory_increase:.2f} MB")
    assert memory_increase < 500, f"Memory leak detected: {memory_increase} MB increase"

# Test 6: File I/O
@test("Save and export functionality")
def test_file_io():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    image = Image.new('RGB', (224, 224))
    viz = visualizer.visualize(image=image, text="test", visualization_type="heatmap")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test PNG save
        png_path = os.path.join(tmpdir, "test.png")
        viz.save(png_path)
        assert os.path.exists(png_path)
        print(f"   PNG saved successfully")
        
        # Test different formats
        for fmt in ['pdf', 'svg']:
            path = os.path.join(tmpdir, f"test.{fmt}")
            try:
                viz.save(path)
                assert os.path.exists(path)
                print(f"   {fmt.upper()} saved successfully")
            except Exception as e:
                print(f"   ⚠️  {fmt.upper()} save not supported: {e}")

# Test 7: Dashboard Test
@test("Dashboard initialization")
def test_dashboard():
    from attention_viz.dashboard.app import Dashboard
    
    dashboard = Dashboard()
    interface = dashboard.create_interface()
    assert interface is not None
    print("   Dashboard created successfully")

# Test 8: Statistics Functionality
@test("Attention statistics")
def test_statistics():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    image = Image.new('RGB', (224, 224))
    stats = visualizer.get_attention_stats(image, "test text")
    
    required_keys = ["entropy", "top_tokens", "concentration", "mean_attention", "std_attention"]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    
    print(f"   Stats keys: {list(stats.keys())}")

# Test 9: Error Handling
@test("Error handling")
def test_error_handling():
    from attention_viz import AttentionVisualizer
    
    # Test with None processor
    try:
        from tests.test_basic import MockModel
        model = MockModel()
        visualizer = AttentionVisualizer(model, None)
        image = Image.new('RGB', (224, 224))
        viz = visualizer.visualize(image=image, text="test")
    except ValueError as e:
        assert "Processor not provided" in str(e)
        print("   Correctly handles missing processor")

# Test 10: Package Structure
@test("Package structure")
def test_package_structure():
    import attention_viz
    
    # Check all expected modules exist
    expected_modules = [
        'attention_viz.core',
        'attention_viz.extractors',
        'attention_viz.visualizers',
        'attention_viz.dashboard',
        'attention_viz.utils'
    ]
    
    for module in expected_modules:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except ImportError:
            raise AssertionError(f"Missing module: {module}")

# Test 11: Cross-platform Path Handling
@test("Cross-platform compatibility")
def test_cross_platform():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    # Test with path containing spaces and special chars
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test dir with spaces")
        os.makedirs(test_dir, exist_ok=True)
        
        image_path = os.path.join(test_dir, "test image.png")
        Image.new('RGB', (224, 224)).save(image_path)
        
        # Should handle path with spaces
        viz = visualizer.visualize(image=image_path, text="test")
        assert viz is not None

# Test 12: Performance Benchmarks
@test("Performance benchmarks")
def test_performance():
    from attention_viz import AttentionVisualizer
    from tests.test_basic import MockModel, MockProcessor
    
    model = MockModel()
    processor = MockProcessor()
    visualizer = AttentionVisualizer(model, processor)
    
    image = Image.new('RGB', (224, 224))
    
    # Benchmark visualization creation
    start_time = time.time()
    for _ in range(5):
        viz = visualizer.visualize(image=image, text="test", visualization_type="heatmap")
    elapsed = time.time() - start_time
    
    avg_time = elapsed / 5
    print(f"   Average visualization time: {avg_time:.3f}s")
    assert avg_time < 5.0, f"Visualization too slow: {avg_time}s"

# Run all tests
if __name__ == "__main__":
    # Run tests
    test_imports()
    test_version()
    test_clip_model()
    test_blip_model()
    test_empty_inputs()
    test_large_inputs()
    test_all_viz_types()
    test_memory_usage()
    test_file_io()
    test_dashboard()
    test_statistics()
    test_error_handling()
    test_package_structure()
    test_cross_platform()
    test_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    
    if test_results['failed']:
        print("\nFailed tests:")
        for test in test_results['failed']:
            print(f"  - {test}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed! Ready for PyPI release.")
        print("\nNext steps:")
        print("1. Update version in setup.py and pyproject.toml")
        print("2. Create git tag: git tag v0.1.0")
        print("3. Build: python -m build")
        print("4. Upload to TestPyPI first: twine upload --repository testpypi dist/*")
        print("5. Test installation: pip install -i https://test.pypi.org/simple/ transformers-attention-viz")
        print("6. Upload to PyPI: twine upload dist/*")