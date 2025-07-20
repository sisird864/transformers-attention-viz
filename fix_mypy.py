#!/usr/bin/env python3
"""
Fix all mypy errors in the project
"""
import re

# Fix 1: Update pyproject.toml
print("1. Updating Python version in pyproject.toml...")
with open('pyproject.toml', 'r') as f:
    content = f.read()
content = content.replace('python_version = "3.8"', 'python_version = "3.9"')
with open('pyproject.toml', 'w') as f:
    f.write(content)

# Fix 2: Fix tensor_to_numpy imports in base.py
print("2. Fixing imports in base.py...")
with open('attention_viz/extractors/base.py', 'r') as f:
    content = f.read()

# Replace the try/except import block
content = re.sub(
    r'# Import from parent package\ntry:.*?from utils import tensor_to_numpy',
    'from ..utils import tensor_to_numpy',
    content,
    flags=re.DOTALL
)
# Also add type imports at the top
if 'from typing import' in content:
    content = content.replace(
        'from typing import Dict, List, Optional, Union, Any',
        'from typing import Dict, List, Optional, Union, Any\nimport torch'
    )

# Add type annotations
content = content.replace(
    'self.hooks = []',
    'self.hooks: List[torch.utils.hooks.RemovableHandle] = []'
)
content = content.replace(
    'self.attention_maps = defaultdict(list)',
    'self.attention_maps: Dict[str, List[torch.Tensor]] = defaultdict(list)'
)

with open('attention_viz/extractors/base.py', 'w') as f:
    f.write(content)

# Fix 3: Fix tensor_to_numpy imports in core.py
print("3. Fixing imports in core.py...")
with open('attention_viz/core.py', 'r') as f:
    content = f.read()

# Replace the try/except import block
content = re.sub(
    r'try:.*?from utils import tensor_to_numpy',
    'from .utils import tensor_to_numpy',
    content,
    flags=re.DOTALL
)

with open('attention_viz/core.py', 'w') as f:
    f.write(content)

# Fix 4: Add type annotations to evolution.py
print("4. Adding type annotations to evolution.py...")
with open('attention_viz/visualizers/evolution.py', 'r') as f:
    content = f.read()

# Add Dict import if not present
if 'from typing import' in content and 'Dict' not in content:
    content = content.replace(
        'from typing import',
        'from typing import Dict,'
    )

# Add type annotation
content = content.replace(
    '        metrics = {',
    '        metrics: Dict[str, List[float]] = {'
)

with open('attention_viz/visualizers/evolution.py', 'w') as f:
    f.write(content)

# Fix 5: Add type annotations to comparison.py
print("5. Adding type annotations to comparison.py...")
with open('attention_viz/visualizers/comparison.py', 'r') as f:
    content = f.read()

# Add type annotation
content = content.replace(
    '        head_stats = {',
    '        head_stats: Dict[str, List[float]] = {'
)

with open('attention_viz/visualizers/comparison.py', 'w') as f:
    f.write(content)

print("\n✅ All mypy fixes applied!")
print("\nNow run:")
print("  git add -A")
print("  git commit -m 'Fix mypy type checking errors'")
print("  git push")