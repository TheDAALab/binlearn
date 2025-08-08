#!/usr/bin/env python3
"""
Analyze utils usage across the binlearn codebase to identify what's actually used.
"""

import os
import re
from pathlib import Path


def find_utils_imports(base_path):
    """Find all imports from utils modules in the codebase."""
    imports = set()

    # Search in base and methods directories
    for subdir in ["base", "methods"]:
        search_dir = Path(base_path) / "binlearn" / subdir
        if search_dir.exists():
            for py_file in search_dir.glob("*.py"):
                with open(py_file, "r") as f:
                    content = f.read()

                    # Find direct imports from ..utils.module_name
                    pattern = r"from \.\.utils\.([a-z_]+) import ([^)]+(?:\([^)]+\))?)"
                    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                    for module_name, imported_items in matches:
                        # Clean up imported items (remove parentheses and whitespace)
                        cleaned_items = re.sub(r"[(),\s]+", " ", imported_items).strip()
                        for item in cleaned_items.split():
                            if item:
                                imports.add(f"{module_name}.{item}")

                    # Find imports from ..utils (consolidated)
                    pattern = r"from \.\.utils import \((.*?)\)"
                    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        # Clean up and split items
                        cleaned_items = re.sub(r"[(),\s\n]+", " ", match).strip()
                        for item in cleaned_items.split(","):
                            item = item.strip()
                            if item and not item.startswith("#"):
                                imports.add(f"consolidated.{item}")

    return sorted(imports)


# Find all imports
base_path = "/home/gykovacs/workspaces/binlearn"
used_imports = find_utils_imports(base_path)

print("ACTUALLY USED IMPORTS FROM UTILS:")
print("=" * 60)
for imp in used_imports:
    print(f"  {imp}")

print(f"\nTotal unique imports: {len(used_imports)}")
