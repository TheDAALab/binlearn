#!/usr/bin/env python3
"""
More precise analysis of utils usage.
"""

import ast
import os
from pathlib import Path

def analyze_imports_in_file(filepath):
    """Extract imports from a Python file using AST parsing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('..utils'):
                    module_parts = node.module.split('.')
                    if len(module_parts) >= 3:  # ['', '', 'utils', 'module_name']
                        utils_module = '.'.join(module_parts[2:])  # 'utils' or 'utils.module_name'
                        for alias in node.names:
                            imports.append(f"{utils_module}.{alias.name}")
                    elif node.module == '..utils':
                        for alias in node.names:
                            imports.append(f"utils.{alias.name}")
        
        return imports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def find_all_utils_usage():
    """Find all actual utils imports across the codebase."""
    base_path = Path('/home/gykovacs/workspaces/binlearn')
    all_imports = set()
    
    # Check base and methods directories
    for subdir in ['base', 'methods']:
        dir_path = base_path / 'binlearn' / subdir
        if dir_path.exists():
            for py_file in dir_path.glob('*.py'):
                if py_file.name != '__init__.py':  # Skip init files for now
                    imports = analyze_imports_in_file(py_file)
                    for imp in imports:
                        all_imports.add(imp)
                        print(f"{py_file.name}: {imp}")
    
    return sorted(all_imports)

print("PRECISE UTILS USAGE ANALYSIS:")
print("=" * 60)
used_imports = find_all_utils_usage()

print(f"\nUNIQUE IMPORTS SUMMARY:")
print("=" * 40)
module_counts = {}
for imp in used_imports:
    module = imp.split('.')[1] if '.' in imp else imp
    if module not in module_counts:
        module_counts[module] = []
    module_counts[module].append(imp.split('.', 1)[1] if '.' in imp else imp)

for module, functions in sorted(module_counts.items()):
    print(f"\n{module}:")
    for func in sorted(set(functions)):
        print(f"  - {func}")

print(f"\nTotal unique imports: {len(used_imports)}")
print(f"Modules used: {len(module_counts)}")
