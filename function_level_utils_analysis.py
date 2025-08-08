#!/usr/bin/env python3
"""
Analyze function usage within utils files to identify unused functions.
"""

import ast
import os
from pathlib import Path
import re

def get_functions_in_file(filepath):
    """Extract all function definitions from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return functions, classes
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return [], []

def find_usage_in_codebase(item_name, base_path):
    """Find usage of a function/class across the codebase."""
    usage_count = 0
    usage_files = []
    
    # Search in all Python files
    for root, dirs, files in os.walk(base_path):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple text search - look for the item name
                    # This will catch both imports and usage
                    if re.search(r'\b' + re.escape(item_name) + r'\b', content):
                        usage_count += 1
                        usage_files.append(filepath.replace(base_path, ''))
                        
                except Exception:
                    continue
    
    return usage_count, usage_files

def analyze_utils_functions():
    """Analyze all utils files for unused functions."""
    base_path = '/home/gykovacs/workspaces/binlearn'
    utils_path = Path(base_path) / 'binlearn' / 'utils'
    
    results = {}
    
    for py_file in utils_path.glob('*.py'):
        if py_file.name == '__init__.py':
            continue
            
        functions, classes = get_functions_in_file(py_file)
        results[py_file.name] = {
            'functions': [],
            'classes': [],
            'unused_functions': [],
            'unused_classes': []
        }
        
        # Check usage for each function
        for func_name in functions:
            if not func_name.startswith('_'):  # Skip private functions for now
                usage_count, usage_files = find_usage_in_codebase(func_name, base_path)
                results[py_file.name]['functions'].append({
                    'name': func_name,
                    'usage_count': usage_count,
                    'usage_files': usage_files
                })
                
                # Consider unused if only found in its own definition file
                if usage_count <= 1 or (usage_count == 2 and any('__init__.py' in f for f in usage_files)):
                    results[py_file.name]['unused_functions'].append(func_name)
        
        # Check usage for each class  
        for class_name in classes:
            usage_count, usage_files = find_usage_in_codebase(class_name, base_path)
            results[py_file.name]['classes'].append({
                'name': class_name,
                'usage_count': usage_count,
                'usage_files': usage_files
            })
            
            if usage_count <= 1 or (usage_count == 2 and any('__init__.py' in f for f in usage_files)):
                results[py_file.name]['unused_classes'].append(class_name)
    
    return results

# Run the analysis
print("FUNCTION-LEVEL UTILS CLEANUP ANALYSIS:")
print("=" * 60)

results = analyze_utils_functions()

total_unused = 0
for filename, data in results.items():
    unused_funcs = data['unused_functions'] 
    unused_classes = data['unused_classes']
    
    if unused_funcs or unused_classes:
        print(f"\n📁 {filename}:")
        if unused_funcs:
            print(f"  Unused functions ({len(unused_funcs)}):")
            for func in unused_funcs:
                print(f"    - {func}")
                total_unused += 1
        if unused_classes:
            print(f"  Unused classes ({len(unused_classes)}):")
            for cls in unused_classes:
                print(f"    - {cls}")
                total_unused += 1
    else:
        print(f"\n✅ {filename}: All functions/classes are used")

print(f"\nSUMMARY:")
print(f"Total potentially unused items: {total_unused}")

if total_unused > 0:
    print(f"\nNOTE: This is a conservative analysis. Some items marked as 'unused'")
    print(f"might be used in ways not detected by simple text search.")
    print(f"Manual verification is recommended before removal.")
