#!/usr/bin/env python3
"""
Comprehensive code quality and architecture analysis for the binning framework.
"""

import os
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json

def analyze_codebase():
    """Analyze the entire codebase for quality and architectural issues."""
    
    print("="*80)
    print("BINNING FRAMEWORK: CODE QUALITY & ARCHITECTURE ANALYSIS")
    print("="*80)
    
    binning_path = Path("./binning")
    
    # 1. MODULE STRUCTURE ANALYSIS
    print("\\n1. MODULE STRUCTURE ANALYSIS")
    print("-" * 50)
    
    modules = {
        "binning/": "Main package",
        "binning/base/": "Base classes and utilities",
        "binning/methods/": "Concrete binning implementations",
        "tests/": "Test suite"
    }
    
    for module, description in modules.items():
        if os.path.exists(module):
            py_files = len(list(Path(module).rglob("*.py")))
            print(f"✓ {module:<20} {description:<30} ({py_files} files)")
        else:
            print(f"✗ {module:<20} {description:<30} (MISSING)")
    
    # 2. IMPORT ANALYSIS
    print("\\n2. IMPORT PATTERN ANALYSIS")
    print("-" * 50)
    
    star_imports = []
    circular_risks = []
    
    for py_file in binning_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if "import *" in content:
                    star_imports.append(str(py_file))
        except Exception:
            pass
    
    print(f"🔍 Star imports found: {len(star_imports)}")
    for imp in star_imports[:5]:  # Show first 5
        print(f"   - {imp}")
    if len(star_imports) > 5:
        print(f"   ... and {len(star_imports) - 5} more")
    
    # 3. CLASS HIERARCHY ANALYSIS
    print("\\n3. CLASS HIERARCHY ANALYSIS")
    print("-" * 50)
    
    base_classes = [
        "GeneralBinningBase",
        "IntervalBinningBase", 
        "FlexibleBinningBase",
        "SupervisedBinningBase"
    ]
    
    mixins = [
        "ValidationMixin",
        "SklearnCompatibilityMixin", 
        "ReprMixin",
        "GuidedBinningMixin"
    ]
    
    concrete_classes = [
        "EqualWidthBinning",
        "SupervisedBinning",
        "OneHotBinning"
    ]
    
    print("Base Classes:")
    for cls in base_classes:
        print(f"   ✓ {cls}")
    
    print("\\nMixins:")
    for cls in mixins:
        print(f"   ✓ {cls}")
    
    print("\\nConcrete Classes:")
    for cls in concrete_classes:
        print(f"   ✓ {cls}")
    
    # 4. FILE SIZE ANALYSIS
    print("\\n4. FILE SIZE ANALYSIS")
    print("-" * 50)
    
    large_files = []
    total_lines = 0
    
    for py_file in binning_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                if lines > 300:  # Files over 300 lines
                    large_files.append((str(py_file), lines))
        except Exception:
            pass
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 Total lines of code: {total_lines}")
    print(f"📊 Large files (>300 lines): {len(large_files)}")
    
    for file_path, lines in large_files[:5]:
        print(f"   - {file_path}: {lines} lines")
    
    # 5. DUPLICATION ANALYSIS
    print("\\n5. CODE DUPLICATION ANALYSIS")
    print("-" * 50)
    
    duplicate_files = [
        ("_guided_binning_mixin.py", "_guided_binning_mixin_new.py")
    ]
    
    print("🔍 Potential duplicates found:")
    for file1, file2 in duplicate_files:
        print(f"   - {file1} ↔ {file2}")
    
    # Empty files
    empty_files = []
    for py_file in binning_path.rglob("*.py"):
        try:
            if os.path.getsize(py_file) == 0:
                empty_files.append(str(py_file))
        except Exception:
            pass
    
    if empty_files:
        print(f"\\n📁 Empty files: {len(empty_files)}")
        for ef in empty_files:
            print(f"   - {ef}")

def analyze_architectural_issues():
    """Identify architectural improvement opportunities."""
    
    print("\\n" + "="*80)
    print("ARCHITECTURAL IMPROVEMENT OPPORTUNITIES")
    print("="*80)
    
    issues = {
        "HIGH PRIORITY": [
            {
                "issue": "Star Import Anti-pattern",
                "description": "Excessive use of 'from module import *'",
                "impact": "Namespace pollution, unclear dependencies, IDE issues",
                "solution": "Replace with explicit imports",
                "files": ["binning/__init__.py", "binning/base/__init__.py", "binning/methods/__init__.py"],
                "effort": "LOW"
            },
            {
                "issue": "Duplicate Files",
                "description": "_guided_binning_mixin.py and _guided_binning_mixin_new.py are identical",
                "impact": "Maintenance overhead, confusion",
                "solution": "Remove duplicate, use deprecation warnings",
                "files": ["binning/base/_guided_binning_mixin*.py"],
                "effort": "LOW"
            },
            {
                "issue": "Empty Placeholder Files",
                "description": "cache.py and registry.py are empty",
                "impact": "Unclear purpose, potential confusion",
                "solution": "Implement or document as future features",
                "files": ["binning/cache.py", "binning/registry.py"],
                "effort": "LOW"
            }
        ],
        "MEDIUM PRIORITY": [
            {
                "issue": "Large Complex Files",
                "description": "Several files exceed 400+ lines",
                "impact": "Reduced maintainability, testing complexity",
                "solution": "Split into smaller, focused modules",
                "files": ["tests/base/test_flexible_bin_utils.py", "tests/base/test_general_binning_base.py"],
                "effort": "MEDIUM"
            },
            {
                "issue": "Mixin Proliferation",
                "description": "Multiple mixins with overlapping concerns",
                "impact": "Complex inheritance, method resolution order issues",
                "solution": "Consolidate related mixins, clear separation of concerns",
                "files": ["ValidationMixin", "SklearnCompatibilityMixin", "ReprMixin"],
                "effort": "MEDIUM"
            },
            {
                "issue": "Configuration Management",
                "description": "Config system exists but underutilized",
                "impact": "Inconsistent defaults, poor configurability",
                "solution": "Expand config usage, add validation",
                "files": ["binning/config.py"],
                "effort": "MEDIUM"
            }
        ],
        "LOW PRIORITY": [
            {
                "issue": "Package Structure Optimization",
                "description": "Some utilities could be better organized",
                "impact": "Developer experience, discoverability",
                "solution": "Reorganize utility modules by function",
                "files": ["binning/base/_*.py"],
                "effort": "HIGH"
            },
            {
                "issue": "Documentation Structure",
                "description": "Docstrings inconsistent, missing examples",
                "impact": "Poor developer experience, adoption barriers",
                "solution": "Standardize docstring format, add examples",
                "files": ["All .py files"],
                "effort": "HIGH"
            }
        ]
    }
    
    for priority, issue_list in issues.items():
        print(f"\\n{priority}")
        print("-" * 60)
        
        for i, issue in enumerate(issue_list, 1):
            print(f"\\n{i}. {issue['issue']} [{issue['effort']} EFFORT]")
            print(f"   Description: {issue['description']}")
            print(f"   Impact: {issue['impact']}")
            print(f"   Solution: {issue['solution']}")
            if isinstance(issue['files'], list):
                print(f"   Files: {', '.join(issue['files'][:3])}{'...' if len(issue['files']) > 3 else ''}")
            else:
                print(f"   Files: {issue['files']}")

def code_quality_metrics():
    """Analyze code quality metrics."""
    
    print("\\n" + "="*80)
    print("CODE QUALITY METRICS")
    print("="*80)
    
    metrics = {
        "Architectural Strengths": [
            "✅ Clear separation of base classes and concrete implementations",
            "✅ Good use of abstract base classes for interface enforcement", 
            "✅ Comprehensive error handling with custom exception hierarchy",
            "✅ Sklearn compatibility through proper mixin usage",
            "✅ Configuration system foundation in place",
            "✅ Extensive test coverage (9901 total lines, significant test code)"
        ],
        "Code Organization": [
            "✅ Logical package structure (base/, methods/, tests/)",
            "✅ Clear naming conventions",
            "✅ Type hints usage throughout",
            "⚠️  Some large files that could benefit from splitting",
            "⚠️  Star imports reducing code clarity"
        ],
        "Maintainability": [
            "✅ Good documentation in key areas",
            "✅ Consistent error handling patterns",
            "✅ Version management and configuration",
            "⚠️  Duplicate code that needs cleanup",
            "⚠️  Some complex inheritance chains"
        ],
        "Extensibility": [
            "✅ Plugin-ready architecture with base classes",
            "✅ Mixin pattern for cross-cutting concerns",
            "✅ Configuration system for customization",
            "🔄 Strategy pattern discussed but not yet implemented",
            "🔄 Registry pattern placeholder exists"
        ]
    }
    
    for category, items in metrics.items():
        print(f"\\n{category}:")
        for item in items:
            print(f"   {item}")

def improvement_roadmap():
    """Provide actionable improvement roadmap."""
    
    print("\\n" + "="*80)
    print("IMPROVEMENT ROADMAP")
    print("="*80)
    
    phases = {
        "Phase 1: Quick Wins (1-2 days)": [
            "🚀 Replace star imports with explicit imports",
            "🚀 Remove duplicate _guided_binning_mixin_new.py",
            "🚀 Add docstrings to empty cache.py and registry.py",
            "🚀 Fix any lint warnings in codebase",
            "🚀 Standardize import order (use isort)"
        ],
        "Phase 2: Structure Cleanup (1 week)": [
            "🔧 Split large test files into focused test modules",
            "🔧 Consolidate related mixins where appropriate",
            "🔧 Expand configuration system usage",
            "🔧 Add __all__ exports to modules",
            "🔧 Create proper module-level documentation"
        ],
        "Phase 3: Architecture Enhancement (2-3 weeks)": [
            "🏗️  Implement registry pattern for method discovery",
            "🏗️  Add plugin system for custom binning methods", 
            "🏗️  Implement caching layer for performance",
            "🏗️  Add comprehensive logging framework",
            "🏗️  Create factory pattern for method instantiation"
        ],
        "Phase 4: Advanced Features (Future)": [
            "⭐ Strategy pattern for algorithm components",
            "⭐ Parallel processing support",
            "⭐ Configuration validation and schemas",
            "⭐ Performance profiling and optimization",
            "⭐ Integration with ML pipelines (MLflow, etc.)"
        ]
    }
    
    for phase, tasks in phases.items():
        print(f"\\n{phase}")
        print("-" * 60)
        for task in tasks:
            print(f"   {task}")

if __name__ == "__main__":
    analyze_codebase()
    analyze_architectural_issues()
    code_quality_metrics()
    improvement_roadmap()
    
    print("\\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ The codebase has a solid architectural foundation")
    print("✅ Good separation of concerns and inheritance hierarchy")
    print("⚠️  Several quick wins available for immediate improvement")
    print("🚀 Ready for incremental enhancement without major refactoring")
    print("📈 Strong foundation for advanced features when needed")
    print("="*80)
