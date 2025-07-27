#!/usr/bin/env python3
"""
Development quality checks for the binning framework.
Run this script to identify code quality issues and improvement opportunities.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ ISSUES FOUND")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def check_imports():
    """Check for remaining star imports."""
    print("\n🔍 Checking for star imports")
    print("-" * 60)
    
    star_imports = []
    binning_path = Path("./binning")
    
    for py_file in binning_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if "from " in content and " import *" in content:
                    # Look for actual star imports, not comments
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if line.strip().startswith("from ") and " import *" in line and not line.strip().startswith("#"):
                            star_imports.append(f"{py_file}:{i}: {line.strip()}")
        except Exception:
            pass
    
    if star_imports:
        print("❌ Star imports found:")
        for imp in star_imports:
            print(f"   {imp}")
        return False
    else:
        print("✅ No star imports found")
        return True


def check_test_structure():
    """Analyze test file sizes."""
    print("\n🔍 Analyzing test file structure")
    print("-" * 60)
    
    large_tests = []
    tests_path = Path("./tests")
    
    for py_file in tests_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                lines = len(f.readlines())
                if lines > 500:  # Large test files
                    large_tests.append((str(py_file), lines))
        except Exception:
            pass
    
    if large_tests:
        print("⚠️  Large test files found (consider splitting):")
        for file_path, lines in sorted(large_tests, key=lambda x: x[1], reverse=True):
            print(f"   {file_path}: {lines} lines")
    else:
        print("✅ All test files are reasonably sized")
    
    return len(large_tests) == 0


def main():
    """Run all quality checks."""
    print("="*80)
    print("BINNING FRAMEWORK: DEVELOPMENT QUALITY CHECKS")
    print("="*80)
    
    checks = [
        ("python -m py_compile binning/__init__.py", "Python syntax check"),
        ("python -c 'import binning; print(\"✅ Package imports successfully\")'", "Import validation"),
    ]
    
    results = []
    
    # Run command-based checks
    for cmd, desc in checks:
        results.append(run_command(cmd, desc))
    
    # Run custom checks
    results.append(check_imports())
    results.append(check_test_structure())
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All quality checks passed!")
        print("✅ Code is ready for development")
    else:
        print(f"⚠️  {total - passed} out of {total} checks need attention")
        print("🔧 Review the issues above and run again")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Fix any failing checks above")
    print("   2. Run full test suite: python -m pytest tests/")
    print("   3. Consider running: black binning/ tests/")
    print("   4. Review large files for splitting opportunities")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
