#!/usr/bin/env python3
"""
Migration verification script for setup.py -> pyproject.toml transition.

This script helps verify that the migration from setup.py to pyproject.toml
was successful by comparing configurations and testing build capabilities.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, check=True
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return None


def check_pyproject_toml():
    """Check if pyproject.toml exists and is valid."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found!")
        return False

    try:
        import tomllib

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Check required sections
        required_sections = ["build-system", "project"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False

        print("✅ pyproject.toml is valid")
        return True
    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return False


def check_build_tools():
    """Check if modern build tools are available."""
    tools = ["build", "pip", "twine"]
    available_tools = []

    for tool in tools:
        if run_command(f"which {tool}") is not None:
            available_tools.append(tool)
            print(f"✅ {tool} is available")
        else:
            print(f"⚠️  {tool} is not available (install with: pip install {tool})")

    return len(available_tools) >= 2  # At least build and pip


def test_package_build():
    """Test building the package with modern tools."""
    print("\n🔨 Testing package build...")

    # Clean previous builds
    run_command("rm -rf build/ dist/ *.egg-info/", capture_output=False)

    # Test build
    result = run_command("python -m build --sdist --wheel")
    if result is not None:
        print("✅ Package build successful")

        # Check if files were created
        dist_files = list(Path("dist").glob("*"))
        if dist_files:
            print(f"✅ Generated files: {[f.name for f in dist_files]}")
            return True
        else:
            print("❌ No files generated in dist/")
            return False
    else:
        print("❌ Package build failed")
        return False


def test_editable_install():
    """Test editable installation."""
    print("\n📦 Testing editable installation...")

    result = run_command("pip install -e .")
    if result is not None:
        print("✅ Editable installation successful")

        # Test import
        try:
            import binning

            print(f"✅ Package import successful (version: {binning.__version__})")
            return True
        except ImportError as e:
            print(f"❌ Package import failed: {e}")
            return False
    else:
        print("❌ Editable installation failed")
        return False


def check_legacy_files():
    """Check for legacy files that should be removed after migration."""
    legacy_files = ["setup.py", "setup.cfg", "MANIFEST.in"]
    found_legacy = []

    for file in legacy_files:
        if Path(file).exists():
            found_legacy.append(file)

    if found_legacy:
        print(f"\n⚠️  Legacy files found: {found_legacy}")
        print("Consider removing these after verifying the migration is complete")
        return False
    else:
        print("\n✅ No legacy files found")
        return True


def verify_dependencies():
    """Verify that dependencies are correctly specified."""
    print("\n📋 Verifying dependencies...")

    try:
        import tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        deps = config.get("project", {}).get("dependencies", [])
        optional_deps = config.get("project", {}).get("optional-dependencies", {})

        print(f"✅ Main dependencies: {len(deps)}")
        print(f"✅ Optional dependency groups: {list(optional_deps.keys())}")

        # Check if essential dependencies are present
        essential = ["numpy", "scipy", "scikit-learn"]
        missing = []
        for dep in essential:
            if not any(dep in d for d in deps):
                missing.append(dep)

        if missing:
            print(f"⚠️  Missing essential dependencies: {missing}")
            return False
        else:
            print("✅ All essential dependencies present")
            return True

    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False


def main():
    """Run all migration verification checks."""
    print("🚀 Binning Package Migration Verification")
    print("=" * 50)

    checks = [
        ("pyproject.toml validation", check_pyproject_toml),
        ("Build tools availability", check_build_tools),
        ("Dependencies verification", verify_dependencies),
        ("Package build test", test_package_build),
        ("Editable install test", test_editable_install),
        ("Legacy files check", check_legacy_files),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n🔍 {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 MIGRATION VERIFICATION SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 Migration verification completed successfully!")
        print("Your package is ready for the modern Python ecosystem!")
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please review and fix the issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
