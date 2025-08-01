# Dynamic Versioning with setuptools_scm - Complete Guide

## 🏷️ Version Information

Your binning package now uses dynamic versioning with setuptools_scm. Here's what you get:

### Current Version Status
- **Version**: 0.2.1.dev0+gd7882f2.d20250801
- **Status**: Development (dev0 = 0 commits ahead of v0.2.0 tag)
- **Commit**: d7882f2 (short hash)
- **Date**: 20250801 (August 1, 2025)

## 🚀 Usage Scenarios

### 1. Development Workflow
```bash
# Make changes and commit
git add .
git commit -m "Add new feature"

# Version automatically becomes: 0.2.1.dev1+g<new-hash>.d20250801
python -m setuptools_scm
```

### 2. Pre-release Workflow
```bash
# Create alpha release
git tag v0.3.0a1
python -m setuptools_scm  # → 0.3.0a1

# Create beta release  
git tag v0.3.0b1
python -m setuptools_scm  # → 0.3.0b1

# Create release candidate
git tag v0.3.0rc1
python -m setuptools_scm  # → 0.3.0rc1
```

### 3. Release Workflow
```bash
# Use the automated release script
./scripts/release.sh patch    # Creates v0.2.1
./scripts/release.sh minor    # Creates v0.3.0  
./scripts/release.sh major    # Creates v1.0.0

# Or manually
git tag v0.3.0
python -m setuptools_scm     # → 0.3.0 (exact)
```

### 4. Build and Deploy
```bash
# Build with dynamic version
python -m build

# Install with dynamic version
pip install -e .

# Version in code automatically matches
python -c "import binning; print(binning.__version__)"
```

## 🔧 CI/CD Integration

### GitHub Actions Features
- ✅ Full history fetch (`fetch-depth: 0`) for accurate versioning
- ✅ Automatic version detection in builds
- ✅ Different workflows for dev/release versions
- ✅ TestPyPI for development builds
- ✅ PyPI for tagged releases
- ✅ Automatic GitHub releases

### Version-aware Deployment
```yaml
# Development builds → TestPyPI
if: github.ref == 'refs/heads/main'

# Tagged releases → PyPI  
if: startsWith(github.ref, 'refs/tags/v')
```

## 📊 Version Metadata

The enhanced version system provides detailed information:

```python
import binning._version as v

# Basic version info
print(v.__version__)        # "0.2.1.dev0+gd7882f2.d20250801"
print(v.__version_tuple__)  # (0, 2, 1, 'dev0', 'gd7882f2.d20250801')

# Enhanced metadata
info = v.get_version_info()
print(info['is_release'])      # False (for dev versions)
print(info['is_prerelease'])   # False (True for alpha/beta/rc)
print(info['is_development'])  # True (for dev versions)
```

## 🎯 Best Practices

### Tag Naming Convention
- **Release**: `v1.0.0`, `v1.2.3`
- **Pre-release**: `v1.0.0a1`, `v1.0.0b2`, `v1.0.0rc1`
- **Avoid**: `release-1.0`, `1.0.0` (without 'v' prefix)

### Release Process
1. **Development**: Work on main branch, automatic dev versions
2. **Testing**: Create pre-release tags for testing
3. **Release**: Create final version tag
4. **Hotfix**: Create patch tags from release branches

### CI/CD Configuration
- Always use `fetch-depth: 0` in checkout actions
- Include setuptools_scm in build requirements
- Use version-aware deployment logic
- Test builds before releasing

## 🔄 Migration Benefits

### Before (Manual versioning)
- ❌ Manual version bumping in setup.py
- ❌ Easy to forget version updates
- ❌ Version mismatches between code and tags
- ❌ Complex release workflows

### After (Dynamic versioning)
- ✅ Automatic version management
- ✅ Git tags as single source of truth
- ✅ Automatic dev/pre-release versions
- ✅ CI/CD ready out of the box
- ✅ PEP 440 compliance
- ✅ Detailed version metadata

## 🛠️ Troubleshooting

### Common Issues
1. **No version tags**: Falls back to `fallback_version = "0.0.0"`
2. **Shallow clone**: Use `fetch-depth: 0` in CI
3. **Dirty working tree**: Adds `+dirty` suffix
4. **Wrong tag format**: Use `v` prefix and semantic versioning

### Debug Commands
```bash
# Check current version
python -m setuptools_scm

# Verbose output
python -m setuptools_scm --verbose

# Check git state
git describe --tags --dirty --always
```

Your binning package is now fully modernized with automated versioning! 🎉
