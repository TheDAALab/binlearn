# GitHub Workflows

This directory contains the CI/CD pipelines for the binning framework.

## Workflows

### 🔨 `build.yml` - Build & Test Pipeline
**Triggers:** Every push and pull request  
**Purpose:** Continuous integration - validates code quality and runs tests

**What it does:**
- ✅ Runs tests across Python 3.8-3.13
- ✅ Performs ruff linting checks
- ✅ Runs mypy type checking  
- ✅ Generates code coverage reports
- ✅ Validates package can be built
- ✅ Provides quality gate for merging

**Use this for:** Day-to-day development and PR validation

### 🚀 `release.yml` - Release Pipeline  
**Triggers:** GitHub releases or manual dispatch  
**Purpose:** Publishes releases to PyPI

**What it does:**
- ✅ Validates release conditions
- ✅ Builds source and wheel distributions
- ✅ Publishes to TestPyPI first
- ✅ Publishes to PyPI
- ✅ Creates GitHub release (for manual triggers)
- ✅ Provides release summary

**Use this for:** Publishing new versions

## How to Use

### For Development (Every Commit)
The build pipeline runs automatically on every push and PR. Just commit your code!

```bash
git add .
git commit -m "Your changes"
git push
```

The build pipeline will:
1. Run all tests across Python versions
2. Check code quality with ruff and mypy
3. Generate coverage reports
4. Validate the package builds correctly

### For Releases

#### Option 1: GitHub Release (Recommended)
1. Go to GitHub → Releases → "Create a new release"
2. Create a new tag (e.g., `v1.0.0`)
3. Fill in release notes
4. Click "Publish release"

The release pipeline will automatically:
1. Build the package
2. Test on TestPyPI
3. Publish to PyPI

#### Option 2: Manual Trigger
1. Go to Actions → Release → "Run workflow"
2. Enter version (e.g., `v1.0.0`)
3. Click "Run workflow"

This will also create a GitHub release for you.

## Environments

Make sure you have these GitHub environments configured:

- **`testpypi`** - Contains `TEST_PYPI_API_TOKEN`
- **`pypi`** - Contains `PYPI_API_TOKEN`

## Migration Notes

- The old `ci.yml` has been backed up as `ci.yml.backup`
- New structure separates concerns: build vs release
- Build pipeline is lighter and faster for development
- Release pipeline is comprehensive for production deployment
