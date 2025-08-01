#!/bin/bash
# release.sh - Automated release script with dynamic versioning

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log_error "Not in a git repository!"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_error "You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Get current version info
CURRENT_VERSION=$(python -m setuptools_scm)
log_info "Current version: $CURRENT_VERSION"

# Parse command line arguments
RELEASE_TYPE=${1:-patch}  # major, minor, patch, alpha, beta, rc

case $RELEASE_TYPE in
    major|minor|patch)
        log_info "Creating $RELEASE_TYPE release..."
        ;;
    alpha|beta|rc)
        log_info "Creating $RELEASE_TYPE pre-release..."
        ;;
    *)
        log_error "Invalid release type: $RELEASE_TYPE"
        log_info "Usage: $0 [major|minor|patch|alpha|beta|rc]"
        exit 1
        ;;
esac

# Get the last tag
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
log_info "Last tag: $LAST_TAG"

# Calculate next version based on type
calculate_next_version() {
    local last_tag=$1
    local release_type=$2
    
    # Remove 'v' prefix if present
    local version=${last_tag#v}
    
    # Split version into components
    IFS='.' read -ra VERSION_PARTS <<< "$version"
    local major=${VERSION_PARTS[0]}
    local minor=${VERSION_PARTS[1]:-0}
    local patch=${VERSION_PARTS[2]:-0}
    
    case $release_type in
        major)
            echo "v$((major + 1)).0.0"
            ;;
        minor)
            echo "v${major}.$((minor + 1)).0"
            ;;
        patch)
            echo "v${major}.${minor}.$((patch + 1))"
            ;;
        alpha)
            echo "v${major}.${minor}.$((patch + 1))a1"
            ;;
        beta)
            echo "v${major}.${minor}.$((patch + 1))b1"
            ;;
        rc)
            echo "v${major}.${minor}.$((patch + 1))rc1"
            ;;
    esac
}

# Calculate next version
NEXT_VERSION=$(calculate_next_version "$LAST_TAG" "$RELEASE_TYPE")
log_info "Next version will be: $NEXT_VERSION"

# Confirm with user
read -p "Proceed with release $NEXT_VERSION? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "Release cancelled."
    exit 0
fi

# Run tests
log_info "Running tests..."
if ! python -m pytest; then
    log_error "Tests failed! Aborting release."
    exit 1
fi
log_success "Tests passed!"

# Check code quality
log_info "Checking code quality..."
if ! ruff check binning/; then
    log_error "Code quality checks failed! Aborting release."
    exit 1
fi
log_success "Code quality checks passed!"

# Build package
log_info "Building package..."
if ! python -m build; then
    log_error "Package build failed! Aborting release."
    exit 1
fi
log_success "Package built successfully!"

# Create and push tag
log_info "Creating tag $NEXT_VERSION..."
git tag "$NEXT_VERSION"
git push origin "$NEXT_VERSION"
log_success "Tag created and pushed!"

# Show final version
FINAL_VERSION=$(python -m setuptools_scm)
log_success "Release completed! Final version: $FINAL_VERSION"

# Optional: Upload to PyPI
read -p "Upload to PyPI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Uploading to PyPI..."
    python -m twine upload dist/*
    log_success "Uploaded to PyPI!"
fi

log_success "🎉 Release $NEXT_VERSION completed successfully!"
