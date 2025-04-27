# List all available recipes
default:
    just --list

# Check that all programs are installed
[group("installation")]
check-versions:
    uv --version
    just --version

# Create uv virtual environment
[group("installation")]
create-venv:
    uv sync

# Install pre-commit hooks
[group("installation")]
install-pre-commit:
    uv run pre-commit install

# Setup repo
[group("installation")]
setup: create-venv install-pre-commit

# Run pre-commit hooks
[group("linting & formatting")]
pre-commit:
    uv run pre-commit run --all

# Run tests
[group("testing")]
test:
    uv run pytest

# Run tests with coverage
[group("testing")]
test-cov:
    uv run pytest --cov=pyprune --cov-report=html

# Create a new version tag (will trigger publish.yaml workflow)
[group("packaging")]
publish:
    #!/usr/bin/env bash
    # Get last tag from git
    CURRENT_VERSION=$(git describe --tags --abbrev=0)
    echo "Current version: $CURRENT_VERSION"

    # Remove 'v' prefix if it exists
    VERSION_NUMBER=$(echo $CURRENT_VERSION | sed 's/^v//')

    # Split version into major.minor.patch
    IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION_NUMBER"

    # Increment patch version
    NEW_PATCH=$((PATCH + 1))
    NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

    # Create git tag (always with 'v' prefix)
    NEW_TAG="v$NEW_VERSION"
    echo "New version: $NEW_TAG"

    # Create and push the new tag
    git tag -a "$NEW_TAG" -m "Release version $NEW_VERSION"
    git push origin "$NEW_TAG"
