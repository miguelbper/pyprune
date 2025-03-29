# List all available recipes
default:
    just --list

# Check that all programs are installed
[group("installation")]
check-versions:
    uv --version
    just --version
    direnv --version

# Allow direnv to load environment variables
[group("installation")]
direnv-allow:
    direnv allow

# Create uv virtual environment
[group("installation")]
create-venv:
    uv sync

# Install pre-commit hooks
[group("installation")]
install-pre-commit:
    uv run pre-commit install

# Setup environment variables (reminder)
[group("installation")]
reminder-env-vars:
    @echo "\033[1;33mRemember to setup the environment variables by editing the .envrc file!\033[0m"

# Setup repo
[group("installation")]
setup: direnv-allow create-venv install-pre-commit reminder-env-vars

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

# Increment version (+ trigger a release GitHub actions workflow)
[group("packaging")]
increment-version:
    #!/usr/bin/env bash
    # Get current version from pyproject.toml
    CURRENT_VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)

    # Split version into major.minor.patch
    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

    # Increment patch version
    NEW_PATCH=$((PATCH + 1))
    NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

    # Update version in pyproject.toml
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
    uv sync

    # Create git tag
    git add pyproject.toml
    git commit -m "Update version to $NEW_VERSION"
    git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
    git push --follow-tags origin main
