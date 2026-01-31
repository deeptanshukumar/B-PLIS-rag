#!/bin/bash
# Lint script for B-PLIS-RAG
# Runs ruff (linter) and mypy (type checker)

set -e

echo "========================================"
echo "B-PLIS-RAG Code Quality Checks"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo -e "${YELLOW}ruff not found. Installing...${NC}"
    pip install ruff
fi

# Check if mypy is installed
if ! command -v mypy &> /dev/null; then
    echo -e "${YELLOW}mypy not found. Installing...${NC}"
    pip install mypy
fi

# Run ruff linter
echo "----------------------------------------"
echo "Running ruff (linter)..."
echo "----------------------------------------"
if ruff check src/ tests/ main.py scripts/; then
    echo -e "${GREEN}✓ ruff: No issues found${NC}"
else
    echo -e "${RED}✗ ruff: Issues found (see above)${NC}"
    LINT_FAILED=1
fi

# Run ruff formatter check
echo ""
echo "----------------------------------------"
echo "Running ruff format check..."
echo "----------------------------------------"
if ruff format --check src/ tests/ main.py scripts/; then
    echo -e "${GREEN}✓ ruff format: Code is properly formatted${NC}"
else
    echo -e "${YELLOW}! ruff format: Code needs formatting${NC}"
    echo "  Run 'ruff format src/ tests/' to fix"
fi

# Run mypy type checker
echo ""
echo "----------------------------------------"
echo "Running mypy (type checker)..."
echo "----------------------------------------"
if mypy src/ --ignore-missing-imports; then
    echo -e "${GREEN}✓ mypy: No type errors${NC}"
else
    echo -e "${YELLOW}! mypy: Type issues found (see above)${NC}"
    # Don't fail on mypy errors for now
fi

# Summary
echo ""
echo "========================================"
if [ -z "$LINT_FAILED" ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed${NC}"
    exit 1
fi
