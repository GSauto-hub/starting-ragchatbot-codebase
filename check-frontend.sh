#!/usr/bin/env bash
# Frontend code quality checks
set -e

FRONTEND_DIR="$(dirname "$0")/frontend"

echo "=== Frontend Quality Checks ==="
echo ""

# Check if node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install --prefix "$FRONTEND_DIR"
fi

echo "Running Prettier format check..."
npx --prefix "$FRONTEND_DIR" prettier --check "$FRONTEND_DIR/index.html" "$FRONTEND_DIR/style.css" "$FRONTEND_DIR/script.js"
echo "✓ Formatting consistent"

echo ""
echo "=== All checks passed ==="
