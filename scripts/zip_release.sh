#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/zip_release.sh [output.zip]
# Default output: release.zip

OUT=${1:-release.zip}

EXCLUDE=(
  "venv/*"
  "__pycache__/*"
  "logs/*"
  "output/*"
)

# Create zip excluding patterns
zip -r "$OUT" . -x "${EXCLUDE[@]}"

echo "Created $OUT (excluded: ${EXCLUDE[*]})"
