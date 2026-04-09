#!/bin/sh
set -eu

mkdir -p "$SHARED_MODELS_DIR"
cp -a "$MODELS_DIR"/. "$SHARED_MODELS_DIR"/

echo "Copied model bundle into $SHARED_MODELS_DIR"
find "$SHARED_MODELS_DIR" -maxdepth 1 -type f -print
