#!/usr/bin/env bash
# Download and extract Imagenette 160px to data/imagenette2-160/.
#
# Run from the project root:
#   bash scripts/download_data.sh
#
# After completion the dataset layout is:
#   data/imagenette2-160/train/<class>/<image>.JPEG
#   data/imagenette2-160/val/<class>/<image>.JPEG
set -e

DEST="data/imagenette2-160"
ARCHIVE="data/imagenette2-160.tgz"
URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# Skip download if dataset is already present
if [ -d "$DEST" ]; then
    echo "Dataset already exists at $DEST — skipping download."
    exit 0
fi

mkdir -p data

echo "Downloading imagenette2-160 (~100 MB) ..."
wget --show-progress -O "$ARCHIVE" "$URL"

echo "Extracting to data/ ..."
tar -xzf "$ARCHIVE" -C data/
rm "$ARCHIVE"

echo "Done.  Dataset ready at $DEST"
