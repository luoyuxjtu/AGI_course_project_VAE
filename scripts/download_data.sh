#!/usr/bin/env bash
# Download COCO 2017 train and val images to data/coco2017/.
#
# Run from the project root:
#   bash scripts/download_data.sh
#
# Sizes (approximate):
#   train2017.zip  ~18 GB  →  ~118,000 images
#   val2017.zip     ~1 GB  →    ~5,000 images
#
# After completion the layout is:
#   data/coco2017/train2017/*.jpg
#   data/coco2017/val2017/*.jpg
#
# Re-running is safe: each split is skipped if its directory already exists.
set -e

DEST="data/coco2017"
mkdir -p "$DEST"

# ------------------------------------------------------------------ #
# Helper: download one zip and extract it                              #
# ------------------------------------------------------------------ #
download_split() {
    local url="$1"
    local zip_name="$2"
    local split_dir="$3"
    local zip_path="$DEST/$zip_name"
    local out_dir="$DEST/$split_dir"

    if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null)" ]; then
        echo "  $split_dir already exists and is non-empty — skipping download."
        return
    fi

    echo ""
    echo "=== Downloading $zip_name ==="
    wget --show-progress -O "$zip_path" "$url"

    echo "=== Extracting $zip_name → $DEST/ ==="
    unzip -q "$zip_path" -d "$DEST"
    rm "$zip_path"

    echo "  Done: $out_dir  ($(ls "$out_dir" | wc -l | tr -d ' ') files)"
}

# ------------------------------------------------------------------ #
# Download train and val splits                                        #
# ------------------------------------------------------------------ #
download_split \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "train2017.zip" \
    "train2017"

download_split \
    "http://images.cocodataset.org/zips/val2017.zip" \
    "val2017.zip" \
    "val2017"

echo ""
echo "Dataset ready at $DEST"
echo "  train2017: $(ls "$DEST/train2017" 2>/dev/null | wc -l | tr -d ' ') images"
echo "  val2017:   $(ls "$DEST/val2017"   2>/dev/null | wc -l | tr -d ' ') images"
