#!/usr/bin/env bash
# Download and extract Imagenette 160px to data/
set -e

mkdir -p data
cd data

echo "Downloading imagenette2-160..."
wget -q --show-progress https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz

echo "Extracting..."
tar -xzf imagenette2-160.tgz
rm imagenette2-160.tgz

echo "Done. Dataset at: data/imagenette2-160/{train,val}/<class>/<image>.JPEG"
