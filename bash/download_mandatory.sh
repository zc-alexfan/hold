#!/bin/bash
set -e

mkdir -p downloads/mandatory

echo "Downloading mandatory files..."
python scripts/download.py --url_file ./bash/assets/urls/mandatory.txt --out_folder downloads/mandatory

echo "Downloading MANO models..."
python scripts/download.py --url_file ./bash/assets/urls/mano.txt --out_folder downloads/mandatory