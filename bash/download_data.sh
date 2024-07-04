#!/bin/bash
set -e

mkdir -p downloads/data

echo "Downloading mandatory files..."
python scripts/download.py --url_file ./bash/assets/urls/data.txt --out_folder downloads/data