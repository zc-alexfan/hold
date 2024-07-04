#!/bin/bash
set -e

mkdir -p downloads/ckpts

echo "Downloading checkpoints ..."
python scripts/download.py --url_file ./bash/assets/urls/ckpts.txt --out_folder downloads/ckpts