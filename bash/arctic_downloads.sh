#!/bin/bash
set -e

echo "Downloading ARCTIC challenge: data"
mkdir -p downloads/arctic_data

python scripts/download.py --url_file ./bash/assets/urls/arctic_data.txt --out_folder downloads/arctic_data

echo "Downloading ARCTIC challenge: HOLD baseline"
mkdir -p downloads/arctic_ckpts
python scripts/download.py --url_file ./bash/assets/urls/arctic_ckpts.txt --out_folder downloads/arctic_ckpts
