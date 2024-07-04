#!/bin/bash
set -e

echo "Downloading full resolution images"
mkdir -p downloads/arctic/images_zips

python scripts/download.py --url_file ./bash/assets/urls/arctic_images.txt --out_folder downloads/arctic/images_zips

echo "Downloading smaller files"
mkdir -p downloads/arctic
python scripts/download.py --url_file ./bash/assets/urls/arctic_misc.txt --out_folder downloads/arctic

echo "Downloading SMPLX"
mkdir -p downloads
python scripts/download.py --url_file ./bash/assets/urls/smplx.txt --out_folder downloads
mv models/smplx ../code/body_models 
cd ..

mv unpack code/arctic_data