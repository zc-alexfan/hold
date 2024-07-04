#!/bin/bash
set -e

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install easydict matplotlib chumpy loguru opencv-python-headless numpy

cd submodules/pytorch3d
git checkout 35badc08
python setup.py install
cd ../..

cd submodules/smplx
python setup.py install
pip install aitviewer==1.13.0
pip install numpy==1.23.5
cd ../..