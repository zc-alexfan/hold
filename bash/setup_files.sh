#!/bin/bash
set -e
# -- preprocessed datasets
mv unpack/data code/

# MANO
mkdir -p code/body_models
mv unpack/mandatory/mano_v1_2/models/* code/body_models
mv downloads/mandatory/*.npy code/body_models/
mv downloads/mandatory/*.pkl code/body_models/

# pre-trained hand shapes
mkdir -p code/saved_models
mv unpack/mandatory/5c09be8ac code/saved_models 
mv unpack/mandatory/75268d864 code/saved_models 

# ours trained models
mkdir -p code/logs
mv unpack/ckpts/* code/logs

# clean
find unpack -delete
cd generator; ln -s ../code/data # link folder
cd ..