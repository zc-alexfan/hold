# Getting Started

General Requirements:

- Ubuntu 20.04.6 LTS
- Python 3.8
- torch 1.9.1
- CUDA 11.1 (check nvcc --version)
- pytorch3d 0.7.4
- pytorch-lightning 1.5.7
- aitviewer 1.13.0

These requirements are non-strict. However, they have been tested on our end. Therefore, it is a good starting point. 

## Preliminary

**PyTorch Lightning**: To avoid boilerplate code, we use [pytorch lightning (PL)](https://pytorch-lightning.readthedocs.io/en/1.5.7/common/trainer.html) to handle the main logic for training and evaluation. Feel free to consult the documentation, should you have any questions.

**Comet logger**: To keep track of experiments and visualize results, our code logs experiments using [`comet.ml`](https://comet.ml). If you wish to use own logger service, you mostly modify the code in `common/comet_utils.py`. This code is only meant as a guideline; you are free to modify it to whatever extent you deem necessary.

To configure the comet logger, you need to first register an account and create a private project. An API code will be provided for you to log the experiment. Then you export the API code and the workspace ID:

```bash
export COMET_API_KEY="your_api_key_here"
export COMET_WORKSPACE="your_workspace_here"
```

It might be a good idea to add these commands to your `~/.bashrc` file, so you don't have to load the environment every time you login to your machine. Add these lines to the end of `~/.bashrc`.

Each experiment is tracked with a 9-character ID. When the training procedure starts, a random ID (e.g., `837e1e5b2`) is assigned to the experiment and a folder (e.g., `logs/837e1e5b2`) to save information on this folder.

```bash
sudo apt-get install ffmpeg # needed to convert rendered files to mp4
```

### CUDA

Before starting, check your CUDA `nvcc` version:

```bash
nvcc --version # should be 11.1
```

You can install nvcc and cuda via [runfile](https://developer.nvidia.com/cuda-11-1-0-download-archive). If `nvcc --version` is still not `11.1`, check whether you are referring the right nvcc with `which nvcc`. Assuming you have an NVIDIA driver installed, usually, you only need to run the following command to install `nvcc` (as an example):

```bash
sudo bash cuda_11.1.0_510.39.01_linux.run --toolkit --silent --override
```

After the installation, make sure the paths pointing to the current cuda toolkit location. For example:

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export CPATH="/usr/local/cuda-11.1/include:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64/"
```

## HOLD environment

Install packages: 

```shell
# -- conda
ENV_NAME=hold_env
conda create -n $ENV_NAME python=3.8
conda activate $ENV_NAME

# -- requirements
pip install -r requirements.txt
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# -- torch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# -- submodules
mkdir submodules && cd submodules

# --- pytroch3d
git clone https://github.com/facebookresearch/pytorch3d.git 
cd pytorch3d
git checkout 35badc08
python setup.py install
cd ..

# --- kaolin
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.10.0
python setup.py install
cd ../../

# -- hold custom c++ extensions (mise)
cd ./code/
python setup.py build_ext --inplace
cd ..

# --- smplx (custom)
cd submodules
git clone https://github.com/zc-alexfan/smplx.git
cd smplx
python setup.py install
cd ../..

# -- override non-compatible packages
pip install setuptools==59.5.0
pip install numpy==1.23.5
pip install scikit-image==0.18.1
```

Setup viewer:

```bash
# aitviewer
conda create --name aitviewer python=3.8.1
conda activate aitviewer
bash ./bash/setup_viewer.sh
```

Clean up:

```bash
find submodules -delete
```

Create shortcuts. See example below:

```bash
alias cdroot='cd ~/hold/generator'
alias pyhold='~/miniconda3/envs/hold_env/bin/python'
alias pyait='~/miniconda3/envs/aitviewer/bin/python'
```

## Download data and models

This section provides instructions on downloading data and models used in our HOLD project.

⚠️ Register accounts on [HOLD](https://hold.is.tue.mpg.de/register.php), and [MANO](https://mano.is.tue.mpg.de/), and then export your the username and password with the following commands:

```bash
export HOLD_USERNAME=<ENTER_YOUR_HOLD_EMAIL>
export HOLD_PASSWORD=<ENTER_YOUR_HOLD_PASSWD>
export MANO_USERNAME=<ENTER_YOUR_MANO_EMAIL>
export MANO_PASSWORD=<ENTER_YOUR_MANO_PASSWD>
```

Before starting, check if your credentials are exported correctly (following the commands above).

```bash
echo $HOLD_USERNAME
echo $HOLD_PASSWORD
echo $MANO_USERNAME
echo $MANO_PASSWORD
```

⚠️ If the echo is empty, `export` your credentials following the instructions above before moving forward.

Run each command below to download your files of interest:

```bash
conda activate hold_env
bash ./bash/download_mandatory.sh # files that are mandatory
bash ./bash/download_ckpts.sh # models trained on videos (optional)
bash ./bash/download_data.sh # pre-processsed data (optional)
python scripts/checksum.py # verify checksums
python scripts/unzip_download.py # unzip downloads
```

Unpack downloads:

```bash
bash ./bash/setup_files.sh
```

Now you should have all dependencies needed to train HOLD on preprocessed data. See potential sequences via `ls code/data`. For example, you can run the following command to train on a given in-the-wild sequence: 

```bash
cd code
seq_name=hold_bottle1_itw
python train.py --case $seq_name --eval_every_epoch 1 --num_sample 64
```

If you have OOM issue, you can decrease `--num_sample` for lower memory requirements but it might impact performance. If no errors raised with the command above, your HOLD environment is good to go. 

If you need to run on HO3D, setup HO3D data following [here](ho3d.md). 

## External dependencies

If you want to reconstruct a custom video sequence with HOLD, you will need to setup the following dependencies. Here we provide tested instructions to install them. For additional installation related issues, refer to the original repo.

Create independent environments:

```bash
cd ./generator
bash ./install/conda.sh
```

Install dependencies:

```bash
# -- Segment-and-Track-Anything (required)
conda activate sam-track
bash ./install/sam.sh 
cd Segment-and-Track-Anything
python app.py # this should run with no errors
cd ..

# -- Hierarchical-Localization (required)
conda activate hloc
bash ./install/hloc.sh 

# -- hand_detector.d2 (right-hand only)
conda activate 100doh
bash ./install/100doh.sh 

# -- MeshTransformer (right-hand only)
conda activate metro
bash ./install/metro.sh 
```

If you want to reconstruct two-hand videos, install CUDA 11.7 (required by detectron2 in hamer) via [runfile](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux) and source them: 

```bash
sudo bash cuda_11.7*.run --toolkit --silent --override
export CUDA_HOME=/usr/local/cuda-11.7
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export CPATH="/usr/local/cuda-11.7/include:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64/"
```

Install hamer:

```bash
conda activate hamer
bash ./install/hamer.sh 
```

## Source paths

Create aliases for the installed dependencies. See example below:

```bash
alias pymetro='~/miniconda3/envs/metro/bin/python'
alias pyhamer='~/miniconda3/envs/hamer/bin/python'
alias pysam='~/miniconda3/envs/sam-track/bin/python'
alias pydoh='~/miniconda3/envs/100doh/bin/python'
alias pycolmap='~/miniconda3/envs/hloc/bin/python'
```

Feel free to put them inside your `~/.zshrc` or `~/.bashrc` depending on your shell. 

By default, `python` refers to `pyhold` in all documentations for simplicity.
