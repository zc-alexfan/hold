set -e
cd MeshTransformer
git submodule update --init --recursive

# setup folders
mkdir -p ./models  # pre-trained models
mkdir -p ./datasets  # datasets
mkdir -p ./predictions  # prediction outputs


pip install -r requirements.txt
python setup.py build develop
pip install torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy==1.22.1
pip install ./manopth/.

# setup files
cp ../../code/body_models/* metro/modeling/data/
bash scripts/download_models.sh
cd ..