set -e
cd hand_detector.d2

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install opencv-python gdown
pip install Pillow==9.5.0 

gdown https://drive.google.com/uc\?id\=1OqgexNM52uxsPG3i8GuodDOJAGFsYkPg
mkdir -p models
mv model_0529999.pth models
cd ..