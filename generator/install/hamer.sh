set -e
cd hamer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e ".[all]" 
pip install -v -e third-party/ViTPose
pip install pillow==9.1.0
# bash fetch_demo_data.sh
cd ..
mkdir -p hamer/_DATA/data/mano
cp -r ../code/body_models/* hamer/_DATA/data/mano