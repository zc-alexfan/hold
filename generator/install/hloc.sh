set -e
cd Hierarchical-Localization

pip install -e .
pip install trimesh
pip uninstall pycolmap
pip install pycolmap==0.4.0
cd ..