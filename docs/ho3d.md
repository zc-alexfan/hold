# Setup HO3D data

Since we cannot redistribute HO3D, the HO3D dataset you downloaded from us do not contain RGB images or 3D annotations. If you want to train on HO3D, download HO3Dv3 [here](https://github.com/shreyashampali/ho3d) and put into this path `./downloads/ho3d_v3/HO3D_v3.zip`.

Unpack HO3D:

```bash
# under root directory
mkdir -p downloads/ho3d_v3
cd downloads/ho3d_v3
unzip HO3D_v3.zip
cd ../..

mkdir -p generator/assets
mv downloads/ho3d_v3/ generator/assets/
```

Preprocess HO3D annotation: 

```bash
python scripts/process_ho3d.py
```

Copy missing HO3D RGB images:

```bash
python scripts/copy_ho3d_frames.py
```

Finally, put the HO3D/YCB models under `./generator/assets/ho3d_v3/models` so that for each object we have `./generator/assets/ho3d_v3/models/{obj_name}/textured_simple.obj`.

