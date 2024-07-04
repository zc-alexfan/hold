import os
import os.path as op
import zipfile
from glob import glob

from tqdm import tqdm


def unzip(zip_p, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(out_dir)


def main():
    fnames = glob(op.join("downloads/", "**/*"), recursive=True)
    all_zips = [fname for fname in fnames if ".zip" in fname]
    out_dir = "./unpack/"
    os.makedirs(out_dir, exist_ok=True)

    # unzip zip files
    for zip_p in all_zips:
        if "images_zips" in zip_p:
            out_dir = zip_p.replace('downloads/', 'unpack/').replace('.zip', '')
            out_dir = out_dir.replace('/images_zips/', '/images/')
        else: 
            out_dir = op.dirname(zip_p).replace('downloads/', 'unpack/')
        print(f"Unzipping {zip_p} to {out_dir}")
        unzip(zip_p, out_dir)


if __name__ == "__main__":
    main()
