from PIL import Image
import os
from tqdm import tqdm
from glob import glob
import os.path as op



def copy_frames(dataset):
    ho3d_seq = dataset.split('_')[1]
    src_dir = f"generator/assets/ho3d_v3/train/{ho3d_seq}/rgb/"
    
    # correspondence
    with open(f'code/data/{dataset}/build/corres.txt', 'r') as f:
        corres = f.readlines()
    corres = sorted([line.strip() for line in corres])
    
    # sanity check
    masks_ps = glob(f'code/data/{dataset}/build/mask/*')
    assert len(corres) == len(masks_ps)
    
    src_ps = [op.join(src_dir, fid) for fid in corres]

    pbar = tqdm(enumerate(src_ps), total=len(src_ps))
    for fid, src_p in pbar:
        im = Image.open(src_p)
        out_p = f'code/data/{dataset}/build/image/{fid:04d}.png'
        os.makedirs(op.dirname(out_p), exist_ok=True)
        pbar.set_description(f"{src_p} -> {out_p}")
        im.save(out_p)
        
if __name__ == "__main__":

    ho3d_datasets = glob('./code/data/*')
    ho3d_datasets = [op.basename(dataset) for dataset in ho3d_datasets if 'ho3d' in dataset]
    ho3d_datasets = [dataset for dataset in ho3d_datasets if '.zip' not in dataset]
    print("Found datasets: ", ho3d_datasets)
    print('---------------------')
    for idx, dataset in enumerate(ho3d_datasets):
        print(f"{idx+1}/{len(ho3d_datasets)}: {dataset}")
        copy_frames(dataset)
