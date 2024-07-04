from glob import glob
from PIL import Image
import os
def create_seq(sid, seq_name, view):
    seq_name_view = f'{sid}/{seq_name}/{view}'
    out_seq_name = 'arctic_' +  seq_name_view.replace('/', '_')
    src_folder = f'/home/zfan/arctic/data/arctic_data/data/images/{seq_name_view}'
    fnames = sorted(glob(src_folder + '/*.jpg'))
    fnames = fnames[100:400]
    out_folder = f"./data/{out_seq_name}/raw_images"
    for i, fname in enumerate(fnames):
        img = Image.open(fname)
        out_p = f"{out_folder}/{os.path.basename(fname)}"
        os.makedirs(os.path.dirname(out_p), exist_ok=True)  
        img.save(out_p)
    os.makedirs(f"./data/{out_seq_name}/processed/sam/right", exist_ok=True)
    os.makedirs(f"./data/{out_seq_name}/processed/sam/left", exist_ok=True)
    os.makedirs(f"./data/{out_seq_name}/processed/sam/object", exist_ok=True)
seqs = [
"box_grab_01",
"capsulemachine_grab_01",
"espressomachine_grab_01",
"ketchup_grab_01",
"laptop_grab_01",
"microwave_grab_01",
"mixer_grab_01",
"notebook_grab_01",
"phone_grab_01",
"scissors_grab_01",
"waffleiron_grab_01",
]
views = list(range(1, 9))
from tqdm import tqdm

for seq in tqdm(seqs):
    for view in views:
        create_seq('s03', seq, view)
