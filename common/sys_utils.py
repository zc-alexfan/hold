import subprocess
import os
import pickle
import argparse
from glob import glob
import pygit2
from tqdm import tqdm
import numpy as np
import os.path as op
import cv2 as cv
import logging
import time

from loguru import logger

def copy(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    copy(s, d)
                else:
                    with open(s, 'rb') as source_file:
                        with open(d, 'wb') as dest_file:
                            contents = source_file.read()
                            dest_file.write(contents)
        else:
            with open(src, 'rb') as source_file:
                with open(dst, 'wb') as dest_file:
                    contents = source_file.read()
                    dest_file.write(contents)


def copy_repo_folder(src_files, dst_folder, filter_keywords):
    src_files = [
        f for f in src_files if not any(keyword in f for keyword in filter_keywords)
    ]
    dst_files = [op.join(dst_folder, op.basename(f)) for f in src_files]
    for src_f, dst_f in zip(src_files, dst_files):
        logger.info(f"FROM: {src_f}\nTO:{dst_f}")
        copy(src_f, dst_f)


class Timer:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.elapse = time.time() - self.start_time

    def reset(self):
        self.start_time = None
        self.elapse = -1


def fetch_logger(filename, filemode="w"):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=filename, mode=filemode)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def get_branch():
    return str(pygit2.Repository(".").head.shorthand)


def get_commit_hash():
    return str(pygit2.Repository(".").head.target)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def mkdir_p(exp_path):
    os.makedirs(exp_path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def count_files(path):
    """
    Non-recursively count number of files in a folder.
    """
    files = glob(path)
    return len(files)


def get_host_name():
    results = subprocess.run(["cat", "/etc/hostname"], stdout=subprocess.PIPE)
    return str(results.stdout.decode("utf-8").rstrip())


class Email:
    def __init__(self, address, password, default_recipient):
        import yagmail

        self.client = yagmail.SMTP(address, password)
        self.email_address = address
        self.email_password = password
        self.default_recipient = default_recipient

    def notify(self, subject, body):
        if "bdb.BdbQuit" in body:
            return
        self.client.send(self.default_recipient, subject, body)
        print("Email sent.")


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def fetch_lmdb_reader(db_path):
    import lmdb

    env = lmdb.open(
        db_path,
        subdir=op.isdir(db_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = env.begin(write=False)
    return txn


def read_lmdb_image(txn, fname):
    import lmdb

    image_bin = txn.get(fname.encode("ascii"))
    if image_bin is None:
        return image_bin
    image = np.fromstring(image_bin, dtype=np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


def package_lmdb(lmdb_name, map_size, fnames, keys, write_frequency=5000):
    import lmdb

    """
    Package image files into a lmdb database.
    fnames are the paths to each file and also the key to fetch the images.
    lmdb_name is the name of the lmdb database file
    map_size: recommended to set to len(fnames)*num_types_per_image*10
    keys: the key of each image in dict
    """
    assert len(fnames) == len(keys)
    db = lmdb.open(lmdb_name, map_size=map_size)
    txn = db.begin(write=True)
    for idx, (fname, key) in tqdm(enumerate(zip(fnames, keys)), total=len(fnames)):
        img = cv.imread(fname)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        status, encoded_image = cv.imencode(".png", img, [cv.IMWRITE_JPEG_QUALITY, 100])
        assert status
        txn.put(key.encode("ascii"), encoded_image.tostring())

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_pyarrow(fnames))
        txn.put(b"__len__", dumps_pyarrow(len(fnames)))
    db.sync()
    db.close()


def copy_repo(dst_repo_p):
    dst_folder = dst_repo_p

    if not op.exists(dst_folder):
        logger.info("Copying repo")
        src_files = glob("./*")
        os.makedirs(dst_folder)

        logger.info("Copying ../common")
        src_common = op.join('..', 'common')
        dst_common = op.join(dst_folder, '..', 'common')
        copy(src_common, dst_common)

        logger.info("Copying repo files")
        filter_keywords = [".ipynb", ".obj", ".pt", "run_scripts", ".sub", ".txt", 'generator', 'lightning_logs']
        copy_repo_folder(src_files, dst_folder, filter_keywords)
        logger.info("Done")
