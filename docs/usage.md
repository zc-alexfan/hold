# Usage

Here we document examples to train HOLD models, evaluate and visualize the results using source code under `./code/` on preprocessed dataset such as HO3D. 

## Training

Here we detail some options in the `argparse` parsers we use. There are other options in the argparser. You can check `src/utils/parsers` for more details.

- `-f`: Run on a mini training and validation set (i.e., a dry-run). 
- `--mute`: Do not create a new experiment on the remote comet logger.
- `--case`: preprocessed sequence to train on
- `--log_every`: moving average window for loss tracking
- `--shape_init`: hand shape pre-trained network ID under `code/saved_models/`
- `--num_sample`: number of 3D points for each node in the scene to sample along the ray
- `--num_epoch`: number of epochs to train
- `--tempo_len`: number of pairs of images to train per epoch
- `--barf_s`: iteration step to start turning on BARF
- `--barf_e`: iteration step at which the BARF embedding will equal to fouriere positional embeddings
- `--no_barf`: do not use BARF low-pass positional embeddings
- `--offset`: at each `dataset.__getitem__` we randomly sample pairs of images from a sequence that are `offset` away from each other in the video
- `--no_meshing`: turn off marching cube
- `--no_vis`: turn off rendering
- `--load_pose`: load hand and object pose and scale parameters, but do not load network weights
- `--eval_every_epoch`: epoch intervals to run evaluation loop

For example, here we train on an in-the-wild (`itw`) sequence called `hold_bottle1_itw` for 200 epochs (default) and render every 1 epoch. At each step, we sample 64 points along each ray (`--num_sample`) for each of the hand and object. However, we do not log the experiment online on Comet (`--mute`): 

```bash
seq_name=hold_bottle1_itw
pyhold train.py --case $seq_name --eval_every_epoch 1 --num_sample 64 --mute
```

If you want to have a dry-run on a HO3D sequence `hold_MC1_ho3d` for debugging purpose, use the flag `-f`. You can also disable online logging with `--mute` or visulization with `--no_vis`:

```bash
seq_name=hold_MC1_ho3d
pyhold train.py --case $seq_name --eval_every_epoch 1 --num_sample 64 -f --mute --no_vis
```

Since `-f` is enabled, the following values are set for fast debugging: 

```bash
args.num_workers = 0
args.eval_every_epoch = 1
args.num_sample = 8
args.tempo_len = 50
args.log_every = 1
```

Each experiment is tracked with a 9-character ID. When the training procedure starts, a random ID (e.g., `837e1e5b2`) is assigned to the experiment and a folder (e.g., `logs/837e1e5b2`) to save information on this folder. 


## Visualization: Checkpoint viewer

Given a checkpoint from the previous training, you can visualize the results with our checkpoint visualizer that is built with the [very-advanced AIT viewer](https://github.com/eth-ait/aitviewer). 

<p align="center">
    <img src="./static/aitviewer.gif" alt="Image" width="80%"/>
</p>

In particular, run this command:

```bash
exp_id=cb20a1702 # replace with any experiment id
pyait visualize_ckpt.py --ckpt_p logs/$exp_id/checkpoints/last.ckpt --ours
```

You can also visualize the groundtruth:

```bash
seq_name=hold_MC1_ho3d # folder name in ./data/; this is required for GT viewing
exp_id=cb20a1702
pyait visualize_ckpt.py --ckpt_p logs/$exp_id/checkpoints/last.ckpt --gt_ho3d --seq_name $seq_name
```

Or both:

```bash
seq_name=hold_MC1_ho3d # folder name in ./data/; this is required for GT viewing
exp_id=cb20a1702
pyait visualize_ckpt.py --ckpt_p logs/$exp_id/checkpoints/last.ckpt --gt_ho3d --ours --seq_name $seq_name
```

[AITViewer](https://github.com/eth-ait/aitviewer) has lots of useful builtin controls. For an explanation of the frontend and control, visit [here](https://eth-ait.github.io/aitviewer/frontend.html). Here we assume you are in interactive mode (`--headless` is turned off).

- To play/pause the animation, hit `<SPACE>`.
- To center around an object, click the mesh you want to center, press `X`.
- To go between the previous and the current frame, press `<` and `>`.

More documentation can be found in [aitviewer github](https://github.com/eth-ait/aitviewer) and in [viewer docs](https://eth-ait.github.io/aitviewer/frontend.html).


We provided the pre-trained models in our CVPR papers. Their experiment hashcode and the sequence name pairs can be found in [`docs/data_doc.md`](data_doc.md).

## Full HOLD pipeline

The `Training` section above only to document how to use the `train.py` script. It does not perform the proper full training on HOLD. Recall, HOLD consists of three stages: pre-training, pose refinement, and a final training stage. To do that you can run the following commands:

```bash
seq_name=hold_MC1_ho3d
pyhold train.py --case $seq_name --num_epoch 100 --shape_init 5c09be8ac # this yield exp_id
pyhold optimize_ckpt.py --write_gif --batch_size 51 --iters 300  --ckpt_p logs/$exp_id/checkpoints/last.ckpt
pyhold train.py --case $seq_name --num_epoch 200 --load_pose logs/$exp_id/checkpoints/last.pose_ref --shape_init 5c09be8ac # this yield another exp_id
```

The code by default use pre-trained hand shape network for two-hand setting (`--shape_init 75268d864`), which has different weights than what we used in CVPR (`--shape_init 5c09be8ac`).

Note that each step above creates independent checkpoints, you can visualize results at each step with our checkpoint viewer for sanity check:

```bash
exp_id=aaaaaaaaa # checkpoint from pre-training
pyait visualize_ckpt.py --ckpt_p logs/$exp_id/checkpoints/last.ckpt --ours
pyait visualize_ckpt.py --ckpt_p logs/$exp_id/checkpoints/last.pose_ref --ours
exp_id=bbbbbbbbb # checkpoint from final training
pyait visualize_ckpt.py --ckpt_p logs/$exp_id/checkpoints/last.ckpt --ours
```

Important flags of `optimize_ckpt.py`:

- `--write_gif`: Dump the optimization results to gif for visualization
- `--batch_size`: Number of frames to optimize at once
- `--iters`: number of iterations to optimize each batch on
- `--inspect_idx`: only optimize on 1 batch for debugging; the batch is chosen by specifying a frame idx that is within this batch

## Rendering sequence

To render the visualization for the entire sequence you can use the following:

```bash
python render.py --case $seq_name  --load_ckpt logs/$exp_id/checkpoints/last.ckpt  --mute --agent_id -1 --render_downsample 4
```

Flags:

- `--render_downsample 4`: downsample the image by 4 times for faster rendering
- `--agent_id -1`: if it is `-1`, all frames will be rendered by this program; otherwise it is the node id that is rendering multiple batches of frames in paralellel in a cluster. For example, `--agent_id 0` means this program will run for the first batch by node `0` in a cluster.

Encode as mp4 files: 

```bash
bash ./create_videos.sh $exp_id
```

## Evaluation

To evaluate on HO3D, simply provide the experiment ID and run our evaluation code. For example, if your final training ID is `524dcb8d4` you can run: 

```bash
pyhold evaluate.py --sd_p logs/524dcb8d4/checkpoints/last.ckpt
```

This will provide the results as a json file. 

For example, 

```bash
(hold_env) ➜  code git: ✗ cat logs/524dcb8d4/checkpoints/last.ckpt.metric.json 
{
    "cd_icp": 0.9491690920636109,
    "cd_ra": 1.9679582405194538,
    "cd_right": 7.160760344183221,
    "f10_icp": 90.74656674714542,
    "f10_ra": 72.52117287466974,
    "f10_right": 51.067675348112715,
    "f5_icp": 79.25239562342499,
    "f5_ra": 36.541843512664556,
    "f5_right": 25.163817812580625,
    "mpjpe_ra_r": 19.590365779194137,
    "mrrpe_ho": 24.78926374328456,
    "timestamp": "06-30 21:09",
    "seq_name": "hold_SMu40_ho3d"
}
```

Here, `mpjpe_ra_r`, `cd_icp`, `f10_icp`, `cd_right` are `MPJPE`, `CD`, `F10`, and `CD_h` in the paper respectively. Suppose that you have evaluated multiple sequences, you can use average them via: 

```bash
python summarize_metrics.py $exp_id1 $exp_id2 ...
```

For example:

```bash
python summarize_metrics.py 81a2bea9a 20b7fc070 524dcb8d4 
```
