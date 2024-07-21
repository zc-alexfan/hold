# How to preprocess an ARCTIC sequence

This document gives instructions to preprocess an ARCTIC video sequences for the HOLD baseline. It mainly serves as a guideline. You may modify it for better ARCTIC two-hand reconstruction. 

> ⚠️ We require several dependencies to create a custom sequence. See the [setup page](setup.md) for details before moving on from here. 

## Pipeline overview

Overall, the preprocessing pipeline is as follows:

1. Create dataset
1. Image segmentation
1. Hand pose estimation
1. Object pose estimation
1. Hand-object alignment
1. Build dataset
1. Train HOLD on custom dataset
1. Evaluate trained model

This preprocessing pipeline yields different artifacts. The created files and folders are explained in the [data documentation page](data_doc.md).

## Create dataset

Given a new sequence, for example, `arctic_s05_espressomachine_grab_01_8` (arctic sequence from subject `s05` with sequence name `espressomachine_grab_01` in camera `8`), create the folling folder structure:

```bash
cdroot
seq_name=arctic_s05_espressomachine_grab_01_8

mkdir -p ./data/$seq_name/images
mkdir -p ./data/$seq_name/processed/sam/object
mkdir -p ./data/$seq_name/processed/sam/right
mkdir -p ./data/$seq_name/processed/sam/left
```

If you downloaded HOLD data using our scripts, you can find the clip `arctic_s05_espressomachine_grab_01_8` in the `./data/` directory. Follow the instructions using this clip, which was created by taking a sub-sequence where both hands and the object are visible. To create more custom sequences of your choice, download the original [ARCTIC dataset](https://github.com/zc-alexfan/arctic) and place the images of the subsequence in `./data/$seq_name/images`.

If you want to crop the images with a bounding box, you can use this script:

```bash
python scripts_arctic/crop_arctic_videos.py --input_dir ./data/arctic_s05_espressomachine_grab_01_8/images
```

You will be prompted with a window to draw bounding box to crop and press `c` in the keyboard to crop the video.

For the HANDS2024 challenge test set evaluation, you don't need to extract RGB images, as the selected images are available [here](arctic.md).

## Segmentation

The goal of this step is to extract hand and object masks for the input video. In particular, we use SAM-Track by first selecting the entity of interest in the first video frame. Then SAM-Track will annotate the rest of the video.

Launch SAM-track server to label segmentation for starting frame:

```bash
cdroot; cd Segment-and-Track-Anything
pysam app.py
```

Label the object:

- Open the server page.
- Click `Image-Seq type input`
- Upload the zip version of `./data/$seq_name/images`
- Click `extract`
- Select `Click` and `Positive` to label the object.
- Select `Click` and `Negative` to label region to avoid. 
- Click `Start Tracking`
- After the tracking is complete, you can copy the files under `./generator/Segment-and-Track-Anything/tracking_results/images/*` to the desination path (`./data/$seq_name/processed/sam/*`).

After copying the segmentation files, we expect file structure like this:

```bash
➜  cd ./data/$seq_name/processed/sam/object; ls
images_masks
```

Now we repeat the same process to label the hand(s) and save results to the corresponding folder. After you have all masks, the command below will merge them and create object-only images:

```bash
cdroot; pyhold scripts/validate_masks.py --seq_name $seq_name
```

## Hand pose estimation

Since HAMER has hand detection, we can directly estimate 3D left and right hand poses. Run the commands below to estimate hand meshes and register MANO to them:

```bash
cdroot; cd hamer
pyhamer demo.py --seq_name $seq_name --batch_size=2  --full_frame --body_detector regnety
```

Register MANO model to predicted meshes: 

```bash
cdroot
pyhold scripts/register_mano.py --seq_name $seq_name --save_mesh --use_beta_loss
```

After registeration, run this to linearly interpolate missing frames:

```bash
pyhold scripts/validate_hamer.py --seq_name $seq_name
```

## Object pose estimation

Run HLoc to obtain object pose and point cloud:

```bash
cdroot; pycolmap scripts/colmap_estimation.py --num_pairs 40 --seq_name $seq_name
```

## Hand-object alignment

Since HLoc (SfM) reconstructs object up to a scale, we need to estimate the object scale and align the hand and object in the same space through a fitting process below. Using HLoc intrinsics, we fit the hands such that their 2D projection is consistent with the new intrinsics `--mode h`; We freeze the hand and find the object scale and translations to encourage hand-object contact `--mode o`; Now that object is to scale, we jointly optimize both `--mode ho`.

```bash
cdroot
pyhold scripts/align_hands_object.py --seq_name $seq_name --colmap_k --mode h  --is_arctic --config confs/arctic.yaml
pyhold scripts/align_hands_object.py --seq_name $seq_name --colmap_k --mode o  --is_arctic --config confs/arctic.yaml
pyhold scripts/align_hands_object.py --seq_name $seq_name --colmap_k --mode ho  --is_arctic --config confs/arctic.yaml
```

You can visualize the results at each stage with our custom viewer to debug any fitting issue:

```bash
cdroot
pyait scripts/visualize_fits.py --seq_name $seq_name
```

You can adjust the fitting weights here (`confs/arctic.yaml`) if your sequence does not work out of the box.

⚠️Warning: This visualization is usually the final step for quality assurance. Ideally, you will expect perfect object point cloud 2D reprojection, a reasonable scale of the object point cloud in side view, hand location is roughly near the object. If they all look good, it is good to build the dataset for training.

## Build dataset

Finally, we have all the artifacts needed. We can compile them into a dataset: 

```bash
cdroot; pyhold scripts/build_dataset.py --seq_name $seq_name --no_fixed_shift --rebuild
```

This "compilation" creates a "build" of the dataset under `./data/$seq_name/build/`. Files within "build" is all you need for HOLD to train. It also packs all needed data into a zip file, which you can transfer to your remote cluster to train HOLD on.

## Final remark

Now your sequence is built, you can train, and evaluate locally. See [here](arctic.md) for details.
