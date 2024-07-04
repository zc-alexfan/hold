
## Preprocessing for ARCTIC benchmark

This document gives instructions to preprocess an ARCTIC video sequences for the HOLD baseline. It mainly serves as a guideline. You may modify it for better ARCTIC two-hand reconstruction. 

> ⚠️ We require several dependencies to create a custom sequence. See the [setup page](setup.md) for details before moving on from here. 

### Pipeline overview

Overall, the preprocessing pipeline is as follows:

0. Create dataset
1. Image segmentation
2. Hand pose estimation
3. Object pose estimation
4. Hand-object alignment
5. Build dataset

This is the same for single-hand and two-hand cases. This preprocessing pipeline yields different artifacts. The created files and folders are explained in the [data documentation page](data_doc.md).

### Create dataset

Given a new sequence called `hold_grape_0404`, we first create a folder for it, copy it to the folder, and parse its frames:

```bash
cdroot
seq_name=hold_grape_0404
mkdir -p ./data/$seq_name
cp my/video/path/video.mp4 ./data/$seq_name/video.mp4
```

Extract images from video: 

```
python scripts/init_dataset.py --video_path ./data/$seq_name/video.mp4 --skip_every 2 # extract every 2 frames
```

The option `--skip_every` allows you to downsample frames if the video is too long.

### Segmentation

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
- After the tracking is complete, you can copy the files under `./generator/Segment-and-Track-Anything/tracking_results/images/*` to the desination path (see below).

The destination path has been created when you parsed the video:

```bash
./data/$seq_name/processed/sam/object
```

After copying the segmentation files, we expect file structure like this:

```bash
➜  cd ./data/$seq_name/processed/sam/object; ls
images_masks
```

Now we repeat the same process to label the hand(s) and save results to the corresponding folder. After you have all masks, the command below will merge them and create object-only images:

```bash
cdroot; pyhold scripts/validate_masks.py --seq_name $seq_name
```

### Hand pose estimation

Since HAMER has hand detection, we can directly estimate 3D left and right hand poses. Run the commands below to estimate hand meshes and register MANO to them:

```bash
cdroot; cd hamer
pyhamer demo.py --seq_name $seq_name --batch_size=2  --full_frame --body_detector regnety
```

Register MANO model to predicted meshes: 

```bash
cdroot
pyhold scripts/register_mano.py --seq_name $seq_name --save_mesh #--hand_type right  --use_beta_loss
```

Note: If your video has only 1 hand, you can use ``--hand_type right` or `--hand_type left` to register the corresponding hand; If not specified, this option will default to registering left and right hands. The rest of the code will behave differently based on the number of hand types registered in this step. The flag `--use_beta_loss` encourages the hand shape to be near zero and often has faster convergence.

After registeration, run this to linearly interpolate missing frames:

```bash
pyhold scripts/validate_hamer.py --seq_name $seq_name
```

### Object pose estimation

Run HLoc to obtain object pose and point cloud:

```bash
cdroot; pycolmap scripts/colmap_estimation.py --num_pairs 40 --seq_name $seq_name
```

### Hand-object alignment

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

In our CVPR experiments, we use the same loss weights for all sequences, but you can adjust the fitting weights here (`confs/generic.yaml`) if your sequence does not work out of the box.

Warning⚠️: This visualization is usually the final step for quality assurance. Ideally, you will expect perfect object point cloud 2D reprojection, a reasonable scale of the object point cloud in side view, hand location is roughly near the object. If they all look good, it is good to build the dataset for training.

### Build dataset

Finally, we have all the artifacts needed. We can compile them into a dataset: 

```bash
cdroot; pyhold scripts/build_dataset.py --seq_name $seq_name --no_fixed_shift --rebuild
```

This "compilation" creates a "build" of the dataset under `./data/$seq_name/build/`. Files within "build" is all you need for HOLD to train. It also packs all needed data into a zip file, which you can transfer to your remote cluster to train HOLD on.

## Train on ARCTIC

Train on ARCTIC sequence with name `seq_name`:

```bash
pyhold train.py --case $seq_name --num_epoch 100 --shape_init 5c09be8ac # this yield exp_id 
pyhold optimize_ckpt.py --write_gif --batch_size 51 --iters 300  --ckpt_p logs/$exp_id/checkpoints/last.ckpt
pyhold train.py --case $seq_name --num_epoch 200 --load_pose logs/$exp_id/checkpoints/last.pose_ref --shape_init 5c09be8ac # this yield another exp_id
```

See more details on [usage](docs/usage.md).

## Evaluation on ARCTIC

Since ARCTIC test set is hidden, you cannot find subject 3 ground-truth annotations here. To evaluate on subject 3, you can submit `arctic_preds.zip` to our [evaluation server](https://arctic-leaderboard.is.tuebingen.mpg.de/) following the submission instructions below. 

(Skip if evaluated online): However, if you wish to evaluate locally for other subjects, you need to first convert raw ARCTIC annotations to meshes and joints via the following script. It will save the processed meshes under `./arctic_data/processed/`:

```bash
pyhold scripts_arctic/process_arctic.py
```

Then you can export the prediction for each experiment (indicated by `exp_id`) via:

```bash
pyhold scripts_arctic/export_arctic.py --sd_p logs/$exp_id/checkpoints/last.ckpt
```

This will dump the prediction of the model for the experiment `exp_id` under `./arctic_preds`. 

Finally, package the predictions:

```bash
zip -r arctic_preds.zip arctic_preds
```

and evaluate:

```bash
python scripts_arctic/evaluate_arctic.py 
```

