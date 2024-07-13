# ARCTIC benchmark using HOLD

## Data preparation for [HANDS2024 competition](https://hands-workshop.org/challenge2024.html#challenge2)

To benchmark this challenge, we adapt HOLD to two-hand manipulation settings and use 9 videos from ARCTIC dataset's rigid object collection, one per object (excluding small objects such as scissors and phone), and sourced from the test set for this challenge. Here we provide instructions to reproduce our HOLD baseline, and to produce data to upload to our evaluation server for test set evaluation.

Download ARCTIC clips and HOLD checkpoints:

```bash
./bash/arctic_downloads.sh
python scripts/unzip_download.py
mkdir -p code/logs
mkdir -p code/data

mv unpack/arctic_ckpts/* code/logs/
mv unpack/arctic_data/* code/data/
```

This should put the pre-trained HOLD models under `./code/logs` and the ARCTIC clips using `./code/data`.

To visualize pre-trained checkpoints of HOLD (for example, `5c224a94e`), you can use our visualization script:

```bash
python visualize_ckpt.py --ckpt_p logs/5c224a94e/checkpoints/last.ckpt --ours
```

To train HOLD on an ARCTIC sequence, you can use the following:

```bash
pyhold train.py --case $seq_name --num_epoch 100 --shape_init 75268d864 # this yield exp_id 
pyhold optimize_ckpt.py --write_gif --batch_size 51 --iters 300  --ckpt_p logs/$exp_id/checkpoints/last.ckpt
pyhold train.py --case $seq_name --num_epoch 200 --load_pose logs/$exp_id/checkpoints/last.pose_ref --shape_init 75268d864 # this yield another exp_id
```

See more details on [usage](docs/usage.md).


## Evaluation on ARCTIC

Since ARCTIC test set is hidden, you cannot find subject 3 ground-truth annotations here. To evaluate on subject 3, you can submit `arctic_preds.zip` to our [evaluation server](https://arctic-leaderboard.is.tuebingen.mpg.de/) following the submission instructions below. 

Then you can export the prediction for each experiment (indicated by `exp_id`) via:

```bash
pyhold scripts_arctic/export_arctic.py --sd_p logs/$exp_id/checkpoints/last.ckpt
```

This will dump the prediction of the model for the experiment `exp_id` under `./arctic_preds`. 

Finally, package the predictions:

```bash
zip -r arctic_preds.zip arctic_preds
```

