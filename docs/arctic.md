# Bimanual category-agnostic reconstruction (WIP!!)

**Motivation**: Humans interact with various objects daily, making holistic 3D capture of these interactions crucial for modeling human behavior. Recently, HOLD has shown promise in category-agnostic hand-object reconstruction but is limited to single-hand interaction. To address the natural use of both hands, we introduce the **bimanual category-agnostic reconstruction** task, where participants must reconstruct both hands and the object in 3D from a video clip without relying on pre-scanned templates. This task is more challenging due to severe hand-object occlusion and dynamic hand-object contact in bimanual manipulation. 

What you will find in this document:

- A HOLD baseline for two-hand manipulation settings.
    - Use 9 videos from ARCTIC dataset's rigid object collection (used for HANDS2024 competition).
    - One video per object, excluding small objects like scissors and phones.
    - Videos sourced from the test set.
- Instructions to reproduce our HOLD baseline.
- Code to evaluate on our ARCTIC test set as well as a custom evaluation set.

> IMPORTANT⚠️: If you're participating in our HANDS2024 challenge, sign up on the workshop website to join our mailing list, as all important information will be communicated through it.

<!-- Assuming that you learned how to train, visualize, and preprocess a custom single-hand sequence, to get started on this bimanual task, we suggest the following steps:

1. Download our preprocessed ARCTIC clips to train a model per sequence (see below)
1. (Unavailable yet) Upload your prediction (the `arctic_preds.zip` file) to our leaderboard to reproduce the results:

```bash
cat /tmp/tmpx3qzcyz9/avg_results.json
{	'cd_h': 109.34839048857026, 
	'cd_icp': 2.100823341816449, 
	'cd_l': 102.9745147216744, 
	'cd_r': 115.72226625546607, 
	'f10_icp': 65.87858542756224, 
	'f5_icp': 41.574190964361314, 
	'mpjpe_ra_h': 25.872253841824, 
	'mpjpe_ra_l': 27.116696039835613, 
	'mpjpe_ra_r': 24.627809312608505, 
	'timestamp': '07-13 17:38'}
``` -->



## Training using our preprocessed sequences

Here we have preprocessed ARCTIC clips for you to get started. You can download the ARCTIC clips and pre-trained HOLD models with this command:

```bash
./bash/arctic_downloads.sh
python scripts/unzip_download.py
mkdir -p code/logs
mkdir -p code/data

mv unpack/arctic_ckpts/* code/logs/
mv unpack/arctic_data/* code/data/
cd code
```

This should put the pre-trained HOLD models under `./code/logs` and the ARCTIC clips using `./code/data`.

To visualize pre-trained checkpoints (for example, our baseline run `5c224a94e`), you can use our visualization script:

```bash
python visualize_ckpt.py --ckpt_p logs/5c224a94e/checkpoints/last.ckpt --ours
```

To train HOLD on an ARCTIC sequence, following HOLD's full pipeline (pre-train, pose refinement, fully-train), you can use the following:

```bash
pyhold train.py --case $seq_name --num_epoch 100 --shape_init 75268d864 # this yield exp_id 
pyhold optimize_ckpt.py --write_gif --batch_size 51 --iters 600  --ckpt_p logs/$exp_id/checkpoints/last.ckpt
pyhold train.py --case $seq_name --num_epoch 200 --load_pose logs/$exp_id/checkpoints/last.pose_ref --shape_init 75268d864 # this yield another exp_id
```

See more details on [usage](usage.md).

## Training using your own preprocessing method

[Here](custom_arctic.md) we show an example of how preprocessing was done. We observed that in general, the higher accuracy of preprocessed hand and object poses are, the better reconstruction quality HOLD has. Therefore, you are encouraged to have your own preprocessing method so long as you use the same set of images from the previous step. You can also follow this to preprocess any custom sequences that are not in the test set (for example, in case you need more examples for publications).

## Evaluation on ARCTIC

### Online evaluation (ARCTIC test set)

Since ARCTIC test set is hidden, you cannot find subject 3 ground-truth annotations here. To evaluate on subject 3, you can submit `arctic_preds.zip` to our [evaluation server](https://arctic-leaderboard.is.tuebingen.mpg.de/) following the submission instructions below. 

Then you can export the prediction for each experiment (indicated by `exp_id`) via:

```bash
pyhold scripts_arctic/extract_preds.py --sd_p logs/$exp_id/checkpoints/last.ckpt
```

This will dump the prediction of the model for the experiment `exp_id` under `./arctic_preds`. To submit to our online server, you must extract predictions for all sequences.

For example, here we extract all predictions of the baseline checkpoints:

```bash
pyhold scripts_arctic/extract_preds.py --sd_p logs/5c224a94e/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/f44e4bf8f/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/09c728594/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/0cc49e42c/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/8239a3dcb/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/a961b659b/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/4052f966a/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/cf4b38269/checkpoints/last.ckpt
pyhold scripts_arctic/extract_preds.py --sd_p logs/1c1fe8646/checkpoints/last.ckpt
```

Finally, package the predictions:

```bash
zip -r arctic_preds.zip arctic_preds
```

Submit this zip file to our [leaderboard](https://arctic-leaderboard.is.tuebingen.mpg.de/leaderboard) for online evaluation. 

### Offline evaluation on non-test-set seqs

Suppose that you want to evaluate offline on sequences that are not in the test set. For example, you may need more sequence evaluation for a paper or you may want to analyze your method quantitatively in details. In that case, you need to prepare ARCTIC groundtruth for your sequence of interest. 

First, download the ARCTIC dataset following instructions [here](https://github.com/zc-alexfan/arctic). Place the arctic data with this folder structure:

```bash
./ # code folder
./arctic_data/arctic/images
./arctic_data/arctic/meta
./arctic_data/arctic/raw_seqs
./arctic_data/models/smplx/SMPLX_FEMALE.npz
./arctic_data/models/smplx/SMPLX_NEUTRAL.npz
./arctic_data/models/smplx/SMPLX_MALE.npz
```

Suppose that you want to evaluate on `s05/box_grab_01`, you can prepare the groundtruth file via:

```bash
python scripts_arctic/process_arctic.py --mano_p arctic_data/arctic/raw_seqs/s05/box_grab_01.mano.npy
```

Modify the sequence to evaluate on in `evaluate_on_arctic.py`:

As an example, this shows to evaluate on subject 5 with sequence name `box_grab_01` for the view `1`:

```python
test_seqs = [
    'arctic_s05_box_grab_01_1', 
    'arctic_s05_waffleiron_grab_01_1', 
]
```

Run the evaluate with: 

```bash
python evaluate_on_arctic.py --zip_p ./arctic_preds.zip --output results
```
