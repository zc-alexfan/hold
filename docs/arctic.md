# ARCTIC benchmark using HOLD (WIP!!!)

Humans interact with various objects daily, making holistic 3D capture of these interactions crucial for modeling human behavior. Most methods for reconstructing hand-object interactions require pre-scanned 3D object templates, which are impractical in real-world scenarios. Recently, HOLD (Fan et al. CVPR'24) has shown promise in category-agnostic hand-object reconstruction but is limited to single-hand interaction.

Since we naturally interact with both hands, we host the bimanual category-agnostic reconstruction task where participants must reconstruct both hands and the object in 3D from a video clip, without relying on pre-scanned templates. This task is more challenging as bimanual manipulation exhibits severe hand-object occlusion and dynamic hand-object contact, leaving rooms for future development.

To benchmark this challenge, we adapt HOLD to two-hand manipulation settings and use 9 videos from ARCTIC dataset's rigid object collection, one per object (excluding small objects such as scissors and phone), and sourced from the test set for this challenge. Here we provide instructions to reproduce our HOLD baseline, and to produce data to upload to our evaluation server for test set evaluation.

> IMPORTANT⚠️: If you are participating in our HANDS2024 challenge, make sure that you've signed up the form on the workshop website to join our mailing list. All important information will be communicated through this mailing list.

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
