# Bimanual category-agnostic reconstruction

**Motivation**: Humans interact with various objects daily, making holistic 3D capture of these interactions crucial for modeling human behavior. Recently, HOLD has shown promise in category-agnostic hand-object reconstruction but is limited to single-hand interaction. To address the natural use of both hands, we introduce the **bimanual category-agnostic reconstruction** task, where participants must reconstruct both hands and the object in 3D from a video clip without relying on pre-scanned templates. This task is more challenging due to severe hand-object occlusion and dynamic hand-object contact in bimanual manipulation. 

What you will find in this document:

- A HOLD baseline for two-hand manipulation settings.
    - Nine preprocessed clips from ARCTIC dataset's rigid object collection (used for HANDS2024 competition).
    - One clip per object, excluding small objects like scissors and phones.
    - Clips sourced from the test set.
- Instructions to reproduce our HOLD baseline.
- Instructions to evaluate on ARCTIC.

> IMPORTANT⚠️: If you're participating in our HANDS2024 challenge, sign up on the workshop website to join our mailing list, as all important information will be communicated through it.


## Training using our preprocessed sequences

Here we have preprocessed ARCTIC clips for you to get started. You can download the ARCTIC clips and pre-trained HOLD models with this command:

```bash
./bash/arctic_downloads.sh
pyhold scripts/unzip_download.py
mkdir -p code/logs
mkdir -p code/data

mv unpack/arctic_ckpts/* code/logs/
mv unpack/arctic_data/* code/data/

find downloads -delete # clean up

cd code
```

This should put the pre-trained HOLD models under `./code/logs` and the ARCTIC clips using `./code/data`.

To visualize pre-trained checkpoints (for example, our baseline, run `5c224a94e`), you can use our visualization script:

```bash
pyhold visualize_ckpt.py --ckpt_p logs/5c224a94e/checkpoints/last.ckpt --ours
```

To train HOLD on a preprocessed sequence, following HOLD's full pipeline (pre-train, pose refinement, fully-train), you can use the following:

```bash
pyhold train.py --case $seq_name --num_epoch 100 --shape_init 75268d864 # this yield exp_id 
pyhold optimize_ckpt.py --write_gif --batch_size 51 --iters 600  --ckpt_p logs/$exp_id/checkpoints/last.ckpt
pyhold train.py --case $seq_name --num_epoch 200 --load_pose logs/$exp_id/checkpoints/last.pose_ref --shape_init 75268d864 # this yield another exp_id
```

See more details on [usage](usage.md).

## Training using your own preprocessing method

[Here](custom_arctic.md) we demonstrate an example of how preprocessing was performed. We observed that, in general, higher accuracy in preprocessed hand and object poses leads to better reconstruction quality in HOLD. Therefore, you are encouraged to use your own preprocessing method as long as you use the same set of images from the previous step. You can also follow this guide to preprocess any custom sequences that are not in the test set (for example, if you need more examples for publications).

## Evaluation on ARCTIC

### Online evaluation (ARCTIC test set)

Since ARCTIC test set is hidden, you cannot find subject 3 ground-truth annotations here. To evaluate on subject 3, you can submit `arctic_preds.zip` to our [evaluation server](https://arctic-leaderboard.is.tuebingen.mpg.de/) following the submission instructions below. 

First, you can export the prediction for each experiment (indicated by `exp_id`) via:

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

To submit this zip file to our [leaderboard](https://arctic-leaderboard.is.tuebingen.mpg.de/leaderboard) for online evaluation, create an account with your academic address. **If you participate in the ECCV24 HANDS challenge, you should use the same email used when registering for the challenge.**. Also, this is a seperate account from HOLD. After activating your account, you may now log into the website.

After logging in, click "My algorithms" at the top to manage your algorithms. To add one, click "Add algorithm" and enter your algorithm details, with only the "short name" field being mandatory. This information will appear on the leaderboard if published. Algorithm scores remain private by default unless published. Click save to create the algorithm.

IMPORTANT: When an algorithm is created, you can submit on multiple sub-tasks. You do not need to create a separate algorithm for each sub-task

After the step above, you'll reach a page to upload prediction results. Choose upload and select the Bi-CAIR (bimanual category-agnostic interaction reconstruction) task for submission. If it is your first attempt, try to submit the `arctic_preds.zip` from our HOLD baseline to the server side works. This should yield to similar numbers as in the leaderboard. Once you tested that this zip file works, you can try with your own `arctic_preds.zip` file. The evaluation time may vary based on the task and number of requests. You'll see the results in a table, which can be downloaded as a JSON file by clicking "evaluation result".

To avoid excessive hyperparameter tuning on the test set, each account can only submit to the server for 10 successful evaluations in total every month. For the ECCV24 competition, we allow more attempts and please email `zicong.fan@inf.ethz.ch` if you ran out of attempts.



**Documentation for pt files in arctic_preds.zip**: Suppose that NUM_IMAGES is the number of images in each clip and you performed marching cube to obtain object template with NUM_VERTS_OBJECT vertices.

- `fnames`: a list of image paths for the clip, for example, `./data/arctic_s03_box_grab_01_1/build/image/0000.png`; Shape: `NUM_IMAGES`
- `full_seq_name`: sequence name, for example, `arctic_s03_box_grab_01_1` 
- `j3d_ra.right`: predicted right hand 3D joint position (in meters) after subtracted by its root; Shape: `torch.Size([NUM_IMAGES, 21, 3])`
- `j3d_ra.left`: predicted left hand 3D joint position (in meters) after subtracted by its root; Shape: `torch.Size([NUM_IMAGES, 21, 3])`
- `v3d_right.object`: predicted object vertices in camera coordinate (meters) after subtracted by right hand root; Shape: `torch.Size([NUM_IMAGES, NUM_VERTS_OBJECT, 3])`
- `v3d_left.object`: predicted object vertices in camera coordinate (meters) after subtracted by left hand root; Shape: `torch.Size([NUM_IMAGES, NUM_VERTS_OBJECT, 3])`
- `v3d_c.left`: predicted left hand vertices in camera coordinate (meters); Shape: `torch.Size([NUM_IMAGES, 778, 3])`
- `v3d_c.right`: predicted right hand vertices in camera coordinate (meters); Shape: `torch.Size([NUM_IMAGES, 778, 3])`
- `v3d_c.object`: predicted object vertices in camera coordinate (meters); Shape: `torch.Size([NUM_IMAGES, NUM_VERTS_OBJECT, 3])`
- `j3d_c.left`: predicted left hand 3D joint position in camera coordinate (in meters) after subtracted by its root; Shape: `torch.Size([NUM_IMAGES, 21, 3])`
- `j3d_c.right`: predicted right hand 3D joint position in camera coordinate (in meters) after subtracted by its root; Shape: `torch.Size([NUM_IMAGES, 21, 3])`

### Offline evaluation on non-test-set seqs

Suppose that you want to evaluate offline on sequences that are not in the test set. For example, you may need more sequence evaluation for a paper or you may want to analyze your method quantitatively in details. In that case, you need to prepare ARCTIC groundtruth for your sequence of interest. 

First, download the ARCTIC dataset following instructions [here](https://github.com/zc-alexfan/arctic). Place the arctic data with this folder structure (smplx npz can be downloaded [here](https://smpl-x.is.tue.mpg.de/)):

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
pyhold scripts_arctic/process_arctic.py --mano_p arctic_data/arctic/raw_seqs/s05/box_grab_01.mano.npy
```

In `evaluate_on_arctic.py`, you can specify the sequences to be evaluated on. As an example, this shows to evaluate on `arctic_s05_box_grab_01_1` (subject 5 with sequence name `box_grab_01` for the view `1`) as well as `arctic_s05_waffleiron_grab_01_1`:

```python
test_seqs = [
    'arctic_s05_box_grab_01_1', 
    'arctic_s05_waffleiron_grab_01_1', 
]
```

After you save the model predictions to `arctic_preds.zip` (see above) for all sequences. Run the evaluate with: 

```bash
pyhold scripts_arctic/evaluate_on_arctic.py --zip_p ./arctic_preds.zip --output results
```
