# Data documentation

## Experiment folders under `./logs`

Each run is saved under an experiment id (for example, here it is `56c1a54d2`):

```bash
56c1a54d2: Experiment ID
56c1a54d2/args.json: Commit, branch information of the code
56c1a54d2/misc: Information needed for pose refinement and visualize_ckpt.py
56c1a54d2/mesh_cano: Hand and object templates in the canonical space
56c1a54d2/checkpoints: Checkpoints containing model weights and pose parameters for hands and objects
56c1a54d2/visuals: visualizations
56c1a54d2/visuals/right.normal: Right hand normal rendering
56c1a54d2/visuals/rgb: RGB rendering
56c1a54d2/visuals/right.mask_prob: Right hand mask rendering
56c1a54d2/visuals/object.mask_prob: Object mask rendering
56c1a54d2/visuals/object.fg_rgb.vis: Object RGB rendering without the background
56c1a54d2/visuals/imap: Instance segmentation rendering
56c1a54d2/visuals/mask_prob: Foreground mask rendering
56c1a54d2/visuals/normal: Foreground normal rendering
56c1a54d2/visuals/fg_rgb.vis: Foreground RGB rendering
56c1a54d2/visuals/right.fg_rgb.vis: Right hand rendering without background
56c1a54d2/visuals/bg_rgb: Background-only rendering
56c1a54d2/visuals/object.normal: object normal rendering
56c1a54d2/train.log: loguru outputts
```

## Sequence folder

- `./data/$seq_name/`: folder to stored artifacts and built dataset for a sequence named `$seq_name`
- `./data/$seq_name/build`: the build dataset, all needed files to run HOLD is here. 
- `./data/$seq_name/images`: images extracted to train HOLD; names of images are the original names in the video.
- `./data/$seq_name/images.zip`: zip of `images`, used by SAM to label segmentation
- `./data/$seq_name/processed`: all intermediate data for preprocessing stored here
- `./data/$seq_name/video.mp4`: input video
- `./data/$seq_name/build/corres.txt`: orignal names of the image files here; correspondances
- `./data/$seq_name/build/data.npy`: packaged data such as camera information, estimated hand-object poses, etc. 
- `./data/$seq_name/build/image`: images being resized and renamed for training HOLD.
- `./data/$seq_name/build/mask`: segmentation masks from SAM but preprocssed to merge hand and object masks.
- `./data/$seq_name/build/vis`: visualization of the build; this allows a sanity check of the processing; not needed for HOLD run. 
- `./data/$seq_name/processed/2d_keypoints`: extracted 2D keypoints for hands by projecting 3D to 2D. 
- `./data/$seq_name/processed/boxes.npy`: bounding boxes detected around the hand
- `./data/$seq_name/processed/colmap`: colmap intermediate results
- `./data/$seq_name/processed/colmap_2d`: 2D projection of COLMAP 3D pointcloud from SfM.
- `./data/$seq_name/processed/crop_image`: images crops from bounding box detector.
- `./data/$seq_name/processed/hold_fit.init.npy`: MANO parameters from hand pose estimation
- `./data/$seq_name/processed/hold_fit.slerp.npy`: SLERP for linear interpolation of poses in missing frames from init
- `./data/$seq_name/processed/hold_fit.aligned.npy`: Hand-object parameters after energy minimization to estimate object scale and align hand-object with contact
- `./data/$seq_name/processed/images_object`: images created by using object masks, used by SfM.
- `./data/$seq_name/processed/j2d.crop.npy`: 2D hand keypoints in crop space
- `./data/$seq_name/processed/j2d.full.npy`: 2D hand keypoints in original image space
- `./data/$seq_name/processed/mano_fit_ckpt`: Checkpoints created during the energy minimization process in hand-object alignment. 
- `./data/$seq_name/processed/hpe_vis`: HAMER hand pose estimation visualization (3D to 2D)
- `./data/$seq_name/processed/masks`: segmentation masks preprocessed from SAM masks.
- `./data/$seq_name/processed/mesh_fit_vis`: visualization of hand mesh registration
- `./data/$seq_name/processed/metro_vis`: rendering of hand overlay onto RGB images of METRO hand pose estimation
- `./data/$seq_name/processed/raw_images`: all images decoded from video
- `./data/$seq_name/processed/sam`: SAM results for segmentation
- `./data/$seq_name/processed/v3d.npy`: MANO vertices from hand pose estimator
- `./data/$seq_name/processed/colmap/intrinsic.npy`: intrinsics from COLMAP (same for all frames)
- `./data/$seq_name/processed/colmap/normalization_mat.npy`: normalization matrix to center the COLMAP point cloud and make it unit length
- `./data/$seq_name/processed/colmap/o2w.npy`: object canonical space to world transformation (a.k.a object poses)
- `./data/$seq_name/processed/colmap/pairs-netvlad.txt`: image frames that converged during SfM; non-converged ones are filled with SLERP
- `./data/$seq_name/processed/colmap/poses.npy`: camera poses (same for all frames as we assume a fixed camera pose)
- `./data/$seq_name/processed/colmap/sparse_points.ply`: raw 3D point clouds from SfM
- `./data/$seq_name/processed/colmap/sparse_points_trim.ply`: removed outliers
- `./data/$seq_name/processed/colmap/sparse_points_normalized.obj`: normalize pointclouds

The below shows the documentation of `./data/$seq_name/build/data.npy`: 

- `seq_name`: name of the dataset folder
- `cameras`: camera view matrix and scaling matrix (always the same for all frames)
- `scene_bounding_sphere`: float; size of the bounding sphere
- `max_radius_ratio`: float; max radius ratio
- `cameras/scale_mat_0`: 4x4; scaling matrix at frame 0 to normalize the scene in a unit sphere
- `cameras/world_mat_0`: 4x4; view matrix at frame 0
- `entities`: foreground nodes in the scene to render (e.g., hands and objects)
- `entities/right`: right hand parameters
- `entities/right/hand_poses`: Tx48; right hand parameters in axis-angles
- `entities/right/hand_trans`: Tx3; right hand translation
- `entities/right/mean_shape`: 10; mean shape of hand
- `entities/object`: object parameters
- `entities/object/obj_scale`: float; estimated scale of object
- `entities/object/pts.cano`: Nx3; object SfM 3D point cloud in canonical space
- `entities/object/norm_mat`: 4x4; normalization matrix to center and scale COLMAP point cloud to the canonical space
- `entities/object/object_poses`: Tx6; first three global rotation in axis-angle, last three translation; object poses from COLMAP

## Checkpoints

| Dataset | Sequence Name                        | Checkpoint |
|---------|--------------------------------------|------------|
| HO3D    | hold_BB12_ho3d                       |  4d0175b3c |
| HO3D    | hold_BB13_ho3d                       |  32f545e48 |
| HO3D    | hold_GSF12_ho3d                      |  db6508d7f |
| HO3D    | hold_GSF13_ho3d                      |  76fbd4d33 |
| HO3D    | hold_ABF12_ho3d                      |  81a2bea9a |
| HO3D    | hold_ABF14_ho3d                      |  20b7fc070 |
| HO3D    | hold_GPMF12_ho3d                     |  00bc6dc5e |
| HO3D    | hold_GPMF14_ho3d                     |  64834e9bb |
| HO3D    | hold_MC1_ho3d                        |  cb20a1702 |
| HO3D    | hold_MC4_ho3d                        |  c8d39e1aa |
| HO3D    | hold_MDF12_ho3d                      |  fd873a597 |
| HO3D    | hold_MDF14_ho3d                      |  28ab63ba1 |
| HO3D    | hold_ShSu10_ho3d                     |  c2316a5be |
| HO3D    | hold_ShSu12_ho3d                     |  14680ffbf |
| HO3D    | hold_SM2_ho3d                        |  d1281c169 |
| HO3D    | hold_SM4_ho3d                        |  b7c26b798 |
| HO3D    | hold_SMu1_ho3d                       |  6ba784f2d |
| HO3D    | hold_SMu40_ho3d                      |  524dcb8d4 |
| HOLD    | hold_bottle1_itw                     |  009c2e923 |
| HOLD    | hold_bottle2_itw                     |  16d067709 |
| HOLD    | hold_kettle1_itw                     |  b76b7f42e |
| HOLD    | hold_mug1_itw                        |  5f1656837 |
| HOLD    | hold_rubricube1_itw                  |  91d2cd532 |
| HOLD    | hold_rubricube2_itw                  |  6abf1a5ae |
| HOLD    | hold_toycar1_itw                     |  2f71e5d77 |
| HOLD    | hold_toycar2_itw                     |  5fdcfc03f |
| ARCTIC  | arctic_s03_box_grab_01_1             | 25e67133f  |
| ARCTIC  | arctic_s03_mixer_grab_01_1           | c4f488427  |
| ARCTIC  | arctic_s03_capsulemachine_grab_01_1  | f3784c2a2  |
| ARCTIC  | arctic_s03_espressomachine_grab_01_1 | a25e55a10  |
| ARCTIC  | arctic_s03_ketchup_grab_01_1         | a7d4f510c  |
| ARCTIC  | arctic_s03_laptop_grab_01_1          | a1fcfb334  |
| ARCTIC  | arctic_s03_microwave_grab_01_1       | a7733f8cd  |
| ARCTIC  | arctic_s03_notebook_grab_01_1        | 2718c9369  |
| ARCTIC  | arctic_s03_waffleiron_grab_01_1      | b98627ca3  |
