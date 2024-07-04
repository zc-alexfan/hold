# Data documentation

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



