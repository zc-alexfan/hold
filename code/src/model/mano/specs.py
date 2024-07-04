from easydict import EasyDict as edict

mano_specs = edict(
    {
        "pose_dim": 45,
        "full_pose_dim": 48,
        "shape_dim": 10,
        "num_full_tfs": 16,
        "num_tfs": 15,
        "total_dim": 62,  # 1 (scale) + 3 (trans) + 48 (fullpose) + 10 (shape)
        "embedding": "fourier",
    }
)
