from easydict import EasyDict as edict

object_specs = edict(
    {
        "pose_dim": 0,
        "full_pose_dim": 3,
        "num_full_tfs": 1,
        "num_tfs": 0,
        "total_dim": 1 + 3 + 3,  # 1 (scale) + 3 (trans) + 3 (fullpose)
        "embedding": "barf",
    }
)
