from PIL import Image
import pickle as pkl
import matplotlib.pyplot as plt
from smplx import MANO
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os.path as op
from smplx import MANO
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle




import sys
sys.path  = ['.'] + sys.path
from common.transforms import convert_gl2cv


def build_mano(is_rhand):
    return MANO(
        "./code/body_models",
        create_transl=False,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=is_rhand,
    )


def process_subject(subject_id, subject_path, out_folder):

    img_folder = "rgb"
    param_folder = "meta"

    curr_subj_path = osp.join(subject_path, subject_id, img_folder)

    # load fnames
    curr_frames = [
        osp.join(curr_subj_path, fname) for fname in sorted(os.listdir(curr_subj_path))
    ]

    param_path = osp.join(subject_path, subject_id, param_folder)

    hand_pose_list = []
    hand_beta_list = []
    hand_transl_list = []
    K_list = []
    obj_trans_list = []
    obj_rot_list = []
    model_r = build_mano(is_rhand=True)
    hand_mean = model_r.hand_mean#.numpy()    
    fnames = []

    for fname in curr_frames:
        param_p = op.join(param_path, op.basename(fname).replace(".jpg", ".pkl"))
        assert op.exists(param_p)
        with open(param_p, "rb") as f:
            data = pickle.load(f)
        
        if data["handPose"] is not None:
            hand_pose = torch.FloatTensor(data["handPose"]).reshape(1, -1)
            hand_pose[0, 3:] -= hand_mean
        else:
            hand_pose = None
#             import pdb; pdb.set_trace()
        if data["handBeta"] is not None:
            hand_beta = torch.FloatTensor(data["handBeta"]).reshape(1, -1)
        else:
            hand_beta = None
            
        if data["handTrans"] is not None:
            hand_transl = torch.FloatTensor(data["handTrans"]).reshape(1, -1)
        else:
            hand_transl = None
            
        if data["handJoints3D"] is not None:
            j3d = torch.FloatTensor(data["handJoints3D"]).reshape(1, -1, 3)
            
        else:
            j3d = None
        
        if data["camMat"] is not None:
            K = torch.FloatTensor(data["camMat"]).reshape(3, 3)
        else:
            K = None
        obj_name = data["objName"]
        
        if data['objTrans'] is not None:
            obj_trans = torch.FloatTensor(data["objTrans"]).reshape(1, -1)
            obj_rot = torch.FloatTensor(data["objRot"]).reshape(1, -1)
            obj_rot_mat = axis_angle_to_matrix(obj_rot)
        else:
            obj_trans = None
            obj_rot = None
            obj_rot_mat = None

        hand_pose_list.append(hand_pose)
        hand_beta_list.append(hand_beta)
        hand_transl_list.append(hand_transl)
        K_list.append(K)
        obj_trans_list.append(obj_trans)
        obj_rot_list.append(obj_rot_mat)
        fnames.append(fname)

        


    def replace_none_with_nan(tensor_list):
        shape = next(item.shape for item in tensor_list if item is not None)
        return [torch.full(shape, float('nan')) if item is None else item for item in tensor_list]

    hand_pose_list = torch.stack(replace_none_with_nan(hand_pose_list)).squeeze()
    hand_beta_list = torch.stack(replace_none_with_nan(hand_beta_list)).squeeze()
    hand_transl_list = torch.stack(replace_none_with_nan(hand_transl_list)).squeeze()
    K_list = torch.stack(replace_none_with_nan(K_list)).squeeze()
    obj_trans_list = torch.stack(replace_none_with_nan(obj_trans_list)).squeeze()
    obj_rot_list = torch.stack(replace_none_with_nan(obj_rot_list)).squeeze()

    hand_valid = (~torch.isnan(hand_pose_list.mean(dim=1))).float()
    obj_valid = (~torch.isnan(obj_trans_list.mean(dim=1))).float()
    assert (obj_valid != hand_valid).sum() == 0
    is_valid = hand_valid

#         hand_pose_list = torch.stack(hand_pose_list).squeeze()
#         hand_beta_list = torch.stack(hand_beta_list).squeeze()
#         hand_transl_list = torch.stack(hand_transl_list).squeeze()
#         K_list = torch.stack(K_list).squeeze()
#         obj_trans_list = torch.stack(obj_trans_list).squeeze()
#         obj_rot_list = torch.stack(obj_rot_list).squeeze()


    #     hand_rot = axis_angle_to_matrix(hand_pose_list[:, :3])
    #     hand_rot_cv, hand_transl_cv = convert_gl2cv(hand_rot.numpy(), hand_transl_list.numpy())
    #     hand_rot_cv = matrix_to_axis_angle(torch.FloatTensor(hand_rot_cv))
    #     hand_transl_cv = torch.FloatTensor(hand_transl_cv)
    #     import pdb; pdb.set_trace()        
    #     hand_pose_list[:, :3] = hand_rot_cv

    out = {}
    out["hand_pose"] = hand_pose_list
    out["hand_beta"] = hand_beta_list
#     out["hand_transl"] = hand_transl_cv
    out["hand_transl"] = hand_transl_list
    out["K"] = K_list
    out["obj_trans"] = obj_trans_list
    out["obj_rot"] = obj_rot_list
    out["obj_name"] = obj_name
    out['is_valid'] = hand_valid
    out["fnames"] = fnames



    out_p = op.join(out_folder, subject_id + ".pt")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    torch.save(out, out_p)
    print('saved:')
    print(out_p)


if __name__ == "__main__":
    split = "train"
    key = "evaluation" if "val" in split else "train"
    data_folder = "./generator/assets/ho3d_v3/"
    out_folder = './generator/assets/ho3d_v3/processed/'
    subject_path = osp.join(data_folder, key)
    subject_ids = sorted(os.listdir(subject_path))

    for subject_id in tqdm(subject_ids):
        process_subject(subject_id, subject_path, out_folder)





