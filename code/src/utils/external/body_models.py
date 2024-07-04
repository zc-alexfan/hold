# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pickle
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from .lbs import blend_shapes, lbs, vertices2joints
from .utils import Struct, to_np, to_tensor
from .vertex_ids import vertex_ids as VERTEX_IDS
from .vertex_joint_selector import VertexJointSelector

ModelOutput = namedtuple(
    "ModelOutput",
    [
        "vertices",
        "faces",
        "joints",
        "full_pose",
        "betas",
        "global_orient",
        "body_pose",
        "expression",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "T",
        "T_weighted",
        "weights",
    ],
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


MANOOutput = namedtuple(
    "MANOOutput",
    [
        "vertices",
        "faces",
        "joints",
        "full_pose",
        "betas",
        "global_orient",
        "hand_pose",
        "T",
        "T_weighted",
        "weights",
        "v_posed",
    ],
)
MANOOutput.__new__.__defaults__ = (None,) * len(MANOOutput._fields)


def forward_layer(
    model,
    betas,
    transl,
    pose,
    global_orient,
    return_verts,
    return_full_pose,
    v_template,
):
    if isinstance(model, MANO):
        output = model.forward(
            betas=betas,
            global_orient=global_orient,
            hand_pose=pose,
            transl=transl,
            return_verts=return_verts,
            return_full_pose=return_full_pose,
            v_template=v_template,
        )
    else:
        raise ValueError("Unknown model type: {}".format(type(model)))
    return output


class SMPL(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10

    def __init__(
        self,
        model_path,
        data_struct=None,
        create_betas=True,
        betas=None,
        create_global_orient=True,
        global_orient=None,
        create_body_pose=True,
        body_pose=None,
        create_transl=True,
        transl=None,
        dtype=torch.float32,
        batch_size=1,
        joint_mapper=None,
        gender="neutral",
        vertex_ids=None,
        pose_blend=True,
        **kwargs,
    ):
        """SMPL model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.gender = gender
        self.pose_blend = pose_blend

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = "SMPL_{}.{ext}".format(gender.upper(), ext="pkl")
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), "Path {} does not exist!".format(smpl_path)

            with open(smpl_path, "rb") as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))
        super(SMPL, self).__init__()
        self.batch_size = batch_size

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS["smplh"]

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs
        )

        self.faces = data_struct.f
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long),
        )

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.NUM_BETAS], dtype=dtype)
            else:
                if "torch.Tensor" in str(type(betas)):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                "betas", nn.Parameter(default_betas, requires_grad=True)
            )

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                if "torch.Tensor" in str(type(global_orient)):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient, requires_grad=True)
            self.register_parameter("global_orient", global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype
                )
            else:
                if "torch.Tensor" in str(type(body_pose)):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose, dtype=dtype)
            self.register_parameter(
                "body_pose", nn.Parameter(default_body_pose, requires_grad=True)
            )

        if create_transl:
            if transl is None:
                default_transl = torch.zeros(
                    [batch_size, 3], dtype=dtype, requires_grad=True
                )
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                "transl", nn.Parameter(default_transl, requires_grad=True)
            )

        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(data_struct.v_template), dtype=dtype)
        )

        # The shape components
        shapedirs = data_struct.shapedirs[:, :, : self.NUM_BETAS]
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer("J_regressor", j_regressor)

        # if self.gender == 'neutral':
        #     joint_regressor = to_tensor(to_np(
        #     data_struct.cocoplus_regressor), dtype=dtype).permute(1,0)
        #     self.register_buffer('joint_regressor', joint_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.bone_parents = to_np(data_struct.kintree_table[0])

        self.register_buffer(
            "lbs_weights", to_tensor(to_np(data_struct.weights), dtype=dtype)
        )

    def create_mean_pose(self, data_struct):
        pass

    @torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_T_hip(self, betas=None):
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        J = vertices2joints(self.J_regressor, v_shaped)
        T_hip = J[0, 0]
        return T_hip

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        return "Number of betas: {}".format(self.NUM_BETAS)

    def forward(
        self,
        betas=None,
        body_pose=None,
        global_orient=None,
        transl=None,
        return_verts=True,
        return_full_pose=False,
        displacement=None,
        v_template=None,
        **kwargs,
    ):
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        betas: torch.tensor, optional, shape Bx10
            If given, ignore the member variable `betas` and use it
            instead. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None and hasattr(self, "transl"):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        if v_template is None:
            v_template = self.v_template

        if displacement is not None:
            vertices, joints_smpl, T_weighted, W, T = lbs(
                betas,
                full_pose,
                v_template + displacement,
                self.shapedirs,
                self.posedirs,
                self.J_regressor,
                self.parents,
                self.lbs_weights,
                dtype=self.dtype,
                pose_blend=self.pose_blend,
            )
        else:
            # T: transform mat of all joints wrt to root joint
            vertices, joints_smpl, T_weighted, W, T = lbs(
                betas,
                full_pose,
                v_template,
                self.shapedirs,
                self.posedirs,
                self.J_regressor,
                self.parents,
                self.lbs_weights,
                dtype=self.dtype,
                pose_blend=self.pose_blend,
            )

        # if self.gender is not 'neutral':
        joints = self.vertex_joint_selector(vertices, joints_smpl)
        # else:
        # joints = torch.matmul(vertices.permute(0,2,1),self.joint_regressor).permute(0,2,1)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints_smpl = joints_smpl + transl.unsqueeze(dim=1)
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = ModelOutput(
            vertices=vertices if return_verts else None,
            faces=self.faces,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints_smpl,
            betas=self.betas,
            full_pose=full_pose if return_full_pose else None,
            T=T,
            T_weighted=T_weighted,
            weights=W,
        )
        return output


class MANO(SMPL):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(
        self,
        model_path: str,
        is_rhand=True,
        data_struct=None,
        create_hand_pose=True,
        hand_pose=None,
        use_pca=True,
        num_pca_comps=6,
        flat_hand_mean=False,
        batch_size=1,
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed=True,
        ext: str = "pkl",
        pose_blend=True,
        # **kwargs
    ):
        """MANO model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_hand_pose: bool, optional
            Flag for creating a member variable for the pose of the right
            hand. (default = True)
        hand_pose: torch.tensor, optional, BxP
            The default value for the right hand pose member variable.
            (default = None)
        num_pca_comps: int, optional
            The number of PCA components to use for each hand.
            (default = 6)
        flat_hand_mean: bool, optional
            If False, then the pose of the hand is initialized to False.
        batch_size: int, optional
            The batch size used for creating the member variables
        dtype: torch.dtype, optional
            The data type for the created variables
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.num_pca_comps = num_pca_comps
        self.is_rhand = is_rhand
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = "MANO_{}.{ext}".format(
                    "RIGHT" if is_rhand else "LEFT", ext=ext
                )
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = (
                    True if "RIGHT" in os.path.basename(model_path) else False
                )
            assert osp.exists(mano_path), "Path {} does not exist!".format(mano_path)

            if ext == "pkl":
                with open(mano_path, "rb") as mano_file:
                    model_data = pickle.load(mano_file, encoding="latin1")
            elif ext == "npz":
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError("Unknown extension: {}".format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]

        super(MANO, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            batch_size=batch_size,
            vertex_ids=vertex_ids,
            use_compressed=use_compressed,
            dtype=dtype,
            ext=ext,
            pose_blend=pose_blend,
        )
        # , **kwargs)

        # add only MANO tips to the extra joints
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS["mano"].values()), dtype=torch.long
        )

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean

        hand_components = data_struct.hands_components[:num_pca_comps]

        self.np_hand_components = hand_components

        if self.use_pca:
            self.register_buffer(
                "hand_components", torch.tensor(hand_components, dtype=dtype)
            )

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean

        self.register_buffer("hand_mean", to_tensor(hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros(
                    [batch_size, hand_pose_dim], dtype=dtype
                )
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(default_hand_pose, requires_grad=True)
            self.register_parameter("hand_pose", hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(data_struct, flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = pose_mean.clone().to(dtype)
        # pose_mean_tensor = torch.tensor(pose_mean, dtype=dtype)
        self.register_buffer("pose_mean", pose_mean_tensor)

    def name(self):
        return "MANO"

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)
        return pose_mean

    def extra_repr(self):
        msg = [super(MANO, self).extra_repr()]
        if self.use_pca:
            msg.append(f"Number of PCA components: {self.num_pca_comps}")
        msg.append(f"Flat hand mean: {self.flat_hand_mean}")
        return "\n".join(msg)

    def forward(
        self,
        betas=None,
        global_orient=None,
        hand_pose=None,
        transl=None,
        return_verts=True,
        return_full_pose=False,
        displacement=None,
        v_template=None,
        # **kwargs
    ):
        """Forward pass for the MANO model"""
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        betas = betas if betas is not None else self.betas
        hand_pose = hand_pose if hand_pose is not None else self.hand_pose

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None:
            if hasattr(self, "transl"):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum("bi,ij->bj", [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean
        if v_template is None:
            v_template = self.v_template
        if displacement is not None:
            assert False, "Not implemented"
            vertices, joints, T_weighted, W, T = lbs(
                betas,
                full_pose,
                v_template + displacement,
                self.shapedirs,
                self.posedirs,
                self.J_regressor,
                self.parents,
                self.lbs_weights,
                dtype=self.dtype,
                pose_blend=self.pose_blend,
            )
        else:
            vertices, joints, T_weighted, W, T, v_posed = lbs(
                betas,
                full_pose,
                self.v_template,
                self.shapedirs,
                self.posedirs,
                self.J_regressor,
                self.parents,
                self.lbs_weights,
                dtype=self.dtype,
                pose_blend=self.pose_blend,
            )
        # # Add pre-selected extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices if return_verts else None,
            faces=self.faces,
            global_orient=global_orient,
            hand_pose=hand_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
            T=T,
            T_weighted=T_weighted,
            weights=W,
            v_posed=v_posed,
        )

        return output
