import numpy as np
import torch
from kaolin.ops.mesh import index_vertices_by_faces

import src.engine.volsdf_utils as volsdf_utils
import src.utils.debug as debug
from src.model.renderables.node import Node
from src.datasets.utils import get_camera_params
from common.body_models import seal_mano_v
from common.body_models import seal_mano_f
from src.utils.meshing import generate_mesh
from src.model.mano.deformer import MANODeformer
from src.model.mano.server import MANOServer
import src.hold.hold_utils as hold_utils


class MANONode(Node):
    def __init__(self, args, opt, betas, sdf_bounding_sphere, node_id):
        if node_id == "right":
            class_id = 2
            self.is_rhand = True
        elif node_id == "left":
            class_id = 3
            self.is_rhand = False
        else:
            assert False

        deformer = MANODeformer(max_dist=0.1, K=15, betas=betas, is_rhand=self.is_rhand)
        server = MANOServer(betas=betas, is_rhand=self.is_rhand)

        from src.model.mano.params import MANOParams
        from src.model.mano.specs import mano_specs

        params = MANOParams(
            args.n_images,
            {
                "betas": 10,
                "global_orient": 3,
                "transl": 3,
                "pose": 45,
            },
            node_id,
        )
        params.load_params(args.case)
        super(MANONode, self).__init__(
            args,
            opt,
            mano_specs,
            sdf_bounding_sphere,
            opt.implicit_network,
            opt.rendering_network,
            deformer,
            server,
            class_id,
            node_id,
            params,
        )

        self.mesh_v_cano = self.server.verts_c
        self.mesh_f_cano = torch.tensor(
            self.server.human_layer.faces.astype(np.int64)
        ).cuda()
        self.mesh_face_vertices = index_vertices_by_faces(
            self.mesh_v_cano, self.mesh_f_cano
        )

        self.mesh_v_cano_div = None
        self.mesh_f_cano_div = None
        self.canonical_mesh = None

    def sample_points(self, input):
        node_id = self.node_id
        full_pose = input[f"{node_id}.full_pose"]
        output = self.server(
            input[f"{node_id}.params"][:, 0],
            input[f"{node_id}.transl"],
            full_pose,
            input[f"{node_id}.betas"],
        )

        debug.debug_world2pix(self.args, output, input, self.node_id)
        cond = {"pose": full_pose[:, 3:] / np.pi}  # pose-dependent shape
        if self.training:
            if input["current_epoch"] < 20:
                cond = {"pose": full_pose[:, 3:] * 0.0}  # no pose for shape

        ray_dirs, cam_loc = get_camera_params(
            input["uv"], input["extrinsics"], input["intrinsics"]
        )
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        deform_info = {
            "cond": cond,
            "tfs": output["tfs"],
            "verts": output["verts"],
        }
        z_vals = self.ray_sampler.get_z_vals(
            volsdf_utils.sdf_func_with_deformer,
            self.deformer,
            self.implicit_network,
            ray_dirs,
            cam_loc,
            self.density,
            self.training,
            deform_info,
        )

        # fg samples to points
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        out = {}
        out["idx"] = input["idx"]
        out["output"] = output
        out["cond"] = cond
        out["ray_dirs"] = ray_dirs
        out["cam_loc"] = cam_loc
        out["deform_info"] = deform_info
        out["z_vals"] = z_vals
        out["points"] = points
        out["tfs"] = output["tfs"]
        out["batch_size"] = batch_size
        out["num_pixels"] = num_pixels
        return out

    def spawn_cano_mano(self, sample_dict_h):
        mesh_v_cano = sample_dict_h["output"]["v_posed"]
        mesh_vh_cano = seal_mano_v(mesh_v_cano)
        mesh_fh_cano = seal_mano_f(self.mesh_f_cano, self.is_rhand)

        mesh_vh_cano, mesh_fh_cano = hold_utils.subdivide_cano(
            mesh_vh_cano, mesh_fh_cano
        )
        self.mesh_v_cano_div = mesh_vh_cano
        self.mesh_f_cano_div = mesh_fh_cano

    def meshing_cano(self, pose=None):
        if pose is None:
            cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}
        else:
            cond = {"pose": pose / np.pi}
        assert cond["pose"].shape[0] == 1, "only support batch size 1"
        v_min_max = np.array([[-0.0814, -0.0280, -0.0742], [0.1171, 0.0349, 0.0971]])
        mesh_canonical = generate_mesh(
            lambda x: hold_utils.query_oc(self.implicit_network, x, cond),
            v_min_max,
            point_batch=10000,
            res_up=1,
            res_init=64,
        )
        return mesh_canonical
