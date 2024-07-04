import torch
from torch import nn
from PIL import Image


import src.hold.loss_terms as loss_terms


class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.milestone = 30000
        self.im_w = None
        self.im_h = None

    def forward(self, batch, model_outputs):
        device = batch["idx"].device

        # equal scoring for now
        image_scores = torch.ones(batch["idx"].shape).float().to(device)

        if self.im_w is None:
            im = Image.open(batch["im_path"][0][0])
            self.im_w = im.size[0]
            self.im_h = im.size[1]

        rgb_gt = batch["gt.rgb"].view(-1, 3).cuda()
        mask_gt = batch["gt.mask"].view(-1)

        valid_pix = torch.ones_like(mask_gt).float()

        nan_filter = ~torch.any(model_outputs["rgb"].isnan(), dim=1)
        rgb_loss = loss_terms.get_rgb_loss(
            model_outputs["rgb"][nan_filter],
            rgb_gt[nan_filter],
            valid_pix[nan_filter],
            image_scores,
        )

        sem_loss = loss_terms.get_sem_loss(
            model_outputs["semantics"], mask_gt, valid_pix, image_scores
        )

        opacity_sparse_loss = 0.0
        for key in model_outputs.search("index_off_surface").keys():
            node_id = key.split(".")[0]
            opacity_sparse_loss += loss_terms.get_opacity_sparse_loss(
                model_outputs[f"{node_id}.mask_prob"],
                model_outputs[f"{node_id}.index_off_surface"],
                image_scores,
            )

        eikonal_loss = 0.0
        for key in model_outputs.search("grad_theta").keys():
            eikonal_loss += loss_terms.get_eikonal_loss(model_outputs[key])

        # if "pts2mano_sdf_cano" in model_outputs:
        mano_cano_loss = 0.0
        for key in model_outputs.search("pts2mano_sdf_cano").keys():
            node_id = key.split(".")[0]
            gt_sdf = model_outputs[f"{node_id}.pts2mano_sdf_cano"].detach()
            pred_sdf = model_outputs[f"{node_id}.pred_sdf"]
            mano_cano_loss += loss_terms.get_mano_cano_loss(
                pred_sdf, gt_sdf, limit=0.01, scores=image_scores
            )

        progress = min(
            self.milestone, model_outputs["step"]
        )  # will not increase after the milestone

        # sem: [1.1, 0.1]
        # normal_l1: [0.05, 0.01]
        # normal_cos: [0.05, 0.005]

        w_sem = torch.linspace(1.1, 0.1, self.milestone + 1)[progress]
        w_sparse = torch.linspace(0.0, 1.0, self.milestone + 1)[progress]
        loss_dict = {
            "loss/rgb": rgb_loss * 1.0,
            "loss/sem": sem_loss * w_sem,
        }

        low_bnd_eikonal = 0.0004  # 400u
        low_bnd_eikonal = 0.0008  # 400u

        eikonal_loss *= 0.00001
        if eikonal_loss > low_bnd_eikonal:
            loss_dict["loss/eikonal"] = eikonal_loss

        loss_dict["loss/mano_cano"] = mano_cano_loss * 5.0
        loss_dict["loss/opacity_sparse"] = opacity_sparse_loss * w_sparse
        loss_dict["loss"] = sum([loss_dict[k] for k in loss_dict.keys()])
        return loss_dict
