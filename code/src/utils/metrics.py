import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
import common.comet_utils as comet_utils


class Metrics(nn.Module):
    def __init__(self, experiment):
        super().__init__()
        # metrics to evaluate
        self.metrics = ["psnr"]

        self.eval_fns = {
            "psnr": self.evaluate_psnr,
        }
        self.metric_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.experiment = experiment

    def evaluate_psnr(self, preds, targets):
        # psnr
        pred_rgb = preds["rgb"]
        pred_rgb = pred_rgb
        gt_rgb = targets["gt.rgb"].view(-1, 3)
        psnr = self.metric_psnr(pred_rgb, gt_rgb)
        return psnr

    def forward(self, preds, targets, global_step, epoch):
        with torch.no_grad():
            metrics = {}
            for k in self.metrics:
                metrics["metrics/" + k] = self.eval_fns[k](preds, targets)
            comet_utils.log_dict(
                self.experiment, metrics, step=global_step, epoch=epoch
            )
            return metrics
