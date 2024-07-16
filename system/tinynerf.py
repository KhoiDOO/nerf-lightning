from .utils import compute_accumulated_transmittance, psnr, savenormimg
from typing import Any, Dict, Mapping
from .base import BaseSystem
from torch import Tensor

import os
import torch
import numpy as np

class TinyNerf(BaseSystem):
    def __init__(self, cfg: Dict, *args: Any, **kwargs: Any) -> BaseSystem:
        super().__init__(cfg, *args, **kwargs)
    
    def forward(self, ray_origins, ray_directions, near, far, bins_count, *args: Any, **kwargs: Any) -> Tensor:
        device = ray_origins.device
    
        t = torch.linspace(near, far, bins_count, device=device).expand(ray_origins.shape[0], bins_count)
        # Perturb sampling along each ray.
        mid = (t[:, :-1] + t[:, 1:]) / 2.
        lower = torch.cat((t[:, :1], mid), -1)
        upper = torch.cat((mid, t[:, -1:]), -1)
        u = torch.rand(t.shape, device=device)
        t = lower + (upper - lower) * u  # [batch_size, nb_bins]
        delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

        # Compute the 3D points along each ray
        x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
        # Expand the ray_directions tensor to match the shape of x
        ray_directions = ray_directions.expand(bins_count, ray_directions.shape[0], 3).transpose(0, 1) 

        colors, sigma = self.model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
        colors = colors.reshape(x.shape)
        sigma = sigma.reshape(x.shape[:-1])

        alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
        weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
        # Compute the pixel values as a weighted sum of colors along each ray
        c = (weights * colors).sum(dim=1)
        weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
        return c + 1 - weight_sum.unsqueeze(-1)

    def training_step(self, batch:Tensor) -> Tensor:
        ray_origins = batch[:, :3]
        ray_directions = batch[:, 3:6]
        ground_truth_px_values = batch[:, 6:]
        
        preds = self(ray_origins, ray_directions, **self.args)
        loss = self.criterion(preds, ground_truth_px_values)
        snr = psnr(loss)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/psnr", snr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch:Tensor) -> Tensor:
        ray_origins = batch[:, :3]
        ray_directions = batch[:, 3:6]
        ground_truth_px_values = batch[:, 6:]
        
        preds = self(ray_origins, ray_directions, **self.args)
        loss = self.criterion(preds, ground_truth_px_values)
        snr = psnr(loss)

        self.log("valid/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid/psnr", snr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    # def test_step(self, batch, batch_idx):
    #     ray_origins = batch[:, :3]
    #     ray_directions = batch[:, 3:6]

    #     preds:Tensor = self(ray_origins, ray_directions, **self.args)

    #     self.pxs.append(preds[0].cpu())

    #     if (batch_idx + 1) % self.px_cnt == 0:
    #         index = batch_idx // self.px_cnt
    #         img = torch.cat(self.pxs).numpy().reshape(self.H, self.W, 3)
    #         savenormimg(filename=os.path.join(self.render_dir, f'{index}.png'), img=img)
    #         self.pxs = []

    def test_step(self, batch: Tensor):
        
        B = batch.size(dim=0)
        self.accumulated_batch_size += B
        
        ray_origins = batch[:, :3]
        ray_directions = batch[:, 3:6]

        preds:Tensor = self(ray_origins, ray_directions, **self.args)

        self.generated_pixels.append(preds.cpu())

        if self.accumulated_batch_size >= self.image_pixel_total:
            cat_pred = torch.cat(self.generated_pixels, dim=0)
            img = cat_pred[:self.image_pixel_total, :].numpy().reshape(self.H, self.W, 3)

            self.accumulated_batch_size -= self.image_pixel_total
            self.generated_pixels = [cat_pred[self.image_pixel_total:, :]] if self.accumulated_batch_size > 0 else []

            savenormimg(filename=os.path.join(self.render_dir, f'{self.inference_index}.png'), img=img)
            self.inference_index += 1