from omegaconf import OmegaConf
import torch
import imageio.v3 as iio
import numpy as np
import cv2

def compute_accumulated_transmittance(alphas: torch.Tensor):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def saveimg(filename:str, img:cv2.Mat):
    iio.imwrite(uri=filename, image=img)

def savenormimg(filename:str, img:cv2.Mat):
    img = (img * 255).astype(np.uint8)
    saveimg(filename=filename, img=img)

def parse_structure(template, cfg):
    return OmegaConf.structured(template(**cfg))