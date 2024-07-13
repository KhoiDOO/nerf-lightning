from omegaconf import OmegaConf
import torch

def compute_accumulated_transmittance(alphas: torch.Tensor):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def parse_structure(template, cfg):
    return OmegaConf.structured(template(**cfg))