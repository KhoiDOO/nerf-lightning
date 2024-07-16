from dataclasses import dataclass, field
from .func import parse_structure
from torch import nn
from typing import *
from .loss import MSE

import torch.nn.functional as F
import torch

@dataclass
class OptConfig:
    name:str = 'Adam'
    args:Dict = field(default_factory=dict) 

@dataclass
class LossConfig:
    name:str = 'mse'
    args:Dict = field(default_factory=dict)

def psnr(loss: torch.Tensor) -> float:
    if loss == 0.0:
        return -10.0 * torch.log10(torch.tensor(1e-06)).mean(dim=-1).item()
    return -10.0 * torch.log10(loss).mean(dim=-1).item()

def parse_loss(cfg: Dict)->nn.Module:
    cfg:LossConfig = parse_structure(OptConfig, cfg)
    if cfg.name == 'mse':
        return MSE(**cfg.args)

def parse_optimizer(cfg: Dict, model:nn.Module)->torch.optim.Optimizer:
    cfg:OptConfig = parse_structure(OptConfig, cfg)
    params = model.parameters()
    optim = getattr(torch.optim, cfg.name)(params, **cfg.args)
    return optim

def parse_scheduler(cfg: Dict, optimizer: torch.optim.Optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.name)(optimizer, **cfg.args)
    return lr_scheduler