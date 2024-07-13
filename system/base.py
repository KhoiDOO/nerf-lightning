from dataclasses import dataclass, field
from typing import *
from .utils import parse_optimizer, parse_scheduler, parse_loss
from utils import parse_structure
from torch import nn

import pytorch_lightning as pl
import torch
import model
import os


@dataclass
class SystemConfig:
    model_type:str = 'NerfModel'
    model:Dict = field(default_factory=dict)
    optimizer:Dict = field(default_factory=dict)
    scheduler:Dict = field(default_factory=dict)
    loss:Dict = field(default_factory=dict)
    args:Dict = field(default_factory=dict)


class BaseSystem(pl.LightningModule):
    def __init__(self, cfg: Dict, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg:SystemConfig = parse_structure(SystemConfig, cfg)
        self.model: nn.Module = getattr(model, self.cfg.model_type)(**self.cfg.model)
        self.criterion = parse_loss(self.cfg.loss)
        self.args = self.cfg.args
    
    def set_save_dir(self, path:str):
        self.save_dir = path
    
    def configure_optimizers(self):
        if self.model is None:
            raise ValueError(f'self.model is not initialized')
        optimizer = parse_optimizer(self.cfg.optimizer, self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": parse_scheduler(self.cfg.scheduler, optimizer)
            }
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        print('[INFO]: Experiment Started')
    
    def on_fit_end(self) -> None:
        super().on_fit_end()
        with open(os.path.join(self.save_dir, 'done.txt'), 'w') as file:
            file.close()