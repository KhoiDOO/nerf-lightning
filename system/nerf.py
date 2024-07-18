from .utils import compute_accumulated_transmittance, psnr, savenormimg
from typing import Any, Dict, Mapping
from .base import BaseSystem
from torch import nn, Tensor

import os
import torch
import numpy as np
import model

class Nerf(BaseSystem):
    def __init__(self, cfg: Dict, *args: Any, **kwargs: Any) -> BaseSystem:
        super().__init__(cfg, *args, **kwargs)
        self.coarse_model:nn.Module = getattr(model, self.cfg.model_type)(**self.cfg.model)
        