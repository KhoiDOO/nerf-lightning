from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import callbacks
from omegaconf import OmegaConf
from .config import parse_structure, ExpCfg
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import *


@dataclass
class TrainerConfig:
    devices:list = 0
    max_epochs:int = 1
    check_val_every_n_epoch:int = 1
    enable_progress_bar:bool = True
    accumulate_grad_batches:int = 1
    default_root_dir:str = None

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

@dataclass
class LoggerConfig:
    names:List[str] = None
    args:OrderedDict = field(default_factory=OrderedDict)

@dataclass
class CallbackConfig:
    names:List[str] = None
    args:OrderedDict = field(default_factory=OrderedDict)

def parse_logger(cfg:LoggerConfig):
    _loggers = []
    for name, (_, item) in zip(cfg.names, cfg.args.items()):
        logger = getattr(pl_loggers, name)(**item)
        _loggers.append(logger)
    return _loggers

def parse_callback(cfg:CallbackConfig):
    _callbacks = []
    for name, (_, item) in zip(cfg.names, cfg.args.items()):
        callback = getattr(callbacks, name)(**item)
        _callbacks.append(callback)
    return _callbacks

def traincfg_resolve(cfg: Dict):
    trainer_cfg:TrainerConfig = parse_structure(TrainerConfig, cfg.trainer)
    logger_cfg:LoggerConfig = parse_structure(LoggerConfig, cfg.logger)
    callback_cfg:CallbackConfig = parse_structure(CallbackConfig, cfg.callback)

    logger_lst = parse_logger(logger_cfg)
    callback_lst = parse_callback(callback_cfg)

    print(f'[INFO]: Loggers: {logger_lst}')
    print(f'[INFO]: Callbacks: {callback_lst}')

    return trainer_cfg, logger_lst, callback_lst