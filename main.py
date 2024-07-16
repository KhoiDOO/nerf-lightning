from pytorch_lightning import Trainer
from system import parse_system, BaseSystem
from data import RandomCameraModule
from utils import load_cfg, traincfg_resolve, ExpCfg, TrainerConfig
from typing import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='YAML Config file path')
    parser.add_argument('--mode', type=str, required=True, help='mode in experiment')

    args = parser.parse_args()

    cfg:ExpCfg = load_cfg(args.config)

    dm = RandomCameraModule(cfg=cfg.data)
    print(f'[INFO]: DataModule: {dm}')

    system: BaseSystem = parse_system(cfg.system_type)(cfg.system)
    system.set_save_dir(cfg.trial_dir)
    print(f'[INFO]: SystemModule: {system}')

    trainer_cfg, logger_lst, callback_lst = traincfg_resolve(cfg=cfg.train)
    trainer = Trainer(**trainer_cfg, logger=logger_lst, callbacks=callback_lst)

    if args.mode == 'train':
        trainer.fit(model=system, datamodule=dm)
        trainer.test(model=system, datamodule=dm)
    else:
        raise ValueError(f'Mode {args.mode} is currently not supported')