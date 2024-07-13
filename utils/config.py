from omegaconf import OmegaConf
from dataclasses import dataclass, field
from datetime import datetime

import os

def time_string():
    return datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

OmegaConf.register_new_resolver("path_append", lambda a, b: os.path.join(a, b))
OmegaConf.register_new_resolver("get_trial_dir", lambda save_dir: os.path.join(os.getcwd(), save_dir, time_string()))
OmegaConf.register_new_resolver("get_run_id", lambda save_dir: save_dir.split('/')[-1])

@dataclass
class ExpCfg:
    name:str = 'default'
    save_dir:str = 'runs'
    trial_dir:str = None
    save_cfg_path:str = None
    seed:int = 0

    data:dict = field(default_factory=dict)

    system_type:str = 'TinyNerf'
    system:dict = field(default_factory=dict)

    train:dict = field(default_factory=dict)

    def __post_init__(self):
        print('[INFO]: Experiment Configured')
        os.makedirs(self.trial_dir, exist_ok=True)
        print(f'[INFO]: Experiment Directory is created at {self.trial_dir}')
        self.dump(self.save_cfg_path)
        print(f'[INFO]: Experiment YAML Config is saved at {self.save_cfg_path}')
    
    def dump(self, path:str):
        with open(path, "w") as fp:
            OmegaConf.save(config=self, f=fp)

def load_cfg(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    scfg = parse_structure(ExpCfg, cfg)
    print(f'[INFO]: Configuration: \n{OmegaConf.to_yaml(scfg)}')
    return scfg

def parse_structure(template, cfg):
    return OmegaConf.structured(template(**cfg))