import torch.utils
from torch.utils.data import Dataset, DataLoader, default_collate
from typing import *
from dataclasses import dataclass, field

import torch.utils.data
from utils import parse_structure

import pytorch_lightning as pl
import numpy as np
import torch


@dataclass
class DatasetConfig:
    data_source: str = 'data/src'
    train_path:str = None
    valid_path:str = None
    batch_size:int = 32
    shuffle:bool = True
    num_workers:int = 24
    # near:int = 2
    # far:int = 6
    # nb_bins:int = 192
    # eval_height:int = 400
    # eval_width:int = 400


# class RandomCameraDataset(Dataset):
#     def __init__(self, cfg: DatasetConfig, train: bool = True) -> None:
#         super().__init__()
        
#         self.cfg:DatasetConfig = cfg
#         self.train:bool = train
#         self.path:str = cfg.train_path if train else cfg.valid_path
#         self.data:torch.Tensor = torch.from_numpy(np.load(self.path, allow_pickle=True))

#     def __getitem__(self, index) -> Any:
#         return self.data[index]

#     def __len__(self):
#         return self.data.shape[0]

#     def collate(self, batch):
#         batch = torch.utils.data.default_collate(batch)
#         ray_origins = batch[:, :3]
#         ray_directions = batch[:, 3:6]
#         ground_truth_px_values = batch[:, 6:]

#         return {
#             'ray_o': ray_origins,
#             'ray_d': ray_directions,
#             'pxval': ground_truth_px_values,
#             'height': self.cfg.eval_height,
#             'width': self.cfg.eval_width,
#             'near': self.cfg.near,
#             'far': self.cfg.far,
#             'bins': self.cfg.nb_bins
#         }


# class RandomCameraModule(pl.LightningDataModule):
#     cfg: DatasetConfig

#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg:DatasetConfig = parse_structure(DatasetConfig, cfg)
    
#     def setup(self, stage=None) -> None:
#         if stage in [None, "fit"]:
#             self.train_dataset = RandomCameraDataset(self.cfg, train=True)
#         if stage in [None, "fit", "validate"]:
#             self.val_dataset = RandomCameraDataset(self.cfg, train=False)
#         if stage in [None, "test", "predict"]:
#             self.test_dataset = RandomCameraDataset(self.cfg, train=False)

#     def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
#         return DataLoader(dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn)

#     def train_dataloader(self) -> DataLoader:
#         return self.general_loader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.train_dataset.collate)

#     def val_dataloader(self) -> DataLoader:
#         return self.general_loader(self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate)

#     def test_dataloader(self) -> DataLoader:
#         return self.general_loader(self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate)

#     def predict_dataloader(self) -> DataLoader:
#         return self.general_loader(self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate)


class RandomCameraModule(pl.LightningDataModule):
    cfg: DatasetConfig

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg:DatasetConfig = parse_structure(DatasetConfig, cfg)
        self.train_path = cfg.train_path
        self.valid_path = cfg.valid_path

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = torch.from_numpy(np.load(self.train_path, allow_pickle=True))
        if stage in [None, "fit", "validate"]:
            self.val_dataset = torch.from_numpy(np.load(self.valid_path, allow_pickle=True))
        if stage in [None, "test", "predict"]:
            self.test_dataset = torch.from_numpy(np.load(self.valid_path, allow_pickle=True))

    def general_loader(self, dataset, batch_size) -> DataLoader:
        return DataLoader(dataset, num_workers=self.cfg.num_workers, batch_size=batch_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, num_workers=self.cfg.num_workers, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, num_workers=self.cfg.num_workers, batch_size=self.cfg.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, num_workers=self.cfg.num_workers, batch_size=self.cfg.batch_size)