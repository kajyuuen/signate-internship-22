from src.nn.lightning_data_module.crowdfunding_dataset import CrowdfundingDataset
from typing import Optional

import pandas as pd
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class CrowdfundingDataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name: str,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 device: str = "cuda:0",
                 split_ratio: float = 0.8,
                 max_seq_len: int = 512):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.max_seq_len = max_seq_len
        self.train_df = train_df
        self.test_df = test_df
        self.device = device
        self.split_ratio = split_ratio
    
    def setup(self, stage: Optional[str]):
        labeled_dataset = CrowdfundingDataset(self.train_df, self.tokenizer, self.max_seq_len)
        test_dataset = CrowdfundingDataset(self.test_df, self.tokenizer, self.max_seq_len)

        n_samples = len(labeled_dataset)
        train_size = int(n_samples * self.split_ratio)
        valid_size = n_samples - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(labeled_dataset, [train_size, valid_size])
        self.train_dataset, self.valid_dataset, self.test_dataset = train_dataset, valid_dataset, test_dataset

    def train_dataloader(self):
        train_iter = DataLoader(self.train_dataset, batch_size=16, num_workers=4, shuffle=True)
        return train_iter

    def val_dataloader(self):
        valid_iter = DataLoader(self.valid_dataset, batch_size=16, num_workers=4, shuffle=False)
        return valid_iter

    def test_dataloader(self):
        test_iter = DataLoader(self.test_dataset, batch_size=16, num_workers=4, shuffle=False)
        return test_iter
